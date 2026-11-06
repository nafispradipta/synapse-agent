"""Graph and full-text storage: nodes, edges, and the lookups built on them."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional

from synapse.stores.constants import VALID_STATUSES, bounded_strength, bounded_unit


class GraphStore:
    """Nodes, edges, and full-text search over them."""

    def add_node(
        self,
        uri: str,
        l0_abstract: str,
        l1_overview: str = "",
        l2_target_path: Optional[str] = None,
        category: str = "general",
        importance: float = 0.5,
        strength: float = 1.0,
        node_id: Optional[str] = None,
        status: str = "active",
        salience: float = 0.5,
        confidence: float = 1.0,
        created_at: Optional[int] = None,
        access_count: int = 0,
    ) -> str:
        """
        Add a new cognitive node to the synapse graph.
        Returns the generated or provided node_id.
        """
        if self.allowed_categories is not None:
            if category not in self.allowed_categories:
                raise ValueError(
                    f"Invalid category '{category}'. Must be one of: {sorted(self.allowed_categories)}"
                )
        elif not category or not isinstance(category, str):
            raise ValueError("Category must be a non-empty string.")
        if status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. Must be one of: {sorted(VALID_STATUSES)}"
            )

        uid = node_id or uuid.uuid4().hex
        now = int(time.time())
        created_ts = int(created_at) if created_at is not None else now

        with self._write():
            self._conn.execute(
                """
                INSERT INTO synapse_nodes (
                    id, uri, l0_abstract, l1_overview, l2_target_path,
                    category, importance, strength, created_at, last_accessed, status,
                    salience, confidence, access_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uid,
                    uri,
                    l0_abstract,
                    l1_overview or "",
                    l2_target_path,
                    category,
                    bounded_unit(importance),
                    bounded_strength(strength),
                    created_ts,
                    created_ts,
                    status,
                    bounded_unit(salience),
                    bounded_unit(confidence),
                    int(access_count),
                ),
            )

            # Auto-vectorize and insert into synapse_vectors in the same transaction (384-dim unified)
            try:
                vec: Optional[List[float]] = None
                text_to_embed = f"{l0_abstract} {l1_overview or ''}".strip()
                if self.embedder is not None and text_to_embed:
                    if hasattr(self.embedder, "embed") and callable(self.embedder.embed):
                        vec = self.embedder.embed(text_to_embed)
                    elif hasattr(self.embedder, "vectorize") and callable(self.embedder.vectorize):
                        vec = self.embedder.vectorize(text_to_embed)
                    elif callable(self.embedder):
                        vec = self.embedder(text_to_embed)
                if vec is None and text_to_embed:
                    from synapse.retriever import vectorize_text
                    vec = vectorize_text(text_to_embed, dim=384)

                if vec:
                    model = None
                    if self.embedder is not None:
                        model = getattr(self.embedder, "model_name", None)
                    vec_json = json.dumps(vec)
                    self._conn.execute(
                        """
                        INSERT INTO synapse_vectors (id, vector_json, dim, updated_at, model)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            vector_json = excluded.vector_json,
                            dim = excluded.dim,
                            updated_at = excluded.updated_at,
                            model = excluded.model
                        """,
                        (uid, vec_json, len(vec), now, model),
                    )
            except Exception:
                pass

            self._conn.commit()
            self._invalidate_vector_cache()
        return uid

    def get_node(self, node_id_or_uri: str, touch: bool = False) -> Optional[Dict[str, Any]]:
        """
        Retrieve a node by its ID or URI.
        Optionally updates last_accessed timestamp when touch=True.
        """
        with self._write():
            cursor = self._conn.execute(
                "SELECT * FROM synapse_nodes WHERE id = ? OR uri = ?",
                (node_id_or_uri, node_id_or_uri),
            )
            row = cursor.fetchone()
            if row is None and isinstance(node_id_or_uri, str) and node_id_or_uri.startswith("synapse://"):
                rest = node_id_or_uri[len("synapse://"):]
                if rest.startswith("memory/"):
                    alt_uri = f"synapse://{rest[len('memory/'):]}"
                else:
                    alt_uri = f"synapse://memory/{rest}"
                cursor = self._conn.execute(
                    "SELECT * FROM synapse_nodes WHERE uri = ?", (alt_uri,)
                )
                row = cursor.fetchone()

            if row is None:
                return None

            node = dict(row)
            if touch:
                now = int(time.time())
                self._conn.execute(
                    "UPDATE synapse_nodes SET last_accessed = ?, access_count = COALESCE(access_count, 0) + 1 WHERE id = ?",
                    (now, node["id"]),
                )
                self._conn.commit()
                node["last_accessed"] = now
                node["access_count"] = (node.get("access_count") or 0) + 1

            return node

    def record_node_access(
        self,
        node_id: str,
        delta_salience: float = 0.02,
        delta_strength: float = 0.05,
        bump_edges: bool = True,
        delta_edge: float = 0.05,
    ) -> bool:
        """
        Hebbian Reinforcement hook: records node access/recall, increments access_count,
        strengthens node salience and strength, and optionally reinforces connected edges.
        """
        with self._write():
            now = int(time.time())
            try:
                self._conn.execute(
                    """
                    UPDATE synapse_nodes
                    SET last_accessed = ?,
                        access_count = COALESCE(access_count, 0) + 1,
                        salience = MIN(1.0, COALESCE(salience, 0.5) + ?),
                        strength = MIN(5.0, COALESCE(strength, 1.0) + ?)
                    WHERE id = ? OR uri = ?
                    """,
                    (now, delta_salience, delta_strength, node_id, node_id),
                )
                if bump_edges:
                    self._conn.execute(
                        """
                        UPDATE synapse_edges
                        SET weight = MIN(1.0, COALESCE(weight, 0.5) + ?),
                            last_fired = ?,
                            access_count = COALESCE(access_count, 0) + 1
                        WHERE source_id = ? OR target_id = ?
                        """,
                        (delta_edge, now, node_id, node_id),
                    )
                self._conn.commit()
                return True
            except Exception:
                self._rollback_if_outermost()
                return False

    def update_node(self, node_id: str, **kwargs: Any) -> bool:
        """
        Update fields of a synapse node by ID or URI.
        Returns True if a row was updated, False otherwise.
        """
        if not kwargs:
            return False

        allowed_fields = {
            "uri",
            "l0_abstract",
            "l1_overview",
            "l2_target_path",
            "category",
            "importance",
            "strength",
            "status",
            "created_at",
            "last_accessed",
            "salience",
            "confidence",
            "access_count",
        }

        updates: Dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in allowed_fields:
                if k == "category":
                    if self.allowed_categories is not None and v not in self.allowed_categories:
                        raise ValueError(f"Invalid category '{v}'. Allowed: {sorted(self.allowed_categories)}")
                    elif not v or not isinstance(v, str):
                        raise ValueError("Category must be a non-empty string.")
                if k == "status" and v not in VALID_STATUSES:
                    raise ValueError(f"Invalid status '{v}'. Allowed: {sorted(VALID_STATUSES)}")
                if k in ("importance", "salience", "confidence"):
                    updates[k] = bounded_unit(v)
                elif k == "strength":
                    updates[k] = bounded_strength(v)
                else:
                    updates[k] = v

        if not updates:
            return False

        if "last_accessed" not in updates:
            updates["last_accessed"] = int(time.time())

        set_clause = ", ".join(f"{col} = ?" for col in updates.keys())
        values = list(updates.values())
        values.extend([node_id, node_id])

        with self._write():
            cursor = self._conn.execute(
                f"UPDATE synapse_nodes SET {set_clause} WHERE id = ? OR uri = ?",
                values,
            )
            self._conn.commit()
            if "status" in updates:
                self._invalidate_vector_cache()
            if {"l0_abstract", "l1_overview"} & set(updates):
                self._refresh_or_drop_vector(node_id)
            return cursor.rowcount > 0

    def _refresh_or_drop_vector(self, node_id: str) -> None:
        """Re-embed when text changed so recall never ranks stale vectors.

        Runs after the UPDATE commits; the vector may briefly be stale, which
        is bounded and self-heals on the next write or cache rebuild. Falls
        back to deleting the row so ``list_nodes_missing_vectors`` can backfill.
        """
        node = self.get_node(node_id, touch=False)
        target_id = str(node["id"]) if node else node_id
        self._conn.execute(
            "DELETE FROM synapse_vectors WHERE id = ?",
            (target_id,),
        )
        self._conn.commit()

        embedder = self._get_embedder()
        if embedder is None or not node:
            self._invalidate_vector_cache()
            return

        text_to_embed = f"{node.get('l0_abstract') or ''} {node.get('l1_overview') or ''}".strip()
        vector = None
        model = getattr(embedder, "model_name", None)
        try:
            vector = embedder.embed(text_to_embed)
        except Exception:
            vector = None
        if vector:
            self.upsert_vector(target_id, vector, model=model)
        else:
            self._invalidate_vector_cache()

    def delete_node(self, node_id: str) -> bool:
        """
        Delete a node by ID or URI.
        Triggers cascade deletion of associated edges and FTS index entries.
        Returns True if deleted, False otherwise.
        """
        with self._write():
            if self._incremental_cache and self._vector_cache_valid:
                row = self._conn.execute(
                    "SELECT id FROM synapse_nodes WHERE id = ? OR uri = ?", (node_id, node_id)
                ).fetchone()
                if row:
                    self._incremental_delete(row["id"])
            else:
                self._invalidate_vector_cache()

            cursor = self._conn.execute(
                "DELETE FROM synapse_nodes WHERE id = ? OR uri = ?",
                (node_id, node_id),
            )
            self._conn.commit()
            if not self._incremental_cache:
                self._invalidate_vector_cache()
            return cursor.rowcount > 0

    def supersede_node(self, node_id: str, superseded_by: Optional[str] = None) -> bool:
        """
        Mark a node as superseded and reduce its strength to 0.05.
        Optionally link with 'supersedes' and 'superseded_by' edges.
        """
        updated = self.update_node(node_id, status="superseded", strength=0.05)
        if updated and superseded_by:
            try:
                self.add_edge(superseded_by, node_id, relation="supersedes", weight=1.0)
                self.add_edge(node_id, superseded_by, relation="superseded_by", weight=1.0)
            except Exception:
                pass
        return updated

    def get_active_nodes(self, category: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """List active nodes, optionally filtered by category."""
        return self.list_nodes(category=category, status="active", limit=limit)

    def list_nodes(
        self,
        category: Optional[str] = None,
        status: Optional[str] = "active",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List nodes matching optional category and status filters.
        """
        clauses = []
        params: List[Any] = []

        if category is not None:
            clauses.append("category = ?")
            params.append(category)

        if status is not None:
            clauses.append("status = ?")
            params.append(status)

        where_str = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT * FROM synapse_nodes
            {where_str}
            ORDER BY last_accessed DESC, created_at DESC
            LIMIT ?
        """
        params.append(max(1, int(limit)))

        with self._write():
            cursor = self._conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 0.5,
        increment_access: bool = False,
    ) -> bool:
        """
        Create or update a directed edge between two synapse nodes.
        Returns True on success, False if foreign key constraint fails.
        """
        now = int(time.time())
        with self._write():
            try:
                self._conn.execute(
                    """
                    INSERT INTO synapse_edges (source_id, target_id, relation, weight, last_fired, access_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_id, target_id, relation) DO UPDATE SET
                        weight = excluded.weight,
                        last_fired = excluded.last_fired,
                        access_count = COALESCE(synapse_edges.access_count, 0) + (CASE WHEN ? THEN 1 ELSE 0 END)
                    """,
                    (source_id, target_id, relation, float(weight), now, 1 if increment_access else 0, increment_access),
                )
                self._conn.commit()
                return True
            except sqlite3.IntegrityError:
                self._rollback_if_outermost()
                return False

    def get_edges(self, node_id: str, direction: str = "both") -> List[Dict[str, Any]]:
        """
        Retrieve edges connected to a node.
        direction: 'out' (outgoing), 'in' (incoming), or 'both'.
        """
        if direction not in ("out", "in", "both"):
            raise ValueError(f"Invalid direction '{direction}'. Must be 'out', 'in', or 'both'.")

        with self._write():
            if direction == "out":
                cursor = self._conn.execute(
                    "SELECT * FROM synapse_edges WHERE source_id = ? ORDER BY weight DESC",
                    (node_id,),
                )
            elif direction == "in":
                cursor = self._conn.execute(
                    "SELECT * FROM synapse_edges WHERE target_id = ? ORDER BY weight DESC",
                    (node_id,),
                )
            else:
                cursor = self._conn.execute(
                    "SELECT * FROM synapse_edges WHERE source_id = ? OR target_id = ? ORDER BY weight DESC",
                    (node_id, node_id),
                )

            return [dict(row) for row in cursor.fetchall()]

    def prune_decayed_edges(
        self,
        decay_rate: float = 0.05,
        prune_threshold: float = 0.15,
        idle_days_threshold: float = 14.0,
    ) -> Dict[str, int]:
        """
        Apply exponential decay to dormant edge weights (> idle_days_threshold days since last_fired)
        and prune edges below prune_threshold, protecting edges connected to high-salience nodes (>= 0.90).
        """
        now = int(time.time())
        cutoff_seconds = int(idle_days_threshold * 86400)
        cutoff_ts = now - cutoff_seconds

        with self._write():
            cursor = self._conn.execute(
                """
                SELECT source_id, target_id, relation, COALESCE(weight, 0.5), COALESCE(last_fired, 0)
                FROM synapse_edges
                WHERE (last_fired IS NULL OR last_fired < ?)
                  AND source_id NOT IN (SELECT id FROM synapse_nodes WHERE salience >= 0.90)
                  AND target_id NOT IN (SELECT id FROM synapse_nodes WHERE salience >= 0.90)
                """,
                (cutoff_ts,),
            )
            rows = cursor.fetchall()
            decayed = 0
            pruned = 0

            for row in rows:
                s_id, t_id, rel, weight, last_f = row[0], row[1], row[2], float(row[3]), row[4]
                last_ts = last_f or (now - cutoff_seconds)
                elapsed_days = max(0.0, (now - last_ts) / 86400.0)
                new_weight = weight * math.exp(-decay_rate * elapsed_days)

                if new_weight < prune_threshold:
                    self._conn.execute(
                        "DELETE FROM synapse_edges WHERE source_id = ? AND target_id = ? AND relation = ?",
                        (s_id, t_id, rel),
                    )
                    pruned += 1
                else:
                    self._conn.execute(
                        "UPDATE synapse_edges SET weight = ? WHERE source_id = ? AND target_id = ? AND relation = ?",
                        (round(new_weight, 4), s_id, t_id, rel),
                    )
                    decayed += 1

            self._conn.commit()
            return {"processed": len(rows), "decayed": decayed, "pruned": pruned}

    def _sanitize_fts_query(self, raw_query: str) -> str:
        """
        Sanitize user search query for FTS5 syntax safety while enabling prefix search.
        """
        tokens = re.findall(r"[\w\-]+", raw_query)
        if not tokens:
            return ""
        return " ".join(f'"{t}"*' for t in tokens)

    def search_fts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Execute full-text search against synapse_fts indexed fields
        (l0_abstract, l1_overview) ranked by BM25 relevance score.
        """
        if not query or not query.strip():
            return []

        tokens = re.findall(r"[\w\-]+", query)
        if not tokens:
            return []

        formatted_query = self._sanitize_fts_query(query)
        if not formatted_query:
            return []

        with self._write():
            try:
                cursor = self._conn.execute(
                    """
                    SELECT n.*, f.rank
                    FROM synapse_fts f
                    JOIN synapse_nodes n ON n.id = f.id
                    WHERE synapse_fts MATCH ?
                    ORDER BY f.rank ASC
                    LIMIT ?
                    """,
                    (formatted_query, max(1, int(limit))),
                )
                results = [dict(row) for row in cursor.fetchall()]
                if not results and len(tokens) > 1:
                    or_query = " OR ".join(f'"{t}"*' for t in tokens)
                    cursor = self._conn.execute(
                        """
                        SELECT n.*, f.rank
                        FROM synapse_fts f
                        JOIN synapse_nodes n ON n.id = f.id
                        WHERE synapse_fts MATCH ?
                        ORDER BY f.rank ASC
                        LIMIT ?
                        """,
                        (or_query, max(1, int(limit))),
                    )
                    results = [dict(row) for row in cursor.fetchall()]
                return results
            except sqlite3.OperationalError:
                # Fallback to plain quoted phrase if complex query fails
                cursor = self._conn.execute(
                    """
                    SELECT n.*, f.rank
                    FROM synapse_fts f
                    JOIN synapse_nodes n ON n.id = f.id
                    WHERE synapse_fts MATCH ?
                    ORDER BY f.rank ASC
                    LIMIT ?
                    """,
                    (f'"{query}"', max(1, int(limit))),
                )
                return [dict(row) for row in cursor.fetchall()]

    def resolve_wikilink_node(
        self,
        raw: str,
        clean: str,
        clean_slug: str,
        clean_stem: str,
        stem: str,
    ) -> Optional[Dict[str, Any]]:
        """Resolve a wikilink string to an active node via uri or vault path."""
        with self._write():
            cursor = self._conn.execute(
                """
                SELECT * FROM synapse_nodes
                WHERE status = 'active' AND (
                    id = ?
                    OR uri = ?
                    OR uri = ?
                    OR uri = ?
                    OR l2_target_path = ?
                    OR l2_target_path = ?
                    OR l2_target_path LIKE ?
                    OR l2_target_path LIKE ?
                    OR l2_target_path LIKE ?
                    OR uri LIKE ?
                    OR uri LIKE ?
                    OR uri LIKE ?
                )
                LIMIT 1
                """,
                (
                    raw,
                    f"vault:{clean}",
                    f"synapse://sop/{clean_slug}",
                    f"synapse://sop/{clean_stem}",
                    raw,
                    f"{clean}.md",
                    f"%/{stem}.md",
                    f"%/{clean}.md",
                    f"%/{clean}",
                    f"%/{clean_slug}",
                    f"%/{clean_stem}",
                    f"%:{clean}",
                ),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def find_active_node_by_l2_paths(self, paths: List[str]) -> Optional[Dict[str, Any]]:
        """Return the first active node whose l2_target_path matches any candidate."""
        candidates = [str(p) for p in paths if p]
        if not candidates:
            return None
        clause = " OR ".join("l2_target_path = ?" for _ in candidates)
        with self._write():
            cursor = self._conn.execute(
                f"SELECT * FROM synapse_nodes WHERE ({clause}) AND status = 'active' LIMIT 1",
                tuple(candidates),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_active_vault_targets(self) -> List[Dict[str, Any]]:
        """Active nodes that carry a vault path, for wikilink index building."""
        with self._write():
            cursor = self._conn.execute(
                """
                SELECT id, uri, l2_target_path FROM synapse_nodes
                WHERE l2_target_path IS NOT NULL AND status = 'active'
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def find_node_by_episode_ref(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Find a node previously consolidated from the given episode."""
        episode_id = str(episode_id)
        with self._write():
            cursor = self._conn.execute(
                """
                SELECT id, uri FROM synapse_nodes
                WHERE l2_target_path = ? OR uri LIKE ? OR uri LIKE ?
                LIMIT 1
                """,
                (
                    f"episode:{episode_id}",
                    f"%://%/ep_{episode_id}%",
                    f"%://%/episode_{episode_id}%",
                ),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_node_by_l2_target_path(self, path: str) -> Optional[Dict[str, Any]]:
        """Look up a node by its exact l2_target_path."""
        with self._write():
            cursor = self._conn.execute(
                "SELECT id, uri FROM synapse_nodes WHERE l2_target_path = ? LIMIT 1",
                (str(path),),
            )
            row = cursor.fetchone()
            return dict(row) if row else None
