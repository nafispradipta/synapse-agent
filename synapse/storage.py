"""
SynapseStorage - Persistent Graph & FTS Storage Engine for Cognitive Synapse Core.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    _NUMPY_AVAILABLE = False

VALID_CATEGORIES: Set[str] = {
    "preference",
    "decision",
    "architecture",
    "incident",
    "sop",
    "general",
}

VALID_STATUSES: Set[str] = {
    "active",
    "superseded",
    "archived",
    "flagged",
}


class SynapseStorage:
    """
    SQLite-backed graph and full-text search storage engine for the
    Synapse Memory system.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        embedder: Optional[Any] = None,
        allowed_categories: Optional[Union[Set[str], List[str]]] = None,
        allow_custom_categories: bool = False,
    ) -> None:
        if allow_custom_categories:
            self.allowed_categories: Optional[Set[str]] = set(allowed_categories) if allowed_categories else None
        else:
            self.allowed_categories = set(allowed_categories) if allowed_categories is not None else set(VALID_CATEGORIES)
        if db_path is None:
            self.is_memory = False
            raw_path = os.environ.get("SYNAPSE_DB_PATH") or str(Path.home() / ".synapse" / "synapse.db")
            self.db_path = str(Path(raw_path).expanduser().resolve())
        elif str(db_path) == ":memory:":
            self.is_memory = True
            self.db_path = ":memory:"
        else:
            self.is_memory = False
            self.db_path = str(Path(db_path).expanduser().resolve())

        if not self.is_memory:
            db_dir = os.path.dirname(self.db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        self._configure_pragmas()
        self._init_schema()

        # Embedder & Vectorizer configuration
        self.embedder = embedder
        self._embedder_instance: Optional[Any] = None

        # Vector matrix acceleration & cache structures
        self.use_numpy: bool = _NUMPY_AVAILABLE
        self._vector_matrix: Optional[Any] = None
        self._raw_vector_matrix: Optional[Any] = None
        self._vector_node_ids: List[str] = []
        self._matrix_by_dim: Dict[int, Any] = {}
        self._vector_cache_valid: bool = False
        self._incremental_cache: bool = False
        self._cached_vector_version: int = -1

    def _configure_pragmas(self) -> None:
        """Apply performance and integrity pragmas."""
        with self._lock:
            if not self.is_memory:
                self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")

    def _init_schema(self) -> None:
        """Create tables, indexes, FTS5 virtual table, and sync triggers."""
        with self._lock:
            cursor = self._conn.cursor()

            # 1. Synapse Nodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS synapse_nodes (
                    id TEXT PRIMARY KEY,
                    uri TEXT UNIQUE NOT NULL,
                    l0_abstract TEXT NOT NULL,
                    l1_overview TEXT,
                    l2_target_path TEXT,
                    category TEXT DEFAULT 'general',
                    importance REAL DEFAULT 0.5,
                    strength REAL DEFAULT 1.0,
                    created_at INTEGER,
                    last_accessed INTEGER,
                    status TEXT DEFAULT 'active' CHECK(status IN ('active', 'superseded', 'archived', 'flagged')),
                    salience REAL DEFAULT 0.5,
                    confidence REAL DEFAULT 1.0,
                    access_count INTEGER DEFAULT 0
                );
            """)

            # Auto-migrate synapse_nodes if columns are missing in existing databases
            cursor.execute("PRAGMA table_info(synapse_nodes);")
            existing_cols = {row[1] for row in cursor.fetchall()}
            if "salience" not in existing_cols:
                cursor.execute("ALTER TABLE synapse_nodes ADD COLUMN salience REAL DEFAULT 0.5;")
            if "confidence" not in existing_cols:
                cursor.execute("ALTER TABLE synapse_nodes ADD COLUMN confidence REAL DEFAULT 1.0;")
            if "access_count" not in existing_cols:
                cursor.execute("ALTER TABLE synapse_nodes ADD COLUMN access_count INTEGER DEFAULT 0;")

            # 1b. Synapse Superseded Archive (Abstract Rollup historical storage)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS synapse_superseded_archive (
                    id TEXT PRIMARY KEY,
                    uri TEXT,
                    l0_abstract TEXT,
                    l1_overview TEXT,
                    l2_target_path TEXT,
                    category TEXT,
                    importance REAL,
                    strength REAL,
                    created_at INTEGER,
                    last_accessed INTEGER,
                    salience REAL,
                    confidence REAL,
                    access_count INTEGER,
                    archived_at INTEGER,
                    rollup_id TEXT
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_superseded_archive_cat ON synapse_superseded_archive(category);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_superseded_archive_rollup ON synapse_superseded_archive(rollup_id);")

            # Auto-migrate synapse_nodes to remove category check constraint and ensure 'flagged' status
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='synapse_nodes';")
            node_table_row = cursor.fetchone()
            if node_table_row and node_table_row[0] and ("CHECK(category IN" in node_table_row[0] or "'flagged'" not in node_table_row[0]):
                cursor.execute("PRAGMA legacy_alter_table=ON;")
                cursor.execute("PRAGMA foreign_keys=OFF;")
                cursor.execute("ALTER TABLE synapse_nodes RENAME TO _synapse_nodes_old;")
                cursor.execute("""
                    CREATE TABLE synapse_nodes (
                        id TEXT PRIMARY KEY,
                        uri TEXT UNIQUE NOT NULL,
                        l0_abstract TEXT NOT NULL,
                        l1_overview TEXT,
                        l2_target_path TEXT,
                        category TEXT DEFAULT 'general',
                        importance REAL DEFAULT 0.5,
                        strength REAL DEFAULT 1.0,
                        created_at INTEGER,
                        last_accessed INTEGER,
                        status TEXT DEFAULT 'active' CHECK(status IN ('active', 'superseded', 'archived', 'flagged')),
                        salience REAL DEFAULT 0.5,
                        confidence REAL DEFAULT 1.0
                    );
                """)
                cursor.execute("""
                    INSERT INTO synapse_nodes (
                        id, uri, l0_abstract, l1_overview, l2_target_path,
                        category, importance, strength, created_at, last_accessed,
                        status, salience, confidence
                    )
                    SELECT
                        id, uri, l0_abstract, l1_overview, l2_target_path,
                        category, importance, strength, created_at, last_accessed,
                        status, salience, confidence
                    FROM _synapse_nodes_old;
                """)
                cursor.execute("DROP TABLE _synapse_nodes_old;")
                cursor.execute("PRAGMA foreign_keys=ON;")
                cursor.execute("PRAGMA legacy_alter_table=OFF;")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synapse_nodes_category ON synapse_nodes(category);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synapse_nodes_status ON synapse_nodes(status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synapse_nodes_last_accessed ON synapse_nodes(last_accessed);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synapse_nodes_salience ON synapse_nodes(salience);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synapse_nodes_access_count ON synapse_nodes(access_count);")

            # 2. Synapse Edges table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS synapse_edges (
                    source_id TEXT,
                    target_id TEXT,
                    relation TEXT NOT NULL,
                    weight REAL DEFAULT 0.5,
                    last_fired INTEGER,
                    access_count INTEGER DEFAULT 0,
                    PRIMARY KEY (source_id, target_id, relation),
                    FOREIGN KEY(source_id) REFERENCES synapse_nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY(target_id) REFERENCES synapse_nodes(id) ON DELETE CASCADE
                );
            """)

            cursor.execute("PRAGMA table_info(synapse_edges);")
            existing_edge_cols = {row[1] for row in cursor.fetchall()}
            if "access_count" not in existing_edge_cols:
                cursor.execute("ALTER TABLE synapse_edges ADD COLUMN access_count INTEGER DEFAULT 0;")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synapse_edges_target ON synapse_edges(target_id);")

            # 3. FTS5 Virtual Table for full-text search (Nodes)
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS synapse_fts USING fts5(
                    id UNINDEXED,
                    l0_abstract,
                    l1_overview,
                    tokenize = 'porter unicode61'
                );
            """)

            # 4. Triggers to keep synapse_fts synchronized with synapse_nodes
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS synapse_nodes_ai AFTER INSERT ON synapse_nodes BEGIN
                    INSERT INTO synapse_fts (id, l0_abstract, l1_overview)
                    VALUES (new.id, new.l0_abstract, coalesce(new.l1_overview, ''));
                END;
            """)

            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS synapse_nodes_ad AFTER DELETE ON synapse_nodes BEGIN
                    DELETE FROM synapse_fts WHERE id = old.id;
                END;
            """)

            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS synapse_nodes_au AFTER UPDATE ON synapse_nodes BEGIN
                    DELETE FROM synapse_fts WHERE id = old.id;
                    INSERT INTO synapse_fts (id, l0_abstract, l1_overview)
                    VALUES (new.id, new.l0_abstract, coalesce(new.l1_overview, ''));
                END;
            """)

            # 5. Episodes Table (Episodic Memory Cognitive Architecture)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    situation TEXT,
                    action_taken TEXT,
                    outcome TEXT,
                    lessons_learned TEXT,
                    salience REAL DEFAULT 0.5,
                    confidence REAL DEFAULT 1.0,
                    created_at INTEGER,
                    last_accessed INTEGER,
                    tags TEXT
                );
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodes_last_accessed ON episodes(last_accessed);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodes_salience ON episodes(salience);")

            # 6. FTS5 Virtual Table for Episodes
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
                    id UNINDEXED,
                    title,
                    situation,
                    action_taken,
                    outcome,
                    lessons_learned,
                    tags,
                    tokenize = 'porter unicode61'
                );
            """)

            # 7. Triggers to keep episodes_fts synchronized with episodes
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
                    INSERT INTO episodes_fts (id, title, situation, action_taken, outcome, lessons_learned, tags)
                    VALUES (
                        new.id,
                        coalesce(new.title, ''),
                        coalesce(new.situation, ''),
                        coalesce(new.action_taken, ''),
                        coalesce(new.outcome, ''),
                        coalesce(new.lessons_learned, ''),
                        coalesce(new.tags, '')
                    );
                END;
            """)

            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
                    DELETE FROM episodes_fts WHERE id = old.id;
                END;
            """)

            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE ON episodes BEGIN
                    DELETE FROM episodes_fts WHERE id = old.id;
                    INSERT INTO episodes_fts (id, title, situation, action_taken, outcome, lessons_learned, tags)
                    VALUES (
                        new.id,
                        coalesce(new.title, ''),
                        coalesce(new.situation, ''),
                        coalesce(new.action_taken, ''),
                        coalesce(new.outcome, ''),
                        coalesce(new.lessons_learned, ''),
                        coalesce(new.tags, '')
                    );
                END;
            """)

            # 8. Synapse Vectors table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS synapse_vectors (
                    id TEXT PRIMARY KEY,
                    vector_json TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    updated_at INTEGER,
                    FOREIGN KEY(id) REFERENCES synapse_nodes(id) ON DELETE CASCADE
                );
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synapse_vectors_updated_at ON synapse_vectors(updated_at);")

            # 8b. Vector cache metadata table and change-tracking triggers
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS _synapse_vector_meta (
                    id INTEGER PRIMARY KEY,
                    version INTEGER
                );
            """)
            cursor.execute("INSERT OR IGNORE INTO _synapse_vector_meta (id, version) VALUES (1, 0);")

            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS synapse_vectors_version_ai AFTER INSERT ON synapse_vectors BEGIN
                    UPDATE _synapse_vector_meta SET version = version + 1 WHERE id = 1;
                END;
            """)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS synapse_vectors_version_au AFTER UPDATE ON synapse_vectors BEGIN
                    UPDATE _synapse_vector_meta SET version = version + 1 WHERE id = 1;
                END;
            """)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS synapse_vectors_version_ad AFTER DELETE ON synapse_vectors BEGIN
                    UPDATE _synapse_vector_meta SET version = version + 1 WHERE id = 1;
                END;
            """)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS synapse_nodes_version_ad AFTER DELETE ON synapse_nodes BEGIN
                    UPDATE _synapse_vector_meta SET version = version + 1 WHERE id = 1;
                END;
            """)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS synapse_nodes_version_au_status AFTER UPDATE OF status ON synapse_nodes BEGIN
                    UPDATE _synapse_vector_meta SET version = version + 1 WHERE id = 1;
                END;
            """)

            # 9. Synapse Mental Models table (Hierarchical Consolidated Understanding)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS synapse_mental_models (\n                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source_query TEXT NOT NULL,
                    content TEXT NOT NULL,
                    structured_content TEXT,
                    trigger_mode TEXT DEFAULT 'delta' CHECK(trigger_mode IN ('full', 'delta')),
                    tags TEXT,
                    max_tokens INTEGER DEFAULT 2048,
                    last_refreshed_at INTEGER,
                    last_refreshed_source_query TEXT,
                    history TEXT,
                    created_at INTEGER
                );
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synapse_mental_models_last_refreshed ON synapse_mental_models(last_refreshed_at);")

            # 10. Synapse Observations table (Tier 2 Evidence-Grounded Pattern Consolidation)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS synapse_observations (
                    id TEXT PRIMARY KEY,
                    pattern TEXT NOT NULL,
                    supporting_facts TEXT NOT NULL,
                    proof_count INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 1.0,
                    status TEXT DEFAULT 'active' CHECK(status IN ('active', 'contradicted', 'superseded')),
                    category TEXT,
                    created_at INTEGER,
                    updated_at INTEGER
                );
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synapse_observations_status ON synapse_observations(status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synapse_observations_proof_count ON synapse_observations(proof_count);")

            # 11. Synapse External Research table (Tier 4 Sandboxed Autonomous Research)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS synapse_external_research (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    raw_snippet TEXT,
                    tier_level INTEGER DEFAULT 4,
                    vault_path TEXT,
                    fetched_at INTEGER,
                    status TEXT DEFAULT 'quarantined' CHECK(status IN ('quarantined', 'verified', 'archived', 'rejected'))
                );
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synapse_external_research_topic ON synapse_external_research(topic);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synapse_external_research_status ON synapse_external_research(status);")

            self._conn.commit()

    def _get_embedder(self) -> Any:
        """Lazily initialize or return configured embedder instance (384-dim)."""
        if self.embedder is not None:
            return self.embedder
        if self._embedder_instance is None:
            try:
                from synapse.embedder import SynapseEmbedder
                self._embedder_instance = SynapseEmbedder(dim=384)
            except Exception:
                pass
        return self._embedder_instance

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

        with self._lock:
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
                    float(importance),
                    float(strength),
                    created_ts,
                    created_ts,
                    status,
                    float(salience),
                    float(confidence),
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
                    vec_json = json.dumps(vec)
                    self._conn.execute(
                        """
                        INSERT INTO synapse_vectors (id, vector_json, dim, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            vector_json = excluded.vector_json,
                            dim = excluded.dim,
                            updated_at = excluded.updated_at
                        """,
                        (uid, vec_json, len(vec), now),
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
        with self._lock:
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
        with self._lock:
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
                updates[k] = v

        if not updates:
            return False

        if "last_accessed" not in updates:
            updates["last_accessed"] = int(time.time())

        set_clause = ", ".join(f"{col} = ?" for col in updates.keys())
        values = list(updates.values())
        values.extend([node_id, node_id])

        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE synapse_nodes SET {set_clause} WHERE id = ? OR uri = ?",
                values,
            )
            self._conn.commit()
            if "status" in updates:
                self._invalidate_vector_cache()
            return cursor.rowcount > 0

    def delete_node(self, node_id: str) -> bool:
        """
        Delete a node by ID or URI.
        Triggers cascade deletion of associated edges and FTS index entries.
        Returns True if deleted, False otherwise.
        """
        with self._lock:
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

        with self._lock:
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
        with self._lock:
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
                return False

    def get_edges(self, node_id: str, direction: str = "both") -> List[Dict[str, Any]]:
        """
        Retrieve edges connected to a node.
        direction: 'out' (outgoing), 'in' (incoming), or 'both'.
        """
        if direction not in ("out", "in", "both"):
            raise ValueError(f"Invalid direction '{direction}'. Must be 'out', 'in', or 'both'.")

        with self._lock:
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

        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT source_id, target_id, relation, weight, last_fired
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

        with self._lock:
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

    # -------------------------------------------------------------------------
    # Episodic Memory Operations (Cognitive Synapse Architecture)
    # -------------------------------------------------------------------------

    def add_episode(
        self,
        title: str,
        situation: str = "",
        action_taken: str = "",
        outcome: str = "",
        lessons_learned: str = "",
        salience: float = 0.5,
        confidence: float = 1.0,
        tags: Optional[Union[str, List[str]]] = None,
        episode_id: Optional[str] = None,
    ) -> str:
        """
        Add a new structured episodic memory (Situation -> Action -> Outcome -> Lesson).
        Returns the generated or provided episode_id.
        """
        uid = episode_id or uuid.uuid4().hex
        now = int(time.time())

        tags_str = ""
        if tags is not None:
            if isinstance(tags, list):
                tags_str = json.dumps(tags)
            else:
                tags_str = str(tags).strip()

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO episodes (
                    id, title, situation, action_taken, outcome, lessons_learned,
                    salience, confidence, created_at, last_accessed, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uid,
                    title,
                    situation or "",
                    action_taken or "",
                    outcome or "",
                    lessons_learned or "",
                    float(salience),
                    float(confidence),
                    now,
                    now,
                    tags_str,
                ),
            )
            self._conn.commit()
        return uid

    def get_episode(self, episode_id: str, touch: bool = False) -> Optional[Dict[str, Any]]:
        """
        Retrieve an episodic memory by ID.
        Optionally updates last_accessed timestamp when touch=True.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM episodes WHERE id = ?",
                (episode_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            ep = dict(row)
            if touch:
                now = int(time.time())
                self._conn.execute(
                    "UPDATE episodes SET last_accessed = ? WHERE id = ?",
                    (now, ep["id"]),
                )
                self._conn.commit()
                ep["last_accessed"] = now

            return ep

    def update_episode(self, episode_id: str, **kwargs: Any) -> bool:
        """
        Update fields of an episodic memory by ID.
        Returns True if a row was updated, False otherwise.
        """
        if not kwargs:
            return False

        allowed_fields = {
            "title",
            "situation",
            "action_taken",
            "outcome",
            "lessons_learned",
            "salience",
            "confidence",
            "tags",
            "created_at",
            "last_accessed",
        }

        updates: Dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in allowed_fields:
                if k == "tags" and isinstance(v, list):
                    updates[k] = json.dumps(v)
                elif k in ("salience", "confidence"):
                    updates[k] = float(v)
                else:
                    updates[k] = v

        if not updates:
            return False

        if "last_accessed" not in updates:
            updates["last_accessed"] = int(time.time())

        set_clause = ", ".join(f"{col} = ?" for col in updates.keys())
        values = list(updates.values())
        values.append(episode_id)

        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE episodes SET {set_clause} WHERE id = ?",
                values,
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def delete_episode(self, episode_id: str) -> bool:
        """
        Delete an episodic memory by ID.
        Triggers cascade removal from episodes_fts.
        Returns True if deleted, False otherwise.
        """
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM episodes WHERE id = ?",
                (episode_id,),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def list_episodes(
        self,
        limit: int = 100,
        min_salience: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        List episodes ordered by salience and recency.
        """
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT * FROM episodes
                WHERE salience >= ?
                ORDER BY salience DESC, last_accessed DESC, created_at DESC
                LIMIT ?
                """,
                (float(min_salience), max(1, int(limit))),
            )
            return [dict(row) for row in cursor.fetchall()]

    def search_episodes(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Execute full-text search against episodes_fts indexed fields
        (title, situation, action_taken, outcome, lessons_learned, tags)
        ranked by BM25 relevance score.
        """
        if not query or not query.strip():
            return []

        tokens = re.findall(r"[\w\-]+", query)
        if not tokens:
            return []

        formatted_query = self._sanitize_fts_query(query)
        if not formatted_query:
            return []

        with self._lock:
            try:
                cursor = self._conn.execute(
                    """
                    SELECT e.*, f.rank
                    FROM episodes_fts f
                    JOIN episodes e ON e.id = f.id
                    WHERE episodes_fts MATCH ?
                    ORDER BY f.rank ASC, e.salience DESC
                    LIMIT ?
                    """,
                    (formatted_query, max(1, int(limit))),
                )
                results = [dict(row) for row in cursor.fetchall()]
                if not results and len(tokens) > 1:
                    or_query = " OR ".join(f'"{t}"*' for t in tokens)
                    cursor = self._conn.execute(
                        """
                        SELECT e.*, f.rank
                        FROM episodes_fts f
                        JOIN episodes e ON e.id = f.id
                        WHERE episodes_fts MATCH ?
                        ORDER BY f.rank ASC, e.salience DESC
                        LIMIT ?
                        """,
                        (or_query, max(1, int(limit))),
                    )
                    results = [dict(row) for row in cursor.fetchall()]
                return results
            except sqlite3.OperationalError:
                cursor = self._conn.execute(
                    """
                    SELECT e.*, f.rank
                    FROM episodes_fts f
                    JOIN episodes e ON e.id = f.id
                    WHERE episodes_fts MATCH ?
                    ORDER BY f.rank ASC, e.salience DESC
                    LIMIT ?
                    """,
                    (f'"{query}"', max(1, int(limit))),
                )
                return [dict(row) for row in cursor.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        """
        Return aggregate statistics of the Synapse storage.
        """
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM synapse_nodes;")
            node_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM episodes;")
            episode_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM synapse_edges;")
            edge_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM synapse_fts;")
            fts_entry_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM episodes_fts;")
            episodes_fts_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM synapse_vectors;")
            vector_count = cursor.fetchone()[0]

            cursor.execute("SELECT category, COUNT(*) FROM synapse_nodes GROUP BY category;")
            category_counts = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute("SELECT status, COUNT(*) FROM synapse_nodes GROUP BY status;")
            status_counts = {row[0]: row[1] for row in cursor.fetchall()}

            return {
                "node_count": node_count,
                "episode_count": episode_count,
                "edge_count": edge_count,
                "fts_entry_count": fts_entry_count,
                "episodes_fts_count": episodes_fts_count,
                "vector_count": vector_count,
                "categories": category_counts,
                "statuses": status_counts,
            }

    def upsert_vector(self, node_id: str, vector: List[float]) -> bool:
        """
        Store or update the embedding vector for a synapse node.
        Returns True on success, False if node does not exist or vector is empty.
        """
        if not vector or not node_id:
            return False

        with self._lock:
            try:
                now = int(time.time())
                vec_json = json.dumps(vector)
                dim = len(vector)
                cursor = self._conn.execute(
                    """
                    INSERT INTO synapse_vectors (id, vector_json, dim, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        vector_json = excluded.vector_json,
                        dim = excluded.dim,
                        updated_at = excluded.updated_at
                    """,
                    (node_id, vec_json, dim, now),
                )
                self._conn.commit()
                success = cursor.rowcount > 0
                if success:
                    if self._incremental_cache and self._vector_cache_valid:
                        node_row = self._conn.execute(
                            "SELECT status FROM synapse_nodes WHERE id = ?", (node_id,)
                        ).fetchone()
                        if node_row and node_row["status"] == "active":
                            self._incremental_upsert(node_id, vector)
                        else:
                            self._incremental_delete(node_id)
                    else:
                        self._invalidate_vector_cache()
                return success
            except sqlite3.IntegrityError:
                return False

    def get_vector(self, node_id: str) -> Optional[List[float]]:
        """
        Retrieve the embedding vector for a node ID.
        Returns None if not found or invalid.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT vector_json FROM synapse_vectors WHERE id = ?",
                (node_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            try:
                return json.loads(row["vector_json"])
            except Exception:
                return None

    @property
    def incremental_cache(self) -> bool:
        """Whether incremental update mode is enabled for the vector cache."""
        return self._incremental_cache

    @incremental_cache.setter
    def incremental_cache(self, value: bool) -> None:
        self._incremental_cache = bool(value)

    @property
    def vector_matrix(self) -> Optional[Any]:
        """Contiguous float32 numpy matrix cache for active vectors."""
        with self._lock:
            self._ensure_vector_matrix_cache()
            return self._vector_matrix

    @property
    def raw_vector_matrix(self) -> Optional[Any]:
        """Contiguous float32 raw (unnormalized) numpy matrix cache for active vectors."""
        with self._lock:
            self._ensure_vector_matrix_cache()
            return self._raw_vector_matrix

    @property
    def vector_node_ids(self) -> List[str]:
        """Node IDs corresponding to the rows of vector_matrix."""
        with self._lock:
            self._ensure_vector_matrix_cache()
            return list(self._vector_node_ids)

    @property
    def vector_ids(self) -> List[str]:
        """Alias for vector_node_ids."""
        return self.vector_node_ids

    @property
    def _vector_ids(self) -> List[str]:
        return self._vector_node_ids

    @_vector_ids.setter
    def _vector_ids(self, val: List[str]) -> None:
        self._vector_node_ids = val

    def get_vector_matrix(
        self, normalized: bool = True
    ) -> Tuple[Optional[Any], List[str]]:
        """
        Return the cached vector matrix and corresponding node IDs.
        If normalized=True, returns unit-normalized matrix for dot product search.
        If normalized=False, returns raw stored vectors.
        """
        with self._lock:
            self._ensure_vector_matrix_cache()
            mat = self._vector_matrix if normalized else self._raw_vector_matrix
            return mat, list(self._vector_node_ids)

    def _get_vector_data_version(self) -> int:
        try:
            row = self._conn.execute(
                "SELECT version FROM _synapse_vector_meta WHERE id = 1"
            ).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return -1

    def _invalidate_vector_cache(self) -> None:
        """Thread-safe vector matrix cache invalidation."""
        self._vector_cache_valid = False
        self._vector_matrix = None
        self._raw_vector_matrix = None
        self._vector_node_ids = []
        self._matrix_by_dim = {}
        self._cached_vector_version = -1

    def invalidate_vector_cache(self) -> None:
        """Explicitly invalidate the vector matrix cache in a thread-safe manner."""
        with self._lock:
            self._invalidate_vector_cache()

    def rebuild_vector_cache(self) -> Tuple[Optional[Any], List[str]]:
        """Explicitly rebuild the vector matrix cache and return (matrix, node_ids)."""
        with self._lock:
            self._build_vector_matrix_cache()
            return self._vector_matrix, list(self._vector_node_ids)

    def _ensure_vector_matrix_cache(self, force: bool = False) -> None:
        """Ensure vector matrix cache is built and up to date with DB state."""
        if not self.use_numpy or not _NUMPY_AVAILABLE or np is None:
            return

        db_version = self._get_vector_data_version()
        if (
            force
            or not self._vector_cache_valid
            or self._vector_matrix is None
            or (db_version != -1 and db_version != self._cached_vector_version)
        ):
            self._build_vector_matrix_cache()
            self._cached_vector_version = db_version

    def _build_vector_matrix_cache(self) -> None:
        """
        Build contiguous float32 numpy matrix cache for active vectors.
        Must be called under self._lock.
        """
        if not self.use_numpy or not _NUMPY_AVAILABLE or np is None:
            self._vector_cache_valid = True
            return

        cursor = self._conn.execute(
            """
            SELECT v.id, v.vector_json
            FROM synapse_vectors v
            JOIN synapse_nodes n ON n.id = v.id
            WHERE n.status = 'active'
            """
        )
        rows = cursor.fetchall()

        node_ids: List[str] = []
        vectors: List[List[float]] = []
        for row in rows:
            try:
                vec = json.loads(row["vector_json"])
                if isinstance(vec, list) and len(vec) > 0:
                    node_ids.append(row["id"])
                    vectors.append(vec)
            except Exception:
                continue

        if not vectors:
            self._vector_matrix = np.empty((0, 0), dtype=np.float32)
            self._raw_vector_matrix = np.empty((0, 0), dtype=np.float32)
            self._vector_node_ids = []
            self._matrix_by_dim = {}
            self._vector_cache_valid = True
            self._cached_vector_version = self._get_vector_data_version()
            return

        dims = [len(v) for v in vectors]
        unique_dims = set(dims)

        if len(unique_dims) == 1:
            dim = dims[0]
            raw_matrix = np.ascontiguousarray(vectors, dtype=np.float32)
            norms = np.linalg.norm(raw_matrix, axis=1, keepdims=True)
            zero_mask = (norms == 0.0).flatten()
            safe_norms = np.where(norms == 0.0, 1.0, norms)
            norm_matrix = np.ascontiguousarray(raw_matrix / safe_norms, dtype=np.float32)
            norm_matrix[zero_mask] = 0.0

            self._raw_vector_matrix = raw_matrix
            self._vector_matrix = norm_matrix
            self._vector_node_ids = node_ids
            self._matrix_by_dim = {dim: (norm_matrix, node_ids, raw_matrix)}
        else:
            self._matrix_by_dim = {}
            dominant_dim = max(unique_dims, key=dims.count)
            for d in unique_dims:
                d_indices = [i for i, dim_len in enumerate(dims) if dim_len == d]
                d_ids = [node_ids[i] for i in d_indices]
                d_vecs = [vectors[i] for i in d_indices]
                d_raw = np.ascontiguousarray(d_vecs, dtype=np.float32)
                d_norms = np.linalg.norm(d_raw, axis=1, keepdims=True)
                d_mask = (d_norms == 0.0).flatten()
                d_safe_norms = np.where(d_norms == 0.0, 1.0, d_norms)
                d_norm = np.ascontiguousarray(d_raw / d_safe_norms, dtype=np.float32)
                d_norm[d_mask] = 0.0
                self._matrix_by_dim[d] = (d_norm, d_ids, d_raw)

            dom_norm, dom_ids, dom_raw = self._matrix_by_dim[dominant_dim]
            self._vector_matrix = dom_norm
            self._raw_vector_matrix = dom_raw
            self._vector_node_ids = dom_ids

        self._vector_cache_valid = True
        self._cached_vector_version = self._get_vector_data_version()

    def _incremental_upsert(self, node_id: str, vector: List[float]) -> None:
        """Incrementally update or insert a vector into the matrix cache."""
        if not self.use_numpy or not _NUMPY_AVAILABLE or np is None:
            return

        vec_raw = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vec_raw))
        vec_norm = (vec_raw / norm) if (norm > 0 and not math.isnan(norm)) else np.zeros_like(vec_raw)

        if node_id in self._vector_node_ids:
            idx = self._vector_node_ids.index(node_id)
            if self._vector_matrix is not None and self._vector_matrix.shape[1] == len(vec_raw):
                self._vector_matrix[idx] = vec_norm
                if self._raw_vector_matrix is not None:
                    self._raw_vector_matrix[idx] = vec_raw
            else:
                self._invalidate_vector_cache()
        else:
            if self._vector_matrix is None or len(self._vector_node_ids) == 0:
                self._vector_matrix = np.ascontiguousarray([vec_norm], dtype=np.float32)
                self._raw_vector_matrix = np.ascontiguousarray([vec_raw], dtype=np.float32)
                self._vector_node_ids = [node_id]
                self._vector_cache_valid = True
            elif self._vector_matrix.shape[1] == len(vec_raw):
                self._vector_matrix = np.ascontiguousarray(
                    np.vstack([self._vector_matrix, vec_norm.reshape(1, -1)]), dtype=np.float32
                )
                if self._raw_vector_matrix is not None:
                    self._raw_vector_matrix = np.ascontiguousarray(
                        np.vstack([self._raw_vector_matrix, vec_raw.reshape(1, -1)]), dtype=np.float32
                    )
                self._vector_node_ids.append(node_id)
                self._vector_cache_valid = True
            else:
                self._invalidate_vector_cache()

    def _incremental_delete(self, node_id: str) -> None:
        """Incrementally remove a node from the matrix cache."""
        if not self.use_numpy or not _NUMPY_AVAILABLE or np is None:
            return

        if node_id in self._vector_node_ids:
            idx = self._vector_node_ids.index(node_id)
            self._vector_node_ids.pop(idx)
            if self._vector_matrix is not None and len(self._vector_node_ids) > 0:
                self._vector_matrix = np.ascontiguousarray(
                    np.delete(self._vector_matrix, idx, axis=0), dtype=np.float32
                )
                if self._raw_vector_matrix is not None:
                    self._raw_vector_matrix = np.ascontiguousarray(
                        np.delete(self._raw_vector_matrix, idx, axis=0), dtype=np.float32
                    )
            else:
                self._vector_matrix = np.empty((0, 0), dtype=np.float32)
                self._raw_vector_matrix = np.empty((0, 0), dtype=np.float32)
                self._vector_node_ids = []

    def search_vectors(
        self,
        query_vector: Union[List[float], Any],
        top_k: int = 10,
        use_numpy: Optional[bool] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search nodes by cosine similarity against stored vectors.
        Uses contiguous float32 numpy matrix acceleration (matrix @ q with argpartition/argsort)
        for instant sub-millisecond retrieval when available, with graceful pure-Python fallback.
        """
        should_use_numpy = (
            use_numpy if use_numpy is not None else (self.use_numpy and _NUMPY_AVAILABLE and np is not None)
        )
        if should_use_numpy:
            try:
                return self._search_vectors_numpy(query_vector, top_k)
            except Exception:
                return self._search_vectors_fallback(query_vector, top_k)
        return self._search_vectors_fallback(query_vector, top_k)

    def _search_vectors_numpy(
        self, query_vector: Union[List[float], Any], top_k: int = 10
    ) -> List[Tuple[str, float]]:
        if query_vector is None or len(query_vector) == 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0 or math.isnan(q_norm) or math.isinf(q_norm):
            return []

        q_unit = np.ascontiguousarray(q / q_norm, dtype=np.float32)
        q_dim = len(q)

        with self._lock:
            self._ensure_vector_matrix_cache()

            if self._vector_matrix is None or len(self._vector_node_ids) == 0:
                return []

            if self._vector_matrix.shape[1] == q_dim:
                matrix = self._vector_matrix
                node_ids = self._vector_node_ids
            elif q_dim in self._matrix_by_dim:
                matrix, node_ids, _ = self._matrix_by_dim[q_dim]
            else:
                return []

        if len(node_ids) == 0 or matrix.shape[0] == 0:
            return []

        # Vector dot product: matrix (N, dim) @ q_unit (dim,) -> scores (N,)
        scores = matrix @ q_unit
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=-1.0)
        scores = np.clip(scores, -1.0, 1.0)

        N = len(scores)
        k = min(max(1, int(top_k)), N)

        if k >= N:
            best_indices = np.argsort(scores, kind="stable")[::-1]
        else:
            top_indices = np.argpartition(scores, -k)[-k:]
            sorted_sub = np.argsort(scores[top_indices], kind="stable")[::-1]
            best_indices = top_indices[sorted_sub]

        return [(node_ids[i], float(scores[i])) for i in best_indices[:k]]

    def _search_vectors_fallback(
        self, query_vector: Union[List[float], Any], top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Pure-Python fallback for vector search."""
        if query_vector is None or len(query_vector) == 0:
            return []

        q_norm = math.sqrt(sum(x * x for x in query_vector))
        if q_norm == 0.0:
            return []

        scored: List[Tuple[str, float]] = []

        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT v.id, v.vector_json
                FROM synapse_vectors v
                JOIN synapse_nodes n ON n.id = v.id
                WHERE n.status = 'active'
                """
            )
            rows = cursor.fetchall()

        for row in rows:
            node_id = row["id"]
            try:
                vec = json.loads(row["vector_json"])
            except Exception:
                continue

            if not vec or len(vec) != len(query_vector):
                continue

            dot = sum(a * b for a, b in zip(query_vector, vec))
            v_norm = math.sqrt(sum(b * b for b in vec))
            if v_norm == 0.0:
                sim = 0.0
            else:
                sim = dot / (q_norm * v_norm)
                sim = max(-1.0, min(1.0, sim))

            scored.append((node_id, float(sim)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, int(top_k))]

    def optimize_and_compact(self) -> Dict[str, Any]:
        """
        Execute database optimization and compaction:
        PRAGMA wal_checkpoint(TRUNCATE), PRAGMA optimize, and VACUUM.
        """
        results: Dict[str, Any] = {
            "status": "ok",
            "wal_checkpoint": "skipped" if self.is_memory else "none",
            "optimize": "ok",
            "vacuum": "ok",
            "timestamp": int(time.time()),
        }
        with self._lock:
            if not self.is_memory:
                try:
                    chk = self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);").fetchone()
                    results["wal_checkpoint"] = list(chk) if chk else "done"
                except Exception as e:
                    results["wal_checkpoint"] = f"error: {e}"

            try:
                self._conn.execute("PRAGMA optimize;")
            except Exception as e:
                results["optimize"] = f"error: {e}"

            try:
                self._conn.execute("VACUUM;")
            except Exception as e:
                results["vacuum"] = f"error: {e}"

        return results

    def set_mental_model(
        self,
        model_id: str,
        name: str,
        source_query: str,
        content: str,
        structured_content: Optional[str] = None,
        trigger_mode: str = "delta",
        tags: Optional[List[str]] = None,
        max_tokens: int = 2048,
    ) -> bool:
        """Create or update a consolidated mental model."""
        now = int(time.time())
        tags_json = json.dumps(tags or [])
        with self._lock:
            cursor = self._conn.execute(
                "SELECT content, history FROM synapse_mental_models WHERE id = ?",
                (model_id,),
            )
            row = cursor.fetchone()
            if row:
                old_content = row["content"]
                try:
                    history = json.loads(row["history"] or "[]")
                except Exception:
                    history = []
                if old_content and old_content != content:
                    history.append({"content": old_content, "archived_at": now})
                    history = history[-10:]  # Cap last 10 versions

                self._conn.execute(
                    """
                    UPDATE synapse_mental_models
                    SET name = ?, source_query = ?, content = ?, structured_content = ?,
                        trigger_mode = ?, tags = ?, max_tokens = ?, last_refreshed_at = ?,
                        last_refreshed_source_query = ?, history = ?
                    WHERE id = ?
                    """,
                    (
                        name,
                        source_query,
                        content,
                        structured_content,
                        trigger_mode,
                        tags_json,
                        max_tokens,
                        now,
                        source_query,
                        json.dumps(history),
                        model_id,
                    ),
                )
            else:
                self._conn.execute(
                    """
                    INSERT INTO synapse_mental_models (
                        id, name, source_query, content, structured_content,
                        trigger_mode, tags, max_tokens, last_refreshed_at,
                        last_refreshed_source_query, history, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        model_id,
                        name,
                        source_query,
                        content,
                        structured_content,
                        trigger_mode,
                        tags_json,
                        max_tokens,
                        now,
                        source_query,
                        "[]",
                        now,
                    ),
                )
            self._conn.commit()
            return True

    def get_mental_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a mental model by ID."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM synapse_mental_models WHERE id = ?",
                (model_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            try:
                res["tags"] = json.loads(res.get("tags") or "[]")
            except Exception:
                res["tags"] = []
            return res

    def list_mental_models(self) -> List[Dict[str, Any]]:
        """List all mental models."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, name, source_query, content, trigger_mode, tags, last_refreshed_at, created_at FROM synapse_mental_models ORDER BY name ASC"
            )
            rows = cursor.fetchall()
            results: List[Dict[str, Any]] = []
            for r in rows:
                item = dict(r)
                try:
                    item["tags"] = json.loads(item.get("tags") or "[]")
                except Exception:
                    item["tags"] = []
                results.append(item)
            return results

    def add_or_update_observation(
        self,
        pattern: str,
        supporting_quote: str,
        category: str = "general",
        confidence: float = 1.0,
        observation_id: Optional[str] = None,
        proof_delta: int = 1,
    ) -> Dict[str, Any]:
        """
        Record or reinforce an evidence-grounded observation.
        If a matching pattern already exists, increment proof_count and append quote.
        """
        now = int(time.time())
        with self._lock:
            if observation_id:
                cursor = self._conn.execute(
                    "SELECT * FROM synapse_observations WHERE id = ?",
                    (observation_id,),
                )
            else:
                cursor = self._conn.execute(
                    "SELECT * FROM synapse_observations WHERE pattern = ? AND status = 'active'",
                    (pattern,),
                )
            existing = cursor.fetchone()

            if existing:
                obs_id = existing["id"]
                current_proof = int(existing["proof_count"]) + max(1, int(proof_delta))
                try:
                    facts = json.loads(existing["supporting_facts"] or "[]")
                except Exception:
                    facts = []
                if supporting_quote and supporting_quote not in facts:
                    facts.append(supporting_quote)
                    facts = facts[-15:]  # Keep latest 15 exact quotes

                new_confidence = min(1.0, float(existing["confidence"]) + 0.05)
                self._conn.execute(
                    """
                    UPDATE synapse_observations
                    SET proof_count = ?, supporting_facts = ?, confidence = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (current_proof, json.dumps(facts), new_confidence, now, obs_id),
                )
                self._conn.commit()
                return {
                    "id": obs_id,
                    "pattern": pattern,
                    "proof_count": current_proof,
                    "confidence": new_confidence,
                    "status": existing["status"],
                    "action": "reinforced",
                }
            else:
                obs_id = observation_id or f"obs_{uuid.uuid4().hex[:10]}"
                facts = [supporting_quote] if supporting_quote else []
                self._conn.execute(
                    """
                    INSERT INTO synapse_observations (
                        id, pattern, supporting_facts, proof_count, confidence,
                        status, category, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        obs_id,
                        pattern,
                        json.dumps(facts),
                        1,
                        confidence,
                        "active",
                        category,
                        now,
                        now,
                    ),
                )
                self._conn.commit()
                return {
                    "id": obs_id,
                    "pattern": pattern,
                    "proof_count": 1,
                    "confidence": confidence,
                    "status": "active",
                    "action": "created",
                }

    def list_observations(
        self,
        category: Optional[str] = None,
        status: str = "active",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Retrieve active observations ordered by proof_count descending."""
        with self._lock:
            if category:
                cursor = self._conn.execute(
                    """
                    SELECT * FROM synapse_observations
                    WHERE status = ? AND category = ?
                    ORDER BY proof_count DESC, updated_at DESC LIMIT ?
                    """,
                    (status, category, limit),
                )
            else:
                cursor = self._conn.execute(
                    """
                    SELECT * FROM synapse_observations
                    WHERE status = ?
                    ORDER BY proof_count DESC, updated_at DESC LIMIT ?
                    """,
                    (status, limit),
                )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                try:
                    item["supporting_facts"] = json.loads(item.get("supporting_facts") or "[]")
                except Exception:
                    item["supporting_facts"] = []
                results.append(item)
            return results

    def add_external_research(
        self,
        topic: str,
        title: str,
        source_url: str,
        domain: str,
        summary: str,
        raw_snippet: str = "",
        vault_path: str = "",
        status: str = "quarantined",
    ) -> Dict[str, Any]:
        """Record sandboxed Tier 4 external research into synapse_external_research."""
        research_id = f"res:{uuid.uuid4().hex[:10]}"
        now = int(time.time())
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO synapse_external_research (
                    id, topic, title, source_url, domain, summary, raw_snippet, tier_level, vault_path, fetched_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 4, ?, ?, ?)
                """,
                (research_id, topic, title, source_url, domain, summary, raw_snippet, vault_path, now, status),
            )
            self._conn.commit()
            return {
                "id": research_id,
                "topic": topic,
                "title": title,
                "source_url": source_url,
                "domain": domain,
                "summary": summary,
                "tier_level": 4,
                "vault_path": vault_path,
                "status": status,
            }

    def list_external_research(
        self,
        topic: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List sandboxed external research records."""
        with self._lock:
            query = "SELECT * FROM synapse_external_research WHERE 1=1"
            params: List[Any] = []
            if topic:
                query += " AND topic = ?"
                params.append(topic)
            if status:
                query += " AND status = ?"
                params.append(status)
            query += " ORDER BY fetched_at DESC LIMIT ?"
            params.append(limit)

            cursor = self._conn.execute(query, tuple(params))
            return [dict(r) for r in cursor.fetchall()]

    def promote_external_research(
        self,
        research_id: str,
        target_tier: int = 1,
        mental_model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Promotion Gate: Elevate a quarantined Tier 4 research item to Tier 1 Mental Model
        or Tier 2 Observation upon user confirmation or high usage count.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM synapse_external_research WHERE id = ?", (research_id,)
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Research record '{research_id}' not found.")
            res = dict(row)

            # Update status in research table
            self._conn.execute(
                "UPDATE synapse_external_research SET status = 'verified', tier_level = ? WHERE id = ?",
                (target_tier, research_id),
            )
            self._conn.commit()

        # If promoting to Tier 1 Mental Model
        if target_tier == 1:
            mm_target = mental_model_id or "master-user-preferences"
            # Return instructions for caller engine
            return {
                "action": "promoted",
                "target_tier": 1,
                "research": res,
                "mental_model_id": mm_target,
                "fact_snippet": f"Verified Best Practice ({res['title']}): {res['summary'][:200]}",
            }

        # If promoting to Tier 2 Observation
        obs_pattern = f"Validated Technical Standard: {res['title']} ({res['topic']})"
        return {
            "action": "promoted",
            "target_tier": 2,
            "research": res,
            "observation_pattern": obs_pattern,
            "proof_count": 3,
        }

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> SynapseStorage:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
