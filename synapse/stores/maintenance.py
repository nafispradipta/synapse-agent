"""Scheduled upkeep: decay, rollups, and compaction."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class MaintenanceStore:
    """Lifecycle operations run by consolidation jobs rather than by readers."""

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

    def supersede_low_salience_nodes(
        self, node_ids: List[str], salience_below: float = 0.90
    ) -> int:
        """Supersede the given nodes only while their salience stays below the cap."""
        ids = [str(node_id) for node_id in node_ids if node_id]
        if not ids:
            return 0
        updated = 0
        with self._lock:
            for node_id in ids:
                cursor = self._conn.execute(
                    "UPDATE synapse_nodes SET status = 'superseded' "
                    "WHERE id = ? AND salience < ?",
                    (node_id, float(salience_below)),
                )
                if cursor.rowcount and cursor.rowcount > 0:
                    updated += cursor.rowcount
            self._conn.commit()
            if updated:
                self._invalidate_vector_cache()
        return updated

    def _select_superseded_rows(self) -> List[Dict[str, Any]]:
        cursor = self._conn.execute(
            """
            SELECT id, uri, l0_abstract, l1_overview, l2_target_path, category,
                   importance, strength, created_at, last_accessed, salience,
                   confidence, COALESCE(access_count, 0) as access_count
            FROM synapse_nodes
            WHERE status = 'superseded'
            """
        )
        return [dict(row) for row in cursor.fetchall()]

    def list_superseded_nodes(self) -> List[Dict[str, Any]]:
        """All nodes currently marked superseded, with the rollup column set."""
        with self._lock:
            return self._select_superseded_rows()

    def commit_superseded_rollup(
        self,
        master: Dict[str, Any],
        rollup_id: str,
        archived_at: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Archive every currently superseded row, upsert the master rollup entity,
        then purge superseded nodes and their edges in a single transaction.

        The superseded set is re-read inside the lock so a node superseded after
        the caller's snapshot is archived rather than deleted unarchived.
        """
        timestamp = int(archived_at) if archived_at is not None else int(time.time())
        archived_rows = 0
        with self._lock:
            rows = self._select_superseded_rows()
            cur = self._conn.cursor()
            cur.execute(
                """
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
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_superseded_archive_cat "
                "ON synapse_superseded_archive(category)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_superseded_archive_rollup "
                "ON synapse_superseded_archive(rollup_id)"
            )

            for row in rows:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO synapse_superseded_archive (
                        id, uri, l0_abstract, l1_overview, l2_target_path, category,
                        importance, strength, created_at, last_accessed, salience,
                        confidence, access_count, archived_at, rollup_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["id"],
                        row.get("uri"),
                        row.get("l0_abstract"),
                        row.get("l1_overview"),
                        row.get("l2_target_path"),
                        row.get("category"),
                        row.get("importance"),
                        row.get("strength"),
                        row.get("created_at"),
                        row.get("last_accessed"),
                        row.get("salience"),
                        row.get("confidence"),
                        row.get("access_count", 0),
                        timestamp,
                        rollup_id,
                    ),
                )
                archived_rows += 1

            cur.execute(
                """
                INSERT INTO synapse_nodes (
                    id, uri, l0_abstract, l1_overview, l2_target_path, category,
                    importance, strength, created_at, last_accessed, status,
                    salience, confidence, access_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    l0_abstract = excluded.l0_abstract,
                    l1_overview = excluded.l1_overview,
                    last_accessed = excluded.last_accessed,
                    status = 'archived',
                    salience = excluded.salience
                ON CONFLICT(uri) DO UPDATE SET
                    l0_abstract = excluded.l0_abstract,
                    l1_overview = excluded.l1_overview,
                    last_accessed = excluded.last_accessed,
                    status = 'archived',
                    salience = excluded.salience
                """,
                (
                    master["id"],
                    master["uri"],
                    master.get("l0_abstract", ""),
                    master.get("l1_overview", ""),
                    None,
                    "general",
                    0.75,
                    1.0,
                    timestamp,
                    timestamp,
                    "archived",
                    0.65,
                    1.0,
                    0,
                ),
            )

            cur.execute(
                """
                DELETE FROM synapse_edges
                WHERE source_id IN (SELECT id FROM synapse_nodes WHERE status = 'superseded')
                   OR target_id IN (SELECT id FROM synapse_nodes WHERE status = 'superseded')
                """
            )
            deleted_edges = cur.rowcount or 0

            cur.execute("DELETE FROM synapse_nodes WHERE status = 'superseded'")
            deleted_nodes = cur.rowcount or 0

            try:
                cur.execute("INSERT INTO synapse_fts(synapse_fts) VALUES('optimize')")
            except Exception:
                pass

            self._conn.commit()
            self._invalidate_vector_cache()

        return {
            "archived_rows": archived_rows,
            "deleted_edges": deleted_edges,
            "deleted_nodes": deleted_nodes,
        }
