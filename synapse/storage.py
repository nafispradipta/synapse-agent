"""
SynapseStorage - Persistent Graph & FTS Storage Engine for Cognitive Synapse Core.

The engine is assembled from bounded-context mixins; each one owns a single
storage concern and the facade only wires them together, owns the connection,
and answers cross-context aggregates such as ``get_stats``.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

# Re-exported for callers that import these from the historical module path.
from synapse.stores.constants import (
    VALID_CATEGORIES,
    VALID_STATUSES,
    _NUMPY_AVAILABLE,
)
from synapse.stores.episodes import EpisodeStore
from synapse.stores.graph import GraphStore
from synapse.stores.knowledge import KnowledgeStore
from synapse.stores.maintenance import MaintenanceStore
from synapse.stores.schema import SchemaMixin
from synapse.stores.vectors import VectorStore


class SynapseStorage(
    SchemaMixin,
    GraphStore,
    VectorStore,
    EpisodeStore,
    KnowledgeStore,
    MaintenanceStore,
):
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

        try:
            self._configure_pragmas()
            self._init_schema()
        except BaseException:
            self._conn.close()
            raise

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

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> SynapseStorage:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
