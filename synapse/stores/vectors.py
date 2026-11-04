"""Vector storage and the in-process matrix cache used for similarity search."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from synapse.stores.constants import _NUMPY_AVAILABLE, np


class VectorStore:
    """Embeddings, the in-memory matrix cache, and cosine search."""

    def upsert_vector(
        self,
        node_id: str,
        vector: List[float],
        model: Optional[str] = None,
    ) -> bool:
        """
        Store or update the embedding vector for a synapse node.
        Returns True on success, False if node does not exist or vector is empty.
        """
        if not vector or not node_id:
            return False

        with self._write():
            try:
                now = int(time.time())
                vec_json = json.dumps(vector)
                dim = len(vector)
                cursor = self._conn.execute(
                    """
                    INSERT INTO synapse_vectors (id, vector_json, dim, updated_at, model)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        vector_json = excluded.vector_json,
                        dim = excluded.dim,
                        updated_at = excluded.updated_at,
                        model = excluded.model
                    """,
                    (node_id, vec_json, dim, now, model),
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
                self._rollback_if_outermost()
                return False

    def get_vector(self, node_id: str) -> Optional[List[float]]:
        """
        Retrieve the embedding vector for a node ID.
        Returns None if not found or invalid.
        """
        with self._write():
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
        with self._write():
            self._ensure_vector_matrix_cache()
            return self._vector_matrix

    @property
    def raw_vector_matrix(self) -> Optional[Any]:
        """Contiguous float32 raw (unnormalized) numpy matrix cache for active vectors."""
        with self._write():
            self._ensure_vector_matrix_cache()
            return self._raw_vector_matrix

    @property
    def vector_node_ids(self) -> List[str]:
        """Node IDs corresponding to the rows of vector_matrix."""
        with self._write():
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
        with self._write():
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
        with self._write():
            self._invalidate_vector_cache()

    def rebuild_vector_cache(self) -> Tuple[Optional[Any], List[str]]:
        """Explicitly rebuild the vector matrix cache and return (matrix, node_ids)."""
        with self._write():
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

        with self._write():
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

        with self._write():
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

    def list_nodes_missing_vectors(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Active nodes that do not have an embedding vector stored yet."""
        with self._write():
            cursor = self._conn.execute(
                """
                SELECT n.id, n.l0_abstract, n.l1_overview
                FROM synapse_nodes n
                LEFT JOIN synapse_vectors v ON v.id = n.id
                WHERE n.status = 'active' AND v.id IS NULL
                LIMIT ?
                """,
                (int(limit),),
            )
            return [dict(row) for row in cursor.fetchall()]
