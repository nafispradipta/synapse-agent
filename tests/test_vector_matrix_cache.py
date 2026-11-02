"""
Tests for Vector Matrix Acceleration & Cache in SynapseStorage.
Covers:
1. Contiguous float32 numpy matrix cache for active vectors (matrix shape (N, dim), node_ids list).
2. Thread-safe invalidation or incremental update when upsert_vector or delete_node is called.
3. Fast vector dot product (matrix @ q) with np.argpartition/argsort for sub-millisecond retrieval.
4. Pure-Python fallback maintained when numpy is disabled or unavailable.
5. Zero regressions with existing storage behavior and edge cases.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import time
from typing import List

import pytest
import numpy as np

from synapse.storage import SynapseStorage, _NUMPY_AVAILABLE


@pytest.fixture
def storage() -> SynapseStorage:
    """Create an in-memory SynapseStorage instance."""
    s = SynapseStorage(":memory:")
    yield s
    s.close()


class TestVectorMatrixCacheStructure:
    def test_empty_cache_initialization(self, storage: SynapseStorage) -> None:
        """Verify that an empty storage initialized cache is empty and safe."""
        assert storage.vector_matrix.shape == (0, 0)
        assert storage.vector_node_ids == []
        assert storage.vector_ids == []
        assert storage._vector_ids == []
        assert storage.search_vectors([1.0, 0.0]) == []

    def test_matrix_cache_structure_and_contiguity(self, storage: SynapseStorage) -> None:
        """Verify contiguous float32 matrix cache shape and data type."""
        n1 = storage.add_node(uri="synapse://test/n1", l0_abstract="Node 1", category="general")
        n2 = storage.add_node(uri="synapse://test/n2", l0_abstract="Node 2", category="general")

        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        storage.upsert_vector(n1, v1)
        storage.upsert_vector(n2, v2)

        mat = storage.vector_matrix
        ids = storage.vector_node_ids

        assert isinstance(mat, np.ndarray)
        assert mat.dtype == np.float32
        assert mat.flags["C_CONTIGUOUS"] is True
        assert mat.shape == (2, 3)
        assert ids == [n1, n2]

    def test_raw_vs_normalized_matrix_accessors(self, storage: SynapseStorage) -> None:
        """Test get_vector_matrix returns normalized or raw arrays as requested."""
        nid = storage.add_node(uri="synapse://test/unnorm", l0_abstract="Unnorm", category="general")
        raw_vec = [3.0, 4.0]  # norm is 5.0
        storage.upsert_vector(nid, raw_vec)

        norm_mat, norm_ids = storage.get_vector_matrix(normalized=True)
        raw_mat, raw_ids = storage.get_vector_matrix(normalized=False)

        assert norm_ids == [nid]
        assert raw_ids == [nid]

        # Raw matrix should have original magnitudes
        assert pytest.approx(raw_mat[0][0], abs=1e-4) == 3.0
        assert pytest.approx(raw_mat[0][1], abs=1e-4) == 4.0

        # Normalized matrix should be unit length: [3/5, 4/5] = [0.6, 0.8]
        assert pytest.approx(norm_mat[0][0], abs=1e-4) == 0.6
        assert pytest.approx(norm_mat[0][1], abs=1e-4) == 0.8
        assert pytest.approx(float(np.linalg.norm(norm_mat[0])), abs=1e-4) == 1.0


class TestVectorSimilaritySearchCorrectness:
    def test_cosine_similarity_ordering(self, storage: SynapseStorage) -> None:
        """Test exact cosine similarity ranking against known vector angles."""
        n_exact = storage.add_node(uri="synapse://arch/exact", l0_abstract="Exact", category="architecture")
        n_close = storage.add_node(uri="synapse://arch/close", l0_abstract="Close", category="architecture")
        n_ortho = storage.add_node(uri="synapse://arch/ortho", l0_abstract="Ortho", category="architecture")
        n_opp = storage.add_node(uri="synapse://arch/opp", l0_abstract="Opposite", category="architecture")

        storage.upsert_vector(n_exact, [1.0, 0.0])
        storage.upsert_vector(n_close, [0.8, 0.6])
        storage.upsert_vector(n_ortho, [0.0, 1.0])
        storage.upsert_vector(n_opp, [-1.0, 0.0])

        hits = storage.search_vectors([1.0, 0.0], top_k=4)

        assert len(hits) == 4
        assert hits[0][0] == n_exact
        assert pytest.approx(hits[0][1], abs=1e-3) == 1.0

        assert hits[1][0] == n_close
        assert pytest.approx(hits[1][1], abs=1e-3) == 0.8

        assert hits[2][0] == n_ortho
        assert pytest.approx(hits[2][1], abs=1e-3) == 0.0

        assert hits[3][0] == n_opp
        assert pytest.approx(hits[3][1], abs=1e-3) == -1.0

    def test_top_k_argpartition_and_argsort_bounds(self, storage: SynapseStorage) -> None:
        """Test various top_k bounds (k=1, k < N, k == N, k > N)."""
        node_ids = []
        for i in range(10):
            nid = storage.add_node(uri=f"synapse://test/{i}", l0_abstract=f"Node {i}", category="general")
            # Create unit vectors rotated along circle
            theta = i * (math.pi / 20.0)  # angles 0 to ~81 deg
            storage.upsert_vector(nid, [math.cos(theta), math.sin(theta)])
            node_ids.append(nid)

        query = [1.0, 0.0]

        # Case 1: k = 1 (minimum top_k)
        hits_k1 = storage.search_vectors(query, top_k=1)
        assert len(hits_k1) == 1
        assert hits_k1[0][0] == node_ids[0]

        # Case 2: k = 3 (k < N, tests argpartition path)
        hits_k3 = storage.search_vectors(query, top_k=3)
        assert len(hits_k3) == 3
        scores_k3 = [h[1] for h in hits_k3]
        assert scores_k3 == sorted(scores_k3, reverse=True)
        assert hits_k3[0][0] == node_ids[0]
        assert hits_k3[1][0] == node_ids[1]
        assert hits_k3[2][0] == node_ids[2]

        # Case 3: k == N (k = 10, tests full sort path)
        hits_k10 = storage.search_vectors(query, top_k=10)
        assert len(hits_k10) == 10
        scores_k10 = [h[1] for h in hits_k10]
        assert scores_k10 == sorted(scores_k10, reverse=True)

        # Case 4: k > N (k = 50, clamps to available nodes)
        hits_k50 = storage.search_vectors(query, top_k=50)
        assert len(hits_k50) == 10

    def test_numpy_array_as_query(self, storage: SynapseStorage) -> None:
        """Ensure search_vectors accepts a numpy array directly as query_vector."""
        nid = storage.add_node(uri="synapse://test/np_q", l0_abstract="Np Query", category="general")
        storage.upsert_vector(nid, [0.5, 0.5])

        q_np = np.array([0.5, 0.5], dtype=np.float32)
        hits = storage.search_vectors(q_np, top_k=5)
        assert len(hits) == 1
        assert hits[0][0] == nid
        assert pytest.approx(hits[0][1], abs=1e-3) == 1.0


class TestCacheInvalidationAndIncrementalUpdates:
    def test_invalidation_on_upsert(self, storage: SynapseStorage) -> None:
        """Upserting a vector invalidates cache; subsequent search rebuilds it."""
        n1 = storage.add_node(uri="synapse://test/inv1", l0_abstract="Inv 1", category="general")
        storage.upsert_vector(n1, [1.0, 0.0])

        # Run search to build cache
        hits1 = storage.search_vectors([1.0, 0.0], top_k=10)
        assert len(hits1) == 1
        assert storage._vector_cache_valid is True

        # Upsert a second vector (invalidation mode default)
        n2 = storage.add_node(uri="synapse://test/inv2", l0_abstract="Inv 2", category="general")
        storage.upsert_vector(n2, [0.0, 1.0])
        assert storage._vector_cache_valid is False

        # Next search rebuilds cache with both nodes
        hits2 = storage.search_vectors([1.0, 0.0], top_k=10)
        assert len(hits2) == 2
        assert storage._vector_cache_valid is True
        retrieved_ids = [h[0] for h in hits2]
        assert n1 in retrieved_ids
        assert n2 in retrieved_ids

    def test_invalidation_on_delete_node(self, storage: SynapseStorage) -> None:
        """Deleting a node invalidates cache and cascades vector removal."""
        n1 = storage.add_node(uri="synapse://test/del1", l0_abstract="Del 1", category="general")
        n2 = storage.add_node(uri="synapse://test/del2", l0_abstract="Del 2", category="general")
        storage.upsert_vector(n1, [1.0, 0.0])
        storage.upsert_vector(n2, [0.0, 1.0])

        storage.search_vectors([1.0, 0.0])
        assert storage._vector_cache_valid is True

        # Delete n1
        storage.delete_node(n1)
        assert storage._vector_cache_valid is False

        # Next search reflects deletion
        hits = storage.search_vectors([1.0, 0.0], top_k=10)
        assert len(hits) == 1
        assert hits[0][0] == n2

    def test_invalidation_on_status_change(self, storage: SynapseStorage) -> None:
        """Changing node status to superseded/archived invalidates cache and excludes node."""
        nid = storage.add_node(uri="synapse://test/stat", l0_abstract="Status Test", category="decision")
        storage.upsert_vector(nid, [1.0, 0.0])

        hits = storage.search_vectors([1.0, 0.0], top_k=10)
        assert len(hits) == 1
        assert hits[0][0] == nid

        # Mark as superseded
        storage.update_node(nid, status="superseded")
        assert storage._vector_cache_valid is False

        # Search should now exclude it
        hits_after = storage.search_vectors([1.0, 0.0], top_k=10)
        assert len(hits_after) == 0

        # Mark back as active
        storage.update_node(nid, status="active")
        hits_reactivated = storage.search_vectors([1.0, 0.0], top_k=10)
        assert len(hits_reactivated) == 1
        assert hits_reactivated[0][0] == nid

    def test_explicit_invalidation_and_rebuild(self, storage: SynapseStorage) -> None:
        """Test manual invalidate_vector_cache() and rebuild_vector_cache()."""
        n1 = storage.add_node(uri="synapse://test/man1", l0_abstract="Man 1", category="general")
        storage.upsert_vector(n1, [0.5, 0.5])
        storage.search_vectors([0.5, 0.5])
        assert storage._vector_cache_valid is True

        storage.invalidate_vector_cache()
        assert storage._vector_cache_valid is False

        mat, ids = storage.rebuild_vector_cache()
        assert storage._vector_cache_valid is True
        assert ids == [n1]
        assert mat.shape == (1, 2)

    def test_incremental_cache_mode(self, storage: SynapseStorage) -> None:
        """Test incremental cache mode updates vector matrix without full table scans."""
        n1 = storage.add_node(uri="synapse://test/inc1", l0_abstract="Inc 1", category="general")
        storage.upsert_vector(n1, [1.0, 0.0])

        # Prime the cache
        storage.search_vectors([1.0, 0.0])
        assert storage._vector_cache_valid is True
        assert storage.vector_matrix.shape == (1, 2)

        # Enable incremental cache
        storage.incremental_cache = True
        assert storage.incremental_cache is True

        # Insert node 2 directly into synapse_nodes so it doesn't auto-vectorize with 64-dim
        n2 = "inc_node_2"
        with storage._lock:
            storage._conn.execute(
                """
                INSERT INTO synapse_nodes (id, uri, l0_abstract, category, status)
                VALUES (?, 'synapse://test/inc2', 'Inc 2', 'general', 'active')
                """,
                (n2,),
            )
            storage._conn.commit()

        # Upsert n2 in incremental mode
        storage.upsert_vector(n2, [0.0, 1.0])

        # Cache remains valid because it was updated incrementally
        assert storage._vector_cache_valid is True
        assert storage.vector_matrix.shape == (2, 2)
        assert n2 in storage.vector_node_ids

        # Update n1's vector
        storage.upsert_vector(n1, [0.8, 0.6])
        assert storage._vector_cache_valid is True
        assert pytest.approx(storage.vector_matrix[0][0], abs=1e-4) == 0.8

        # Delete n2 in incremental mode
        storage.delete_node(n2)
        assert storage._vector_cache_valid is True
        assert storage.vector_matrix.shape == (1, 2)
        assert n2 not in storage.vector_node_ids


class TestPurePythonFallback:
    def test_pure_python_matches_numpy_results(self, storage: SynapseStorage) -> None:
        """Ensure search_vectors with use_numpy=False matches use_numpy=True."""
        nodes = []
        rng = np.random.default_rng(12345)
        for i in range(15):
            nid = storage.add_node(uri=f"synapse://test/equiv_{i}", l0_abstract=f"Equiv {i}", category="general")
            vec = rng.standard_normal(16).tolist()
            storage.upsert_vector(nid, vec)
            nodes.append(nid)

        query = rng.standard_normal(16).tolist()

        res_numpy = storage.search_vectors(query, top_k=5, use_numpy=True)
        res_python = storage.search_vectors(query, top_k=5, use_numpy=False)

        assert len(res_numpy) == len(res_python) == 5
        for (id_np, sim_np), (id_py, sim_py) in zip(res_numpy, res_python):
            assert id_np == id_py
            assert pytest.approx(sim_np, abs=1e-4) == sim_py

    def test_pure_python_fallback_when_numpy_disabled(self, storage: SynapseStorage) -> None:
        """Setting storage.use_numpy = False seamlessly uses pure-Python fallback."""
        n1 = storage.add_node(uri="synapse://test/no_np", l0_abstract="No Np", category="general")
        storage.upsert_vector(n1, [1.0, 0.0])

        storage.use_numpy = False
        hits = storage.search_vectors([1.0, 0.0], top_k=5)
        assert len(hits) == 1
        assert hits[0][0] == n1
        assert pytest.approx(hits[0][1], abs=1e-3) == 1.0

        # Restore
        storage.use_numpy = True

    def test_fallback_when_numpy_is_none(self, storage: SynapseStorage, monkeypatch: pytest.MonkeyPatch) -> None:
        """When numpy module is missing or None, fallback executes cleanly."""
        n1 = storage.add_node(uri="synapse://test/mock_np", l0_abstract="Mock Np", category="general")
        storage.upsert_vector(n1, [0.6, 0.8])

        monkeypatch.setattr("synapse.storage._NUMPY_AVAILABLE", False)
        monkeypatch.setattr("synapse.storage.np", None)

        hits = storage.search_vectors([0.6, 0.8], top_k=5)
        assert len(hits) == 1
        assert hits[0][0] == n1
        assert pytest.approx(hits[0][1], abs=1e-3) == 1.0


class TestSubMillisecondPerformance:
    def test_sub_millisecond_vector_similarity_search(self, storage: SynapseStorage) -> None:
        """Benchmark that 1,000 384-dimensional vectors achieve sub-millisecond retrieval."""
        rng = np.random.default_rng(42)
        N = 1000
        dim = 384

        # Generate random normalized vectors
        matrix = rng.standard_normal((N, dim)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

        # Bulk insert directly into SQLite for fast test fixture setup
        now = int(time.time())
        with storage._lock:
            for i in range(N):
                nid = f"bench_node_{i}"
                storage._conn.execute(
                    """
                    INSERT INTO synapse_nodes (id, uri, l0_abstract, category, status)
                    VALUES (?, ?, ?, 'general', 'active')
                    """,
                    (nid, f"synapse://bench/{i}", f"Benchmark Node {i}"),
                )
                storage._conn.execute(
                    """
                    INSERT INTO synapse_vectors (id, vector_json, dim, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (nid, json.dumps(matrix[i].tolist()), dim, now),
                )
            storage._conn.commit()

        # Prime the cache once
        storage.search_vectors(matrix[0].tolist(), top_k=10)
        assert storage._vector_cache_valid is True

        # Benchmark 50 distinct queries
        queries = rng.standard_normal((50, dim)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        t0 = time.perf_counter()
        for q in queries:
            hits = storage.search_vectors(q.tolist(), top_k=10)
            assert len(hits) == 10
        total_time = time.perf_counter() - t0

        avg_query_ms = (total_time / 50.0) * 1000.0
        print(f"Average search_vectors latency: {avg_query_ms:.4f} ms")

        # Sub-millisecond target: avg_query_ms must be strictly less than 1.0 ms
        assert avg_query_ms < 1.0, f"Search latency {avg_query_ms:.3f} ms exceeds 1ms target"


class TestEdgeCasesAndThreadSafety:
    def test_empty_and_zero_norm_queries(self, storage: SynapseStorage) -> None:
        """Test empty vector and all-zero query vectors."""
        nid = storage.add_node(uri="synapse://test/zero_q", l0_abstract="Zero Q", category="general")
        storage.upsert_vector(nid, [1.0, 0.0])

        assert storage.search_vectors([]) == []
        assert storage.search_vectors([0.0, 0.0]) == []
        assert storage.search_vectors([0.0, 0.0, 0.0]) == []

    def test_zero_magnitude_stored_vector(self, storage: SynapseStorage) -> None:
        """Zero magnitude stored vector should return 0.0 without ZeroDivisionError."""
        nid = storage.add_node(uri="synapse://test/zerostored", l0_abstract="Zero Stored", category="general")
        storage.upsert_vector(nid, [0.0, 0.0, 0.0])

        hits = storage.search_vectors([1.0, 0.0, 0.0], top_k=5)
        assert len(hits) == 1
        assert hits[0][0] == nid
        assert hits[0][1] == 0.0

    def test_heterogeneous_dimensions_handling(self, storage: SynapseStorage) -> None:
        """Different dimensions across active nodes are supported without crashing."""
        n_2d = storage.add_node(uri="synapse://test/2d", l0_abstract="2D Node", category="general")
        n_3d = storage.add_node(uri="synapse://test/3d", l0_abstract="3D Node", category="general")

        storage.upsert_vector(n_2d, [1.0, 0.0])
        storage.upsert_vector(n_3d, [0.0, 1.0, 0.0])

        hits_2d = storage.search_vectors([1.0, 0.0], top_k=5)
        assert len(hits_2d) == 1
        assert hits_2d[0][0] == n_2d

        hits_3d = storage.search_vectors([0.0, 1.0, 0.0], top_k=5)
        assert len(hits_3d) == 1
        assert hits_3d[0][0] == n_3d

    def test_concurrent_search_and_updates_thread_safety(self, storage: SynapseStorage) -> None:
        """Multi-threaded stress test: readers and writers running concurrently."""
        # Prepopulate with 20 nodes
        for i in range(20):
            nid = storage.add_node(uri=f"synapse://test/concurr_{i}", l0_abstract=f"C {i}", category="general")
            storage.upsert_vector(nid, [float(i), 1.0, 2.0])

        stop_event = False
        errors: List[Exception] = []

        def reader_task() -> None:
            nonlocal errors
            for _ in range(50):
                try:
                    hits = storage.search_vectors([1.0, 1.0, 2.0], top_k=5)
                    assert isinstance(hits, list)
                except Exception as e:
                    errors.append(e)

        def writer_task(worker_id: int) -> None:
            nonlocal errors
            for j in range(20):
                try:
                    nid = storage.add_node(
                        uri=f"synapse://test/dyn_{worker_id}_{j}", l0_abstract=f"Dyn {j}", category="general"
                    )
                    storage.upsert_vector(nid, [float(j), 2.0, 3.0])
                    if j % 2 == 0:
                        storage.delete_node(nid)
                except Exception as e:
                    errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            readers = [executor.submit(reader_task) for _ in range(4)]
            writers = [executor.submit(writer_task, w) for w in range(2)]
            concurrent.futures.wait(readers + writers, timeout=10)

        assert len(errors) == 0, f"Encountered thread safety errors: {errors}"
