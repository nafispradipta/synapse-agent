"""
Unit tests for Synapse Cognitive Memory Vector Storage, Semantic Vectorizer & Session Distiller.

Covering:
1. Vector storage (upsert_vector, get_vector, search_vectors, cosine similarity math, cascade deletion).
2. Semantic vectorizer in retriever.py (vectorize_text, cosine_similarity) and Hybrid Dense-Sparse Fusion ranking in retrieve().
3. optimize_and_compact() in storage.py (in-memory and file-backed database).
4. distill_session() in worker.py (rule-based fallback, entity extraction, synaptic edge formation, and mocked LLM responses).
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synapse.retriever import (
    SynapseRetriever,
    cosine_similarity,
    vectorize_text,
)
from synapse.storage import SynapseStorage
from synapse.worker import SynapseWorker, resolve_synapse_llm_config


@pytest.fixture
def mem_storage():
    """In-memory SynapseStorage fixture."""
    storage = SynapseStorage(db_path=":memory:")
    yield storage
    storage.close()


@pytest.fixture
def temp_file_storage():
    """Temporary file-backed SynapseStorage fixture with WAL mode."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_path = tf.name

    storage = SynapseStorage(db_path=temp_path)
    yield storage
    storage.close()

    for p in (temp_path, temp_path + "-wal", temp_path + "-shm"):
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


# ============================================================================
# 1. Vector Storage Tests
# ============================================================================

class TestSynapseVectorStorage:
    """Tests for vector CRUD, cosine similarity math, and cascade deletion."""

    def test_upsert_and_get_vector_success(self, mem_storage: SynapseStorage) -> None:
        """Test inserting, updating, and retrieving embedding vectors."""
        node_id = mem_storage.add_node(
            uri="synapse://architecture/vector_test",
            l0_abstract="Vector storage test node",
            category="architecture",
        )

        test_vec = [0.1, 0.5, -0.3, 0.8]
        ok = mem_storage.upsert_vector(node_id, test_vec)
        assert ok is True

        retrieved = mem_storage.get_vector(node_id)
        assert retrieved is not None
        assert len(retrieved) == 4
        assert [round(x, 4) for x in retrieved] == [0.1, 0.5, -0.3, 0.8]

        # Update vector with new dimensions and values
        new_vec = [0.9, -0.2, 0.4]
        ok_update = mem_storage.upsert_vector(node_id, new_vec)
        assert ok_update is True

        updated_retrieved = mem_storage.get_vector(node_id)
        assert updated_retrieved is not None
        assert len(updated_retrieved) == 3
        assert [round(x, 4) for x in updated_retrieved] == [0.9, -0.2, 0.4]

    def test_upsert_vector_invalid_inputs_and_foreign_key(self, mem_storage: SynapseStorage) -> None:
        """Test upsert_vector rejects empty vectors and respects foreign key constraint."""
        node_id = mem_storage.add_node(
            uri="synapse://pref/fk_test",
            l0_abstract="Foreign key test node",
            category="preference",
        )

        # Empty vector or empty node_id
        assert mem_storage.upsert_vector(node_id, []) is False
        assert mem_storage.upsert_vector("", [0.1, 0.2]) is False

        # Non-existent node_id (violates FOREIGN KEY)
        assert mem_storage.upsert_vector("non_existent_node_id_999", [0.1, 0.2]) is False

    def test_get_vector_edge_cases(self, mem_storage: SynapseStorage) -> None:
        """Test get_vector on missing nodes and corrupted database entries."""
        # Non-existent node
        assert mem_storage.get_vector("ghost_node_123") is None

        # Manually corrupt JSON in synapse_vectors table
        node_id = mem_storage.add_node(
            uri="synapse://decision/corrupt_json",
            l0_abstract="Corrupted vector node",
            category="decision",
        )
        with mem_storage._lock:
            mem_storage._conn.execute(
                "UPDATE synapse_vectors SET vector_json = 'INVALID_JSON{[' WHERE id = ?",
                (node_id,),
            )
            mem_storage._conn.commit()

        assert mem_storage.get_vector(node_id) is None

    def test_search_vectors_cosine_similarity_ordering(self, mem_storage: SynapseStorage) -> None:
        """Test vector similarity search ranks correctly according to cosine similarity."""
        # Create 4 nodes with known orthogonal and parallel unit vectors
        n1 = mem_storage.add_node(uri="synapse://arch/exact_match", l0_abstract="Exact", category="architecture")
        n2 = mem_storage.add_node(uri="synapse://arch/close_match", l0_abstract="Close", category="architecture")
        n3 = mem_storage.add_node(uri="synapse://arch/orthogonal", l0_abstract="Ortho", category="architecture")
        n4 = mem_storage.add_node(uri="synapse://arch/opposite", l0_abstract="Opposite", category="architecture")

        # Query vector: [1.0, 0.0]
        # n1: [1.0, 0.0] -> cos_sim = 1.0
        # n2: [0.8, 0.6] -> cos_sim = 0.8
        # n3: [0.0, 1.0] -> cos_sim = 0.0
        # n4: [-1.0, 0.0] -> cos_sim = -1.0
        mem_storage.upsert_vector(n1, [1.0, 0.0])
        mem_storage.upsert_vector(n2, [0.8, 0.6])
        mem_storage.upsert_vector(n3, [0.0, 1.0])
        mem_storage.upsert_vector(n4, [-1.0, 0.0])

        query = [1.0, 0.0]
        hits = mem_storage.search_vectors(query, top_k=4)

        assert len(hits) == 4
        assert hits[0][0] == n1
        assert pytest.approx(hits[0][1], abs=1e-3) == 1.0
        assert hits[1][0] == n2
        assert pytest.approx(hits[1][1], abs=1e-3) == 0.8
        assert hits[2][0] == n3
        assert pytest.approx(hits[2][1], abs=1e-3) == 0.0
        assert hits[3][0] == n4
        assert pytest.approx(hits[3][1], abs=1e-3) == -1.0

    def test_search_vectors_empty_and_zero_norm(self, mem_storage: SynapseStorage) -> None:
        """Test search_vectors with empty or zero-norm query vector."""
        node_id = mem_storage.add_node(uri="synapse://arch/ztest", l0_abstract="ZTest", category="architecture")
        mem_storage.upsert_vector(node_id, [0.5, 0.5])

        assert mem_storage.search_vectors([]) == []
        assert mem_storage.search_vectors([0.0, 0.0, 0.0]) == []

    def test_search_vectors_status_filtering(self, mem_storage: SynapseStorage) -> None:
        """Test search_vectors excludes non-active nodes (superseded, archived)."""
        active_id = mem_storage.add_node(uri="synapse://pref/act", l0_abstract="Act", status="active")
        superseded_id = mem_storage.add_node(uri="synapse://pref/sup", l0_abstract="Sup", status="superseded")
        archived_id = mem_storage.add_node(uri="synapse://pref/arc", l0_abstract="Arc", status="archived")

        mem_storage.upsert_vector(active_id, [1.0, 0.0])
        mem_storage.upsert_vector(superseded_id, [1.0, 0.0])
        mem_storage.upsert_vector(archived_id, [1.0, 0.0])

        hits = mem_storage.search_vectors([1.0, 0.0], top_k=10)
        returned_ids = [h[0] for h in hits]

        assert active_id in returned_ids
        assert superseded_id not in returned_ids
        assert archived_id not in returned_ids

    def test_search_vectors_zero_magnitude_stored_vector(self, mem_storage: SynapseStorage) -> None:
        """Test search_vectors handles zero-magnitude stored vectors without ZeroDivisionError."""
        node_id = mem_storage.add_node(uri="synapse://arch/zerovec", l0_abstract="Zero", category="architecture")
        # Upsert all-zero vector
        with mem_storage._lock:
            mem_storage._conn.execute(
                "UPDATE synapse_vectors SET vector_json = '[0.0, 0.0]' WHERE id = ?",
                (node_id,),
            )
            mem_storage._conn.commit()

        hits = mem_storage.search_vectors([1.0, 1.0], top_k=5)
        assert len(hits) == 1
        assert hits[0][0] == node_id
        assert hits[0][1] == 0.0

    def test_vector_cascade_deletion(self, mem_storage: SynapseStorage) -> None:
        """Test deleting a node cascades and removes its vector from synapse_vectors."""
        node_id = mem_storage.add_node(
            uri="synapse://sop/cascade_vector",
            l0_abstract="Cascade vector deletion test",
            category="sop",
        )
        mem_storage.upsert_vector(node_id, [0.3, 0.4, 0.5])

        assert mem_storage.get_vector(node_id) is not None
        stats_before = mem_storage.get_stats()
        assert stats_before["vector_count"] >= 1

        # Delete node
        deleted = mem_storage.delete_node(node_id)
        assert deleted is True

        # Vector should be removed via SQLite cascade
        assert mem_storage.get_vector(node_id) is None
        stats_after = mem_storage.get_stats()
        assert stats_after["vector_count"] == stats_before["vector_count"] - 1


# ============================================================================
# 2. Semantic Vectorizer and Hybrid Dense-Sparse Fusion Ranking Tests
# ============================================================================

class TestSemanticVectorizerAndHybridRetriever:
    """Tests for vectorize_text, cosine_similarity, and hybrid fusion retrieve()."""

    def test_vectorize_text_properties(self) -> None:
        """Test vectorize_text determinism, L2 normalization, and dimension support."""
        text = "SQLite WAL mode with persistent FTS5 indexing"

        # Determinism
        vec1 = vectorize_text(text, dim=64)
        vec2 = vectorize_text(text, dim=64)
        assert vec1 == vec2
        assert len(vec1) == 64

        # Unit normalization (L2 norm ~ 1.0)
        l2_norm = math.sqrt(sum(x * x for x in vec1))
        assert pytest.approx(l2_norm, abs=1e-3) == 1.0

        # Custom dimensions
        vec_32 = vectorize_text(text, dim=32)
        assert len(vec_32) == 32
        vec_128 = vectorize_text(text, dim=128)
        assert len(vec_128) == 128

        # Empty and whitespace inputs
        assert vectorize_text("", dim=64) == [0.0] * 64
        assert vectorize_text("   \n\t ", dim=64) == [0.0] * 64
        assert vectorize_text("!!!???", dim=64) == [0.0] * 64

    def test_vectorize_text_semantic_affinity(self) -> None:
        """Test related texts have higher cosine similarity than unrelated texts."""
        base = vectorize_text("Docker container isolation with gVisor sandbox")
        similar = vectorize_text("Container isolation and docker sandboxing")
        unrelated = vectorize_text("Cooking Italian pasta with fresh parmesan cheese")

        sim_related = cosine_similarity(base, similar)
        sim_unrelated = cosine_similarity(base, unrelated)

        assert sim_related > sim_unrelated
        assert sim_related > 0.3

    def test_cosine_similarity_math_boundaries(self) -> None:
        """Test cosine_similarity mathematical boundaries and edge cases."""
        assert cosine_similarity([], [1.0, 2.0]) == 0.0
        assert cosine_similarity([1.0, 2.0], []) == 0.0
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

        v1 = [1.0, 0.0]
        v2 = [1.0, 0.0]
        assert pytest.approx(cosine_similarity(v1, v2), abs=1e-5) == 1.0

        v_opposite = [-1.0, 0.0]
        assert pytest.approx(cosine_similarity(v1, v_opposite), abs=1e-5) == -1.0

        v_ortho = [0.0, 1.0]
        assert pytest.approx(cosine_similarity(v1, v_ortho), abs=1e-5) == 0.0

    def test_retriever_auto_indexing(self, mem_storage: SynapseStorage) -> None:
        """Test retriever indexes missing vectors for active nodes."""
        retriever = SynapseRetriever(storage=mem_storage)

        node_id = mem_storage.add_node(
            uri="synapse://arch/market_data_engine",
            l0_abstract="Auto indexing test node",
            category="architecture",
        )
        # Force remove vector
        with mem_storage._lock:
            mem_storage._conn.execute("DELETE FROM synapse_vectors WHERE id = ?", (node_id,))
            mem_storage._conn.commit()

        assert mem_storage.get_vector(node_id) is None

        # auto_index_missing_vectors should recreate it
        count = retriever.auto_index_missing_vectors()
        assert count == 1
        assert mem_storage.get_vector(node_id) is not None

    def test_hybrid_retrieve_composite_scoring_and_fields(self, mem_storage: SynapseStorage) -> None:
        """Test hybrid retrieve() calculates composite_score, bm25_score, vector_score, etc."""
        mem_storage.add_node(
            uri="synapse://pref/nordic_theme",
            l0_abstract="Nordic dark mode theme with cyan accents.",
            l1_overview="Preferred theme for all code editing and terminal displays.",
            category="preference",
            importance=0.8,
            strength=1.2,
        )

        retriever = SynapseRetriever(storage=mem_storage)
        results = retriever.retrieve("Nordic dark theme", limit=5)

        assert len(results) >= 1
        top = results[0]
        assert "composite_score" in top
        assert "bm25_score" in top
        assert "vector_similarity" in top
        assert "vector_score" in top
        assert "activation_energy" in top
        assert "recency_score" in top

        assert top["composite_score"] > 0.0
        assert top["vector_similarity"] > 0.0
        assert top["bm25_score"] > 0.0

    def test_hybrid_retrieve_dense_fallback_when_no_lexical_hits(self, mem_storage: SynapseStorage) -> None:
        """Test dense vector retrieval retrieves relevant items when BM25 has no exact keyword matches."""
        n1 = mem_storage.add_node(
            uri="synapse://sop/container_sandbox",
            l0_abstract="Containerized execution sandboxing with gVisor namespace isolation.",
            category="sop",
            importance=0.8,
        )
        mem_storage.add_node(
            uri="synapse://pref/font_family",
            l0_abstract="JetBrains Mono font family preference.",
            category="preference",
            importance=0.5,
        )

        retriever = SynapseRetriever(storage=mem_storage)

        # Query with semantic overlap but distinct vocabulary from exact FTS words
        results = retriever.retrieve("containerized sandboxing namespace", limit=5)
        assert len(results) >= 1
        assert results[0]["id"] == n1

    def test_hybrid_retrieve_spreading_activation(self, mem_storage: SynapseStorage) -> None:
        """Test spreading activation propagates energy from seed node to connected neighbor."""
        n1 = mem_storage.add_node(
            uri="synapse://decision/db_wal",
            l0_abstract="Adopted SQLite WAL mode for concurrency.",
            category="decision",
        )
        n2 = mem_storage.add_node(
            uri="synapse://architecture/synapse_engine",
            l0_abstract="SynapseStorage manages tables and indexes.",
            category="architecture",
        )

        # Connect n1 and n2
        mem_storage.add_edge(n1, n2, relation="implements", weight=0.9)

        retriever = SynapseRetriever(storage=mem_storage)
        # Search specifically for WAL mode (seed is n1)
        results = retriever.retrieve("SQLite WAL concurrency", limit=5, max_hops=1)

        result_ids = [r["id"] for r in results]
        assert n1 in result_ids
        # n2 should be discovered and activated via spreading activation
        assert n2 in result_ids

        n2_res = next(r for r in results if r["id"] == n2)
        assert n2_res["activation_energy"] > 0.0

    def test_hybrid_retrieve_touch_reinforcement(self, mem_storage: SynapseStorage) -> None:
        """Test touch=True reinforces node strength and updates last_accessed."""
        node_id = mem_storage.add_node(
            uri="synapse://pref/touch_test",
            l0_abstract="Touch reinforcement test node.",
            category="preference",
            strength=1.0,
        )
        before_node = mem_storage.get_node(node_id)
        assert before_node is not None
        initial_strength = before_node["strength"]

        retriever = SynapseRetriever(storage=mem_storage)

        # Retrieve with touch=False
        retriever.retrieve("Touch reinforcement", touch=False)
        node_after_no_touch = mem_storage.get_node(node_id)
        assert node_after_no_touch["strength"] == initial_strength

        # Retrieve with touch=True
        results = retriever.retrieve("Touch reinforcement", touch=True)
        assert len(results) >= 1
        node_after_touch = mem_storage.get_node(node_id)
        assert node_after_touch["strength"] > initial_strength
        assert node_after_touch["strength"] == pytest.approx(1.1, abs=1e-3)

    def test_hybrid_retrieve_empty_query(self, mem_storage: SynapseStorage) -> None:
        """Test retrieve returns empty list on empty or whitespace queries."""
        retriever = SynapseRetriever(storage=mem_storage)
        assert retriever.retrieve("") == []
        assert retriever.retrieve("   \n\t  ") == []


# ============================================================================
# 3. Storage Optimization & Compaction Tests
# ============================================================================

class TestSynapseStorageOptimizeAndCompact:
    """Tests for optimize_and_compact() on in-memory and file-backed SQLite storage."""

    def test_optimize_and_compact_in_memory(self, mem_storage: SynapseStorage) -> None:
        """Test optimize_and_compact on in-memory database."""
        res = mem_storage.optimize_and_compact()
        assert isinstance(res, dict)
        assert res["status"] == "ok"
        assert res["wal_checkpoint"] == "skipped"
        assert res["optimize"] == "ok"
        assert res["vacuum"] == "ok"
        assert isinstance(res["timestamp"], int)

    def test_optimize_and_compact_file_db(self, temp_file_storage: SynapseStorage) -> None:
        """Test optimize_and_compact on file-backed WAL database maintains integrity."""
        # Populate with data
        n1 = temp_file_storage.add_node(
            uri="synapse://pref/compaction1",
            l0_abstract="Compaction test node 1",
            category="preference",
        )
        n2 = temp_file_storage.add_node(
            uri="synapse://pref/compaction2",
            l0_abstract="Compaction test node 2",
            category="preference",
        )
        temp_file_storage.add_edge(n1, n2, relation="relates", weight=0.7)

        # Delete one node to create free pages
        temp_file_storage.delete_node(n1)

        # Run compaction
        res = temp_file_storage.optimize_and_compact()
        assert res["status"] == "ok"
        assert res["wal_checkpoint"] != "skipped"
        assert res["optimize"] == "ok"
        assert res["vacuum"] == "ok"

        # Verify database remains functional and remaining node exists
        remaining = temp_file_storage.get_node(n2)
        assert remaining is not None
        assert remaining["uri"] == "synapse://pref/compaction2"


# ============================================================================
# 4. Session Distiller Tests
# ============================================================================

class TestSynapseDistiller:
    """Tests for distill_session in SynapseWorker (rule-based and mocked LLM)."""

    def test_extract_entities_from_text(self, mem_storage: SynapseStorage) -> None:
        """Test entity extraction finds tech keywords, backticked identifiers, and capitalized terms."""
        worker = SynapseWorker(storage=mem_storage)
        text = "We deployed `FastAPI` service with `PostgreSQL` and SQLite WAL mode on Docker containers."
        entities = worker._extract_entities_from_text(text)

        assert "FastAPI" in entities
        assert "PostgreSQL" in entities
        assert any("sqlite" in e.lower() for e in entities)
        assert any("docker" in e.lower() for e in entities)
        assert any("wal" in e.lower() for e in entities)

    def test_distill_session_empty_messages(self, mem_storage: SynapseStorage) -> None:
        """Test distill_session with empty input returns empty status dict."""
        worker = SynapseWorker(storage=mem_storage)
        result = worker.distill_session([])
        assert result["status"] == "empty"
        assert result["distilled_count"] == 0
        assert result["nodes_created"] == []
        assert result["edges_created"] == 0
        assert result["method"] == "none"

    def test_distill_session_rule_based_categorization(self, mem_storage: SynapseStorage) -> None:
        """Test rule-based fallback accurately categorizes preference, decision, architecture, incident, and sop."""
        worker = SynapseWorker(storage=mem_storage)

        messages = [
            {"role": "user", "content": "I prefer dark theme with Fira Code font in the editor."},
            {"role": "assistant", "content": "We decided to adopt SQLite WAL mode for the backend database."},
            {"role": "user", "content": "The architecture uses a FastAPI service layer and SynapseStorage."},
            {"role": "assistant", "content": "Fixed critical crash bug when connection timeout occurred."},
            {"role": "user", "content": "Always run pytest before deploying code changes to production."},
        ]

        result = worker.distill_session(messages, use_llm=False)

        assert result["status"] == "success"
        assert result["distilled_count"] >= 5
        assert result["method"] == "rule_based"

        # Check nodes in storage
        nodes = mem_storage.list_nodes(limit=20)
        categories = {n["category"] for n in nodes}
        assert "preference" in categories
        assert "decision" in categories
        assert "architecture" in categories
        assert "incident" in categories
        assert "sop" in categories

    def test_distill_session_synaptic_edges_and_shared_entities(self, mem_storage: SynapseStorage) -> None:
        """Test session distillation forms co-distillation and shared entity edges."""
        worker = SynapseWorker(storage=mem_storage)

        messages = [
            {"role": "user", "content": "We decided to configure `PostgreSQL` connection pool with 20 connections."},
            {"role": "assistant", "content": "Fixed `PostgreSQL` timeout crash bug during high load stress tests."},
        ]

        result = worker.distill_session(messages, use_llm=False)
        assert result["status"] == "success"
        assert result["distilled_count"] == 2
        assert result["edges_created"] >= 2  # Bidirectional session_distilled + shares_entity edges

        node_ids = result["nodes_created"]
        n1, n2 = node_ids[0], node_ids[1]

        edges_n1 = mem_storage.get_edges(n1, direction="out")
        edge_relations = {e["relation"] for e in edges_n1 if e["target_id"] == n2}
        assert "session_distilled" in edge_relations
        assert "shares_entity" in edge_relations

    def test_distill_session_l0_abstract_truncation(self, mem_storage: SynapseStorage) -> None:
        """Test very long sentences are cleanly truncated to maintain concise L0 abstract."""
        worker = SynapseWorker(storage=mem_storage)
        long_sentence = (
            "We decided to implement " + " ".join(["very_long_spec_parameter"] * 50) + " for storage optimization."
        )
        messages = [{"role": "user", "content": long_sentence}]

        result = worker.distill_session(messages, use_llm=False)
        assert result["status"] == "success"
        assert result["distilled_count"] == 1

        node = mem_storage.get_node(result["nodes_created"][0])
        assert node is not None
        assert len(node["l0_abstract"].split()) <= 30

    def test_distill_session_mocked_llm_json_response(self, mem_storage: SynapseStorage) -> None:
        """Test distillation with mocked OpenAI-compatible LLM direct JSON response."""
        worker = SynapseWorker(storage=mem_storage, llm_config={"base_url": "http://mock-llm:8000/v1"})

        llm_response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "memories": [
                                {
                                    "l0_abstract": "User requires TypeScript strict mode enabled.",
                                    "l1_overview": "Strict null checks and noImplicitAny are mandatory across all frontend code.",
                                    "category": "preference",
                                    "importance": 0.9,
                                    "entities": ["TypeScript", "Frontend"],
                                },
                                {
                                    "l0_abstract": "Standard procedure for database migrations.",
                                    "l1_overview": "Always backup SQLite database before running migration scripts.",
                                    "category": "sop",
                                    "importance": 0.85,
                                    "entities": ["SQLite", "Migrations"],
                                },
                            ]
                        })
                    }
                }
            ]
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(llm_response_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        messages = [
            {"role": "user", "content": "Please remember that I always require TypeScript strict mode."},
            {"role": "assistant", "content": "Understood. Also always backup SQLite before database migrations."},
        ]

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = worker.distill_session(messages, use_llm=True)

        assert result["status"] == "success"
        assert result["distilled_count"] == 2
        assert result["method"] == "llm"

        nodes = mem_storage.list_nodes(limit=10)
        assert len(nodes) == 2
        categories = {n["category"] for n in nodes}
        assert "preference" in categories
        assert "sop" in categories

        # Verify vectors were automatically created for both nodes at unified 384 dimension
        for node in nodes:
            vec = mem_storage.get_vector(node["id"])
            assert vec is not None
            assert len(vec) == 384

    def test_distill_session_mocked_llm_markdown_fenced_json(self, mem_storage: SynapseStorage) -> None:
        """Test distillation parses markdown code-fenced JSON responses from LLM."""
        worker = SynapseWorker(storage=mem_storage, llm_config={"base_url": "http://mock-llm:8000/v1"})

        markdown_json_content = (
            "Here are the extracted memories from the session:\n\n"
            "```json\n"
            "{\n"
            '  "memories": [\n'
            "    {\n"
            '      "l0_abstract": "Decided to adopt Redis for session token caching.",\n'
            '      "l1_overview": "Redis was chosen due to sub-millisecond response latency under high load.",\n'
            '      "category": "decision",\n'
            '      "importance": 0.8,\n'
            '      "entities": ["Redis", "SessionCache"]\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n"
        )

        llm_response_payload = {
            "choices": [{"message": {"content": markdown_json_content}}]
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(llm_response_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        messages = [{"role": "user", "content": "We decided to adopt Redis for session caching."}]

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = worker.distill_session(messages, use_llm=True)

        assert result["status"] == "success"
        assert result["distilled_count"] == 1
        assert result["method"] == "llm"

        node = mem_storage.get_node(result["nodes_created"][0])
        assert node is not None
        assert node["category"] == "decision"
        assert "Redis" in node["l0_abstract"]

    def test_distill_session_llm_failure_falls_back_to_rule_based(self, mem_storage: SynapseStorage) -> None:
        """Test network error or HTTP failure in LLM call gracefully falls back to rule-based extraction."""
        worker = SynapseWorker(storage=mem_storage)

        messages = [
            {"role": "user", "content": "I prefer using dark theme and Neovim editor."},
            {"role": "assistant", "content": "We decided to use SQLite WAL mode for concurrency."},
        ]

        # Simulate network failure / connection refused on LLM call
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            result = worker.distill_session(messages, use_llm=True)

        assert result["status"] == "success"
        assert result["distilled_count"] >= 2
        assert result["method"] == "rule_based"

    def test_distill_session_uri_collision_updates_existing(self, mem_storage: SynapseStorage) -> None:
        """Test duplicate distillation with same URI updates the existing node."""
        worker = SynapseWorker(storage=mem_storage)

        messages1 = [{"role": "user", "content": "We decided to adopt SQLite WAL mode for the engine."}]
        res1 = worker.distill_session(messages1, use_llm=False)
        assert res1["distilled_count"] == 1
        node_id1 = res1["nodes_created"][0]

        # Distill same message again
        messages2 = [{"role": "user", "content": "We decided to adopt SQLite WAL mode for the engine."}]
        res2 = worker.distill_session(messages2, use_llm=False)
        assert res2["distilled_count"] == 1
        node_id2 = res2["nodes_created"][0]

        # Should update same node ID
        assert node_id1 == node_id2
        assert mem_storage.get_stats()["node_count"] == 1
