"""
Comprehensive verification test for Temporal Traversal, Neural Embedder, and Full Triad Hierarchy.
"""

import pytest
from synapse.storage import SynapseStorage
from synapse.retriever import SynapseRetriever
from synapse.adapter import SynapseAdapter
from synapse.embedder import SynapseEmbedder


def test_embedder_deterministic_and_neural_ready():
    embedder = SynapseEmbedder(dim=64)
    vec1 = embedder.embed("Master preferred Nordic Monolith design.")
    vec2 = embedder.embed("Master preferred Nordic Monolith design.")

    if embedder.is_neural and embedder.model is not None:
        # Neural backend: vectors are at the model's native dimension (384 for
        # all-MiniLM-L6-v2), not the requested fallback dim.
        native_dim = embedder.native_dim or 384
        assert len(vec1) == native_dim
        # Neural inference is deterministic for identical input.
        assert vec1 == vec2
    else:
        # Fallback CRC32 n-gram vectorizer: deterministic at the requested dim.
        assert len(vec1) == 64
        assert vec1 == vec2  # Pure deterministic


def test_temporal_spreading_activation_boost():
    storage = SynapseStorage(":memory:")
    retriever = SynapseRetriever(storage=storage)

    # Add 3 sequential nodes
    id1 = storage.add_node(uri="synapse://node1", l0_abstract="Setup PostgreSQL database", l1_overview="", status="active")
    id2 = storage.add_node(uri="synapse://node2", l0_abstract="Run database migration v1", l1_overview="", status="active")
    id3 = storage.add_node(uri="synapse://node3", l0_abstract="Verify database health check", l1_overview="", status="active")

    # Add temporal causal edges
    storage.add_edge(id1, id2, relation="preceded_by", weight=0.8)
    storage.add_edge(id2, id3, relation="triggered", weight=0.8)

    # Retrieve starting from node1 topic
    results = retriever.retrieve("Setup PostgreSQL database", limit=3, max_hops=2)
    result_ids = [r["id"] for r in results]
    assert id1 in result_ids
    assert id2 in result_ids  # Propagated via causal temporal edge
