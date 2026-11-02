"""
Tests for Synapse Observations (Tier 2 Proof-Count Pattern Consolidation) and Disposition-Aware Tuning.
"""

import pytest
from synapse.storage import SynapseStorage
from synapse.observation_engine import ObservationEngine
from synapse.adapter import SynapseAdapter
from synapse.retriever import SynapseRetriever


def test_observation_storage_and_reinforcement():
    storage = SynapseStorage(":memory:")
    engine = ObservationEngine(storage)

    # Ingest first evidence
    res1 = engine.ingest_turn_observation(
        "User prefers minimalist zinc dark theme with subtle grid pattern."
    )
    assert res1 is not None
    assert res1["proof_count"] == 1
    assert res1["action"] == "created"

    # Ingest second supporting evidence
    res2 = engine.ingest_turn_observation(
        "User prefers minimalist zinc dark theme for all editor layouts."
    )
    assert res2 is not None
    assert res2["proof_count"] == 2
    assert res2["action"] == "reinforced"

    # List observations
    active_obs = storage.list_observations()
    assert len(active_obs) == 1
    assert active_obs[0]["proof_count"] == 2
    assert len(active_obs[0]["supporting_facts"]) == 2


def test_disposition_tuning():
    storage = SynapseStorage(":memory:")
    # Balanced disposition (3, 3, 3)
    r_bal = SynapseRetriever(storage, disposition={"skepticism": 3, "literalism": 3, "empathy": 3})
    bal_bm25 = r_bal.bm25_weight
    bal_vec = r_bal.vector_weight

    # Skeptical & Literal disposition (5, 5, 2)
    r_skep = SynapseRetriever(storage, disposition={"skepticism": 5, "literalism": 5, "empathy": 2})
    # High skepticism & literalism should increase lexical BM25 verification weight
    assert r_skep.bm25_weight > bal_bm25
    assert r_skep.vector_weight < bal_vec


def test_adapter_recall_includes_observation():
    adapter = SynapseAdapter({"db_path": ":memory:"})
    assert adapter.storage is not None
    assert adapter.observation_engine is not None

    # Ingest an observation
    adapter.observation_engine.ingest_turn_observation(
        "All automated refactoring tasks should be executed via dedicated CLI tool."
    )

    # Recall targeting CLI coding
    recalled = adapter.recall("What are the guidelines for automated CLI refactoring?")
    assert len(recalled) > 0
    assert any("[OBSERVATION" in r for r in recalled)
    assert any("Proof:" in r for r in recalled)
