"""
Tests for Real-Time In-Flight Vectorization in SynapseAdapter.
Validates that retain() and retain_episode() compute dense vectors immediately,
handle model update semantics, and fail gracefully when the embedder is unavailable.
"""

import os
from pathlib import Path
from typing import List, Optional
import pytest

from synapse.adapter import SynapseAdapter
from synapse.embedder import SynapseEmbedder


def test_inflight_vectorization_on_retain(tmp_path: Path) -> None:
    db_file = str(tmp_path / "test_synapse.db")
    adapter = SynapseAdapter(config={"db_path": db_file})

    uri = "synapse://decision/in-flight-vector-test"
    content = "We use real-time in-flight vectorization for fast semantic indexing."
    ok = adapter.retain(
        content=content,
        context="Architecture discussion",
        metadata={"uri": uri, "category": "decision"},
    )
    assert ok is True

    node = adapter.storage.get_node(uri)
    assert node is not None
    node_id = node["id"]

    vec = adapter.storage.get_vector(node_id)
    assert vec is not None
    assert isinstance(vec, list)
    assert len(vec) in (64, 384)


def test_inflight_vectorization_updates_on_re_retain(tmp_path: Path) -> None:
    db_file = str(tmp_path / "test_synapse.db")
    adapter = SynapseAdapter(config={"db_path": db_file})

    uri = "synapse://preference/editor-choice"
    adapter.retain(
        content="We prefer Neovim for daily terminal editing.",
        metadata={"uri": uri, "category": "preference"},
    )
    node = adapter.storage.get_node(uri)
    vec1 = adapter.storage.get_vector(node["id"])
    assert vec1 is not None

    # Re-retain with different content
    adapter.retain(
        content="We switch entirely to VS Code remote containers for web projects.",
        metadata={"uri": uri, "category": "preference"},
    )
    vec2 = adapter.storage.get_vector(node["id"])
    assert vec2 is not None
    assert vec1 != vec2


def test_inflight_vectorization_graceful_fallback(tmp_path: Path) -> None:
    db_file = str(tmp_path / "test_synapse.db")

    class CrashingEmbedder:
        def embed(self, text: str) -> List[float]:
            raise RuntimeError("Embedder service unreachable")

    adapter = SynapseAdapter(config={"db_path": db_file, "embedder": CrashingEmbedder()})

    # retain() should succeed despite crashing embedder
    ok = adapter.retain(
        content="Resilient memory storage despite embedding failure.",
        metadata={"uri": "synapse://resilience/test-crash"},
    )
    assert ok is True

    node = adapter.storage.get_node("synapse://resilience/test-crash")
    assert node is not None

    # retain_episode() should succeed despite crashing embedder
    ep_id = adapter.retain_episode(
        title="Crashing Embedder Incident",
        situation="Embedder crashed during runtime",
        action_taken="Graceful fallback caught exception",
        outcome="Storage succeeded",
        lessons_learned="Always isolate non-critical enrichment pipelines",
    )
    assert ep_id != ""
    ep = adapter.get_episode(ep_id)
    assert ep is not None
    assert ep["title"] == "Crashing Embedder Incident"
