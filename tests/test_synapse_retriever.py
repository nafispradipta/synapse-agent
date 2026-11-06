"""
Unit tests for Synapse Cognitive Memory Retriever and Context Loader (Phase 2).
"""

import os
import sys
import time
from pathlib import Path
import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synapse.storage import SynapseStorage
from synapse.retriever import SynapseRetriever
from synapse.loader import SynapseContextLoader, estimate_tokens, split_sentences
from synapse.adapter import SynapseAdapter


@pytest.fixture
def mem_storage():
    """In-memory SynapseStorage fixture with pre-populated test graph."""
    storage = SynapseStorage(db_path=":memory:")
    yield storage
    storage.close()


@pytest.fixture
def populated_storage(mem_storage):
    """
    Storage populated with nodes and edges across various categories and statuses.
    Graph topology:
    - N1 (Dark Mode pref) --[requires w=0.8]--> N2 (UI Theme architecture) --[implements w=0.6]--> N3 (Nordic Palette decision)
    - N4 (Docker sandbox SOP) --[relates w=0.5]--> N5 (Security Incident)
    - N6 (Superseded old theme)
    - N7 (Archived legacy tool)
    """
    now = int(time.time())

    # Seed Node 1: Lexical match for "dark mode"
    n1_id = mem_storage.add_node(
        uri="synapse://preference/dark_mode",
        l0_abstract="Default UI theme preference set to Nordic Dark mode.",
        l1_overview="User explicitly requested dark mode styling across all TUI panels and card components.",
        l2_target_path="/config/theme.json",
        category="preference",
        importance=0.8,
        strength=1.5,
        status="active",
    )

    # 1-hop Neighbor Node 2: Architecture for theme rendering
    n2_id = mem_storage.add_node(
        uri="synapse://architecture/theme_engine",
        l0_abstract="Theme engine manages visual token palettes and style overrides.",
        l1_overview="Coordinates color tokens between prompt cards, border glows, and syntax highlighting.",
        l2_target_path="/core/theme_engine.py",
        category="architecture",
        importance=0.5,
        strength=1.0,
        status="active",
    )

    # 2-hop Neighbor Node 3: Nordic palette decision
    n3_id = mem_storage.add_node(
        uri="synapse://decision/nordic_palette",
        l0_abstract="Adopted Nordic frost palette for visual harmony.",
        l1_overview="Polar night dark backgrounds with cyan and frost accents for high readability.",
        category="decision",
        importance=0.6,
        strength=1.2,
        status="active",
    )

    # Isolated Cluster Node 4 & 5: Docker & Security
    n4_id = mem_storage.add_node(
        uri="synapse://sop/docker_sandbox",
        l0_abstract="Standard operating procedure for Docker container isolation.",
        l1_overview="All external code execution must occur within gVisor-sandboxed containers.",
        category="sop",
        importance=0.9,
        strength=1.0,
        status="active",
    )

    n5_id = mem_storage.add_node(
        uri="synapse://incident/cve_container_escape",
        l0_abstract="Incident post-mortem on container namespace vulnerability.",
        l1_overview="Hardened seccomp profile applied after vulnerability report in legacy daemon.",
        category="incident",
        importance=0.7,
        strength=0.9,
        status="active",
    )

    # Node 6: Superseded node with "dark mode" keyword
    n6_id = mem_storage.add_node(
        uri="synapse://preference/legacy_dark_mode",
        l0_abstract="Old dark mode configuration from v1.",
        l1_overview="Superseded by Nordic dark mode configuration.",
        category="preference",
        importance=0.5,
        strength=0.5,
        status="superseded",
    )

    # Node 7: Archived node
    n7_id = mem_storage.add_node(
        uri="synapse://general/archived_notes",
        l0_abstract="Archived system notes.",
        l1_overview="General legacy archives.",
        category="general",
        importance=0.3,
        strength=0.5,
        status="archived",
    )

    # Edges: N1 -> N2 (w=0.8), N2 -> N3 (w=0.6), N4 -> N5 (w=0.5)
    mem_storage.add_edge(n1_id, n2_id, relation="requires", weight=0.8)
    mem_storage.add_edge(n2_id, n3_id, relation="implements", weight=0.6)
    mem_storage.add_edge(n4_id, n5_id, relation="mitigates", weight=0.5)

    return {
        "storage": mem_storage,
        "n1": n1_id,
        "n2": n2_id,
        "n3": n3_id,
        "n4": n4_id,
        "n5": n5_id,
        "n6": n6_id,
        "n7": n7_id,
    }


class TestSynapseRetrieverLexicalAndNormalization:
    def test_empty_and_whitespace_query(self, populated_storage):
        retriever = SynapseRetriever(storage=populated_storage["storage"])
        assert retriever.retrieve("") == []
        assert retriever.retrieve("   ") == []

    def test_lexical_seed_retrieval_and_bm25_normalization(self, populated_storage):
        storage = populated_storage["storage"]
        retriever = SynapseRetriever(storage=storage)

        # Query matches "Nordic Dark mode" in N1
        results = retriever.retrieve("Nordic Dark mode", limit=5, max_hops=0)
        assert len(results) >= 1
        top = results[0]
        assert top["id"] == populated_storage["n1"]
        assert top["bm25_score"] == 1.0  # Normalized top score
        assert top["composite_score"] > 0.0
        assert "composite_score" in top
        assert "activation_energy" in top
        assert "recency_score" in top

    def test_status_filtering_ignores_superseded_and_archived(self, populated_storage):
        storage = populated_storage["storage"]
        retriever = SynapseRetriever(storage=storage)

        # "dark mode" is in N1 (active) and N6 (superseded)
        results = retriever.retrieve("dark mode", limit=10, max_hops=0)
        result_ids = [r["id"] for r in results]

        assert populated_storage["n1"] in result_ids
        assert populated_storage["n6"] not in result_ids
        assert populated_storage["n7"] not in result_ids


class TestHebbianSpreadingActivation:
    def test_spreading_activation_1_hop(self, populated_storage):
        storage = populated_storage["storage"]
        retriever = SynapseRetriever(storage=storage)

        # N1 has "Nordic Dark mode". N2 does not have "Nordic Dark mode" in text, but is connected to N1 (w=0.8).
        # When max_hops=1, N2 should be activated and included in results!
        results_0_hops = retriever.retrieve("Nordic Dark mode", limit=5, max_hops=0)
        results_1_hop = retriever.retrieve("Nordic Dark mode", limit=5, max_hops=1)

        ids_0 = [r["id"] for r in results_0_hops]
        ids_1 = [r["id"] for r in results_1_hop]

        assert populated_storage["n1"] in ids_0
        assert populated_storage["n2"] not in ids_0  # No lexical match

        assert populated_storage["n1"] in ids_1
        assert populated_storage["n2"] in ids_1  # Activated via 1-hop edge
        assert populated_storage["n3"] not in ids_1  # 2-hop away, so not in 1-hop

        # Verify N2 has positive activation energy but 0 bm25 score
        n2_res = next(r for r in results_1_hop if r["id"] == populated_storage["n2"])
        assert n2_res["activation_energy"] > 0.0
        assert n2_res["bm25_score"] == 0.0

    def test_spreading_activation_2_hops(self, populated_storage):
        storage = populated_storage["storage"]
        retriever = SynapseRetriever(storage=storage)

        # When max_hops=2, N3 (connected to N2) should also be activated
        results_2_hops = retriever.retrieve("Nordic Dark mode", limit=5, max_hops=2)
        ids_2 = [r["id"] for r in results_2_hops]

        assert populated_storage["n1"] in ids_2
        assert populated_storage["n2"] in ids_2
        assert populated_storage["n3"] in ids_2

        # Check energy decay: N1 energy > N2 energy > N3 energy
        n1_res = next(r for r in results_2_hops if r["id"] == populated_storage["n1"])
        n2_res = next(r for r in results_2_hops if r["id"] == populated_storage["n2"])
        n3_res = next(r for r in results_2_hops if r["id"] == populated_storage["n3"])

        assert n1_res["activation_energy"] >= n2_res["activation_energy"]
        assert n2_res["activation_energy"] >= n3_res["activation_energy"]


class TestRecencyAndStrengthScoring:
    def test_recency_decay_and_strength_weighting(self, mem_storage):
        retriever = SynapseRetriever(storage=mem_storage, half_life_seconds=86400.0)  # 1 day half-life
        now = int(time.time())

        # Fresh node (accessed right now)
        n_fresh = mem_storage.add_node(
            uri="synapse://decision/fresh",
            l0_abstract="Fresh architecture decision on caching.",
            strength=1.0,
            status="active",
        )
        # Old node (accessed 2 days ago = 2 half-lives)
        n_old = mem_storage.add_node(
            uri="synapse://decision/old",
            l0_abstract="Old architecture decision on caching.",
            strength=1.0,
            status="active",
        )
        mem_storage.update_node(n_old, last_accessed=now - (86400 * 2))

        # Query "caching" matching both
        results = retriever.retrieve("caching", limit=5, max_hops=0, touch=False)
        assert len(results) == 2

        fresh_res = next(r for r in results if r["id"] == n_fresh)
        old_res = next(r for r in results if r["id"] == n_old)

        # Fresh node recency score should be ~1.0, old node recency score should be ~0.25 (0.5^2)
        assert fresh_res["recency_score"] > 0.95
        assert old_res["recency_score"] < 0.35
        assert fresh_res["composite_score"] > old_res["composite_score"]

    def test_touch_reinforcement(self, mem_storage):
        retriever = SynapseRetriever(storage=mem_storage)
        nid = mem_storage.add_node(
            uri="synapse://preference/editor_cursor",
            l0_abstract="Cursor style configured to block blinking.",
            strength=1.0,
            status="active",
        )

        initial_node = mem_storage.get_node(nid)
        assert initial_node["strength"] == 1.0

        # Retrieve with touch=True
        results = retriever.retrieve("cursor", limit=1, touch=True)
        assert len(results) == 1
        assert results[0]["strength"] == 1.1

        # Check in storage
        updated_node = mem_storage.get_node(nid)
        assert updated_node["strength"] == 1.1

        # Retrieve with touch=False (strength should remain 1.1)
        results2 = retriever.retrieve("cursor", limit=1, touch=False)
        assert len(results2) == 1
        assert results2[0]["strength"] == 1.1
        assert mem_storage.get_node(nid)["strength"] == 1.1


class TestSynapseContextLoader:
    def test_estimate_tokens_and_split_sentences(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("hello") > 0
        assert estimate_tokens("a" * 40) == 10

        sentences = split_sentences("First sentence. Second sentence! Third sentence? Fourth.")
        assert len(sentences) == 4
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence!"
        assert sentences[2] == "Third sentence?"
        assert sentences[3] == "Fourth."

    def test_synapse_uri_parsing_and_resolving(self, populated_storage):
        storage = populated_storage["storage"]
        loader = SynapseContextLoader(storage=storage)

        parsed = loader.parse_synapse_uri("synapse://preference/dark_mode#v2")
        assert parsed["scheme"] == "synapse"
        assert parsed["category"] == "preference"
        assert parsed["path"] == "dark_mode"
        assert parsed["fragment"] == "v2"

        resolved = loader.resolve_synapse_uri("synapse://preference/dark_mode")
        assert resolved is not None
        assert resolved["id"] == populated_storage["n1"]

        # Non-existent URI
        assert loader.resolve_synapse_uri("synapse://preference/non_existent") is None

    def test_format_l0_and_l1(self, populated_storage):
        storage = populated_storage["storage"]
        loader = SynapseContextLoader(storage=storage)
        node = storage.get_node(populated_storage["n1"])

        l0 = loader.format_l0_abstract(node)
        assert "• [PREFERENCE]" in l0
        assert "synapse://preference/dark_mode" in l0
        assert "[ref: /config/theme.json]" in l0

        l1 = loader.format_l1_overview(node)
        assert "Overview:" in l1
        assert "dark mode styling" in l1

    def test_pack_context_strict_token_budget(self, populated_storage):
        storage = populated_storage["storage"]
        retriever = SynapseRetriever(storage=storage)
        loader = SynapseContextLoader(storage=storage)

        nodes = retriever.retrieve("Nordic Dark mode", limit=5, max_hops=2, touch=False)
        assert len(nodes) >= 3

        # Test generous budget (e.g. 500 tokens)
        packed_500 = loader.pack_context(nodes, token_budget=500, include_l1=True)
        assert len(packed_500) > 0
        assert estimate_tokens(packed_500) <= 500
        assert "### Active Synapse Context:" in packed_500
        assert "Overview:" in packed_500

        # Test tight budget (e.g. 50 tokens)
        packed_50 = loader.pack_context(nodes, token_budget=50, include_l1=False)
        assert len(packed_50) > 0
        assert estimate_tokens(packed_50) <= 50

        # Test impossible budget (e.g. 5 tokens - smaller than header)
        packed_tiny = loader.pack_context(nodes, token_budget=5)
        assert packed_tiny == ""

        # Test empty nodes
        assert loader.pack_context([], token_budget=250) == ""

    def test_pack_context_avoids_mid_sentence_truncation(self, mem_storage):
        loader = SynapseContextLoader(storage=mem_storage)

        # Node with multi-sentence L1 overview
        node = {
            "id": "test_node",
            "uri": "synapse://sop/incident_response",
            "category": "sop",
            "l0_abstract": "Incident triage protocol.",
            "l1_overview": "Step 1: Isolate system. Step 2: Notify on-call. Step 3: Run forensic snapshot.",
            "importance": 0.8,
            "strength": 1.0,
            "status": "active",
        }

        # Pack with a constrained budget that can only fit 1 or 2 sentences
        packed = loader.pack_context([node], token_budget=45, include_l1=True)
        assert estimate_tokens(packed) <= 45
        
        # Verify sentences end with proper terminal punctuation (not truncated mid-sentence)
        overview_line = [line for line in packed.split("\n") if "Overview:" in line]
        if overview_line:
            content = overview_line[0]
            assert content.endswith(".")


class TestTrueHybridRRFAndLockSafety:
    def test_true_hybrid_retrieval_simultaneous_fts_and_vector(self, mem_storage):
        """Test FTS5 and Dense Vector Search run simultaneously and combine via RRF."""
        # Node 1: Matches keyword 'Kubernetes' exactly
        n1 = mem_storage.add_node(
            uri="synapse://architecture/k8s",
            l0_abstract="Kubernetes cluster orchestration platform configuration.",
            category="architecture",
        )
        # Node 2: Semantic match for container orchestration without 'Kubernetes' keyword
        n2 = mem_storage.add_node(
            uri="synapse://architecture/container_infra",
            l0_abstract="Containerized service deployment and pod management infrastructure.",
            category="architecture",
        )

        retriever = SynapseRetriever(storage=mem_storage)
        # Query contains 'Kubernetes' so FTS finds n1. Vector search also finds n2.
        results = retriever.retrieve("Kubernetes orchestration", limit=5, max_hops=0)
        result_ids = [r["id"] for r in results]

        assert n1 in result_ids
        # Both lexical and dense seeds must have rrf_score
        for r in results:
            assert "rrf_score" in r
            assert r["rrf_score"] > 0.0

    def test_retrieval_does_not_mutate_belief_state(self, mem_storage):
        """Retrieval may hide a contradiction in the response, never in storage."""
        n1 = mem_storage.add_node(
            uri="synapse://decision/postgres_use",
            l0_abstract="Always prefer PostgreSQL for production relational persistence.",
            category="decision",
            salience=0.85,
            confidence=0.95,
        )
        n2 = mem_storage.add_node(
            uri="synapse://decision/postgres_avoid",
            l0_abstract="Never use PostgreSQL, avoid postgres in all services.",
            category="decision",
            salience=0.50,
            confidence=0.50,
        )

        retriever = SynapseRetriever(storage=mem_storage)
        results = retriever.retrieve(
            "PostgreSQL database preference", limit=5, touch=False
        )

        assert mem_storage.get_node(n1)["status"] == "active"
        assert mem_storage.get_node(n2)["status"] == "active"
        assert n2 not in [item["id"] for item in results]

        retriever.retrieve("PostgreSQL database preference", limit=5, touch=True)
        assert mem_storage.get_node(n2)["status"] == "active"



def test_retrieve_tolerates_legacy_null_scores(tmp_path):
    # Simulate rows written by older releases / external tooling.
    storage = SynapseStorage(str(tmp_path / "legacy.db"))
    node_id = storage.add_node(
        uri="synapse://general/legacy-null",
        l0_abstract="legacy null scores",
        category="general",
        salience=0.5,
        importance=0.5,
        confidence=1.0,
    )
    with storage._lock:
        storage._conn.execute(
            "UPDATE synapse_nodes SET salience = NULL, importance = NULL, "
            "confidence = NULL, strength = NULL WHERE id = ?",
            (node_id,),
        )
        storage._conn.commit()

    retriever = SynapseRetriever(storage=storage)
    results = retriever.retrieve("legacy null scores", limit=3, touch=False)
    assert results, "NULL legacy fields must not break retrieval"
    assert results[0]["salience"] == 0.5
    storage.close()


def test_recall_logs_retrieval_failure(caplog):
    adapter = SynapseAdapter({"db_path": ":memory:"})

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    adapter.retriever.retrieve = _boom
    with caplog.at_level("WARNING", logger="synapse.adapter"):
        assert adapter.recall("anything") == []
    assert any("Retriever query failed" in r.getMessage() for r in caplog.records)
    adapter.storage.close()

