"""
Unit tests for Hebbian Outcome Feedback Loop (Active Plasticity: LTP and LTD).
Validates:
1. apply_outcome_feedback in SynapseWorker with LTP (+delta) and LTD (-delta_ltd).
2. Node strength reinforcement on LTP.
3. Node confidence penalty and threshold-based node flagging on LTD.
4. Salience Protection: Nodes with salience >= 0.95 are completely protected from confidence decay/penalties.
5. Exposing apply_outcome_feedback in SynapseAdapter (both direct node_ids and recall context).
6. Robustness against various outcome representations (bool, string variants), custom parameters, and edge cases.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synapse.adapter import SynapseAdapter
from synapse.storage import SynapseStorage
from synapse.worker import (
    DEFAULT_CONFIDENCE_PENALTY,
    DEFAULT_LTD_DELTA,
    DEFAULT_LTP_DELTA,
    FLAG_CONFIDENCE_THRESHOLD,
    SALIENCE_PROTECTION_THRESHOLD,
    SynapseWorker,
)


# ============================================================================
# 1. Outcome Parsing and String / Bool Variants
# ============================================================================

def test_outcome_feedback_empty_or_missing_nodes(tmp_path: Path) -> None:
    """Empty or None node_ids should return a safe dictionary with zero updates."""
    storage = SynapseStorage(str(tmp_path / "test_empty.db"))
    worker = SynapseWorker(storage)

    res1 = worker.apply_outcome_feedback([], outcome=True, note="empty list")
    assert res1["is_success"] is True
    assert res1["outcome"] == "success"
    assert res1["edges_updated"] == 0
    assert res1["nodes_updated"] == 0
    assert res1["note"] == "empty list"

    res2 = worker.apply_outcome_feedback([], outcome=False, note="failure empty")
    assert res2["is_success"] is False
    assert res2["outcome"] == "failure"
    assert res2["edges_updated"] == 0
    assert res2["nodes_updated"] == 0


def test_outcome_parsing_variants(tmp_path: Path) -> None:
    """Validate parsing of various outcome strings (case-insensitive) and boolean values."""
    storage = SynapseStorage(str(tmp_path / "test_variants.db"))
    worker = SynapseWorker(storage)

    id_a = storage.add_node(uri="synapse://test/a", l0_abstract="Node A")
    id_b = storage.add_node(uri="synapse://test/b", l0_abstract="Node B")

    success_variants = [True, "success", "SUCCESS", "ltp", "pass", "PASSED", "positive", "reward", "win", "good"]
    for val in success_variants:
        res = worker.apply_outcome_feedback([id_a, id_b], outcome=val)
        assert res["is_success"] is True, f"Failed for variant {val!r}"
        assert res["outcome"] == "success"

    failure_variants = [False, "failure", "FAILURE", "ltd", "fail", "FAILED", "negative", "penalty", "loss", "bad", "error"]
    for val in failure_variants:
        res = worker.apply_outcome_feedback([id_a, id_b], outcome=val)
        assert res["is_success"] is False, f"Failed for variant {val!r}"
        assert res["outcome"] == "failure"


# ============================================================================
# 2. Long-Term Potentiation (LTP / Success)
# ============================================================================

def test_ltp_reinforcement_edges_and_node_strength(tmp_path: Path) -> None:
    """
    On success (LTP):
    - Edges between co-activated nodes are strengthened by +delta (default 0.1).
    - Node strength is reinforced by +delta (default 0.1).
    """
    storage = SynapseStorage(str(tmp_path / "test_ltp.db"))
    worker = SynapseWorker(storage)

    id_a = storage.add_node(uri="synapse://sop/step1", l0_abstract="Step 1", strength=1.0)
    id_b = storage.add_node(uri="synapse://sop/step2", l0_abstract="Step 2", strength=1.0)

    # Initial state: no edges between A and B
    assert len(storage.get_edges(id_a, direction="out")) == 0

    # 1. Apply LTP feedback
    result = worker.apply_outcome_feedback([id_a, id_b], outcome=True, note="Workflow step succeeded")
    assert result["is_success"] is True
    assert result["outcome"] == "success"
    assert result["edges_updated"] == 2  # Bidirectional: A->B and B->A
    assert result["edges_reinforced"] == 2
    assert result["nodes_updated"] == 2
    assert set(result["reinforced_nodes"]) == {id_a, id_b}
    assert result["note"] == "Workflow step succeeded"

    # Verify edge weights: base 0.5 + 0.1 = 0.6
    edges_a = storage.get_edges(id_a, direction="out")
    assert len(edges_a) == 1
    assert edges_a[0]["target_id"] == id_b
    assert round(edges_a[0]["weight"], 2) == 0.6

    edges_b = storage.get_edges(id_b, direction="out")
    assert len(edges_b) == 1
    assert edges_b[0]["target_id"] == id_a
    assert round(edges_b[0]["weight"], 2) == 0.6

    # Verify node strength: 1.0 + 0.1 = 1.1
    node_a = storage.get_node(id_a)
    node_b = storage.get_node(id_b)
    assert node_a is not None and round(node_a["strength"], 2) == 1.1
    assert node_b is not None and round(node_b["strength"], 2) == 1.1

    # 2. Apply second LTP feedback with custom delta=0.2
    result2 = worker.apply_outcome_feedback([id_a, id_b], outcome="success", delta=0.2)
    assert result2["is_success"] is True
    edge_ab = next(e for e in storage.get_edges(id_a, direction="out") if e["target_id"] == id_b)
    assert round(edge_ab["weight"], 2) == 0.8  # 0.6 + 0.2

    node_a_2 = storage.get_node(id_a)
    assert node_a_2 is not None and round(node_a_2["strength"], 2) == 1.3  # 1.1 + 0.2


def test_ltp_edge_weight_capped_at_one(tmp_path: Path) -> None:
    """Edge weight should never exceed 1.0 even after multiple strong LTP reinforcements."""
    storage = SynapseStorage(str(tmp_path / "test_ltp_cap.db"))
    worker = SynapseWorker(storage)

    id_a = storage.add_node(uri="synapse://test/cap_a", l0_abstract="Node Cap A")
    id_b = storage.add_node(uri="synapse://test/cap_b", l0_abstract="Node Cap B")

    # Manually add edge with weight 0.95
    storage.add_edge(id_a, id_b, relation="co_activated", weight=0.95)

    worker.apply_outcome_feedback([id_a, id_b], outcome=True, delta=0.1)
    edge = next(e for e in storage.get_edges(id_a, direction="out") if e["target_id"] == id_b)
    assert edge["weight"] == 1.0


# ============================================================================
# 3. Long-Term Depression (LTD / Failure) & Node Flagging
# ============================================================================

def test_ltd_depression_edges_and_confidence_penalty(tmp_path: Path) -> None:
    """
    On failure (LTD):
    - Edges are depressed by -delta_ltd (default 0.15).
    - Node confidence is penalized by -confidence_penalty (default -0.1).
    - Active nodes whose confidence remains >= 0.3 stay active.
    """
    storage = SynapseStorage(str(tmp_path / "test_ltd.db"))
    worker = SynapseWorker(storage)

    id_a = storage.add_node(uri="synapse://test/ltd_a", l0_abstract="LTD A", confidence=0.8, salience=0.5)
    id_b = storage.add_node(uri="synapse://test/ltd_b", l0_abstract="LTD B", confidence=0.9, salience=0.6)

    # Establish initial edge with weight 0.6
    storage.add_edge(id_a, id_b, relation="co_activated", weight=0.6)
    storage.add_edge(id_b, id_a, relation="co_activated", weight=0.6)

    # Apply LTD
    res = worker.apply_outcome_feedback([id_a, id_b], outcome="failure", note="Command exited with error code 1")
    assert res["is_success"] is False
    assert res["outcome"] == "failure"
    assert res["edges_updated"] == 2
    assert res["edges_depressed"] == 2
    assert res["nodes_updated"] == 2
    assert set(res["penalized_nodes"]) == {id_a, id_b}
    assert res["flagged_nodes"] == []  # Neither dropped below 0.3

    # Check edges depressed: 0.6 - 0.15 = 0.45
    edge_a = next(e for e in storage.get_edges(id_a, direction="out") if e["target_id"] == id_b)
    edge_b = next(e for e in storage.get_edges(id_b, direction="out") if e["target_id"] == id_a)
    assert round(edge_a["weight"], 2) == 0.45
    assert round(edge_b["weight"], 2) == 0.45

    # Check confidence penalized: 0.8 - 0.1 = 0.7, 0.9 - 0.1 = 0.8
    node_a = storage.get_node(id_a)
    node_b = storage.get_node(id_b)
    assert node_a is not None and round(node_a["confidence"], 2) == 0.7
    assert node_b is not None and round(node_b["confidence"], 2) == 0.8
    assert node_a["status"] == "active"
    assert node_b["status"] == "active"


def test_ltd_node_flagged_when_confidence_drops_below_threshold(tmp_path: Path) -> None:
    """
    If confidence drops below 0.3 after LTD penalty, node status is updated to 'flagged'
    and included in flagged_nodes list.
    """
    storage = SynapseStorage(str(tmp_path / "test_flag.db"))
    worker = SynapseWorker(storage)

    # Node with confidence 0.35 -> after -0.1 it becomes 0.25 (< 0.3)
    id_low = storage.add_node(
        uri="synapse://test/fragile",
        l0_abstract="Fragile hypothesis",
        confidence=0.35,
        salience=0.5,
    )
    # Node with confidence 0.8 -> after -0.1 it becomes 0.7 (>= 0.3)
    id_stable = storage.add_node(
        uri="synapse://test/stable",
        l0_abstract="Stable assumption",
        confidence=0.8,
        salience=0.5,
    )

    res = worker.apply_outcome_feedback([id_low, id_stable], outcome=False, note="Hypothesis was contradicted")
    assert res["is_success"] is False
    assert id_low in res["flagged_nodes"]
    assert id_stable not in res["flagged_nodes"]
    assert id_low in res["penalized_nodes"]
    assert id_stable in res["penalized_nodes"]

    # Verify storage status
    low_node = storage.get_node(id_low)
    stable_node = storage.get_node(id_stable)
    assert low_node is not None
    assert stable_node is not None

    assert round(low_node["confidence"], 2) == 0.25
    assert low_node["status"] == "flagged"

    assert round(stable_node["confidence"], 2) == 0.7
    assert stable_node["status"] == "active"


def test_ltd_edge_weight_floor_at_zero(tmp_path: Path) -> None:
    """Edge weights in LTD must not drop below 0.0."""
    storage = SynapseStorage(str(tmp_path / "test_ltd_zero.db"))
    worker = SynapseWorker(storage)

    id_a = storage.add_node(uri="synapse://test/z1", l0_abstract="Node Z1")
    id_b = storage.add_node(uri="synapse://test/z2", l0_abstract="Node Z2")

    storage.add_edge(id_a, id_b, relation="co_activated", weight=0.08)

    worker.apply_outcome_feedback([id_a, id_b], outcome=False, delta_ltd=0.15)
    edge = next(e for e in storage.get_edges(id_a, direction="out") if e["target_id"] == id_b)
    assert edge["weight"] == 0.0


# ============================================================================
# 4. Salience Protection (Master Directives >= 0.95)
# ============================================================================

def test_salience_protection_master_directives(tmp_path: Path) -> None:
    """
    Mandate: Master directives (salience >= 0.95) must be completely protected
    from LTD confidence penalties and decay.
    - salience >= 0.95: confidence remains unchanged, node is NOT flagged, placed in protected_nodes.
    - salience < 0.95: confidence is penalized, can be flagged if below threshold.
    """
    storage = SynapseStorage(str(tmp_path / "test_salience_prot.db"))
    worker = SynapseWorker(storage)

    # 1. Master Directive node: salience 0.98, confidence 1.0
    id_master = storage.add_node(
        uri="synapse://sop/master_security_rule",
        l0_abstract="Master Directive: Always authenticate upstream requests.",
        salience=0.98,
        confidence=1.0,
    )

    # 2. Master Directive node with already low confidence: salience 0.95, confidence 0.2
    # Even though confidence is < 0.3, because salience >= 0.95 it must be protected from LTD penalty & flagging
    id_master_low_conf = storage.add_node(
        uri="synapse://sop/core_constitution",
        l0_abstract="Core Constitution Rule",
        salience=0.95,
        confidence=0.2,
        status="active",
    )

    # 3. Regular node: salience 0.94 (just below 0.95 threshold)
    id_near_threshold = storage.add_node(
        uri="synapse://sop/regular_high_salience",
        l0_abstract="Near-master node",
        salience=0.94,
        confidence=1.0,
    )

    # 4. Fragile regular node: salience 0.5, confidence 0.32
    id_regular = storage.add_node(
        uri="synapse://sop/regular_temporary",
        l0_abstract="Temporary step",
        salience=0.5,
        confidence=0.32,
    )

    # Apply LTD failure outcome to all four
    node_ids = [id_master, id_master_low_conf, id_near_threshold, id_regular]
    res = worker.apply_outcome_feedback(node_ids, outcome="failure", note="Catastrophic cluster outage")

    assert res["is_success"] is False
    # Protected nodes check
    assert id_master in res["protected_nodes"]
    assert id_master_low_conf in res["protected_nodes"]
    assert id_near_threshold not in res["protected_nodes"]
    assert id_regular not in res["protected_nodes"]

    # Penalized nodes check
    assert id_master not in res["penalized_nodes"]
    assert id_master_low_conf not in res["penalized_nodes"]
    assert id_near_threshold in res["penalized_nodes"]
    assert id_regular in res["penalized_nodes"]

    # Flagged nodes check: id_regular (0.32 - 0.1 = 0.22 < 0.3) must be flagged
    assert id_regular in res["flagged_nodes"]
    assert id_master not in res["flagged_nodes"]
    assert id_master_low_conf not in res["flagged_nodes"]

    # Verify storage records
    master_node = storage.get_node(id_master)
    master_low = storage.get_node(id_master_low_conf)
    near_node = storage.get_node(id_near_threshold)
    reg_node = storage.get_node(id_regular)

    assert master_node is not None
    assert master_low is not None
    assert near_node is not None
    assert reg_node is not None

    # Master directive unchanged
    assert master_node["confidence"] == 1.0
    assert master_node["status"] == "active"

    # Master low confidence unchanged
    assert master_low["confidence"] == 0.2
    assert master_low["status"] == "active"

    # Near threshold was penalized (1.0 - 0.1 = 0.9)
    assert round(near_node["confidence"], 2) == 0.9
    assert near_node["status"] == "active"

    # Regular node was penalized and flagged (0.32 - 0.1 = 0.22)
    assert round(reg_node["confidence"], 2) == 0.22
    assert reg_node["status"] == "flagged"


def test_salience_protection_preserves_plasticity_of_associations(tmp_path: Path) -> None:
    """
    Even though the master directive's internal confidence is protected,
    the synaptic association edge between the master directive and the failed action
    MUST still be depressed. (Depress edge weight, protect core node).
    """
    storage = SynapseStorage(str(tmp_path / "test_assoc_plasticity.db"))
    worker = SynapseWorker(storage)

    id_directive = storage.add_node(
        uri="synapse://directive/zero_trust",
        l0_abstract="Zero trust architecture rule",
        salience=0.99,
        confidence=1.0,
    )
    id_misconfigured_action = storage.add_node(
        uri="synapse://action/skip_verify",
        l0_abstract="Action: skip ssl verification",
        salience=0.4,
        confidence=0.7,
    )

    # Pre-existing edge
    storage.add_edge(id_directive, id_misconfigured_action, relation="co_activated", weight=0.7)
    storage.add_edge(id_misconfigured_action, id_directive, relation="co_activated", weight=0.7)

    worker.apply_outcome_feedback([id_directive, id_misconfigured_action], outcome=False, delta_ltd=0.15)

    # Master directive confidence is protected
    assert storage.get_node(id_directive)["confidence"] == 1.0

    # Misconfigured action confidence is penalized
    assert round(storage.get_node(id_misconfigured_action)["confidence"], 2) == 0.6

    # Synaptic edge between them is depressed from 0.7 to 0.55
    edge = next(e for e in storage.get_edges(id_directive, direction="out") if e["target_id"] == id_misconfigured_action)
    assert round(edge["weight"], 2) == 0.55


# ============================================================================
# 5. Single Node Outcome Feedback
# ============================================================================

def test_single_node_outcome_feedback(tmp_path: Path) -> None:
    """Outcome feedback applied to a single node updates node metrics without edges."""
    storage = SynapseStorage(str(tmp_path / "test_single.db"))
    worker = SynapseWorker(storage)

    id_single = storage.add_node(uri="synapse://single/test", l0_abstract="Single node", strength=1.0, confidence=0.8)

    # 1. LTP on single node
    res_ltp = worker.apply_outcome_feedback([id_single], outcome=True, delta=0.1)
    assert res_ltp["edges_updated"] == 0
    assert res_ltp["nodes_updated"] == 1
    assert res_ltp["reinforced_nodes"] == [id_single]
    assert round(storage.get_node(id_single)["strength"], 2) == 1.1

    # 2. LTD on single node
    res_ltd = worker.apply_outcome_feedback([id_single], outcome=False, confidence_penalty=0.1)
    assert res_ltd["edges_updated"] == 0
    assert res_ltd["nodes_updated"] == 1
    assert res_ltd["penalized_nodes"] == [id_single]
    assert round(storage.get_node(id_single)["confidence"], 2) == 0.7


# ============================================================================
# 6. SynapseAdapter Integration
# ============================================================================

def test_adapter_apply_outcome_feedback_direct(tmp_path: Path) -> None:
    """Test SynapseAdapter.apply_outcome_feedback directly forwarding to worker."""
    db_file = str(tmp_path / "test_adapter_direct.db")
    adapter = SynapseAdapter({"db_path": db_file})

    id_a = adapter.storage.add_node(uri="synapse://adapter/a", l0_abstract="Adapter A", strength=1.0)
    id_b = adapter.storage.add_node(uri="synapse://adapter/b", l0_abstract="Adapter B", strength=1.0)

    # Call through adapter
    res = adapter.apply_outcome_feedback(
        node_ids=[id_a, id_b],
        outcome="success",
        note="Integrated adapter test",
    )

    assert res["is_success"] is True
    assert res["edges_updated"] == 2
    assert res["note"] == "Integrated adapter test"

    # Verify storage updated
    node_a = adapter.storage.get_node(id_a)
    assert node_a is not None and round(node_a["strength"], 2) == 1.1


def test_adapter_apply_outcome_feedback_with_recall_context(tmp_path: Path) -> None:
    """
    Test SynapseAdapter.apply_outcome_feedback when node_ids=None:
    it automatically targets nodes co-activated during the most recent recall().
    """
    db_file = str(tmp_path / "test_adapter_recall.db")
    adapter = SynapseAdapter({"db_path": db_file})

    # Register nodes
    id1 = adapter.storage.add_node(
        uri="synapse://sop/docker_deploy",
        l0_abstract="Deploying services using Docker compose in production.",
        confidence=0.8,
    )
    id2 = adapter.storage.add_node(
        uri="synapse://sop/docker_network",
        l0_abstract="Docker network bridge configuration and port bindings.",
        confidence=0.8,
    )

    # Trigger recall
    results = adapter.recall("Docker compose deploy network", limit=5)
    assert len(results) > 0

    # Ensure adapter tracked the retrieved nodes
    assert hasattr(adapter, "_last_retrieved_node_ids")
    assert id1 in adapter._last_retrieved_node_ids
    assert id2 in adapter._last_retrieved_node_ids

    # Call outcome feedback with node_ids=None (implicitly using last recalled nodes)
    feedback_res = adapter.apply_outcome_feedback(
        node_ids=None,
        outcome="failure",
        note="Deployment failed due to port conflict",
    )

    assert feedback_res["is_success"] is False
    assert id1 in feedback_res["penalized_nodes"]
    assert id2 in feedback_res["penalized_nodes"]

    # Verify confidence was penalized in storage
    node1 = adapter.storage.get_node(id1)
    node2 = adapter.storage.get_node(id2)
    assert node1 is not None and round(node1["confidence"], 2) == 0.7
    assert node2 is not None and round(node2["confidence"], 2) == 0.7


def test_adapter_fallback_when_worker_is_none() -> None:
    """Test SynapseAdapter.apply_outcome_feedback safe fallback if worker is None."""
    adapter = SynapseAdapter({"db_path": ":memory:"})
    adapter.worker = None  # Simulate worker initialization failure

    res = adapter.apply_outcome_feedback(node_ids=["dummy_id"], outcome=True, note="Safe test")
    assert res["is_success"] is True
    assert res["edges_updated"] == 0
    assert res["nodes_updated"] == 0
    assert res["note"] == "Safe test"


# ============================================================================
# 7. Triplet Co-Activation and Multi-Edge Depression
# ============================================================================

def test_triplet_ltd_outcome_feedback(tmp_path: Path) -> None:
    """
    Test triplet co-activation in LTD:
    3 unique nodes = 3 pairs * 2 directions = 6 directed edges depressed.
    """
    storage = SynapseStorage(str(tmp_path / "test_triplet.db"))
    worker = SynapseWorker(storage)

    id_1 = storage.add_node(uri="synapse://n/1", l0_abstract="Node 1", confidence=1.0)
    id_2 = storage.add_node(uri="synapse://n/2", l0_abstract="Node 2", confidence=1.0)
    id_3 = storage.add_node(uri="synapse://n/3", l0_abstract="Node 3", confidence=1.0)

    # Duplicate IDs should be deduplicated automatically
    res = worker.apply_outcome_feedback([id_1, id_2, id_3, id_1], outcome="LTD")

    assert res["is_success"] is False
    assert res["edges_updated"] == 6
    assert res["nodes_updated"] == 3
    assert set(res["penalized_nodes"]) == {id_1, id_2, id_3}

    # Verify confidence decreased by 0.1 for all 3 nodes
    for nid in (id_1, id_2, id_3):
        n = storage.get_node(nid)
        assert n is not None and round(n["confidence"], 2) == 0.9


# ============================================================================
# 8. Custom Thresholds & Overrides
# ============================================================================

def test_custom_parameters_override(tmp_path: Path) -> None:
    """Verify callers can override default delta, penalty, and thresholds."""
    storage = SynapseStorage(str(tmp_path / "test_custom_params.db"))
    worker = SynapseWorker(storage)

    id_a = storage.add_node(uri="synapse://c/a", l0_abstract="Custom A", confidence=0.5, salience=0.8)
    id_b = storage.add_node(uri="synapse://c/b", l0_abstract="Custom B", confidence=0.5, salience=0.8)

    # Custom confidence_penalty = 0.25, flag_confidence_threshold = 0.35
    # 0.5 - 0.25 = 0.25 (< 0.35 -> flagged)
    res = worker.apply_outcome_feedback(
        [id_a, id_b],
        outcome=False,
        delta_ltd=0.3,
        confidence_penalty=0.25,
        flag_confidence_threshold=0.35,
        salience_protection_threshold=0.85,  # 0.8 salience is not protected
    )

    assert res["is_success"] is False
    assert id_a in res["flagged_nodes"]
    assert id_b in res["flagged_nodes"]

    node_a = storage.get_node(id_a)
    assert node_a is not None
    assert round(node_a["confidence"], 2) == 0.25
    assert node_a["status"] == "flagged"

    edge = next(e for e in storage.get_edges(id_a, direction="out") if e["target_id"] == id_b)
    assert round(edge["weight"], 2) == 0.2  # 0.5 - 0.3 = 0.2
