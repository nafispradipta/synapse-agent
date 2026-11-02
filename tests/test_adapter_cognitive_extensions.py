"""Tests for SynapseAdapter Cognitive Extensions: Counterfactual Guardrail & Metacognitive Scorer."""

from pathlib import Path
import time
import pytest

from synapse.adapter import SynapseAdapter
from synapse.storage import SynapseStorage


def test_adapter_counterfactual_guardrail_integration(tmp_path: Path) -> None:
    db_file = tmp_path / "test_synapse.db"
    storage = SynapseStorage(str(db_file))

    # Add a failure episode to storage
    storage.add_episode(
        title="Direct Production Service Restart Crash",
        situation="Server under heavy user load during active traffic",
        action_taken="Ran systemctl restart gateway-proxy without draining traffic",
        outcome="Outage 15 minutes, connection reset, user tokens dropped, severe incident",
        lessons_learned="Never restart active proxy while traffic is flowing. Always drain traffic first.",
        salience=0.95,
        confidence=1.0,
        tags=["incident", "failure", "downtime"],
    )

    adapter = SynapseAdapter({"db_path": str(db_file)})

    # Test dangerous action
    eval_result = adapter.evaluate_action(
        action_proposal="systemctl restart gateway-proxy without draining traffic",
        context="Server under heavy user load",
    )
    assert eval_result.is_risky is True
    assert eval_result.risk_level.lower() in ("high", "critical", "moderate")
    assert "Direct Production Service Restart Crash" in eval_result.warning_message
    assert "drain" in eval_result.suggested_mitigation.lower()

    # Test benign safe action
    safe_result = adapter.evaluate_action(
        action_proposal="check system status with htop and memory info",
        context="normal monitoring",
    )
    assert safe_result.is_risky is False
    assert safe_result.risk_level.lower() in ("safe", "low")


def test_adapter_metacognitive_uncertainty_and_recall_annotation(tmp_path: Path) -> None:
    db_file = tmp_path / "test_synapse_meta.db"
    storage = SynapseStorage(str(db_file))

    now = int(time.time())
    stale_time = now - (90 * 86400)  # 90 days ago

    # Node 1: Old unverified fact (stale)
    storage.add_node(
        uri="spec://server/old-bandwidth-quota",
        category="architecture",
        salience=0.4,
        confidence=0.5,
        created_at=stale_time,
        l0_abstract="Server bandwidth limit is 100GB per month",
        l1_overview="Legacy quota specification details.",
    )

    # Node 2: Core directive with high salience (immune to decay / low uncertainty)
    storage.add_node(
        uri="principle://directive/orchestrator-subagent",
        category="sop",
        salience=0.98,
        confidence=1.0,
        created_at=stale_time,
        l0_abstract="Heavy computing workloads must use parallel subagent delegation",
        l1_overview="Core architectural directive.",
    )

    adapter = SynapseAdapter({"db_path": str(db_file), "token_budget": 500})

    # Test score_uncertainty directly
    stale_node = storage.get_node("spec://server/old-bandwidth-quota")
    stale_report = adapter.score_uncertainty(stale_node)
    assert stale_report.is_uncertain is True
    assert stale_report.confidence_level in ("LOW", "MEDIUM")

    master_node = storage.get_node("principle://directive/orchestrator-subagent")
    master_report = adapter.score_uncertainty(master_node)
    assert master_report.is_uncertain is False
    assert master_report.uncertainty_score <= 0.05
    assert master_report.confidence_level == "HIGH"

    # Test recall integration: stale node should have UNCERTAINTY tag in recalled text
    recalled = adapter.recall("bandwidth limit", limit=3)
    assert len(recalled) > 0
    recalled_str = " ".join(recalled)
    assert "UNCERTAINTY" in recalled_str
    assert "MUST VERIFY LIVE" in recalled_str


def test_pluggable_categories_and_custom_taxonomy(tmp_path: Path) -> None:
    """Verify that SynapseAdapter supports 100% pluggable custom categories and unconstrained storage."""
    db_file = tmp_path / "test_pluggable.db"
    vault_dir = tmp_path / "custom_vault"
    vault_dir.mkdir()

    custom_config = {
        "db_path": str(db_file),
        "vault_path": str(vault_dir),
        "categories": ["diagnosis", "treatment", "lab_result", "general"],
        "category_patterns": {
            "diagnosis": r"\b(diagnos|condition|disease|syndrome)\b",
            "treatment": r"\b(therapy|prescription|medication|dosage)\b",
            "lab_result": r"\b(blood|glucose|scan|mri|biopsy)\b",
        },
        "tag_rules": {
            "cardiology": r"\b(heart|cardio|artery)\b",
            "neurology": r"\b(brain|neuro|nerve)\b",
        },
        "vault_routes": {
            "diagnosis": {
                "folder": "clinical_records",
                "index": "00-INDEX-CLINICAL",
                "title": "Clinical Diagnoses",
                "tags": ["clinical", "records"],
            }
        },
    }

    adapter = SynapseAdapter(custom_config)
    assert adapter.storage is not None

    # Test custom category inference
    inferred = adapter._infer_category("Patient condition indicates acute coronary syndrome", "")
    assert inferred == "diagnosis"

    # Test saving custom non-standard category into SQLite storage
    ok = adapter.retain(
        content="Patient diagnosed with hypertension stage 1.",
        category="diagnosis",
        salience=0.9,
    )
    assert ok is True

    nodes = adapter.storage.list_nodes(category="diagnosis")
    assert len(nodes) == 1
    assert nodes[0]["category"] == "diagnosis"

    # Test custom vault routing
    folder, index_stem, title, base_tags = adapter._resolve_vault_target(
        vault_dir,
        "diagnosis",
        "Patient diagnosed with hypertension stage 1.",
        "",
        "synapse://diagnosis/hypertension",
        {},
    )
    assert folder == vault_dir / "clinical_records"
    assert index_stem == "00-INDEX-CLINICAL"
    assert "records" in base_tags
