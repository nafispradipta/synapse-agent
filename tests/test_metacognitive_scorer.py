"""
Comprehensive Unit and Integration Tests for Metacognitive Uncertainty Scorer.
Verifies:
1. MetacognitiveScorer initialization, config handling, and error handling.
2. UncertaintyReport structure, typing, and dataclass/dict hybrid behavior.
3. Freshness decay evaluation across time horizons and timestamp formats.
4. Proof count evaluation (low proof penalty, target proof satisfaction).
5. Factual contradiction detection (flags, context candidates, storage).
6. Salience protection mandate: Master directives (salience >= 0.95) dampened to near 0.
7. Dynamic proportional salience dampening for non-master nodes.
8. annotate_context behavior (abstract/context tag injection, deduplication, thresholds).
9. Zero-hardcode parameterization and customizable templates.
10. End-to-end integration with SynapseStorage.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List
import pytest

from synapse.metacognitive_scorer import (
    MetacognitiveScorer,
    MetacognitiveConfig,
    UncertaintyReport,
    parse_timestamp,
)
from synapse.storage import SynapseStorage


# ============================================================================
# 1. Initialization & Config Tests
# ============================================================================

def test_scorer_initialization_defaults() -> None:
    """Test scorer default initialization with no arguments."""
    scorer = MetacognitiveScorer()
    assert scorer.storage is None
    assert isinstance(scorer.config, MetacognitiveConfig)
    assert scorer.config.salience_master_threshold == 0.95
    assert scorer.config.uncertainty_threshold == 0.60
    assert scorer.config.freshness_half_life_days == 30.0


def test_scorer_initialization_with_custom_config_object() -> None:
    """Test scorer initialization with explicit MetacognitiveConfig."""
    custom_cfg = MetacognitiveConfig(
        freshness_half_life_days=14.0,
        target_proof_count=3,
        uncertainty_threshold=0.50,
        action_advice="PERIKSA DOKUMENTASI RESMI",
    )
    scorer = MetacognitiveScorer(config=custom_cfg)
    assert scorer.config.freshness_half_life_days == 14.0
    assert scorer.config.target_proof_count == 3
    assert scorer.config.uncertainty_threshold == 0.50
    assert scorer.config.action_advice == "PERIKSA DOKUMENTASI RESMI"


def test_scorer_initialization_with_dict_config() -> None:
    """Test scorer initialization with dictionary config."""
    dict_cfg = {
        "freshness_half_life_days": 45.0,
        "salience_master_threshold": 0.90,
        "max_master_uncertainty": 0.02,
    }
    scorer = MetacognitiveScorer(config=dict_cfg)
    assert scorer.config.freshness_half_life_days == 45.0
    assert scorer.config.salience_master_threshold == 0.90
    assert scorer.config.max_master_uncertainty == 0.02


def test_scorer_initialization_invalid_config_type() -> None:
    """Test that invalid config types raise TypeError."""
    with pytest.raises(TypeError, match="Config must be MetacognitiveConfig or dict"):
        MetacognitiveScorer(config="invalid_config_string")  # type: ignore[arg-type]


# ============================================================================
# 2. UncertaintyReport Dataclass / Dict Hybrid Tests
# ============================================================================

def test_uncertainty_report_dual_interface() -> None:
    """Test that UncertaintyReport functions seamlessly as both dataclass and dictionary."""
    report = UncertaintyReport(
        uncertainty_score=0.72,
        confidence_level="LOW",
        is_uncertain=True,
        reasons=["Age decay", "Single proof"],
        tag="[UNCERTAINTY: HIGH - MUST VERIFY LIVE]",
        uncertainty_level="HIGH",
        breakdown={"raw": 0.72},
    )

    # 1. Dataclass attribute access
    assert report.uncertainty_score == 0.72
    assert report.confidence_level == "LOW"
    assert report.is_uncertain is True
    assert len(report.reasons) == 2
    assert report.tag == "[UNCERTAINTY: HIGH - MUST VERIFY LIVE]"

    # 2. Dict key access
    assert isinstance(report, dict)
    assert report["uncertainty_score"] == 0.72
    assert report["confidence_level"] == "LOW"
    assert report["is_uncertain"] is True
    assert report.get("tag") == "[UNCERTAINTY: HIGH - MUST VERIFY LIVE]"
    assert "reasons" in report

    # 3. Serialization and keys
    dict_repr = report.to_dict()
    assert isinstance(dict_repr, dict)
    assert json.loads(json.dumps(dict_repr))["uncertainty_score"] == 0.72

    # 4. Bidirectional mutation
    report.confidence_level = "MEDIUM"
    assert report["confidence_level"] == "MEDIUM"

    report["uncertainty_score"] = 0.45
    assert report.uncertainty_score == 0.45


def test_parse_timestamp_formats() -> None:
    """Test timestamp parser across multiple representations."""
    # Integer epoch
    assert parse_timestamp(1700000000) == 1700000000.0
    # Float epoch
    assert parse_timestamp(1700000000.5) == 1700000000.5
    # String numeric
    assert parse_timestamp("1700000000") == 1700000000.0
    # ISO string
    iso_ts = parse_timestamp("2026-09-01T12:00:00Z")
    assert iso_ts > 0.0
    # None or invalid
    assert parse_timestamp(None) == 0.0
    assert parse_timestamp("") == 0.0
    assert parse_timestamp("not-a-timestamp") == 0.0


# ============================================================================
# 3. Freshness / Age Decay Evaluation Tests
# ============================================================================

def test_freshness_decay_brand_new_memory() -> None:
    """Test that brand new memory has 0 age decay uncertainty."""
    scorer = MetacognitiveScorer()
    curr_time = 1750000000
    node = {
        "id": "node-fresh",
        "created_at": curr_time,
        "proof_count": 5,
        "salience": 0.5,
    }

    report = scorer.score_memory(node, current_time=curr_time)
    assert report.breakdown["freshness_uncertainty"] == 0.0
    assert report.uncertainty_score == 0.0
    assert report.confidence_level == "HIGH"
    assert report.is_uncertain is False


def test_freshness_decay_half_life() -> None:
    """Test that at exactly 1 half-life (30 days), decay is ~0.50."""
    scorer = MetacognitiveScorer(config={"freshness_half_life_days": 30.0})
    curr_time = 1750000000
    age_seconds = 30 * 86400  # 30 days
    node = {
        "id": "node-30days",
        "created_at": curr_time - age_seconds,
        "proof_count": 5,
        "salience": 0.5,
    }

    report = scorer.score_memory(node, current_time=curr_time)
    freshness_u = report.breakdown["freshness_uncertainty"]
    assert pytest.approx(freshness_u, 0.01) == 0.50
    assert any("Age decay" in r for r in report.reasons)


def test_freshness_decay_very_old_memory() -> None:
    """Test that an old memory (120 days = 4 half-lives) has decay near 1.0."""
    scorer = MetacognitiveScorer(config={"freshness_half_life_days": 30.0})
    curr_time = 1750000000
    age_seconds = 120 * 86400
    node = {
        "id": "node-ancient",
        "created_at": curr_time - age_seconds,
        "proof_count": 5,
        "salience": 0.5,
    }

    report = scorer.score_memory(node, current_time=curr_time)
    freshness_u = report.breakdown["freshness_uncertainty"]
    assert freshness_u > 0.90


def test_freshness_missing_timestamp() -> None:
    """Test that missing timestamp falls back to baseline unknown age uncertainty."""
    scorer = MetacognitiveScorer(config={"unknown_age_uncertainty": 0.40})
    node = {
        "id": "node-notime",
        "proof_count": 5,
        "salience": 0.5,
    }

    report = scorer.score_memory(node)
    assert report.breakdown["freshness_uncertainty"] == 0.40
    assert any("Missing timestamp" in r for r in report.reasons)


# ============================================================================
# 4. Proof Count Evaluation Tests
# ============================================================================

def test_proof_count_zero_and_negative() -> None:
    """Test that zero proofs result in maximum proof uncertainty."""
    scorer = MetacognitiveScorer()
    curr_time = 1750000000
    node = {
        "id": "node-unproven",
        "created_at": curr_time,
        "proof_count": 0,
        "salience": 0.5,
    }

    report = scorer.score_memory(node, current_time=curr_time)
    assert report.breakdown["proof_uncertainty"] == 1.0
    assert any("Zero verification proofs" in r for r in report.reasons)


def test_proof_count_scaling() -> None:
    """Test gradual reduction of proof uncertainty as proof count increases toward target."""
    scorer = MetacognitiveScorer(config={"target_proof_count": 5})
    curr_time = 1750000000

    # Proof = 1 -> uncertainty = (5-1)/5 = 0.80
    r1 = scorer.score_memory({"proof_count": 1, "created_at": curr_time}, current_time=curr_time)
    assert pytest.approx(r1.breakdown["proof_uncertainty"], 0.01) == 0.80

    # Proof = 3 -> uncertainty = (5-3)/5 = 0.40
    r3 = scorer.score_memory({"proof_count": 3, "created_at": curr_time}, current_time=curr_time)
    assert pytest.approx(r3.breakdown["proof_uncertainty"], 0.01) == 0.40

    # Proof = 5 -> uncertainty = 0.0
    r5 = scorer.score_memory({"proof_count": 5, "created_at": curr_time}, current_time=curr_time)
    assert r5.breakdown["proof_uncertainty"] == 0.0

    # Proof = 10 -> uncertainty = 0.0
    r10 = scorer.score_memory({"proof_count": 10, "created_at": curr_time}, current_time=curr_time)
    assert r10.breakdown["proof_uncertainty"] == 0.0


def test_proof_count_extracted_from_supporting_facts_or_list() -> None:
    """Test that proof count is properly derived from supporting_facts list."""
    scorer = MetacognitiveScorer(config={"target_proof_count": 3})
    curr_time = 1750000000
    node = {
        "id": "node-list",
        "created_at": curr_time,
        "supporting_facts": ["fact A", "fact B", "fact C"],
    }
    report = scorer.score_memory(node, current_time=curr_time)
    assert report.breakdown["proof_uncertainty"] == 0.0


# ============================================================================
# 5. Contradiction Detection Tests
# ============================================================================

def test_contradiction_explicit_flag() -> None:
    """Test that explicit conflict/contradiction flag increases uncertainty."""
    scorer = MetacognitiveScorer()
    curr_time = 1750000000
    node = {
        "id": "node-conflict",
        "created_at": curr_time,
        "proof_count": 5,
        "has_contradiction": True,
        "salience": 0.5,
    }

    report = scorer.score_memory(node, current_time=curr_time)
    assert report.breakdown["contradiction_uncertainty"] == 1.0
    assert report.is_uncertain is True
    assert report.confidence_level == "LOW"
    assert report.tag == "[UNCERTAINTY: HIGH - MUST VERIFY LIVE]"


def test_contradiction_superseded_status() -> None:
    """Test that superseded node is marked with maximum contradiction uncertainty."""
    scorer = MetacognitiveScorer()
    curr_time = 1750000000
    node = {
        "id": "node-superseded",
        "created_at": curr_time,
        "proof_count": 5,
        "status": "superseded",
        "salience": 0.5,
    }

    report = scorer.score_memory(node, current_time=curr_time)
    assert report.breakdown["contradiction_uncertainty"] == 1.0
    assert any("superseded" in r for r in report.reasons)


def test_contradiction_with_candidate_nodes() -> None:
    """Test contradiction detected across context candidate nodes."""
    scorer = MetacognitiveScorer()
    curr_time = 1750000000

    node_a = {
        "id": "node-a",
        "category": "architecture",
        "l0_abstract": "Server RAM 2GB",
        "l1_overview": "Server specification uses 2GB RAM.",
        "created_at": curr_time,
        "proof_count": 5,
        "salience": 0.5,
    }
    node_b = {
        "id": "node-b",
        "category": "architecture",
        "l0_abstract": "Server RAM 4GB",
        "l1_overview": "Server specification uses 4GB RAM.",
        "created_at": curr_time,
        "proof_count": 5,
        "salience": 0.5,
    }

    # Score node_a in the presence of conflicting candidate node_b
    report = scorer.score_memory(node_a, current_time=curr_time, candidate_nodes=[node_a, node_b])
    assert report.breakdown["contradiction_uncertainty"] == 1.0
    assert any("Contradiction detected" in r for r in report.reasons)


# ============================================================================
# 6. Master Directive Salience Protection Tests (MANDATE)
# ============================================================================

def test_master_directive_salience_protection_mandate() -> None:
    """
    MANDATE TEST:
    Master directives with salience >= 0.95 must be protected from high uncertainty.
    Even with aged decay, low proof count, or contradiction presence,
    uncertainty is dampened to near 0.
    """
    scorer = MetacognitiveScorer()
    curr_time = 1750000000

    # A 1-year old node with single proof count and contradiction flag
    master_node = {
        "id": "node-master-directive",
        "l0_abstract": "CORE DIRECTIVE: Use PostgreSQL for transactional persistence",
        "created_at": curr_time - (365 * 86400),  # 1 year old
        "proof_count": 1,                         # low proof
        "has_contradiction": True,               # conflicting claims exist
        "salience": 0.98,                         # MASTER DIRECTIVE!
    }

    report = scorer.score_memory(master_node, current_time=curr_time)

    # Uncertainty MUST be dampened to near 0 (<= 0.05)
    assert report.uncertainty_score <= 0.05
    assert report.is_uncertain is False
    assert report.confidence_level == "HIGH"
    assert report.tag is None
    assert any("Master directive protected" in r for r in report.reasons)


def test_master_directive_vs_standard_node_comparison() -> None:
    """
    Compare identical conditions between a master directive (salience 0.96)
    and an ordinary memory (salience 0.50).
    """
    scorer = MetacognitiveScorer()
    curr_time = 1750000000

    base_properties = {
        "created_at": curr_time - (90 * 86400),  # 90 days old
        "proof_count": 1,                        # 1 proof
        "has_contradiction": True,
    }

    ordinary_node = dict(base_properties, id="ordinary", salience=0.50)
    master_node = dict(base_properties, id="master", salience=0.96)

    rep_ordinary = scorer.score_memory(ordinary_node, current_time=curr_time)
    rep_master = scorer.score_memory(master_node, current_time=curr_time)

    # Ordinary node should be highly uncertain
    assert rep_ordinary.is_uncertain is True
    assert rep_ordinary.uncertainty_score >= 0.60
    assert rep_ordinary.confidence_level == "LOW"
    assert rep_ordinary.tag == "[UNCERTAINTY: HIGH - MUST VERIFY LIVE]"

    # Master node must remain protected and confident
    assert rep_master.is_uncertain is False
    assert rep_master.uncertainty_score <= 0.05
    assert rep_master.confidence_level == "HIGH"
    assert rep_master.tag is None


# ============================================================================
# 7. Proportional Salience Dampening Tests
# ============================================================================

def test_dynamic_salience_proportional_dampening() -> None:
    """Test that higher salience proportionally dampens uncertainty for non-master nodes."""
    scorer = MetacognitiveScorer()
    curr_time = 1750000000

    node_low_sal = {
        "created_at": curr_time - (30 * 86400),
        "proof_count": 2,
        "salience": 0.10,
    }
    node_high_sal = {
        "created_at": curr_time - (30 * 86400),
        "proof_count": 2,
        "salience": 0.80,
    }

    r_low = scorer.score_memory(node_low_sal, current_time=curr_time)
    r_high = scorer.score_memory(node_high_sal, current_time=curr_time)

    assert r_high.uncertainty_score < r_low.uncertainty_score


# ============================================================================
# 8. Context Annotation (annotate_context) Tests
# ============================================================================

def test_annotate_context_tag_injection() -> None:
    """Test that annotate_context appends live verification tags to uncertain memories."""
    scorer = MetacognitiveScorer()
    curr_time = 1750000000

    nodes = [
        # Node 1: Uncertain (old, 1 proof, low salience)
        {
            "id": "node-1",
            "l0_abstract": "API gateway port is 8080",
            "created_at": curr_time - (120 * 86400),
            "proof_count": 1,
            "salience": 0.20,
        },
        # Node 2: Master Directive (protected)
        {
            "id": "node-2",
            "l0_abstract": "Core Rule: Always use UUID v4",
            "created_at": curr_time - (120 * 86400),
            "proof_count": 1,
            "salience": 0.99,
        },
        # Node 3: Fresh and proven
        {
            "id": "node-3",
            "l0_abstract": "Primary database is PostgreSQL 16",
            "created_at": curr_time,
            "proof_count": 5,
            "salience": 0.70,
        },
    ]

    annotated = scorer.annotate_context(nodes, threshold=0.60, current_time=curr_time)

    assert len(annotated) == 3

    # Node 1: Uncertain, tagged
    assert annotated[0]["is_uncertain"] is True
    assert "[UNCERTAINTY: HIGH - MUST VERIFY LIVE]" in annotated[0]["l0_abstract"]
    assert annotated[0]["l0_abstract"].startswith("API gateway port is 8080")
    assert annotated[0]["uncertainty_tag"] == "[UNCERTAINTY: HIGH - MUST VERIFY LIVE]"

    # Node 2: Master Directive, untouched abstract
    assert annotated[1]["is_uncertain"] is False
    assert "[UNCERTAINTY" not in annotated[1]["l0_abstract"]
    assert annotated[1]["l0_abstract"] == "Core Rule: Always use UUID v4"
    assert annotated[1]["uncertainty_tag"] is None

    # Node 3: Fresh & proven, untouched abstract
    assert annotated[2]["is_uncertain"] is False
    assert "[UNCERTAINTY" not in annotated[2]["l0_abstract"]
    assert annotated[2]["l0_abstract"] == "Primary database is PostgreSQL 16"
    assert annotated[2]["uncertainty_tag"] is None


def test_annotate_context_multiple_fields_and_no_duplication() -> None:
    """Test annotation across abstract, context, and idempotence against double-tagging."""
    scorer = MetacognitiveScorer()
    curr_time = 1750000000

    node = {
        "id": "node-fields",
        "l0_abstract": "Abstract content",
        "context": "Context content",
        "created_at": curr_time - (120 * 86400),
        "proof_count": 0,
        "salience": 0.1,
    }

    # First pass
    pass1 = scorer.annotate_context([node], threshold=0.50)
    tag = "[UNCERTAINTY: HIGH - MUST VERIFY LIVE]"
    assert tag in pass1[0]["l0_abstract"]
    assert tag in pass1[0]["context"]

    # Second pass: ensure tag is not duplicated
    pass2 = scorer.annotate_context(pass1, threshold=0.50)
    assert pass2[0]["l0_abstract"].count(tag) == 1
    assert pass2[0]["context"].count(tag) == 1


def test_annotate_context_empty_list() -> None:
    """Test that annotate_context handles empty input cleanly."""
    scorer = MetacognitiveScorer()
    assert scorer.annotate_context([]) == []


# ============================================================================
# 9. Customization & Zero-Hardcode Parameterization Tests
# ============================================================================

def test_zero_hardcode_custom_weights_and_thresholds() -> None:
    """Test that adjusting weights and thresholds alters scoring behavior dynamically."""
    custom_cfg = MetacognitiveConfig(
        freshness_weight=0.80,
        proof_count_weight=0.10,
        contradiction_weight=0.10,
        uncertainty_threshold=0.40,
        action_advice="CONFIRM WITH TEAM LEAD",
    )
    scorer = MetacognitiveScorer(config=custom_cfg)
    curr_time = 1750000000

    # Node with 45 days age decay
    node = {
        "id": "node-cfg",
        "created_at": curr_time - (45 * 86400),
        "proof_count": 5,
        "salience": 0.20,
    }

    report = scorer.score_memory(node, current_time=curr_time)
    assert report.is_uncertain is True  # because threshold is 0.40 and freshness weight is 0.80
    assert "CONFIRM WITH TEAM LEAD" in (report.tag or "")


# ============================================================================
# 10. Integration with SynapseStorage Tests
# ============================================================================

def test_integration_with_synapse_storage() -> None:
    """Test MetacognitiveScorer against nodes stored in an in-memory SynapseStorage database."""
    storage = SynapseStorage(":memory:")

    curr_time = int(time.time())

    # Insert an older active node in storage
    storage.add_node(
        node_id="storage-node-old",
        uri="spec/server_ram_old",
        l0_abstract="Server RAM is 2GB for production host",
        category="architecture",
        salience=0.5,
        confidence=1.0,
        created_at=curr_time - (60 * 86400),
    )

    # Scorer with storage connected
    scorer = MetacognitiveScorer(storage=storage)

    # Node that contradicts the storage record
    contradicting_candidate = {
        "id": "candidate-new",
        "uri": "spec/server_ram_new",
        "category": "architecture",
        "l0_abstract": "Server RAM is 4GB for production host",
        "salience": 0.5,
        "proof_count": 1,
        "created_at": curr_time,
    }

    report = scorer.score_memory(contradicting_candidate, current_time=curr_time)
    assert report.breakdown["contradiction_uncertainty"] == 1.0
    assert report.is_uncertain is True
    assert report.confidence_level == "LOW"
    assert report.tag == "[UNCERTAINTY: HIGH - MUST VERIFY LIVE]"
