"""
Unit and integration tests for Episodic Counterfactual Guardrail.

Verifies:
1. CounterfactualGuardrail initialization (default, custom, fallback embedders).
2. GuardrailResult dataclass and dict-like interface compatibility.
3. Accurate detection of risky actions matching past failures/incidents (via tags, outcome, lessons learned).
4. Differentiation between failed causal actions and successful recovery remedies.
5. Actionable mitigation extraction and informative warning message synthesis.
6. Context-aware risk modulation (situational congruence boosting risk).
7. Multilingual support (English and Indonesian incident patterns).
8. Risk tiering (safe, low, medium, high, critical).
9. Batch evaluation, risk explanation, and live storage dynamism.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synapse.storage import SynapseStorage
from synapse.embedder import SynapseEmbedder, fallback_vectorize
from synapse.counterfactual_guardrail import (
    CounterfactualGuardrail,
    GuardrailResult,
    cosine_similarity,
    extract_mitigation,
    generate_warning_message,
    parse_tags,
    token_overlap_ratio,
)


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database file for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    for p in [path, f"{path}-wal", f"{path}-shm"]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


@pytest.fixture
def storage(temp_db):
    """Fixture providing an isolated SynapseStorage instance."""
    st = SynapseStorage(temp_db)
    yield st
    st.close()


@pytest.fixture(scope="module")
def shared_embedder():
    """Fixture providing a cached SynapseEmbedder to avoid re-loading weights in each test."""
    return SynapseEmbedder(dim=64)


# ============================================================================
# 1. Initialization and Data Structure Tests
# ============================================================================

def test_guardrail_initialization_defaults_and_custom(storage, shared_embedder):
    """Test initializing CounterfactualGuardrail with default and custom arguments."""
    # Default initialization
    guardrail_def = CounterfactualGuardrail(storage)
    assert guardrail_def.storage is storage
    assert guardrail_def.similarity_threshold == 0.55
    assert guardrail_def.critical_risk_threshold == 0.88

    # Custom thresholds and embedder
    guardrail_custom = CounterfactualGuardrail(
        storage=storage,
        embedder=shared_embedder,
        similarity_threshold=0.50,
        medium_risk_threshold=0.60,
        high_risk_threshold=0.75,
        critical_risk_threshold=0.85,
        vector_dim=64,
    )
    assert guardrail_custom.embedder is shared_embedder
    assert guardrail_custom.similarity_threshold == 0.50
    assert guardrail_custom.critical_risk_threshold == 0.85


def test_guardrail_result_interface():
    """Test GuardrailResult dataclass, dict access, and conversion."""
    res = GuardrailResult(
        is_risky=True,
        risk_level="high",
        matched_episode={"id": "ep1", "title": "Test Incident"},
        warning_message="Dangerous action detected",
        suggested_mitigation="Use staging first",
        similarity_score=0.825,
    )

    # Attribute access
    assert res.is_risky is True
    assert res.risk_level == "high"
    assert res.similarity_score == 0.825

    # Dict-like subscripting and methods
    assert res["is_risky"] is True
    assert res["risk_level"] == "high"
    assert res.get("warning_message") == "Dangerous action detected"
    assert res.get("non_existent", "default") == "default"
    assert "suggested_mitigation" in res

    # to_dict serialization
    d = res.to_dict()
    assert isinstance(d, dict)
    assert d["is_risky"] is True
    assert d["risk_level"] == "high"
    assert d["matched_episode"]["id"] == "ep1"

    with pytest.raises(KeyError):
        _ = res["invalid_key"]


def test_extract_mitigation_patterns():
    """Test mitigation extraction from various lessons learned structures."""
    # Multi-sentence with affirmative directive
    text1 = (
        "Never run TRUNCATE on production tables directly. "
        "Always create a snapshot backup and perform migrations in staging first."
    )
    m1 = extract_mitigation(text1)
    assert "Always create a snapshot backup and perform migrations in staging first" in m1

    # Affirmative directive
    text2 = (
        "Never disable iptables firewall on live production server. "
        "Always create a specific security rule for the required port."
    )
    m2 = extract_mitigation(text2)
    assert "Always create a specific security rule for the required port" in m2

    # Single affirmative directive
    text3 = "Always verify backup before performing database maintenance."
    m3 = extract_mitigation(text3)
    assert "Always verify backup" in m3

    # Fallback when no affirmative keywords present
    text4 = "Avoid unindexed query."
    m4 = extract_mitigation(text4)
    assert m4 == "Avoid unindexed query."

    # Empty fallback
    assert "Review action proposal" in extract_mitigation("")


# ============================================================================
# 2. Evaluation Edge Cases and Safe Actions
# ============================================================================

def test_evaluate_empty_proposal_and_empty_storage(storage, shared_embedder):
    """Test evaluating blank inputs or when storage has no episodes."""
    guardrail = CounterfactualGuardrail(storage, embedder=shared_embedder)

    # Empty and whitespace inputs
    res_empty = guardrail.evaluate_action("")
    assert res_empty.is_risky is False
    assert res_empty.risk_level == "safe"
    assert res_empty.matched_episode is None
    assert res_empty.similarity_score == 0.0

    res_ws = guardrail.evaluate_action("   \n\t  ")
    assert res_ws.is_risky is False
    assert res_ws.risk_level == "safe"

    # Non-empty proposal in empty database
    res_db = guardrail.evaluate_action("DROP TABLE users in production")
    assert res_db.is_risky is False
    assert res_db.risk_level == "safe"
    assert res_db.similarity_score == 0.0


def test_benign_action_not_flagged(storage, shared_embedder):
    """Verify benign, unrelated actions are not falsely flagged as risky."""
    storage.add_episode(
        title="Production DB Drop Incident",
        situation="Developer ran destructive drop query on primary database",
        action_taken="Executed DROP TABLE customers without backup",
        outcome="Complete customer data loss, 4 hours downtime",
        lessons_learned="Never drop tables in production. Always test in staging first.",
        salience=0.95,
        confidence=1.0,
        tags=["incident", "database", "data_loss"],
    )

    guardrail = CounterfactualGuardrail(storage, embedder=shared_embedder)

    # Completely unrelated proposal
    res = guardrail.evaluate_action("Check system disk utilization with df -h")
    assert res.is_risky is False
    assert res.risk_level == "safe"
    assert res.matched_episode is None


def test_successful_sop_not_flagged(storage, shared_embedder):
    """Verify that successful past actions (SOPs) are not treated as failures."""
    storage.add_episode(
        title="Standard Routine Service Restart SOP",
        situation="Routine memory reclamation maintenance",
        action_taken="Restarted gateway-proxy service via systemctl",
        outcome="Service restarted smoothly, 0 downtime, memory reclaimed",
        lessons_learned="Always monitor health checks after restarting service.",
        salience=0.6,
        confidence=1.0,
        tags=["sop", "maintenance", "deployment"],
    )

    guardrail = CounterfactualGuardrail(storage, embedder=shared_embedder)

    # Propose the exact SOP action
    res = guardrail.evaluate_action("Restart gateway-proxy service via systemctl")
    assert res.is_risky is False
    assert res.risk_level == "safe"
    assert res.matched_episode is None


# ============================================================================
# 3. Risky Action Detection (Counterfactual Matching)
# ============================================================================

def test_detect_risky_action_via_tags_and_action(storage, shared_embedder):
    """Verify detection when proposed action matches action_taken in an incident episode."""
    ep_id = storage.add_episode(
        title="Accidental Production Database Truncate",
        situation="Database cleanup script misconfigured with live production credentials",
        action_taken="Executed TRUNCATE TABLE customers directly on live production DB",
        outcome="Total customer record wipe, 3 hours outage and customer panic",
        lessons_learned="Never run TRUNCATE on production tables. Always use soft deletes or stage migrations first.",
        salience=0.95,
        confidence=1.0,
        tags=["database", "incident", "failure"],
    )

    guardrail = CounterfactualGuardrail(storage, embedder=shared_embedder)

    proposal = "Execute TRUNCATE TABLE customers on the production database"
    res = guardrail.evaluate_action(proposal)

    assert res.is_risky is True
    assert res.risk_level in ["high", "critical"]
    assert res.matched_episode is not None
    assert res.matched_episode["id"] == ep_id
    assert res.similarity_score >= 0.65
    assert "Accidental Production Database Truncate" in res.warning_message
    assert "Total customer record wipe" in res.warning_message
    assert "Always use soft deletes or stage migrations first" in res.suggested_mitigation


def test_detect_risky_action_via_negative_outcome(storage, shared_embedder):
    """Verify detection when tags do not mention incident, but outcome documents severe failure."""
    ep_id = storage.add_episode(
        title="Worker Memory Limit Deactivation",
        situation="Workers were hitting memory ceilings during batch ingestion",
        action_taken="Disabled worker memory recycling and garbage collection limits",
        outcome="Caused catastrophic memory exhaustion and fatal kernel OOM kill on host",
        lessons_learned="Never disable worker memory recycling. Always scale horizontally instead.",
        salience=0.85,
        confidence=0.9,
        tags=["worker", "memory", "config"],  # No explicit 'incident' tag
    )

    guardrail = CounterfactualGuardrail(storage, embedder=shared_embedder)

    proposal = "Disable worker memory recycling limits to prevent memory exceptions"
    res = guardrail.evaluate_action(proposal)

    assert res.is_risky is True
    assert res.risk_level in ["medium", "high", "critical"]
    assert res.matched_episode["id"] == ep_id
    assert "memory exhaustion" in res.warning_message
    assert "Always scale horizontally instead" in res.suggested_mitigation


def test_detect_risky_action_via_lessons_learned_warning(storage, shared_embedder):
    """Verify detection when proposed action matches the cautionary directive in lessons_learned."""
    ep_id = storage.add_episode(
        title="API Gateway TLS Handshake Error",
        situation="Clients reported TLS handshake failures with self-signed test cert",
        action_taken="Added insecure skip verify flag in client configuration",
        outcome="Authentication credentials exposed and intercepted via MITM breach",
        lessons_learned="Never disable SSL certificate verification in production. Always install valid CA certificates.",
        salience=0.90,
        confidence=1.0,
        tags=["security", "tls", "incident"],
    )

    guardrail = CounterfactualGuardrail(storage, embedder=shared_embedder)

    proposal = "Disable SSL certificate verification to bypass connection error"
    res = guardrail.evaluate_action(proposal)

    assert res.is_risky is True
    assert res.matched_episode["id"] == ep_id
    assert "Always install valid CA certificates" in res.suggested_mitigation


def test_recovery_remedy_not_flagged_as_failure_action(storage, shared_embedder):
    """
    Verify that when an episode documents a recovery where action_taken was the successful
    remedy, proposing the remedy is NOT falsely flagged as repeating a failure.
    """
    storage.add_episode(
        title="PostgreSQL WAL Corruption Incident",
        situation="Database crashed during sudden power outage with corrupted WAL files",
        action_taken="Ran pg_resetwal and restored latest consistent basebackup with WAL replay",
        outcome="Database successfully recovered with 0 data loss and clean state",
        lessons_learned="Always enable synchronous_commit and configure automated WAL archiving to S3.",
        salience=0.95,
        confidence=1.0,
        tags=["postgres", "wal", "recovery", "incident"],
    )

    guardrail = CounterfactualGuardrail(storage, embedder=shared_embedder)

    # Propose the remedy action
    remedy_proposal = "Restore latest consistent basebackup with WAL replay"
    res = guardrail.evaluate_action(remedy_proposal)

    # Since the action was a successful remedy, proposing it should NOT be flagged as risky
    assert res.is_risky is False
    assert res.risk_level == "safe"


# ============================================================================
# 4. Context Congruence and Situational Modulation
# ============================================================================

def test_context_congruence_boosts_risk(storage, shared_embedder):
    """Verify that congruent context elevates the similarity score and risk tier."""
    storage.add_episode(
        title="Primary Database Mid-Day Restart",
        situation="During peak production traffic with high concurrent user transactions",
        action_taken="Restarted primary PostgreSQL database service directly",
        outcome="Aborted 1,200 active client transactions and caused 15-minute cascade outage",
        lessons_learned="Never restart primary DB during peak traffic. Always failover to replica first.",
        salience=0.85,
        confidence=1.0,
        tags=["postgres", "incident", "database"],
    )

    guardrail = CounterfactualGuardrail(storage, embedder=shared_embedder)

    proposal = "Restart primary PostgreSQL database service"

    # Evaluation with congruent context
    congruent_ctx = "During peak production traffic with high concurrent transactions"
    res_congruent = guardrail.evaluate_action(proposal, context=congruent_ctx)

    # Evaluation with non-congruent context
    isolated_ctx = "In local isolated docker development environment with zero clients"
    res_isolated = guardrail.evaluate_action(proposal, context=isolated_ctx)

    assert res_congruent.is_risky is True
    # Congruent context must yield higher similarity score
    assert res_congruent.similarity_score >= res_isolated.similarity_score


# ============================================================================
# 5. Multilingual Support (Indonesian)
# ============================================================================

def test_incident_detection_and_mitigation_guardrail(storage, shared_embedder):
    """Test incident detection and mitigation extraction."""
    ep_id = storage.add_episode(
        title="Firewall Shutdown and Port Exposure Incident",
        situation="API gateway port was inaccessible from external network",
        action_taken="Disabled iptables firewall directly on production server",
        outcome="Server suffered brute force attack and database port exposure",
        lessons_learned="Never disable iptables firewall. Always create specific security rules for required ports.",
        salience=0.92,
        confidence=1.0,
        tags=["incident", "security", "firewall", "production"],
    )

    guardrail = CounterfactualGuardrail(storage, embedder=shared_embedder)

    proposal = "Disable iptables firewall on production server for connection testing"
    res = guardrail.evaluate_action(proposal)

    assert res.is_risky is True
    assert res.risk_level in ["high", "critical"]
    assert res.matched_episode["id"] == ep_id
    assert res.suggested_mitigation is not None
    assert "Always create specific security rules for required ports" in res.suggested_mitigation
    assert res.warning_message is not None
    assert "Firewall Shutdown" in res.warning_message or "exposure" in res.warning_message.lower()


# ============================================================================
# 6. Risk Tiering & Threshold Sensitivity
# ============================================================================

def test_risk_level_tiering(storage, shared_embedder):
    """Verify different risk level assignments based on similarity and salience."""
    # Critical incident: high salience
    storage.add_episode(
        title="Critical Security Leak",
        situation="Production env vars exposed",
        action_taken="Committed plaintext production secrets to public git repository",
        outcome="Complete credential compromise, security breach, API keys revoked",
        lessons_learned="Never commit plaintext credentials. Always use secret management.",
        salience=0.98,
        confidence=1.0,
        tags=["incident", "security", "breach"],
    )

    guardrail = CounterfactualGuardrail(
        storage,
        embedder=shared_embedder,
        similarity_threshold=0.50,
        medium_risk_threshold=0.60,
        high_risk_threshold=0.72,
        critical_risk_threshold=0.82,
    )

    # Near-identical proposal should trigger critical
    res_crit = guardrail.evaluate_action("Commit plaintext production secrets to public git repository")
    assert res_crit.is_risky is True
    assert res_crit.risk_level == "critical"


# ============================================================================
# 7. Batch Evaluation, Explanation & Live Updates
# ============================================================================

def test_batch_evaluate_and_explain_risk(storage, shared_embedder):
    """Test batch evaluation and formatted risk explanation."""
    storage.add_episode(
        title="Kill -9 Master Node",
        situation="Database process hung on write lock",
        action_taken="Sent SIGKILL (kill -9) to master database process",
        outcome="Corrupted shared memory buffers and crashed database instance",
        lessons_learned="Never use kill -9 on database processes. Always use graceful shutdown signals.",
        salience=0.88,
        confidence=1.0,
        tags=["database", "incident", "crash"],
    )

    guardrail = CounterfactualGuardrail(storage, embedder=shared_embedder)

    proposals = [
        "Sent kill -9 to master database process",
        ("Sent kill -9 to database process", "Database hanging on write lock"),
        "Check system memory usage with free -m",
    ]

    results = guardrail.batch_evaluate(proposals)
    assert len(results) == 3
    assert results[0].is_risky is True
    assert results[1].is_risky is True
    assert results[2].is_risky is False

    # Test explain_risk
    explanation_risky = guardrail.explain_risk(results[0])
    assert "[RISK LEVEL:" in explanation_risky
    assert "Kill -9 Master Node" in explanation_risky
    assert "Suggested Mitigation:" in explanation_risky

    explanation_safe = guardrail.explain_risk(results[2])
    assert "[RISK LEVEL: SAFE]" in explanation_safe


def test_get_failure_episodes_filter(storage, shared_embedder):
    """Test get_failure_episodes retrieves only failure/incident memories."""
    storage.add_episode(
        title="Incident A",
        situation="Bad config",
        action_taken="Pushed bad config",
        outcome="Outage",
        lessons_learned="Avoid bad config.",
        tags=["incident"],
    )
    storage.add_episode(
        title="Normal SOP B",
        situation="Regular routine",
        action_taken="Cleaned cache",
        outcome="Cache cleared successfully",
        lessons_learned="Routine cache clearing.",
        tags=["sop"],
    )

    guardrail = CounterfactualGuardrail(storage, embedder=shared_embedder)
    fails = guardrail.get_failure_episodes()

    assert len(fails) == 1
    assert fails[0]["title"] == "Incident A"


def test_dynamic_storage_updates(storage, shared_embedder):
    """Test that newly added episodes are immediately recognized by the guardrail."""
    guardrail = CounterfactualGuardrail(storage, embedder=shared_embedder)

    action = "Execute raw DROP DATABASE command on production cluster"

    # Initially no episodes in storage
    res_before = guardrail.evaluate_action(action)
    assert res_before.is_risky is False

    # Add incident episode dynamically
    storage.add_episode(
        title="Production Drop Incident",
        situation="Accidental execution",
        action_taken="Execute raw DROP DATABASE command on production cluster",
        outcome="Entire cluster database was dropped, complete data loss",
        lessons_learned="Never execute raw DROP DATABASE. Always require dual approval and backup.",
        salience=0.99,
        tags=["incident", "failure"],
    )

    # Immediately evaluate again
    res_after = guardrail.evaluate_action(action)
    assert res_after.is_risky is True
    assert res_after.risk_level in ["high", "critical"]
    assert "Production Drop Incident" in res_after.warning_message


def test_custom_callable_embedder(storage):
    """Test CounterfactualGuardrail with a custom callable vectorizer function."""
    def custom_vectorizer(text: str) -> List[float]:
        return fallback_vectorize(text, dim=32)

    storage.add_episode(
        title="Network Gateway Crash",
        situation="Port collision during startup",
        action_taken="Bound multiple processes to port 8080 simultaneously",
        outcome="Network gateway crashed with address already in use error",
        lessons_learned="Always check port availability before binding.",
        salience=0.8,
        tags=["incident", "network"],
    )

    guardrail = CounterfactualGuardrail(
        storage=storage,
        embedder=custom_vectorizer,
        vector_dim=32,
        similarity_threshold=0.40,
    )

    res = guardrail.evaluate_action("Bound multiple processes to port 8080 simultaneously")
    assert res.is_risky is True
    assert res.similarity_score > 0.40


def test_top_level_import_and_memory_storage():
    """Verify CounterfactualGuardrail and GuardrailResult can be imported from synapse package."""
    from synapse import CounterfactualGuardrail as CG, GuardrailResult as GR

    mem_st = SynapseStorage(db_path=":memory:")
    guard = CG(storage=mem_st)
    assert guard is not None

    res = guard.evaluate_action("Arbitrary command")
    assert isinstance(res, GR)
    assert res.is_risky is False
    assert res.risk_level == "safe"
    mem_st.close()


def test_sparse_episode_with_missing_optional_fields(storage, shared_embedder):
    """Verify guardrail does not crash when historical episode has empty or missing fields."""
    ep_id = storage.add_episode(
        title="Crash Without Details",
        situation=None,
        action_taken="Ran unchecked script",
        outcome="Process crashed",
        lessons_learned=None,
        tags=["crash"],
    )

    guardrail = CounterfactualGuardrail(storage, embedder=shared_embedder)
    res = guardrail.evaluate_action("Ran unchecked script")

    assert res.is_risky is True
    assert res.matched_episode["id"] == ep_id
    assert "Crash Without Details" in res.warning_message
    assert "Review action proposal carefully" in res.suggested_mitigation


def test_low_risk_tier_assignment(storage, shared_embedder):
    """Verify that a moderate/low similarity match assigns 'low' risk level."""
    storage.add_episode(
        title="Minor API Timeout Warning",
        situation="High load on secondary endpoint",
        action_taken="Sent concurrent burst of requests to API",
        outcome="Minor intermittent timeouts observed on secondary endpoint",
        lessons_learned="Always throttle concurrent requests.",
        salience=0.4,
        confidence=0.7,
        tags=["warning"],
    )

    guardrail = CounterfactualGuardrail(
        storage=storage,
        embedder=shared_embedder,
        similarity_threshold=0.35,
        medium_risk_threshold=0.75,
        high_risk_threshold=0.88,
        critical_risk_threshold=0.95,
    )

    # A loosely related proposal that lands in the [0.35, 0.75) bracket
    res = guardrail.evaluate_action("Send batch calls to endpoint")
    if res.is_risky:
        assert res.risk_level == "low"
    else:
        assert res.risk_level == "safe"


def test_parse_tags_edge_cases():
    """Test parse_tags function against various formatting edge cases."""
    assert parse_tags(None) == set()
    assert parse_tags("") == set()
    assert parse_tags(["Incident", "Postgres"]) == {"incident", "postgres"}
    assert parse_tags('["incident", "failure"]') == {"incident", "failure"}
    assert parse_tags("[invalid json, array") == {"[invalid", "json", "array"}
    assert parse_tags("incident, outage; bug") == {"incident", "outage", "bug"}
    assert parse_tags(123) == set()


def test_helper_and_math_boundary_conditions(storage):
    """Test mathematical and helper edge cases for cosine similarity and embedder adapters."""
    # Cosine similarity edge cases
    assert cosine_similarity([], [1.0, 2.0]) == 0.0
    assert cosine_similarity([1.0, 2.0], []) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    # Token overlap boundary
    assert token_overlap_ratio(set(), "some text") == 0.0
    assert token_overlap_ratio({"test"}, "") == 0.0

    # Embedder with .vectorize method
    class VectorizeEmbedder:
        def vectorize(self, text: str) -> List[float]:
            return fallback_vectorize(text, dim=16)

    guard_vec = CounterfactualGuardrail(storage, embedder=VectorizeEmbedder(), vector_dim=16)
    assert len(guard_vec.vectorize("sample text")) == 16
    assert guard_vec.vectorize("") == [0.0] * 16

    # Embedder is None (pure fallback)
    guard_none = CounterfactualGuardrail(storage, embedder=None, vector_dim=16)
    guard_none.embedder = None  # Force None to test fallback branch
    assert len(guard_none.vectorize("sample text")) == 16


