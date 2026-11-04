"""
Unit tests for Synapse Auto-Contradiction Resolver & Self-Correcting Engine:
- Topic similarity (keyword overlap, vector similarity > 0.82)
- Factual claim conflict detection (numeric specifications, operational status, directive rules)
- Automatic superseding of older/lower-salience nodes (status='superseded', strength=0.05)
- Adapter retain() integration (real-time conflict pre-check & post-save resolution)
- Nightly mass consolidation audit
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synapse.storage import SynapseStorage
from synapse.adapter import SynapseAdapter
from synapse.contradiction_resolver import (
    ContradictionResolver,
    extract_keywords,
    extract_numeric_specs,
    extract_status_flags,
    extract_directive_flags,
    parse_timestamp,
    cosine_similarity,
)
from scripts.nightly_consolidation import run_consolidation


# ============================================================================
# 1. Feature Extraction & Parsing Tests
# ============================================================================

def test_extract_numeric_specs() -> None:
    """Test extraction of numeric attributes and units."""
    t1 = "Production server specification uses 2GB RAM."
    t2 = "Production server specification upgraded to 4GB RAM."
    t3 = "Use port 8080 for API gateway and timeout 30s."
    t4 = "Disk capacity of 100GB with 4 core CPU."

    s1 = extract_numeric_specs(t1)
    s2 = extract_numeric_specs(t2)
    s3 = extract_numeric_specs(t3)
    s4 = extract_numeric_specs(t4)

    assert s1.get("ram") == (2.0, "gb")
    assert s2.get("ram") == (4.0, "gb")
    assert s3.get("port") == (8080.0, "")
    assert s3.get("timeout") == (30.0, "s")
    assert s4.get("disk") == (100.0, "gb")
    assert s4.get("cpu") == (4.0, "core")


def test_extract_status_and_directive_flags() -> None:
    """Test extraction of active/inactive status and mandatory/forbidden directives."""
    active_text = "Redis server service is active and running."
    inactive_text = "Redis server service is inactive and stopped."
    wajib_text = "Mandatory to use PostgreSQL for data persistence."
    jangan_text = "Do not use PostgreSQL for data persistence, avoid."

    assert extract_status_flags(active_text) == "active"
    assert extract_status_flags(inactive_text) == "inactive"
    assert extract_directive_flags(wajib_text) == "mandatory"
    assert extract_directive_flags(jangan_text) == "forbidden"


# ============================================================================
# 2. Topic Similarity & Claim Conflict Detection
# ============================================================================

def test_topic_similarity_and_conflict_detection() -> None:
    """Test that same-topic nodes are correctly matched and claim discrepancies detected."""
    resolver = ContradictionResolver()

    node_old = {
        "id": "node-1",
        "category": "architecture",
        "l0_abstract": "Server RAM 2GB",
        "l1_overview": "Server allocated 2GB RAM for web service.",
        "created_at": 1000,
        "salience": 0.5,
    }
    node_new = {
        "id": "node-2",
        "category": "architecture",
        "l0_abstract": "Server RAM 4GB",
        "l1_overview": "Server allocated 4GB RAM for web service.",
        "created_at": 2000,
        "salience": 0.7,
    }

    # Verify same topic
    is_same, score, method = resolver.is_same_topic(node_old, node_new)
    assert is_same is True

    # Verify conflict detected
    conflict = resolver.check_contradiction(node_old, node_new)
    assert conflict is not None
    assert conflict["conflict_type"] == "numeric_specification"
    assert conflict["attribute"] == "ram"
    assert conflict["val_a"] == 2.0
    assert conflict["val_b"] == 4.0


def test_vector_similarity_threshold() -> None:
    """Test vector similarity topic matching when cosine similarity exceeds 0.82."""
    resolver = ContradictionResolver(vector_threshold=0.82)

    node_a = {
        "id": "v1",
        "category": "architecture",
        "l0_abstract": "Database Cluster",
        "l1_overview": "PostgreSQL replicas and failover.",
    }
    node_b = {
        "id": "v2",
        "category": "architecture",
        "l0_abstract": "Frontend Layout",
        "l1_overview": "React UI components and styling.",
    }

    # Mock high vector similarity > 0.82
    is_same_high, score_high, method_high = resolver.is_same_topic(node_a, node_b, vector_sim=0.88)
    assert is_same_high is True
    assert method_high == "vector_similarity"
    assert score_high == 0.88

    # Low vector similarity < 0.82 (and no keyword overlap)
    is_same_low, _, _ = resolver.is_same_topic(node_a, node_b, vector_sim=0.50)
    assert is_same_low is False


# ============================================================================
# 3. Core Contradiction Resolution: Spec RAM 2GB vs 4GB
# ============================================================================

def test_ram_spec_contradiction_supersedes_old_node(tmp_path: Path) -> None:
    """
    Test the canonical spec contradiction scenario:
    Old node: RAM 2GB -> New node: RAM 4GB.
    Proves that the old node is automatically marked 'superseded' with strength=0.05.
    """
    db_file = str(tmp_path / "test_synapse_contradiction.db")
    storage = SynapseStorage(db_file)
    resolver = ContradictionResolver(storage=storage)

    # 1. Insert initial/old memory node (RAM 2GB)
    old_id = storage.add_node(
        uri="synapse://architecture/vps_spec_old",
        l0_abstract="Server VPS dialokasikan RAM 2GB",
        l1_overview="Spesifikasi server production menggunakan RAM 2GB.",
        category="architecture",
        importance=0.6,
        salience=0.6,
        strength=1.0,
        status="active",
        created_at=1000,
    )
    assert old_id is not None

    # Check node is active initially
    node_before = storage.get_node(old_id)
    assert node_before is not None
    assert node_before["status"] == "active"
    assert node_before["strength"] == 1.0

    # 2. Insert updated/new memory node (RAM 4GB)
    new_id = storage.add_node(
        uri="synapse://architecture/vps_spec_new",
        l0_abstract="Server VPS dialokasikan RAM 4GB",
        l1_overview="Spesifikasi server production di-upgrade menjadi RAM 4GB.",
        category="architecture",
        importance=0.8,
        salience=0.8,
        strength=1.0,
        status="active",
        created_at=2000,
    )
    assert new_id is not None

    # 3. Run contradiction resolver for the new node
    new_node = storage.get_node(new_id)
    resolved = resolver.resolve_node_contradictions(new_node)

    assert len(resolved) == 1
    assert resolved[0]["winner_id"] == new_id
    assert resolved[0]["loser_id"] == old_id
    assert resolved[0]["superseded_node_id"] == old_id
    assert resolved[0]["conflict_type"] == "numeric_specification"

    # 4. Verify in storage: Old node MUST be superseded with strength 0.05
    old_node_after = storage.get_node(old_id)
    assert old_node_after is not None
    assert old_node_after["status"] == "superseded"
    assert old_node_after["strength"] == 0.05

    # 5. Verify in storage: New node MUST remain active with full strength
    new_node_after = storage.get_node(new_id)
    assert new_node_after is not None
    assert new_node_after["status"] == "active"
    assert new_node_after["strength"] == 1.0

    # 6. Verify supersedes graph edge was recorded
    edges = storage.get_edges(new_id, direction="out")
    supersedes_edges = [e for e in edges if e["relation"] == "supersedes" and e["target_id"] == old_id]
    assert len(supersedes_edges) == 1

    storage.close()


# ============================================================================
# 4. Status and Directive Contradiction Resolution
# ============================================================================

def test_status_and_directive_contradiction_resolution(tmp_path: Path) -> None:
    """Test resolution for active/inactive status and mandatory/forbidden directive clashes."""
    db_file = str(tmp_path / "test_status_directive.db")
    storage = SynapseStorage(db_file)
    resolver = ContradictionResolver(storage=storage)

    # Test Status Clash: Redis active vs Redis stopped
    redis_old = storage.add_node(
        uri="synapse://architecture/redis_status_old",
        l0_abstract="Redis service is active on server",
        l1_overview="Redis service status is active and running.",
        category="architecture",
        salience=0.5,
        created_at=1000,
    )
    redis_new = storage.add_node(
        uri="synapse://architecture/redis_status_new",
        l0_abstract="Redis service is stopped",
        l1_overview="Redis service status is stopped and inactive.",
        category="architecture",
        salience=0.6,
        created_at=2000,
    )

    resolved_status = resolver.resolve_node_contradictions(storage.get_node(redis_new))
    assert len(resolved_status) == 1
    assert resolved_status[0]["conflict_type"] == "operational_status"
    assert storage.get_node(redis_old)["status"] == "superseded"
    assert storage.get_node(redis_old)["strength"] == 0.05

    # Test Directive Clash: Mandatory vs Forbidden PostgreSQL
    pg_old = storage.add_node(
        uri="synapse://decision/pg_directive_old",
        l0_abstract="Mandatory to use PostgreSQL",
        l1_overview="Mandatory to use PostgreSQL for primary persistence.",
        category="decision",
        salience=0.7,
        created_at=1500,
    )
    pg_new = storage.add_node(
        uri="synapse://decision/pg_directive_new",
        l0_abstract="Do not use PostgreSQL",
        l1_overview="Do not use PostgreSQL for primary persistence, migrate to SQLite.",
        category="decision",
        salience=0.9,
        created_at=2500,
    )

    resolved_dir = resolver.resolve_node_contradictions(storage.get_node(pg_new))
    assert len(resolved_dir) == 1
    assert resolved_dir[0]["conflict_type"] == "directive_rule"
    assert storage.get_node(pg_old)["status"] == "superseded"
    assert storage.get_node(pg_old)["strength"] == 0.05

    storage.close()


# ============================================================================
# 5. SynapseAdapter.retain() Real-Time Integration & Retrieval Erasure
# ============================================================================

def test_adapter_retain_auto_resolves_contradiction_and_retrieval(tmp_path: Path) -> None:
    """
    Test full end-to-end flow via SynapseAdapter:
    1. Retain RAM 2GB.
    2. Retain RAM 4GB.
    3. Old node is automatically superseded.
    4. Recall only retrieves the new 4GB fact, eradicating the obsolete 2GB hallucination.
    """
    db_file = str(tmp_path / "test_adapter_retain.db")
    adapter = SynapseAdapter({"db_path": db_file})

    # Step 1: Retain old fact
    ok1 = adapter.retain(
        content="Server main-host allocated 2GB RAM for database service.",
        context="Host infrastructure specifications",
        category="architecture",
        salience=0.6,
        created_at=1000,
    )
    assert ok1 is True

    # Retrieve active nodes
    active_nodes_1 = adapter.storage.list_nodes(status="active")
    assert len(active_nodes_1) == 1
    old_node_id = active_nodes_1[0]["id"]
    assert "2GB" in active_nodes_1[0]["l0_abstract"]

    # Sleep slightly to ensure monotonic timestamp
    time.sleep(0.05)

    # Step 2: Retain new conflicting fact
    ok2 = adapter.retain(
        content="Server main-host allocated 4GB RAM for database service.",
        context="Host infrastructure specifications upgraded",
        category="architecture",
        salience=0.8,
        created_at=2000,
    )
    assert ok2 is True

    # Step 3: Verify old node is superseded in storage
    old_node_record = adapter.storage.get_node(old_node_id)
    assert old_node_record is not None
    assert old_node_record["status"] == "superseded"
    assert old_node_record["strength"] == 0.05

    # Verify only 1 active node remains in storage
    active_nodes_2 = adapter.storage.list_nodes(status="active")
    assert len(active_nodes_2) == 1
    assert "4GB" in active_nodes_2[0]["l0_abstract"]

    # Step 4: Verify Recall returns only the 4GB fact and never the superseded 2GB fact
    recalled = adapter.recall("Server main-host RAM", limit=5)
    recalled_text = " ".join(recalled)
    assert "4GB" in recalled_text
    assert "2GB" not in recalled_text

    adapter.storage.close()


# ============================================================================
# 6. Nightly Consolidation Mass Audit Test
# ============================================================================

def test_nightly_consolidation_mass_audit(tmp_path: Path) -> None:
    """
    Test mass audit capability in nightly consolidation.
    Pre-populates storage with multiple dormant contradictory nodes,
    runs run_consolidation(), and verifies all obsolete nodes are superseded.
    """
    db_file = str(tmp_path / "test_nightly_consolidation.db")
    storage = SynapseStorage(db_file)

    # Insert conflicting pair 1: port 8080 (old) vs port 3000 (new)
    n1 = storage.add_node(
        uri="synapse://architecture/gateway_port_old",
        l0_abstract="API gateway port is 8080",
        l1_overview="API gateway configured to listen on port 8080.",
        category="architecture",
        created_at=1000,
        salience=0.5,
    )
    n2 = storage.add_node(
        uri="synapse://architecture/gateway_port_new",
        l0_abstract="API gateway port is 3000",
        l1_overview="API gateway port remapped to listen on port 3000.",
        category="architecture",
        created_at=3000,
        salience=0.7,
    )

    # Insert conflicting pair 2: worker_threads 4 (old) vs worker_threads 8 (new)
    n3 = storage.add_node(
        uri="synapse://architecture/worker_threads_old",
        l0_abstract="Worker threads configuration: 4",
        l1_overview="Allocation of worker_threads: 4 across system.",
        category="architecture",
        created_at=1200,
        salience=0.5,
    )
    n4 = storage.add_node(
        uri="synapse://architecture/worker_threads_new",
        l0_abstract="Worker threads configuration: 8",
        l1_overview="Allocation of worker_threads: 8 across system.",
        category="architecture",
        created_at=3500,
        salience=0.8,
    )

    storage.close()

    # Run consolidation on the test DB (skip external LLM/crawler services and vault scan)
    run_consolidation(db_path=db_file, skip_external=True, skip_vault_sync=True)

    # Re-open storage and verify results
    check_storage = SynapseStorage(db_file)
    try:
        # Check pair 1
        node_n1 = check_storage.get_node(n1)
        node_n2 = check_storage.get_node(n2)
        assert node_n1["status"] == "superseded"
        assert node_n1["strength"] == 0.05
        assert node_n2["status"] == "active"

        # Check pair 2
        node_n3 = check_storage.get_node(n3)
        node_n4 = check_storage.get_node(n4)
        assert node_n3["status"] == "superseded"
        assert node_n3["strength"] == 0.05
        assert node_n4["status"] == "active"

        # Active nodes should only be n2 and n4 (and any system synced notes)
        active_ids = {n["id"] for n in check_storage.list_nodes(status="active")}
        assert n1 not in active_ids
        assert n3 not in active_ids
        assert n2 in active_ids
        assert n4 in active_ids
    finally:
        check_storage.close()


def test_contradiction_software_version_progression(tmp_path: Path) -> None:
    """Test that engine version updates (e.g. PostgreSQL 14 -> PostgreSQL 16) supersede older claims."""
    db_file = str(tmp_path / "test_version_progression.db")
    adapter = SynapseAdapter({"db_path": db_file})
    assert adapter.storage is not None

    adapter.retain(
        content="Primary production database is PostgreSQL 14 with pgvector.",
        category="architecture",
        salience=0.8,
        created_at=1000,
    )
    assert len(adapter.storage.list_nodes(status="active")) == 1

    time.sleep(0.05)
    adapter.retain(
        content="Primary production database is upgraded to PostgreSQL 16 with pgvector.",
        category="architecture",
        salience=0.85,
        created_at=2000,
    )

    active_nodes = adapter.storage.list_nodes(status="active")
    assert len(active_nodes) == 1
    assert "PostgreSQL 16" in active_nodes[0]["l0_abstract"] or "16" in active_nodes[0]["l0_abstract"]

    superseded_nodes = adapter.storage.list_nodes(status="superseded")
    assert len(superseded_nodes) == 1
    assert "14" in superseded_nodes[0]["l0_abstract"]
    adapter.storage.close()


def test_contradiction_task_lifecycle_in_progress_to_completed(tmp_path: Path) -> None:
    """Test that task status transitions (e.g. In Progress -> Completed) automatically supersede in-progress claims."""
    db_file = str(tmp_path / "test_task_lifecycle.db")
    adapter = SynapseAdapter({"db_path": db_file})
    assert adapter.storage is not None

    adapter.retain(
        content="Task #402 database migration is currently in progress by worker agent.",
        category="decision",
        salience=0.7,
        created_at=1000,
    )
    assert len(adapter.storage.list_nodes(status="active")) == 1

    time.sleep(0.05)
    adapter.retain(
        content="Task #402 database migration is completed successfully.",
        category="decision",
        salience=0.8,
        created_at=2000,
    )

    active_nodes = adapter.storage.list_nodes(status="active")
    assert len(active_nodes) == 1
    assert "completed" in active_nodes[0]["l0_abstract"].lower()

    superseded_nodes = adapter.storage.list_nodes(status="superseded")
    assert len(superseded_nodes) == 1
    assert "progress" in superseded_nodes[0]["l0_abstract"].lower()
    adapter.storage.close()


def test_no_false_positive_on_unrelated_tasks(tmp_path: Path) -> None:
    """Verify that unrelated tasks with different subjects are never superseded by global status."""
    db_file = str(tmp_path / "test_unrelated_tasks.db")
    adapter = SynapseAdapter({"db_path": db_file})
    assert adapter.storage is not None

    adapter.retain("Task database migration is currently in progress.", category="decision")
    adapter.retain("Deploy frontend to production is completed.", category="decision")

    active = adapter.storage.list_nodes(status="active")
    assert len(active) == 2
    superseded = adapter.storage.list_nodes(status="superseded")
    assert len(superseded) == 0
    adapter.storage.close()


def test_no_false_positive_on_different_components_same_tech(tmp_path: Path) -> None:
    """Verify that distinct components using the same technology (e.g. backend vs script) are not superseded."""
    db_file = str(tmp_path / "test_diff_components.db")
    adapter = SynapseAdapter({"db_path": db_file})
    assert adapter.storage is not None

    adapter.retain("Backend API service is written in Python 3.11.", category="architecture")
    adapter.retain("Data migration script requires Python 3.12.", category="architecture")

    active = adapter.storage.list_nodes(status="active")
    assert len(active) == 2
    superseded = adapter.storage.list_nodes(status="superseded")
    assert len(superseded) == 0
    adapter.storage.close()


def test_multi_attribute_numerical_conflict(tmp_path: Path) -> None:
    """Verify sentences with multiple attributes (e.g. RAM + disk) properly resolve conflicts on the differing attribute."""
    db_file = str(tmp_path / "test_multi_attr.db")
    adapter = SynapseAdapter({"db_path": db_file})
    assert adapter.storage is not None

    adapter.retain("Server spec: 8 GB RAM and 100 GB disk.", category="architecture")
    adapter.retain("Server spec: 16 GB RAM and 100 GB disk.", category="architecture")

    active = adapter.storage.list_nodes(status="active")
    assert len(active) == 1
    assert "16" in active[0]["l0_abstract"]
    superseded = adapter.storage.list_nodes(status="superseded")
    assert len(superseded) == 1
    assert "8" in superseded[0]["l0_abstract"]
    adapter.storage.close()


def test_task_progression_lifecycle_extensible_terms(tmp_path: Path) -> None:
    """Verify standard task lifecycle states (in progress -> completed) properly supersede."""
    db_file = str(tmp_path / "test_lifecycle.db")
    adapter = SynapseAdapter({"db_path": db_file})
    assert adapter.storage is not None

    adapter.retain("Database migration task is in progress.", category="decision")
    adapter.retain("Database migration task is completed.", category="decision")

    active = adapter.storage.list_nodes(status="active")
    assert len(active) == 1
    assert "completed" in active[0]["l0_abstract"]
    superseded = adapter.storage.list_nodes(status="superseded")
    assert len(superseded) == 1
    assert "progress" in superseded[0]["l0_abstract"]
    adapter.storage.close()


def test_bilingual_status_and_spec_normalization(tmp_path: Path) -> None:
    """Verify Indonesian operational status terms and spec canonicalization work cleanly."""
    from synapse.contradiction_resolver import extract_status_flags, extract_numeric_specs

    assert extract_status_flags("server aktif") == "active"
    assert extract_status_flags("server nonaktif") == "inactive"
    assert extract_status_flags("Migrasi database sedang berjalan") == "in_progress"
    assert extract_status_flags("Migrasi database sudah selesai") == "completed"

    specs = extract_numeric_specs("Server ini menggunakan 8 GB memori dan 100 GB disk.")
    assert "ram" in specs
    assert specs["ram"] == (8.0, "gb")
    assert "disk" in specs
    assert specs["disk"] == (100.0, "gb")


def test_bilingual_task_progression_and_subject_isolation(tmp_path: Path) -> None:
    """Verify Indonesian task lifecycle properly supersedes identical subjects without false positives across different subjects."""
    db_file = str(tmp_path / "test_bilingual.db")
    adapter = SynapseAdapter({"db_path": db_file})
    assert adapter.storage is not None

    # Different subjects: MUST NOT supersede each other!
    adapter.retain("Task migrasi database sedang berjalan.", category="decision")
    adapter.retain("Deploy frontend ke Vercel sudah selesai.", category="decision")

    active = adapter.storage.list_nodes(status="active")
    assert len(active) == 2, "Different subjects must both stay active"

    # Same subject progression: MUST supersede earlier progress state
    adapter.retain("Task migrasi database sudah selesai.", category="decision")

    active = adapter.storage.list_nodes(status="active")
    assert len(active) == 2  # Vercel deploy + database migration completed
    superseded = adapter.storage.list_nodes(status="superseded")
    assert len(superseded) == 1  # The earlier 'sedang berjalan' node
    assert "sedang berjalan" in superseded[0]["l0_abstract"]
    adapter.storage.close()


def test_negation_phrase_status_and_resolution(tmp_path: Path) -> None:
    """Verify 'tidak aktif' / 'not active' phrases correctly resolve operational status contradiction."""
    from synapse.contradiction_resolver import extract_status_flags

    assert extract_status_flags("server aktif") == "active"
    assert extract_status_flags("server tidak aktif") == "inactive"
    assert extract_status_flags("server not active") == "inactive"

    db_file = str(tmp_path / "test_negation_status.db")
    adapter = SynapseAdapter({"db_path": db_file})
    assert adapter.storage is not None

    adapter.retain("Server produksi aktif dan melayani traffic.", category="architecture")
    adapter.retain("Server produksi tidak aktif karena maintenance.", category="architecture")

    active = adapter.storage.list_nodes(status="active")
    assert len(active) == 1
    assert "tidak aktif" in active[0]["l0_abstract"]
    superseded = adapter.storage.list_nodes(status="superseded")
    assert len(superseded) == 1
    assert "Server produksi aktif" in superseded[0]["l0_abstract"]
    adapter.storage.close()


def test_indonesian_category_and_tags_inference(tmp_path: Path) -> None:
    """Verify Indonesian keywords trigger correct category inference and tags."""
    db_file = str(tmp_path / "test_cat_tags.db")
    adapter = SynapseAdapter({"db_path": db_file})

    cat = adapter._infer_category("Deploy gagal karena masalah cache Redis", "")
    assert cat == "incident"

    sop_cat = adapter._infer_category("Panduan langkah backup database PostgreSQL", "")
    assert sop_cat == "sop"

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    (vault_dir / "tags.md").write_text("- finance\n- meeting\n- marketing\n- sop\n")

    tags = adapter._select_vault_tags(
        vault_dir,
        "general",
        "Team meeting to review quarterly budget, financial forecasts, and advertising campaign",
        [],
        {"tags": []},
    )
    assert "finance" in tags
    assert "meeting" in tags
    assert "marketing" in tags


def test_software_version_downgrade_guard(tmp_path: Path) -> None:
    """Verify inadvertent software downgrade without explicit downgrade keyword is rejected."""
    db_file = str(tmp_path / "test_downgrade.db")
    adapter = SynapseAdapter({"db_path": db_file})
    assert adapter.storage is not None

    adapter.retain("Database server uses PostgreSQL 16.", category="architecture", salience=0.5)
    # Higher salience lower version without downgrade keyword -> protected!
    adapter.retain("Database server uses PostgreSQL 14.", category="architecture", salience=0.8)

    active = adapter.storage.list_nodes(status="active")
    assert len(active) == 1
    assert "16" in active[0]["l0_abstract"]

    # Explicit downgrade keyword -> allowed!
    adapter.retain("Rollback database server to PostgreSQL 14.", category="architecture", salience=0.9)
    active = adapter.storage.list_nodes(status="active")
    assert len(active) == 1
    assert "Rollback" in active[0]["l0_abstract"]
    adapter.storage.close()


def test_word_boundary_immunity_for_otomatis_and_automatic(tmp_path: Path) -> None:
    """Verify words containing 'mati' as substring (otomatis, automatic, sistematis) are NOT parsed as inactive."""
    from synapse.contradiction_resolver import extract_status_flags

    assert extract_status_flags("Deploy otomatis via CI") != "inactive"
    assert extract_status_flags("Backup otomatis tiap hari") != "inactive"
    assert extract_status_flags("Proses sistematis") != "inactive"
    assert extract_status_flags("Deployment is automatic") != "inactive"
    assert extract_status_flags("Deploys automatically every night") != "inactive"
    assert extract_status_flags("System is systematic") != "inactive"

    # Actual inactive statements must still work
    assert extract_status_flags("Server produksi mati") == "inactive"
    assert extract_status_flags("Server produksi tidak aktif") == "inactive"

    db_file = str(tmp_path / "test_otomatis_collision.db")
    adapter = SynapseAdapter({"db_path": db_file})
    assert adapter.storage is not None

    adapter.retain("Backup database otomatis berjalan setiap tengah malam.", salience=0.6)
    adapter.retain("Backup database manual berjalan setiap pagi.", salience=0.9)

    active = adapter.storage.list_nodes(status="active")
    assert len(active) == 2, "Automatic and manual backups must both stay active without false supersede"
    adapter.storage.close()


def test_extract_numeric_specs_binds_values_to_their_own_attribute() -> None:
    specs = extract_numeric_specs(
        "Project Alpha service 7 uses port 8007 with timeout 30s."
    )
    assert specs["port"] == (8007.0, "")
    assert specs["timeout"] == (30.0, "s")
    assert "service" not in specs

    specs2 = extract_numeric_specs("Service 3 runs on port 9003 with timeout 45s.")
    assert specs2["port"] == (9003.0, "")
    assert specs2["timeout"] == (45.0, "s")


def test_numeric_facts_for_distinct_services_are_not_contradictions() -> None:
    adapter = SynapseAdapter({"db_path": ":memory:"})
    try:
        for i in range(20):
            adapter.retain(
                f"Project Alpha service {i} uses port {8000 + i} "
                f"and has timeout {30 + i}s.",
                category="architecture",
                salience=0.5,
            )
        statuses = adapter.storage.get_stats()["statuses"]
        assert statuses.get("active") == 20
        assert statuses.get("superseded", 0) == 0
    finally:
        adapter.storage.close()


def test_same_service_spec_change_still_supersedes() -> None:
    adapter = SynapseAdapter({"db_path": ":memory:"})
    try:
        adapter.retain(
            "Service 7 uses port 8007 with timeout 30s.",
            category="architecture",
            salience=0.5,
        )
        adapter.retain(
            "Service 7 uses port 9007 with timeout 45s.",
            category="architecture",
            salience=0.5,
        )
        assert len(adapter.storage.list_nodes(status="active")) == 1
        assert len(adapter.storage.list_nodes(status="superseded")) == 1
    finally:
        adapter.storage.close()
