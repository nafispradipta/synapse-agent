#!/usr/bin/env python3
"""
Unit and integration tests for Episodic-to-Semantic Consolidation Engine.
Verifies:
1. Category classification ('sop' vs 'architecture').
2. Token budget compliance (<40 tokens for L0).
3. Salience elevation (>= 0.90) and Amigdala decay immunity.
4. Synaptic edge synthesis between consolidated nodes.
5. Mental Model auto-refresh with consolidated directives.
6. WAL mode database integrity and idempotency.
"""
import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
import pytest

from synapse.storage import SynapseStorage
from synapse.worker import SynapseWorker
from synapse.loader import estimate_tokens
from synapse.mental_model_engine import MentalModelEngine


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    for p in [path, f"{path}-wal", f"{path}-shm"]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


def test_episodic_to_semantic_consolidation_rules_and_arch(temp_db):
    """Test converting SOP and Architecture episodes into permanent semantic nodes."""
    storage = SynapseStorage(temp_db)
    worker = SynapseWorker(storage)

    # 1. Seed an SOP episode (operational directive)
    sop_ep_id = storage.add_episode(
        title="Service Deployment Protocol",
        situation="Service crashed frequently when redeployed on default port.",
        action_taken="Added graceful shutdown signal trap in service loop.",
        outcome="Redeployment completed cleanly without downtime.",
        lessons_learned="Always use systemd service unit and verify SIGTERM trap is active before deployment.",
        salience=0.85,
        confidence=1.0,
        tags=["systemd", "deployment", "service"],
    )

    # 2. Seed an Architecture episode (system structure/tuning)
    arch_ep_id = storage.add_episode(
        title="Host Kernel Optimization and Memory Tuning",
        situation="Host with 2GB memory experienced kswapd thrashing when two compute-intensive workloads ran concurrently.",
        action_taken="Configured vm.swappiness=25 and vm.watermark_boost_factor=0.",
        outcome="Swap utilization stabilized and system load remained normal.",
        lessons_learned="Tuning kernel swappiness=25 and watermark_boost_factor=0 is the optimal memory architecture for 2GB hosts.",
        salience=0.95,
        confidence=1.0,
        tags=["kernel", "host", "memory", "swap"],
    )

    # Run consolidation
    res = worker.consolidate_episodes_to_semantic(limit=10, update_mental_models=True)

    assert res["status"] == "success"
    assert res["episodes_analyzed"] == 2
    assert res["consolidated_count"] == 2
    assert len(res["nodes_created"]) == 2
    assert res["edges_created"] >= 2
    assert "core-conventions" in res["mental_models_updated"]

    # Verify nodes in storage
    nodes = storage.list_nodes(limit=10)
    assert len(nodes) == 2

    # Map by l2_target_path
    node_map = {n["l2_target_path"]: n for n in nodes}
    assert f"episode:{sop_ep_id}" in node_map
    assert f"episode:{arch_ep_id}" in node_map

    sop_node = node_map[f"episode:{sop_ep_id}"]
    arch_node = node_map[f"episode:{arch_ep_id}"]

    # Verify SOP node attributes
    assert sop_node["category"] == "sop"
    assert sop_node["salience"] >= 0.92
    assert sop_node["strength"] == 1.0
    assert estimate_tokens(sop_node["l0_abstract"]) <= 40
    assert "Always use systemd service" in sop_node["l1_overview"]
    assert "## Permanent Directive (SOP)" in sop_node["l1_overview"]

    # Verify Architecture node attributes
    assert arch_node["category"] == "architecture"
    assert arch_node["salience"] >= 0.95
    assert arch_node["strength"] == 1.0
    assert estimate_tokens(arch_node["l0_abstract"]) <= 40
    assert "Tuning kernel swappiness=25" in arch_node["l1_overview"]
    assert "## Permanent Directive (ARCHITECTURE)" in arch_node["l1_overview"]

    # Verify episodes marked as consolidated
    ep_sop = storage.get_episode(sop_ep_id)
    ep_arch = storage.get_episode(arch_ep_id)
    assert "consolidated" in json.loads(ep_sop["tags"])
    assert "consolidated" in json.loads(ep_arch["tags"])

    # Verify Synaptic Edges exist between the two co-consolidated nodes
    edges = storage.get_edges(sop_node["id"])
    target_ids = [e["target_id"] for e in edges]
    assert arch_node["id"] in target_ids

    # Verify Mental Model updated
    mm = storage.get_mental_model("core-conventions")
    assert mm is not None
    assert "Always use systemd service" in mm["content"] or "Tuning kernel swappiness=25" in mm["content"]

    storage.close()


def test_consolidation_idempotency(temp_db):
    """Verify running consolidation multiple times does not create duplicate nodes."""
    storage = SynapseStorage(temp_db)
    worker = SynapseWorker(storage)

    storage.add_episode(
        title="Cache Invalidation SOP",
        situation="Stale cache displayed on frontend.",
        action_taken="Added ETag header and automatic purge on build.",
        outcome="Frontend consistently displays fresh data.",
        lessons_learned="Must purge CDN cache and validate ETag after every deployment.",
        salience=0.90,
        confidence=1.0,
    )

    # First run: 1 node created
    res1 = worker.consolidate_episodes_to_semantic(limit=10)
    assert res1["status"] == "success"
    assert res1["consolidated_count"] == 1

    # Second run: 0 new nodes (noop / all already consolidated)
    res2 = worker.consolidate_episodes_to_semantic(limit=10)
    assert res2["status"] == "noop"
    assert res2["consolidated_count"] == 0

    assert len(storage.list_nodes()) == 1

    # Force run: updates existing node without duplicating
    res3 = worker.consolidate_episodes_to_semantic(limit=10, force=True)
    assert res3["status"] == "success"
    assert res3["consolidated_count"] == 1
    assert len(res3["nodes_updated"]) == 1
    assert len(storage.list_nodes()) == 1

    storage.close()


def test_wal_mode_and_decay_immunity(temp_db):
    """Verify WAL mode integrity and that consolidated nodes are immune to Hebbian decay."""
    storage = SynapseStorage(temp_db)
    worker = SynapseWorker(storage)

    # Add an episode and consolidate
    ep_id = storage.add_episode(
        title="Database WAL Mode Requirement",
        situation="Concurrency locks occurred on default SQLite rollback journal.",
        action_taken="Executed PRAGMA journal_mode=WAL and busy_timeout=10000.",
        outcome="Write concurrency succeeded without database locks.",
        lessons_learned="Always enable PRAGMA journal_mode=WAL and 10 second timeout on every SQLite DB.",
        salience=0.92,
        confidence=1.0,
    )

    worker.consolidate_episodes_to_semantic(limit=10)

    # Also add a normal ephemeral node that should decay
    past_ts = int(time.time()) - (30 * 86400)
    id_ephemeral = storage.add_node(
        uri="synapse://general/ephemeral_note",
        l0_abstract="Temporary scratchpad note.",
        category="general",
        salience=0.4,
        importance=0.2,
        strength=0.5,
    )
    storage._conn.execute("UPDATE synapse_nodes SET last_accessed = ? WHERE id = ?", (past_ts, id_ephemeral))
    storage._conn.commit()

    # Backdate the consolidated node to test immunity
    storage._conn.execute(
        "UPDATE synapse_nodes SET last_accessed = ? WHERE l2_target_path = ?",
        (past_ts, f"episode:{ep_id}"),
    )
    storage._conn.commit()

    # Run decay cycle
    decay_stats = worker.decay_cycle(decay_rate=0.1, archive_threshold=0.2)
    assert decay_stats["archived"] >= 1  # ephemeral note archived

    # Consolidated node should remain active with strength 1.0
    nodes = [n for n in storage.list_nodes() if n.get("l2_target_path") == f"episode:{ep_id}"]
    assert len(nodes) == 1
    consolidated_node = nodes[0]
    assert consolidated_node["category"] in ("sop", "architecture")
    assert consolidated_node["status"] == "active"
    assert consolidated_node["strength"] == 1.0
    assert consolidated_node["salience"] >= 0.90

    # Verify SQLite WAL integrity check returns 'ok'
    cur = storage._conn.cursor()
    cur.execute("PRAGMA integrity_check")
    assert cur.fetchone()[0] == "ok"

    storage.close()


def test_nightly_consolidation_pipeline_integration(temp_db):
    """Simulate the nightly consolidation script execution with Step 2.5."""
    storage = SynapseStorage(temp_db)
    
    # 1. Add sample episodes
    ep1 = storage.add_episode(
        title="Automated Backup SOP",
        situation="Risk of data loss on server failure.",
        action_taken="Configured automated cron backup to offsite storage.",
        outcome="Daily backups verified and functional.",
        lessons_learned="Automated backups must run daily at 02:00 UTC with hash checksum verification.",
        salience=0.91,
        confidence=1.0,
    )
    storage.close()

    # Run the pipeline logic as in scripts/nightly_consolidation.py
    conn = sqlite3.connect(temp_db)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA busy_timeout=10000;")
    cur.execute("PRAGMA integrity_check;")
    assert cur.fetchone()[0] == "ok"

    # Step 2: Amigdala Protection
    with conn:
        cur.execute("""
            UPDATE synapse_nodes
            SET strength = 1.0, confidence = 1.0
            WHERE salience >= 0.90 OR category IN ('preference', 'decision')
        """)

    # Step 2.5: Episodic Consolidation
    worker_storage = SynapseStorage(temp_db)
    try:
        worker = SynapseWorker(storage=worker_storage)
        c_res = worker.consolidate_episodes_to_semantic(limit=10)
        assert c_res["status"] == "success"
        assert c_res["consolidated_count"] == 1
    finally:
        worker_storage.close()

    # Step 3: Hebbian Decay
    with conn:
        _cutoff_ts = int(time.time()) - (14 * 86400)
        cur.execute("""
            UPDATE synapse_nodes
            SET strength = MAX(0.05, strength - 0.05)
            WHERE salience < 0.80
              AND category NOT IN ('preference', 'decision', 'architecture', 'sop')
              AND status = 'active'
              AND typeof(last_accessed) = 'integer' AND last_accessed < ?
        """, (_cutoff_ts,))

    # Step 7: VACUUM and Post Integrity Check
    cur.execute("VACUUM")
    cur.execute("PRAGMA integrity_check;")
    assert cur.fetchone()[0] == "ok"
    conn.close()

    # Re-open and verify node exists and is protected
    verify_storage = SynapseStorage(temp_db)
    nodes = [n for n in verify_storage.list_nodes() if n.get("l2_target_path") == f"episode:{ep1}"]
    assert len(nodes) == 1
    assert nodes[0]["salience"] >= 0.90
    assert nodes[0]["strength"] == 1.0
    verify_storage.close()

