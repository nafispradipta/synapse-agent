"""
Tests for Human Brain Cognitive Architecture in Synapse Memory Engine.
Validates:
- Episodic memory CRUD & FTS5 full-text search
- Salience & Confidence weighting and decay immunity (salience >= 0.9)
- Contradiction & Metacognition detection
- Dual semantic + episodic session distillation
- L0/L1 episodic context formatting under 250 token budget
- SynapseAdapter episodic API integration
"""

import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
from synapse import (
    SessionManager,
    SynapseAdapter,
    SynapseContextLoader,
    SynapseRetriever,
    SynapseStorage,
    SynapseWorker,
    detect_contradictions,
    estimate_tokens,
)


def test_episodic_storage_crud_and_fts(tmp_path: Path) -> None:
    """Validate full lifecycle and FTS indexing for episodic memories."""
    db_file = str(tmp_path / "synapse_episodes.db")
    storage = SynapseStorage(db_file)

    # 1. Add episodes
    ep1_id = storage.add_episode(
        title="PostgreSQL WAL Corruption Incident",
        situation="Production database crashed during power outage with corrupted WAL files.",
        action_taken="Ran pg_resetwal and restored latest consistent basebackup with WAL replay.",
        outcome="Database successfully recovered with 0 data loss.",
        lessons_learned="Always enable synchronous_commit and configure automated WAL archiving to S3.",
        salience=0.95,
        confidence=1.0,
        tags=["postgres", "wal", "recovery", "incident"],
    )
    assert ep1_id is not None
    assert len(ep1_id) == 32

    ep2_id = storage.add_episode(
        title="Redis Memory Exhaustion",
        situation="Redis cache reached maxmemory 8GB and started evicting critical session tokens.",
        action_taken="Switched eviction policy to volatile-lru and increased memory limit to 16GB.",
        outcome="Session evictions stopped immediately.",
        lessons_learned="Set maxmemory-policy to volatile-lru for session stores and add monitoring at 75%.",
        salience=0.8,
        confidence=0.9,
        tags=["redis", "cache", "memory"],
    )

    # 2. Get episode
    ep1 = storage.get_episode(ep1_id, touch=False)
    assert ep1 is not None
    assert ep1["title"] == "PostgreSQL WAL Corruption Incident"
    assert ep1["salience"] == 0.95
    assert ep1["confidence"] == 1.0
    assert "postgres" in ep1["tags"]

    # 3. Search episodes via FTS5
    hits_wal = storage.search_episodes("WAL corruption")
    assert len(hits_wal) >= 1
    assert hits_wal[0]["id"] == ep1_id
    assert "pg_resetwal" in hits_wal[0]["action_taken"]

    hits_redis = storage.search_episodes("eviction policy volatile-lru")
    assert len(hits_redis) >= 1
    assert hits_redis[0]["id"] == ep2_id

    # 4. Update episode
    ok = storage.update_episode(
        ep2_id,
        salience=0.85,
        lessons_learned="Always set volatile-lru and alert at 70% capacity.",
    )
    assert ok is True
    updated_ep2 = storage.get_episode(ep2_id, touch=False)
    assert updated_ep2["salience"] == 0.85
    assert "70% capacity" in updated_ep2["lessons_learned"]

    # 5. List episodes
    all_eps = storage.list_episodes(limit=10)
    assert len(all_eps) == 2

    # 6. Delete episode
    del_ok = storage.delete_episode(ep1_id)
    assert del_ok is True
    assert storage.get_episode(ep1_id) is None
    assert len(storage.search_episodes("WAL corruption")) == 0


def test_salience_confidence_migration_and_defaults(tmp_path: Path) -> None:
    """Validate salience and confidence default values and auto-migration."""
    db_file = str(tmp_path / "synapse_migration.db")
    storage = SynapseStorage(db_file)

    # Default node creation
    n1_id = storage.add_node(
        uri="synapse://preference/editor_theme",
        l0_abstract="User prefers dark theme in VS Code.",
    )
    n1 = storage.get_node(n1_id)
    assert n1["salience"] == 0.5
    assert n1["confidence"] == 1.0

    # Custom salience & confidence
    n2_id = storage.add_node(
        uri="synapse://decision/master_directive",
        l0_abstract="Master Directives: Never overwrite customer production tables directly.",
        salience=0.98,
        confidence=1.0,
    )
    n2 = storage.get_node(n2_id)
    assert n2["salience"] == 0.98
    assert n2["confidence"] == 1.0

    # Update salience & confidence
    storage.update_node(n1_id, salience=0.75, confidence=0.85)
    n1_updated = storage.get_node(n1_id)
    assert n1_updated["salience"] == 0.75
    assert n1_updated["confidence"] == 0.85


def test_hybrid_retriever_salience_and_episodes(tmp_path: Path) -> None:
    """Validate hybrid retrieval combining semantic nodes and episodic memories with salience weighting."""
    db_file = str(tmp_path / "synapse_retriever_cog.db")
    storage = SynapseStorage(db_file)
    retriever = SynapseRetriever(storage=storage)

    # Add Declarative Node
    storage.add_node(
        uri="synapse://architecture/postgres_config",
        l0_abstract="PostgreSQL database cluster configuration and replication settings.",
        l1_overview="Primary-replica configuration using streaming replication with slots.",
        salience=0.6,
        confidence=1.0,
    )

    # Add Low-confidence Node
    storage.add_node(
        uri="synapse://architecture/unverified_port",
        l0_abstract="PostgreSQL replication port might be 5433.",
        l1_overview="Unconfirmed port change during maintenance.",
        salience=0.5,
        confidence=0.4,
    )

    # Add Master Directive (High Salience)
    storage.add_node(
        uri="synapse://decision/postgres_master_rule",
        l0_abstract="Master Rule: Always use pg_isready before initiating PostgreSQL failover.",
        l1_overview="Mandatory pre-failover health verification step.",
        salience=0.95,
        confidence=1.0,
    )

    # Add Episode
    storage.add_episode(
        title="PostgreSQL Failover Split Brain",
        situation="Automatic failover triggered during network partition causing dual primaries.",
        action_taken="Used pg_rewind to reconcile old primary with new timeline.",
        outcome="Split brain resolved without data divergence.",
        lessons_learned="Always verify quorum with pg_isready and consensus before promoting replica.",
        salience=0.92,
        confidence=1.0,
        tags=["postgres", "failover", "split-brain"],
    )

    results = retriever.retrieve("PostgreSQL failover and replication", limit=5)
    assert len(results) >= 2

    # High salience items should be prioritized at the top
    top_salience = results[0].get("salience", 0.0)
    assert top_salience >= 0.9

    # Verify low confidence node receives verification flag
    low_conf_items = [r for r in results if r.get("confidence", 1.0) < 0.6]
    for item in low_conf_items:
        assert item.get("verification_needed") is True
        assert item.get("verification_flag") is True

    # Verify episodic memories are returned with proper types
    ep_items = [r for r in results if r.get("type") == "episodic"]
    assert len(ep_items) >= 1
    assert "Situation: Automatic failover" in ep_items[0]["l1_overview"]


def test_contradiction_detector() -> None:
    """Validate metacognition contradiction detection between opposing memories."""
    nodes = [
        {
            "id": "node_dark",
            "uri": "synapse://preference/editor_theme",
            "category": "preference",
            "l0_abstract": "User prefers dark mode in UI and editor.",
            "l1_overview": "Dark theme reduces eye strain.",
            "salience": 0.8,
            "confidence": 0.95,
        },
        {
            "id": "node_light",
            "uri": "synapse://preference/editor_theme_old",
            "category": "preference",
            "l0_abstract": "User prefers light mode during daytime.",
            "l1_overview": "Light theme requested in previous session.",
            "salience": 0.4,
            "confidence": 0.7,
        },
    ]

    conflicts = detect_contradictions(nodes)
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict["node_a_id"] == "node_dark"
    assert conflict["node_b_id"] == "node_light"
    assert conflict["winner_id"] == "node_dark"  # Higher salience (0.8 > 0.4)
    assert conflict["salience_diff"] == 0.4


def test_decay_immunity_for_high_salience(tmp_path: Path) -> None:
    """Validate that nodes with salience >= 0.9 NEVER decay and are immune to pruning."""
    db_file = str(tmp_path / "synapse_immunity.db")
    storage = SynapseStorage(db_file)
    worker = SynapseWorker(storage)

    now = int(time.time())
    past = now - (90 * 86400)  # 90 days ago

    # 1. Master Directive (salience = 0.95) -> Immortal
    id_master = storage.add_node(
        uri="synapse://decision/master_backup_rule",
        l0_abstract="Critical Master Directive: Backups must run daily at midnight UTC.",
        salience=0.95,
        importance=0.9,
        strength=1.0,
    )
    storage._conn.execute("UPDATE synapse_nodes SET last_accessed = ? WHERE id = ?", (past, id_master))
    storage._conn.commit()

    # 2. Normal ephemeral node (salience = 0.5, importance = 0.3) -> Decays & Archives
    id_temp = storage.add_node(
        uri="synapse://general/temp_cache_info",
        l0_abstract="Temporary cache cleared on server restart.",
        salience=0.5,
        importance=0.3,
        strength=0.8,
    )
    storage._conn.execute("UPDATE synapse_nodes SET last_accessed = ? WHERE id = ?", (past, id_temp))
    storage._conn.commit()

    stats = worker.decay_cycle(decay_rate=0.1, archive_threshold=0.2)
    assert stats["processed"] == 2
    assert stats["archived"] == 1
    assert stats["protected"] >= 1
    assert stats["immune"] >= 1

    # Master directive node should retain strength=1.0 and active status
    master_node = storage.get_node(id_master)
    assert master_node["status"] == "active"
    assert master_node["strength"] == 1.0

    # Temp node should be archived with reduced strength
    temp_node = storage.get_node(id_temp)
    assert temp_node["status"] == "archived"
    assert temp_node["strength"] < 0.2


def test_distiller_dual_semantic_and_episodic(tmp_path: Path) -> None:
    """Validate session distiller extracts both atomic semantic nodes and structured episodes."""
    db_file = str(tmp_path / "synapse_distiller.db")
    storage = SynapseStorage(db_file)
    worker = SynapseWorker(storage)

    messages = [
        {"role": "user", "content": "We had an incident where the SQLite database locked during concurrent writes."},
        {"role": "assistant", "content": "Fixed by enabling WAL mode via PRAGMA journal_mode=WAL. Always enable WAL mode for multi-threaded SQLite applications."},
        {"role": "user", "content": "We also decided we will use FastAPI for our API backend services and prefer dark mode for the admin UI."},
    ]

    res = worker.distill_session(messages, use_llm=False)
    assert res["status"] == "success"
    assert res["distilled_count"] >= 2
    assert len(res["nodes_created"]) >= 1
    assert len(res["episodes_created"]) >= 1

    # Check extracted episode
    episodes = storage.list_episodes()
    assert len(episodes) >= 1
    ep = episodes[0]
    assert "SQLite" in ep["situation"] or "locked" in ep["situation"]
    assert "WAL" in ep["action_taken"] or "WAL" in ep["lessons_learned"]
    assert ep["salience"] >= 0.8


def test_loader_and_session_episodic_packing() -> None:
    """Validate L0/L1 episodic context packing adheres to strict token budget < 250 tokens."""
    loader = SynapseContextLoader()
    session = SessionManager(session_id="test_cog_session")

    episodes = [
        {
            "id": "ep_1",
            "title": "PostgreSQL Recovery",
            "situation": "Primary node failed after kernel OOM panic.",
            "action_taken": "Promoted standby replica using pg_ctl promote and updated DNS.",
            "outcome": "Service restored in under 45 seconds.",
            "lessons_learned": "Set vm.overcommit_memory=2 to prevent Linux kernel OOM killing postgres processes.",
            "salience": 0.95,
            "confidence": 1.0,
        },
        {
            "id": "ep_2",
            "title": "Redis Eviction Tuning",
            "situation": "High memory consumption triggered evictions.",
            "action_taken": "Tuned maxmemory to 16GB.",
            "outcome": "Evictions stopped.",
            "lessons_learned": "Monitor redis memory proactively.",
            "salience": 0.7,
            "confidence": 0.9,
        },
    ]

    # Test L0 formatting
    l0 = loader.format_episode_l0(episodes[0])
    assert "• [EPISODE [CRITICAL]] PostgreSQL Recovery:" in l0
    assert estimate_tokens(l0) <= 40

    # Test L1 formatting
    l1 = loader.format_episode_l1(episodes[0])
    assert "Situation: Primary node failed" in l1
    assert "Action: Promoted standby" in l1

    # Test packing under 250 tokens
    packed = loader.pack_episodic_context(episodes, token_budget=250)
    assert len(packed) > 0
    assert estimate_tokens(packed) <= 250

    # Test SessionManager integration
    session.set_recalled_episodes(episodes, token_budget=250)
    prompt = session.build_system_prompt(tool_names=["read_file", "search_files"])
    assert "## Relevant Episodic Experiences (Cognitive Episodes):" in prompt
    assert "PostgreSQL Recovery" in prompt


def test_adapter_episodic_and_cognitive_methods(tmp_path: Path) -> None:
    """Validate SynapseAdapter episodic retain/recall methods and reflect introspection."""
    db_file = str(tmp_path / "synapse_adapter_cog.db")
    adapter = SynapseAdapter(config={"db_path": db_file})

    # Retain episode
    ep_id = adapter.retain_episode(
        title="Nginx TLS Expiration Outage",
        situation="Let's Encrypt certificate expired without automated renewal cron firing.",
        action_taken="Manually executed certbot renew --force-renewal and reloaded nginx.",
        outcome="HTTPS connections restored.",
        lessons_learned="Always install systemd timer for certbot and alert 14 days before certificate expiry.",
        salience=0.9,
        confidence=1.0,
        tags=["nginx", "tls", "certbot"],
    )
    assert ep_id != ""

    # Recall episode
    recalled_eps = adapter.recall_episodes("nginx certificate expired")
    assert len(recalled_eps) == 1
    assert recalled_eps[0]["title"] == "Nginx TLS Expiration Outage"

    # Get episode
    ep_obj = adapter.get_episode(ep_id)
    assert ep_obj is not None
    assert ep_obj["outcome"] == "HTTPS connections restored."

    # Retain declarative node
    adapter.retain(
        content="We prefer dark mode for web dashboards.",
        metadata={"category": "preference", "salience": 0.8, "confidence": 1.0},
    )

    # Recall hybrid
    hybrid_recalled = adapter.recall("nginx certbot and dark mode", limit=5)
    assert len(hybrid_recalled) >= 1

    # Reflect
    reflection = adapter.reflect("nginx and tls")
    assert "Cognitive Synapse Reflection" in reflection
    assert "Discovered" in reflection

    # Delete episode
    del_ok = adapter.delete_episode(ep_id)
    assert del_ok is True
    assert adapter.get_episode(ep_id) is None
