"""
Unit tests for Cognitive Synapse Core Phase 3:
- SynapseWorker (Decay Cycle, Vault Cortex Sync, Hebbian Reinforcement)
- SynapseAdapter (BaseMemoryAdapter implementation, Recall, Retain, Reflect, Fallback)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synapse.adapter import LocalFileAdapter, resolve_memory_adapter
from synapse.adapter import SynapseAdapter
from synapse.loader import estimate_tokens
from synapse.storage import SynapseStorage
from synapse.worker import SynapseWorker


# ============================================================================
# 1. SynapseAdapter Tests
# ============================================================================

def test_synapse_adapter_retain_and_infer_categories(tmp_path: Path) -> None:
    """Test retain method extracts L0/L1, infers categories, and persists nodes."""
    db_file = str(tmp_path / "synapse_test.db")
    adapter = SynapseAdapter({"db_path": db_file})

    # Test preference category inference
    ok1 = adapter.retain(
        content="User prefers dark theme and Fira Code font in the editor.",
        context="Settings configuration",
    )
    assert ok1 is True

    # Test decision category inference
    ok2 = adapter.retain(
        content="We decided to adopt SQLite WAL mode for concurrency.",
        context="Architecture discussion",
    )
    assert ok2 is True

    # Test incident category inference
    ok3 = adapter.retain(
        content="Fixed memory leak bug when socket connection crashed unexpectedly.",
        context="Postmortem 2026-08",
    )
    assert ok3 is True

    # Test explicit category override
    ok4 = adapter.retain(
        content="Always run tests before committing code changes.",
        metadata={"category": "sop", "importance": 0.8},
    )
    assert ok4 is True

    # Verify nodes in storage
    nodes = adapter.storage.list_nodes(limit=10)
    assert len(nodes) == 4

    categories = {n["category"] for n in nodes}
    assert "preference" in categories
    assert "decision" in categories
    assert "incident" in categories
    assert "sop" in categories


def test_synapse_adapter_recall_with_budget(tmp_path: Path) -> None:
    """Test recall respects token budget and limits."""
    db_file = str(tmp_path / "synapse_test.db")
    adapter = SynapseAdapter({"db_path": db_file, "token_budget": 200})

    # Insert items
    adapter.retain(
        content="Python asyncio is used for high performance non-blocking I/O operations.",
        context="Core Engine",
    )
    adapter.retain(
        content="PostgreSQL database connection pool settings are configured with max 20 connections.",
        context="Database Engine",
    )
    adapter.retain(
        content="SQLite WAL mode enables concurrent read access while writing.",
        context="Storage Engine",
    )

    # Recall with matching query
    results = adapter.recall("asyncio operations", limit=2)
    assert len(results) >= 1
    assert any("asyncio" in r.lower() for r in results)

    # Test strict small token budget limit
    tight_results = adapter.recall("database SQLite", limit=3, token_budget=30)
    total_tokens = sum(estimate_tokens(r) for r in tight_results)
    assert total_tokens <= 30


def test_synapse_adapter_reflect(tmp_path: Path) -> None:
    """Test reflect synthesizes associative graph memories and active connections."""
    db_file = str(tmp_path / "synapse_test.db")
    adapter = SynapseAdapter({"db_path": db_file})

    id1 = adapter.storage.add_node(
        uri="synapse://architecture/fastapi_gateway",
        l0_abstract="FastAPI Gateway coordinates incoming agent requests.",
        l1_overview="Serves REST and WebSocket endpoints for client communication.",
        category="architecture",
    )
    id2 = adapter.storage.add_node(
        uri="synapse://architecture/synapse_storage",
        l0_abstract="Synapse SQLite storage engine handles local graph state.",
        l1_overview="WAL mode SQLite database with FTS5 and spreading activation.",
        category="architecture",
    )
    adapter.storage.add_edge(id1, id2, relation="coordinates_with", weight=0.85)

    reflection = adapter.reflect("FastAPI Gateway storage")
    assert "Cognitive Synapse Reflection" in reflection
    assert "FastAPI" in reflection
    assert "Synapse" in reflection
    assert "Active Synaptic Connections" in reflection


def test_synapse_adapter_fallback_on_empty(tmp_path: Path) -> None:
    """Test SynapseAdapter falls back to fallback_adapter when Synapse has no results."""
    db_file = str(tmp_path / "synapse_test.db")
    fallback_file = str(tmp_path / "FALLBACK_MEMORY.md")
    fallback = LocalFileAdapter({"memory_file": str(fallback_file)})
    fallback.retain("Fallback secret knowledge about quantum computing.")

    adapter = SynapseAdapter({
        "db_path": db_file,
        "fallback_adapter": fallback,
    })

    # Query something not in synapse but in fallback
    results = adapter.recall("quantum computing", limit=2)
    assert len(results) == 1
    assert "quantum computing" in results[0]


def test_synapse_adapter_fallback_on_error(tmp_path: Path) -> None:
    """Test SynapseAdapter gracefully falls back when storage encounters an error."""
    fallback_file = str(tmp_path / "FALLBACK_MEMORY.md")
    fallback = LocalFileAdapter({"memory_file": str(fallback_file)})
    fallback.retain("Emergency protocol for system recovery.")

    adapter = SynapseAdapter({
        "db_path": ":memory:",
        "fallback_adapter": fallback,
    })
    # Force storage to None to simulate backend unavailability
    adapter.storage = None
    adapter.retriever = None

    # Test recall fallback
    recalled = adapter.recall("recovery protocol")
    assert len(recalled) == 1
    assert "Emergency protocol" in recalled[0]

    # Test retain fallback
    ok = adapter.retain("New memory while storage offline")
    assert ok is True
    assert "New memory while storage offline" in (tmp_path / "FALLBACK_MEMORY.md").read_text(encoding="utf-8")

    # Test reflect fallback
    refl = adapter.reflect("recovery")
    assert "Reflection not supported" in refl or "Emergency" in refl or "recovery" in refl


def test_resolve_memory_adapter_synapse() -> None:
    """Test resolve_memory_adapter resolves SynapseAdapter."""
    adapter1 = resolve_memory_adapter({"memory_backend": {"type": "synapse"}})
    assert isinstance(adapter1, SynapseAdapter)
    assert adapter1.name == "synapse"

    adapter2 = resolve_memory_adapter({"memory_backend": {"type": "cognitive"}})
    assert isinstance(adapter2, SynapseAdapter)

    old_env = os.environ.get("MEMORY_BACKEND")
    try:
        os.environ["MEMORY_BACKEND"] = "synapse"
        adapter3 = resolve_memory_adapter({})
        assert isinstance(adapter3, SynapseAdapter)

        os.environ["MEMORY_BACKEND"] = "synapse_core"
        adapter4 = resolve_memory_adapter({})
        assert isinstance(adapter4, SynapseAdapter)
    finally:
        if old_env is not None:
            os.environ["MEMORY_BACKEND"] = old_env
        else:
            os.environ.pop("MEMORY_BACKEND", None)


# ============================================================================
# 2. SynapseWorker Tests
# ============================================================================

def test_synapse_worker_decay_cycle(tmp_path: Path) -> None:
    """Test decay cycle degrades strength and archives low-importance decaying nodes."""
    storage = SynapseStorage(str(tmp_path / "synapse_worker.db"))
    worker = SynapseWorker(storage)

    now = int(time.time())
    past_timestamp = now - (60 * 86400)  # 60 days ago

    # Node 1: Normal node from 60 days ago (low importance) -> should decay and archive
    id1 = storage.add_node(
        uri="synapse://general/old_trivia",
        l0_abstract="Old trivia about legacy server version 1.0.",
        importance=0.3,
        strength=0.8,
    )
    # Update last_accessed to 60 days ago
    storage._conn.execute("UPDATE synapse_nodes SET last_accessed = ? WHERE id = ?", (past_timestamp, id1))
    storage._conn.commit()

    # Node 2: High importance node from 60 days ago -> should decay but stay active (protected)
    id2 = storage.add_node(
        uri="synapse://decision/core_principles",
        l0_abstract="Core architecture decisions for Autonomous Agent.",
        importance=0.9,
        strength=0.8,
    )
    storage._conn.execute("UPDATE synapse_nodes SET last_accessed = ? WHERE id = ?", (past_timestamp, id2))
    storage._conn.commit()

    # Node 3: Fresh node accessed now -> should not decay significantly
    id3 = storage.add_node(
        uri="synapse://preference/current_theme",
        l0_abstract="User prefers dark theme today.",
        importance=0.5,
        strength=1.0,
    )

    stats = worker.decay_cycle(decay_rate=0.08, archive_threshold=0.2, importance_threshold=0.8)
    assert stats["processed"] == 3
    assert stats["decayed"] >= 2
    assert stats["archived"] == 1
    assert stats["protected"] == 1

    node1 = storage.get_node(id1)
    assert node1["status"] == "archived"
    assert node1["strength"] < 0.2

    node2 = storage.get_node(id2)
    assert node2["status"] == "active"
    assert node2["strength"] < 0.2

    node3 = storage.get_node(id3)
    assert node3["status"] == "active"
    assert node3["strength"] == 1.0


def test_synapse_worker_hebbian_reinforce(tmp_path: Path) -> None:
    """Test hebbian_reinforce creates and strengthens edges between co-activated nodes."""
    storage = SynapseStorage(str(tmp_path / "synapse_hebbian.db"))
    worker = SynapseWorker(storage)

    id_a = storage.add_node(uri="synapse://sop/step_a", l0_abstract="Step A of workflow.")
    id_b = storage.add_node(uri="synapse://sop/step_b", l0_abstract="Step B of workflow.")
    id_c = storage.add_node(uri="synapse://sop/step_c", l0_abstract="Step C of workflow.")

    # Reinforce single node or empty list should do nothing
    assert worker.hebbian_reinforce([id_a]) == 0

    # Reinforce pair [id_a, id_b]
    count1 = worker.hebbian_reinforce([id_a, id_b], delta=0.1)
    assert count1 == 2  # Bidirectional: A->B and B->A

    edges_a = storage.get_edges(id_a, direction="out")
    assert len(edges_a) == 1
    assert edges_a[0]["target_id"] == id_b
    assert edges_a[0]["weight"] == 0.6  # 0.5 default + 0.1 delta

    # Reinforce triplet [id_a, id_b, id_c]
    count2 = worker.hebbian_reinforce([id_a, id_b, id_c], delta=0.1)
    assert count2 == 6  # 3 pairs * 2 directions

    # Check updated edge weight between id_a and id_b
    edges_a_updated = storage.get_edges(id_a, direction="out")
    edge_ab = next(e for e in edges_a_updated if e["target_id"] == id_b)
    assert round(edge_ab["weight"], 2) == 0.7


def test_synapse_worker_vault_cortex_sync(tmp_path: Path) -> None:
    """Test vault_cortex_sync scans markdown SOP files, extracts L0/L1, and registers nodes."""
    storage = SynapseStorage(str(tmp_path / "synapse_vault.db"))
    worker = SynapseWorker(storage)

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    (vault_dir / "engineering").mkdir()

    # Markdown 1 with Frontmatter
    doc1 = vault_dir / "engineering" / "deploy_guide.md"
    doc1.write_text(
        "---\n"
        "title: Production Deploy Guide\n"
        "summary: Automated Blue-Green deployment pipeline for cloud infrastructure.\n"
        "tags: [ops, cloud]\n"
        "---\n\n"
        "# Deployment Guide\n\n"
        "## Pre-checks\n"
        "- Verify database migration status.\n"
        "- Check health endpoints.\n\n"
        "## Rollout\n"
        "- Deploy container images.\n",
        encoding="utf-8",
    )

    # Markdown 2 with Headings and Body
    doc2 = vault_dir / "incident_response.md"
    doc2.write_text(
        "# Incident Response SOP\n\n"
        "This protocol outlines the immediate actions for Sev-1 outages.\n\n"
        "## Steps\n"
        "- Acknowledge alert in monitoring dashboard within 5 minutes.\n"
        "- Open incident war room channel.\n",
        encoding="utf-8",
    )

    # Initial sync
    stats = worker.vault_cortex_sync(vault_path=str(vault_dir))
    assert stats["scanned"] == 2
    assert stats["created"] == 2
    assert stats["updated"] == 0
    assert stats["errors"] == 0

    nodes = storage.list_nodes(category="sop")
    assert len(nodes) == 2

    node1 = storage.get_node("synapse://sop/engineering/deploy_guide")
    assert node1 is not None
    assert "Blue-Green deployment" in node1["l0_abstract"]
    assert "Pre-checks" in node1["l1_overview"]
    assert "Rollout" in node1["l1_overview"]
    assert node1["l2_target_path"] == str(doc1.resolve())

    # Modify doc2 and re-sync
    doc2.write_text(
        "# Incident Response SOP Updated\n\n"
        "Updated protocol for high priority incidents.\n",
        encoding="utf-8",
    )
    sync2_stats = worker.vault_cortex_sync(vault_path=str(vault_dir))
    assert sync2_stats["scanned"] == 2
    assert sync2_stats["created"] == 0
    assert sync2_stats["updated"] == 2


def test_synapse_worker_vault_cortex_sync_preserves_frontmatter_category(tmp_path: Path) -> None:
    """Test vault_cortex_sync preserves category from frontmatter and uses uniform URI."""
    storage = SynapseStorage(str(tmp_path / "synapse_vault_cat.db"))
    worker = SynapseWorker(storage)

    vault_dir = tmp_path / "vault_cat"
    vault_dir.mkdir()

    # Markdown with category: architecture
    arch_doc = vault_dir / "system_architecture.md"
    arch_doc.write_text(
        "---\n"
        "title: System Architecture Spec\n"
        "category: architecture\n"
        "tags: [architecture, core]\n"
        "---\n\n"
        "# System Architecture\n\n"
        "Core design principles and component hierarchy.\n",
        encoding="utf-8",
    )

    # Markdown with category: preference
    pref_doc = vault_dir / "user_preferences.md"
    pref_doc.write_text(
        "---\n"
        "title: User Preferences\n"
        "category: preference\n"
        "tags: [preference, theme]\n"
        "---\n\n"
        "# Theme Preference\n\n"
        "Nordic dark mode preferred.\n",
        encoding="utf-8",
    )

    stats = worker.vault_cortex_sync(vault_path=str(vault_dir))
    assert stats["scanned"] == 2
    assert stats["created"] == 2

    # Check uniform URI scheme synapse://memory/{category}/{slug}
    arch_node = storage.get_node("synapse://memory/architecture/system_architecture")
    assert arch_node is not None
    assert arch_node["category"] == "architecture"

    pref_node = storage.get_node("synapse://memory/preference/user_preferences")
    assert pref_node is not None
    assert pref_node["category"] == "preference"


def test_synapse_recall_origin_path_and_uri_ref_tags(tmp_path: Path) -> None:
    """Test recall formatting includes [ref: <l2_target_path or uri>] reference source tags."""
    db_file = str(tmp_path / "synapse_ref_tags.db")
    adapter = SynapseAdapter({"db_path": db_file})

    # Node with explicit l2_target_path (e.g. from vault SOP)
    adapter.storage.add_node(
        uri="synapse://sop/backup_procedure",
        l0_abstract="Run nightly PostgreSQL database backups using pg_dump.",
        l1_overview="Backups are stored compressed in S3 bucket.",
        l2_target_path="/home/user/vault/sops/backup.md",
        category="sop",
        importance=0.9,
        strength=1.5,
    )

    # Node without l2_target_path (falls back to URI)
    adapter.storage.add_node(
        uri="synapse://preference/editor_theme",
        l0_abstract="User prefers Nordic Dark syntax highlighting.",
        l1_overview="All development editor themes aligned to dark palette.",
        category="preference",
        importance=0.8,
        strength=1.2,
    )

    results_sop = adapter.recall("PostgreSQL database backups", limit=1)
    assert len(results_sop) == 1
    assert "[ref: /home/user/vault/sops/backup.md]" in results_sop[0]

    results_pref = adapter.recall("Nordic Dark syntax", limit=1)
    assert len(results_pref) == 1
    assert "[ref: synapse://preference/editor_theme]" in results_pref[0]


def test_extract_vault_l1_markdown_table_parsing(tmp_path: Path) -> None:
    """Test _extract_vault_l1 parses markdown tables into structured overview points."""
    storage = SynapseStorage(str(tmp_path / "synapse_table_vault.db"))
    worker = SynapseWorker(storage)

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()

    doc = vault_dir / "disaster_recovery.md"
    doc.write_text(
        "# Disaster Recovery Runbook\n\n"
        "## Critical Services\n"
        "| Service | RTO | Action |\n"
        "| :--- | :--- | :--- |\n"
        "| PostgreSQL | 15m | Failover to standby replica |\n"
        "| Redis Cache | 5m | Restart cluster nodes |\n\n"
        "## Key Directives\n"
        "- Notify incident commander immediately.\n",
        encoding="utf-8",
    )

    stats = worker.vault_cortex_sync(vault_path=str(vault_dir))
    assert stats["scanned"] == 1
    assert stats["created"] == 1

    node = storage.get_node("synapse://sop/disaster_recovery")
    assert node is not None
    assert "Table [Service | RTO | Action]" in node["l1_overview"]
    assert "PostgreSQL -> 15m (Action: Failover to standby replica)" in node["l1_overview"]
    assert "Redis Cache -> 5m (Action: Restart cluster nodes)" in node["l1_overview"]
    assert "Critical Services" in node["l1_overview"]
    assert "Key Directives" in node["l1_overview"]
    assert node["l2_target_path"] == str(doc.resolve())


def test_session_system_prompt_synapse_ref_attribution_guideline(tmp_path: Path) -> None:
    """Test session.build_system_prompt includes 1-line source attribution guideline for synapse/vault."""
    from synapse.session import SessionManager

    manager = SessionManager(session_id="test_prompt_ref", config_dir=tmp_path)
    manager.set_recalled_memories([
        "• [SOP] synapse://sop/backup: Run backups nightly [ref: /vault/sops/backup.md]"
    ])

    prompt = manager.build_system_prompt()
    assert "When answering using facts recalled from Synapse / Vault memory with [ref: <source>], provide a clean 1-line source attribution at the bottom of the answer (e.g. Reference: <source>)." in prompt

