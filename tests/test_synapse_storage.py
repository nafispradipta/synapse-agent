"""
Unit tests for Synapse Cognitive Memory Storage (Phase 1).
"""

import os
import sys
import tempfile
import time
from pathlib import Path
import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synapse.storage import SynapseStorage, VALID_CATEGORIES, VALID_STATUSES


@pytest.fixture
def mem_storage():
    """In-memory SynapseStorage fixture."""
    storage = SynapseStorage(db_path=":memory:")
    yield storage
    storage.close()


@pytest.fixture
def temp_file_storage():
    """Temporary file-backed SynapseStorage fixture."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_path = tf.name

    storage = SynapseStorage(db_path=temp_path)
    yield storage
    storage.close()
    if os.path.exists(temp_path):
        os.remove(temp_path)
    wal_path = temp_path + "-wal"
    if os.path.exists(wal_path):
        os.remove(wal_path)
    shm_path = temp_path + "-shm"
    if os.path.exists(shm_path):
        os.remove(shm_path)


class TestSynapseStorageInit:
    def test_in_memory_initialization(self, mem_storage):
        assert mem_storage.is_memory is True
        stats = mem_storage.get_stats()
        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0
        assert stats["fts_entry_count"] == 0

    def test_file_db_initialization_and_pragmas(self, temp_file_storage):
        assert temp_file_storage.is_memory is False
        cursor = temp_file_storage._conn.cursor()
        
        # Verify WAL journal mode
        journal_mode = cursor.execute("PRAGMA journal_mode;").fetchone()[0]
        assert journal_mode.lower() == "wal"

        # Verify Foreign Keys ON
        fk = cursor.execute("PRAGMA foreign_keys;").fetchone()[0]
        assert fk == 1

        # Verify tables exist
        tables = [
            r[0]
            for r in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        ]
        assert "synapse_nodes" in tables
        assert "synapse_edges" in tables
        assert "synapse_fts" in tables

    def test_context_manager(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            path = tf.name
        try:
            with SynapseStorage(db_path=path) as storage:
                uid = storage.add_node(
                    uri="synapse://pref/test",
                    l0_abstract="Context manager test",
                    category="preference",
                )
                assert storage.get_node(uid) is not None
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestNodeCRUDAndValidation:
    def test_add_and_get_node(self, mem_storage):
        node_id = mem_storage.add_node(
            uri="synapse://pref/dark_mode",
            l0_abstract="Dark mode UI theme preference",
            l1_overview="User explicitly requested dark mode with high contrast.",
            l2_target_path="/config/theme.json",
            category="preference",
            importance=0.85,
            strength=0.9,
            node_id="custom_node_01",
            status="active",
        )
        assert node_id == "custom_node_01"

        # Retrieve by node_id
        node_by_id = mem_storage.get_node("custom_node_01")
        assert node_by_id is not None
        assert node_by_id["uri"] == "synapse://pref/dark_mode"
        assert node_by_id["l0_abstract"] == "Dark mode UI theme preference"
        assert node_by_id["l1_overview"] == "User explicitly requested dark mode with high contrast."
        assert node_by_id["l2_target_path"] == "/config/theme.json"
        assert node_by_id["category"] == "preference"
        assert node_by_id["importance"] == 0.85
        assert node_by_id["strength"] == 0.9
        assert node_by_id["status"] == "active"
        assert isinstance(node_by_id["created_at"], int)
        assert isinstance(node_by_id["last_accessed"], int)

        # Retrieve by uri
        node_by_uri = mem_storage.get_node("synapse://pref/dark_mode")
        assert node_by_uri is not None
        assert node_by_uri["id"] == "custom_node_01"

        # Non-existent node
        assert mem_storage.get_node("non_existent") is None

    def test_touch_on_get_node(self, mem_storage):
        node_id = mem_storage.add_node(
            uri="synapse://pref/touch_test",
            l0_abstract="Touch test abstract",
            category="preference",
        )
        # Manually backdate last_accessed
        mem_storage._conn.execute(
            "UPDATE synapse_nodes SET last_accessed = ? WHERE id = ?",
            (1000, node_id),
        )
        mem_storage._conn.commit()

        node_before = mem_storage.get_node(node_id, touch=False)
        assert node_before["last_accessed"] == 1000

        node_after = mem_storage.get_node(node_id, touch=True)
        assert node_after["last_accessed"] > 1000

    def test_invalid_category_and_status(self, mem_storage):
        with pytest.raises(ValueError, match="Invalid category"):
            mem_storage.add_node(
                uri="synapse://invalid/cat",
                l0_abstract="Bad category",
                category="invalid_category",
            )

        with pytest.raises(ValueError, match="Invalid status"):
            mem_storage.add_node(
                uri="synapse://invalid/status",
                l0_abstract="Bad status",
                status="destroyed",
            )

    def test_update_node(self, mem_storage):
        node_id = mem_storage.add_node(
            uri="synapse://arch/gateway",
            l0_abstract="API Gateway architecture",
            l1_overview="Kong gateway routing",
            category="architecture",
            importance=0.7,
        )

        success = mem_storage.update_node(
            node_id,
            l0_abstract="Updated API Gateway",
            l1_overview="Envoy gateway routing",
            importance=0.95,
            category="decision",
            status="superseded",
        )
        assert success is True

        updated = mem_storage.get_node(node_id)
        assert updated["l0_abstract"] == "Updated API Gateway"
        assert updated["l1_overview"] == "Envoy gateway routing"
        assert updated["importance"] == 0.95
        assert updated["category"] == "decision"
        assert updated["status"] == "superseded"

        # Updating invalid category / status raises ValueError
        with pytest.raises(ValueError):
            mem_storage.update_node(node_id, category="bogus")

        with pytest.raises(ValueError):
            mem_storage.update_node(node_id, status="unknown")

        # Updating non-existent node returns False
        assert mem_storage.update_node("missing_id", importance=0.1) is False

    def test_delete_node(self, mem_storage):
        node_id = mem_storage.add_node(
            uri="synapse://sop/deploy",
            l0_abstract="Deployment SOP",
            category="sop",
        )
        assert mem_storage.get_node(node_id) is not None

        deleted = mem_storage.delete_node(node_id)
        assert deleted is True
        assert mem_storage.get_node(node_id) is None

        # Deleting non-existent returns False
        assert mem_storage.delete_node("missing_id") is False

    def test_list_nodes_filtering(self, mem_storage):
        for cat in VALID_CATEGORIES:
            mem_storage.add_node(
                uri=f"synapse://test/{cat}",
                l0_abstract=f"Abstract for {cat}",
                category=cat,
                status="active" if cat != "incident" else "archived",
            )

        # List active nodes (default status='active')
        active_nodes = mem_storage.list_nodes()
        assert len(active_nodes) == len(VALID_CATEGORIES) - 1

        # List specific category
        pref_nodes = mem_storage.list_nodes(category="preference")
        assert len(pref_nodes) == 1
        assert pref_nodes[0]["category"] == "preference"

        # List archived status
        archived_nodes = mem_storage.list_nodes(status="archived")
        assert len(archived_nodes) == 1
        assert archived_nodes[0]["category"] == "incident"

        # List all statuses
        all_nodes = mem_storage.list_nodes(status=None)
        assert len(all_nodes) == len(VALID_CATEGORIES)


class TestSynapseEdgesAndCascade:
    def test_add_and_get_edges(self, mem_storage):
        n1 = mem_storage.add_node(
            uri="synapse://arch/db",
            l0_abstract="Database Layer",
            category="architecture",
        )
        n2 = mem_storage.add_node(
            uri="synapse://arch/cache",
            l0_abstract="Redis Cache Layer",
            category="architecture",
        )
        n3 = mem_storage.add_node(
            uri="synapse://arch/service",
            l0_abstract="User Microservice",
            category="architecture",
        )

        # Add directed edges
        assert mem_storage.add_edge(n3, n1, relation="reads_writes", weight=0.9) is True
        assert mem_storage.add_edge(n3, n2, relation="caches_via", weight=0.8) is True
        assert mem_storage.add_edge(n2, n1, relation="falls_back_to", weight=0.6) is True

        # Test outgoing edges from n3
        out_n3 = mem_storage.get_edges(n3, direction="out")
        assert len(out_n3) == 2
        assert {e["target_id"] for e in out_n3} == {n1, n2}

        # Test incoming edges to n1
        in_n1 = mem_storage.get_edges(n1, direction="in")
        assert len(in_n1) == 2
        assert {e["source_id"] for e in in_n1} == {n3, n2}

        # Test both directions for n2
        both_n2 = mem_storage.get_edges(n2, direction="both")
        assert len(both_n2) == 2
        sources = {e["source_id"] for e in both_n2}
        targets = {e["target_id"] for e in both_n2}
        assert n3 in sources
        assert n1 in targets

    def test_edge_conflict_update(self, mem_storage):
        n1 = mem_storage.add_node(uri="synapse://node/1", l0_abstract="Node 1")
        n2 = mem_storage.add_node(uri="synapse://node/2", l0_abstract="Node 2")

        mem_storage.add_edge(n1, n2, relation="depends_on", weight=0.4)
        edges_before = mem_storage.get_edges(n1, direction="out")
        assert edges_before[0]["weight"] == 0.4

        # Updating same edge with new weight
        mem_storage.add_edge(n1, n2, relation="depends_on", weight=0.95)
        edges_after = mem_storage.get_edges(n1, direction="out")
        assert len(edges_after) == 1
        assert edges_after[0]["weight"] == 0.95

    def test_foreign_key_violation_returns_false(self, mem_storage):
        n1 = mem_storage.add_node(uri="synapse://node/real", l0_abstract="Real node")
        success = mem_storage.add_edge(n1, "non_existent_target", relation="links_to")
        assert success is False

    def test_invalid_direction_raises_error(self, mem_storage):
        n1 = mem_storage.add_node(uri="synapse://node/dtest", l0_abstract="Test node")
        with pytest.raises(ValueError, match="Invalid direction"):
            mem_storage.get_edges(n1, direction="sideways")

    def test_cascade_delete_removes_edges(self, mem_storage):
        n1 = mem_storage.add_node(uri="synapse://node/c1", l0_abstract="Node C1")
        n2 = mem_storage.add_node(uri="synapse://node/c2", l0_abstract="Node C2")
        n3 = mem_storage.add_node(uri="synapse://node/c3", l0_abstract="Node C3")

        mem_storage.add_edge(n1, n2, relation="rel1")
        mem_storage.add_edge(n2, n3, relation="rel2")
        mem_storage.add_edge(n3, n1, relation="rel3")

        assert mem_storage.get_stats()["edge_count"] == 3

        # Deleting n2 should cascade and remove (n1 -> n2) and (n2 -> n3)
        mem_storage.delete_node(n2)
        assert mem_storage.get_stats()["edge_count"] == 1
        remaining_edges = mem_storage.get_edges(n3, direction="both")
        assert len(remaining_edges) == 1
        assert remaining_edges[0]["source_id"] == n3
        assert remaining_edges[0]["target_id"] == n1


class TestFTS5SyncAndSearch:
    def test_fts5_triggers_sync(self, mem_storage):
        # Insert
        n1 = mem_storage.add_node(
            uri="synapse://pref/editor",
            l0_abstract="VSCode keybindings with Vim mode",
            l1_overview="Prefers standard modal editing with custom escape leader key.",
            category="preference",
        )
        assert mem_storage.get_stats()["fts_entry_count"] == 1

        # Search match
        results = mem_storage.search_fts("vim")
        assert len(results) == 1
        assert results[0]["id"] == n1
        assert "rank" in results[0]

        # Update node text
        mem_storage.update_node(
            n1,
            l0_abstract="Emacs keybindings with Evil mode",
            l1_overview="Switched from Vim to Emacs evil keybindings.",
        )
        assert mem_storage.get_stats()["fts_entry_count"] == 1

        # Search old term should return 0 results (or lower/empty)
        assert len(mem_storage.search_fts("vscode")) == 0
        # Search new term should return 1 result
        emacs_res = mem_storage.search_fts("emacs")
        assert len(emacs_res) == 1
        assert emacs_res[0]["id"] == n1

        # Delete node removes FTS entry
        mem_storage.delete_node(n1)
        assert mem_storage.get_stats()["fts_entry_count"] == 0
        assert len(mem_storage.search_fts("emacs")) == 0

    def test_fts5_search_ranking(self, mem_storage):
        n1 = mem_storage.add_node(
            uri="synapse://doc/1",
            l0_abstract="Authentication OAuth2 token refresh policy",
            l1_overview="OAuth2 tokens refresh automatically every 15 minutes.",
        )
        n2 = mem_storage.add_node(
            uri="synapse://doc/2",
            l0_abstract="Database connection pool configuration",
            l1_overview="Postgres database pool size max 20 connections.",
        )
        n3 = mem_storage.add_node(
            uri="synapse://doc/3",
            l0_abstract="OAuth2 multi-tenant authentication provider setup",
            l1_overview="Authentication server implementing full OAuth2 standard with PKCE.",
        )

        results = mem_storage.search_fts("OAuth2 authentication")
        assert len(results) >= 2
        # Verify returned nodes contain expected IDs
        result_ids = [r["id"] for r in results]
        assert n1 in result_ids
        assert n3 in result_ids
        assert n2 not in result_ids
        # Verify rank is present and sorted
        ranks = [r["rank"] for r in results]
        assert ranks == sorted(ranks)

    def test_fts5_search_special_characters_and_edge_cases(self, mem_storage):
        mem_storage.add_node(
            uri="synapse://incident/01",
            l0_abstract="C++ memory leak in lib-core/v2.1",
            l1_overview="Incident report: null pointer exception at line 42 (std::vector).",
            category="incident",
        )

        # Empty / whitespace
        assert mem_storage.search_fts("") == []
        assert mem_storage.search_fts("   ") == []

        # Queries with punctuation and special chars should not crash FTS5
        res1 = mem_storage.search_fts("C++")
        assert len(res1) == 1

        res2 = mem_storage.search_fts("lib-core/v2.1: memory leak")
        assert len(res2) == 1

        res3 = mem_storage.search_fts('"null pointer" exception (std::vector)?')
        assert len(res3) == 1

    def test_get_stats(self, mem_storage):
        mem_storage.add_node(uri="synapse://p1", l0_abstract="P1", category="preference")
        mem_storage.add_node(uri="synapse://p2", l0_abstract="P2", category="preference")
        d1 = mem_storage.add_node(uri="synapse://d1", l0_abstract="D1", category="decision")
        
        mem_storage.add_edge(d1, mem_storage.get_node("synapse://p1")["id"], relation="based_on")

        stats = mem_storage.get_stats()
        assert stats["node_count"] == 3
        assert stats["edge_count"] == 1
        assert stats["fts_entry_count"] == 3
        assert stats["categories"]["preference"] == 2
        assert stats["categories"]["decision"] == 1
        assert stats["statuses"]["active"] == 3


class TestSynapsePerformance:
    def test_sub_millisecond_operations(self, mem_storage):
        # 1. Benchmark Node Inserts
        start = time.perf_counter()
        node_ids = []
        for i in range(100):
            nid = mem_storage.add_node(
                uri=f"synapse://perf/node_{i}",
                l0_abstract=f"Performance test node abstract {i} with searchable keywords",
                l1_overview=f"Detailed overview explaining component architecture {i}",
                category="architecture" if i % 2 == 0 else "general",
                importance=0.5 + (i % 50) / 100.0,
            )
            node_ids.append(nid)
        total_insert_time = time.perf_counter() - start
        avg_insert_ms = (total_insert_time / 100) * 1000
        # Average insert ms should be well under 1 ms in memory / fast file storage
        assert avg_insert_ms < 1.0, f"Avg insert time {avg_insert_ms:.3f} ms exceeds 1ms target"

        # 2. Benchmark Point Lookups
        start = time.perf_counter()
        for nid in node_ids:
            node = mem_storage.get_node(nid)
            assert node is not None
        total_get_time = time.perf_counter() - start
        avg_get_ms = (total_get_time / 100) * 1000
        assert avg_get_ms < 0.2, f"Avg get time {avg_get_ms:.3f} ms exceeds 0.2ms target"

        # 3. Benchmark Edge Additions
        start = time.perf_counter()
        for i in range(len(node_ids) - 1):
            mem_storage.add_edge(
                node_ids[i],
                node_ids[i + 1],
                relation="connects_to",
                weight=0.75,
            )
        total_edge_time = time.perf_counter() - start
        avg_edge_ms = (total_edge_time / 99) * 1000
        assert avg_edge_ms < 1.0, f"Avg edge insert time {avg_edge_ms:.3f} ms exceeds 1ms target"

        # 4. Benchmark FTS Queries
        start = time.perf_counter()
        for i in range(50):
            res = mem_storage.search_fts(f"architecture {i}", limit=5)
            assert len(res) >= 0
        total_fts_time = time.perf_counter() - start
        avg_fts_ms = (total_fts_time / 50) * 1000
        assert avg_fts_ms < 2.0, f"Avg FTS search time {avg_fts_ms:.3f} ms exceeds 2.0ms target"
