"""
Unit and integration tests for Hebbian Reinforcement & Abstract Rollup.
"""

import pytest
import sqlite3
import time
from pathlib import Path
from synapse.storage import SynapseStorage
from synapse.worker import SynapseWorker
from synapse.retriever import SynapseRetriever


def test_hebbian_node_access_reinforcement():
    """Test that accessing/recalling a node increments access_count, reinforces salience and strength."""
    storage = SynapseStorage(":memory:")
    node_id = storage.add_node(
        uri="synapse://test/hebbian_node",
        l0_abstract="Hebbian test abstract",
        category="architecture",
        importance=0.6,
        strength=1.0,
        salience=0.5,
    )

    node = storage.get_node(node_id, touch=False)
    assert node is not None
    assert node.get("access_count", 0) == 0

    # Record node access
    success = storage.record_node_access(node_id, delta_salience=0.05, delta_strength=0.1)
    assert success is True

    node_after = storage.get_node(node_id, touch=False)
    assert node_after is not None
    assert node_after["access_count"] == 1
    assert round(node_after["salience"], 2) == 0.55
    assert round(node_after["strength"], 2) == 1.10

    # Second access
    storage.record_node_access(node_id, delta_salience=0.05, delta_strength=0.1)
    node_second = storage.get_node(node_id, touch=False)
    assert node_second is not None
    assert node_second["access_count"] == 2
    assert round(node_second["salience"], 2) == 0.60
    assert round(node_second["strength"], 2) == 1.20


def test_hebbian_edge_reinforcement_on_access():
    """Test that connected edges are reinforced and edge access_count is incremented."""
    storage = SynapseStorage(":memory:")
    n1 = storage.add_node(uri="synapse://test/n1", l0_abstract="Node 1", category="general")
    n2 = storage.add_node(uri="synapse://test/n2", l0_abstract="Node 2", category="general")

    storage.add_edge(n1, n2, relation="relates_to", weight=0.5)

    # Initial edge check
    edges = storage.get_edges(n1, direction="out")
    assert len(edges) == 1
    assert edges[0]["weight"] == 0.5
    assert edges[0].get("access_count", 0) == 0

    # Access n1 with edge bumping enabled
    storage.record_node_access(n1, bump_edges=True, delta_edge=0.08)

    edges_after = storage.get_edges(n1, direction="out")
    assert round(edges_after[0]["weight"], 2) == 0.58
    assert edges_after[0]["access_count"] == 1


def test_abstract_rollup_superseded_nodes(tmp_path: Path):
    """Test that rollup_superseded_nodes archives superseded nodes and prunes them from active tables."""
    db_file = tmp_path / "test_rollup.db"
    storage = SynapseStorage(str(db_file))
    worker = SynapseWorker(storage)

    # Create active nodes
    act1 = storage.add_node(uri="synapse://active/1", l0_abstract="Active node 1", category="architecture", status="active")
    act2 = storage.add_node(uri="synapse://active/2", l0_abstract="Active node 2", category="sop", status="active")

    # Create superseded nodes across domains
    sup1 = storage.add_node(uri="synapse://sup/1", l0_abstract="Legacy gateway port 8080", category="architecture", status="superseded")
    sup2 = storage.add_node(uri="synapse://sup/2", l0_abstract="Old dark mode CSS override", category="preference", status="superseded")
    sup3 = storage.add_node(uri="synapse://sup/3", l0_abstract="Deprecated backup script cron", category="sop", status="superseded")

    # Add edges to superseded nodes
    storage.add_edge(act1, sup1, relation="mitigates", weight=0.6)
    storage.add_edge(sup1, sup2, relation="relates_to", weight=0.4)

    assert len(storage.list_nodes(status="superseded")) == 3
    assert len(storage.list_nodes(status="active")) == 2

    # Execute rollup
    result = worker.rollup_superseded_nodes(
        min_count=1,
        backup_snapshot=True,
        snapshot_dir=str(tmp_path),
    )

    assert result["status"] == "success"
    assert result["rolled_up_count"] == 3
    assert "master_entity_id" in result

    # Verify no superseded nodes remain in synapse_nodes
    remaining_sup = storage.list_nodes(status="superseded")
    assert len(remaining_sup) == 0

    # Verify active nodes are untouched
    remaining_act = storage.list_nodes(status="active")
    assert len(remaining_act) == 2

    # Verify master archive rollup entity exists
    master_node = storage.get_node(result["master_entity_id"])
    assert master_node is not None
    assert master_node["status"] == "archived"
    assert "Consolidated Archive Rollup" in master_node["l0_abstract"]

    # Verify superseded records exist in synapse_superseded_archive
    cur = storage._conn.cursor()
    cur.execute("SELECT COUNT(*) FROM synapse_superseded_archive")
    assert cur.fetchone()[0] == 3

    # Verify edges to superseded nodes were pruned
    cur.execute("SELECT COUNT(*) FROM synapse_edges WHERE source_id = ? OR target_id = ?", (sup1, sup1))
    assert cur.fetchone()[0] == 0

    # Verify PRAGMA integrity_check
    cur.execute("PRAGMA integrity_check")
    assert cur.fetchone()[0] == "ok"
