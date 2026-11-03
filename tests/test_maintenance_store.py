"""
Storage-level contract for the nightly maintenance operations.

These are the operations ``scripts/nightly_consolidation.py`` used to execute
as raw SQL against its own connection. Moving them behind the storage layer is
only safe if their effects are covered here, including the decay rule that has
to cope with both integer and ISO-text ``last_accessed`` columns.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synapse.storage import SynapseStorage


@pytest.fixture
def storage():
    s = SynapseStorage(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def file_storage(tmp_path):
    s = SynapseStorage(db_path=str(tmp_path / "maintenance.db"))
    yield s
    s.close()


STALE_TS = 1_600_000_000
STALE_ISO = datetime.fromtimestamp(STALE_TS, tz=timezone.utc).isoformat()
FRESH_TS = int(datetime.now(tz=timezone.utc).timestamp())


def _set_last_accessed(storage, node_id, value):
    storage._conn.execute("UPDATE synapse_nodes SET last_accessed = ? WHERE id = ?", (value, node_id))
    storage._conn.commit()


def test_lock_critical_nodes_hard_locks_high_salience_and_reserved_categories(storage):
    high = storage.add_node(uri="synapse://m/a", l0_abstract="important", category="general")
    pref = storage.add_node(uri="synapse://m/b", l0_abstract="prefs", category="preference")
    decision = storage.add_node(uri="synapse://m/c", l0_abstract="decide", category="decision")
    mundane = storage.add_node(uri="synapse://m/d", l0_abstract="mundane", category="general")

    for n in (high, pref, decision, mundane):
        storage.update_node(n, strength=0.1, confidence=0.2)
    storage.update_node(high, salience=0.95)

    locked = storage.lock_critical_nodes()

    assert locked >= 3
    assert storage.get_node(high)["strength"] == 1.0
    assert storage.get_node(high)["confidence"] == 1.0
    assert storage.get_node(pref)["strength"] == 1.0
    assert storage.get_node(decision)["confidence"] == 1.0
    assert storage.get_node(mundane)["strength"] == 0.1


def test_lock_critical_nodes_returns_zero_when_nothing_matches(storage):
    storage.add_node(uri="synapse://m/quiet", l0_abstract="mundane", category="general")
    assert storage.lock_critical_nodes() == 0


def test_apply_hebbian_decay_weakens_dormant_integer_timestamped_nodes(storage):
    n = storage.add_node(uri="synapse://m/decay", l0_abstract="dormant",
                         category="general", salience=0.5)
    _set_last_accessed(storage, n, STALE_TS)

    assert storage.apply_hebbian_decay(idle_days=14) == 1
    assert storage.get_node(n)["strength"] == pytest.approx(0.95)


def test_apply_hebbian_decay_weakens_dormant_iso_timestamped_nodes(storage):
    n = storage.add_node(uri="synapse://m/decay-iso", l0_abstract="dormant iso",
                         category="general", salience=0.5)
    _set_last_accessed(storage, n, STALE_ISO)

    assert storage.apply_hebbian_decay(idle_days=14) == 1
    assert storage.get_node(n)["strength"] == pytest.approx(0.95)


def test_apply_hebbian_decay_leaves_critical_and_fresh_nodes_alone(storage):
    fresh = storage.add_node(uri="synapse://m/fresh", l0_abstract="fresh",
                             category="general", salience=0.5)
    critical = storage.add_node(uri="synapse://m/critical", l0_abstract="critical",
                                category="general", salience=0.85)
    reserved = storage.add_node(uri="synapse://m/reserved", l0_abstract="reserved",
                                category="architecture", salience=0.4)
    _set_last_accessed(storage, fresh, FRESH_TS)
    _set_last_accessed(storage, critical, STALE_TS)
    _set_last_accessed(storage, reserved, STALE_TS)

    storage.apply_hebbian_decay(idle_days=14)

    assert storage.get_node(fresh)["strength"] == pytest.approx(1.0)
    assert storage.get_node(critical)["strength"] == pytest.approx(1.0)
    assert storage.get_node(reserved)["strength"] == pytest.approx(1.0)


def test_apply_hebbian_decay_stops_at_the_floor(storage):
    n = storage.add_node(uri="synapse://m/floor", l0_abstract="floor",
                         category="general", salience=0.5)
    storage.update_node(n, strength=0.05)
    _set_last_accessed(storage, n, STALE_TS)

    storage.apply_hebbian_decay(idle_days=14)

    assert storage.get_node(n)["strength"] == pytest.approx(0.05)


def test_prune_exhausted_nodes_deletes_only_dead_nodes(storage):
    dead = storage.add_node(uri="synapse://m/dead", l0_abstract="dead",
                             category="general")
    keep_strength = storage.add_node(uri="synapse://m/keep1", l0_abstract="keep1",
                                      category="general")
    reserved = storage.add_node(uri="synapse://m/keep2", l0_abstract="keep2",
                                category="preference")
    storage.update_node(dead, salience=0.1, strength=0.05)
    storage.update_node(keep_strength, salience=0.1, strength=0.5)
    storage.update_node(reserved, salience=0.1, strength=0.05)

    assert storage.prune_exhausted_nodes() == 1
    assert storage.get_node(dead) is None
    assert storage.get_node(keep_strength) is not None
    assert storage.get_node(reserved) is not None


def test_prune_weak_edges_removes_only_edges_below_the_weight_floor(storage):
    strong_a = storage.add_node(uri="synapse://m/sa", l0_abstract="sa", salience=0.5)
    strong_b = storage.add_node(uri="synapse://m/sb", l0_abstract="sb", salience=0.5)
    weak = storage.add_edge(strong_a, strong_b, relation="related", weight=0.1)
    strong = storage.add_edge(strong_a, strong_b, relation="supports", weight=0.6)

    assert storage.prune_weak_edges(min_weight=0.15) == 1
    remaining = {row["relation"] for row in storage.get_edges(strong_a)}
    assert "supports" in remaining
    assert "related" not in remaining


def test_prune_weak_edges_does_not_delete_edges_anchored_to_high_salience_nodes(storage):
    hub = storage.add_node(uri="synapse://m/hub", l0_abstract="hub", salience=0.95)
    quiet = storage.add_node(uri="synapse://m/quiet", l0_abstract="quiet", salience=0.5)
    storage.add_edge(hub, quiet, relation="anchored", weight=0.1)

    assert storage.prune_weak_edges(min_weight=0.15) == 0
    assert [row["relation"] for row in storage.get_edges(quiet)] == ["anchored"]


def test_upsert_vault_node_inserts_then_refreshes_without_resetting_vitality(storage):
    inserted = storage.upsert_vault_node(
        node_id="vault:alpha",
        uri="vault://alpha.md",
        l0_abstract="Alpha title",
        l1_overview="v1 summary",
        l2_target_path="/vault/alpha.md",
        category="general",
        created_at=STALE_TS,
    )
    assert inserted == "vault:alpha"

    storage.update_node(inserted, salience=0.8)
    storage.record_node_access(inserted)

    before = storage.get_node(inserted)

    refreshed = storage.upsert_vault_node(
        node_id="vault:alpha-v2",
        uri="vault://alpha.md",
        l0_abstract="Alpha title v2",
        l1_overview="v2 summary",
        l2_target_path="/vault/alpha.md",
        category="general",
    )

    assert refreshed == inserted
    node = storage.get_node(refreshed)
    assert node["l0_abstract"] == "Alpha title v2"
    assert node["l1_overview"] == "v2 summary"
    assert node["salience"] == pytest.approx(before["salience"])
    assert node["created_at"] == before["created_at"]
    assert node["last_accessed"] == before["last_accessed"]
    assert node["access_count"] == before["access_count"]


def test_upsert_vault_node_reactivates_superseded_nodes(storage):
    inserted = storage.upsert_vault_node(
        node_id="vault:beta",
        uri="vault://beta.md",
        l0_abstract="Beta",
        l1_overview="old",
        l2_target_path="/vault/beta.md",
        category="general",
    )
    storage.supersede_node(inserted)

    storage.upsert_vault_node(
        node_id="vault:beta",
        uri="vault://beta.md",
        l0_abstract="Beta",
        l1_overview="new",
        l2_target_path="/vault/beta.md",
        category="general",
    )

    assert storage.get_node(inserted)["status"] == "active"


def test_optimize_fts_indices_reports_each_index(storage):
    storage.add_node(uri="synapse://m/fts", l0_abstract="searchable node")
    assert storage.optimize_fts_indices() == {"synapse_fts": True, "episodes_fts": True}


def test_verify_integrity_returns_ok(storage):
    assert storage.verify_integrity() == "ok"


def test_wal_checkpoint_runs_on_a_file_database(file_storage):
    result = file_storage.wal_checkpoint()
    assert result["busy"] >= 0
    assert result["checkpointed"] >= 0
