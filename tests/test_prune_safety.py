"""Prune must never delete a superseded row before the rollup archives it."""

from __future__ import annotations

from synapse.storage import SynapseStorage


def test_prune_leaves_superseded_rows_for_the_archive():
    storage = SynapseStorage(":memory:")
    try:
        superseded = storage.add_node(
            uri="synapse://prune/old",
            l0_abstract="Old low-salience fact",
            category="general",
            salience=0.2,
        )
        storage.supersede_node(superseded)
        assert len(storage.list_superseded_nodes()) == 1

        assert storage.prune_exhausted_nodes() == 0
        assert len(storage.list_superseded_nodes()) == 1
        assert storage._conn.execute(
            "SELECT COUNT(*) FROM synapse_superseded_archive"
        ).fetchone()[0] == 0
    finally:
        storage.close()


def test_prune_still_deletes_exhausted_active_rows():
    storage = SynapseStorage(":memory:")
    try:
        exhausted = storage.add_node(
            uri="synapse://prune/active",
            l0_abstract="Exhausted active fact",
            category="general",
            salience=0.2,
        )
        storage.update_node(exhausted, strength=0.05)
        assert storage.prune_exhausted_nodes() == 1
        assert storage.get_node(exhausted) is None
    finally:
        storage.close()
