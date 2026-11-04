"""Decay/prune must not crash on NULL edge weights."""

from __future__ import annotations

from synapse.storage import SynapseStorage


def test_null_weight_edge_is_decayed_and_pruned() -> None:
    storage = SynapseStorage(":memory:")
    try:
        source = storage.add_node(uri="synapse://null/a", l0_abstract="a")
        target = storage.add_node(uri="synapse://null/b", l0_abstract="b")
        storage.add_edge(source, target, relation="co_activated", weight=0.5)
        storage._conn.execute(
            "UPDATE synapse_edges SET weight = NULL, last_fired = 1"
        )
        storage._conn.commit()

        stats = storage.prune_decayed_edges(idle_days_threshold=0)
        assert isinstance(stats, dict)
        assert storage.get_edges(source) == []
    finally:
        storage.close()
