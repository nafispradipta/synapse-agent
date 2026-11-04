"""Vectors must follow the text that produced them."""

from __future__ import annotations

from synapse.storage import SynapseStorage


def test_update_node_text_change_invalidates_vector() -> None:
    storage = SynapseStorage(":memory:")
    try:
        nid = storage.add_node(
            uri="synapse://prov/1",
            l0_abstract="sqlite wal mode concurrency guidance",
            category="architecture",
            salience=0.5,
        )
        before = storage.get_vector(nid)
        assert before is not None

        storage.update_node(nid, l0_abstract="completely unrelated banana recipe text")
        after = storage.get_vector(nid)
        assert after is None or after != before
    finally:
        storage.close()


def test_upsert_vector_records_model_and_dim() -> None:
    storage = SynapseStorage(":memory:")
    try:
        nid = storage.add_node(uri="synapse://prov/2", l0_abstract="provenance check")
        assert storage.upsert_vector(nid, [0.1] * 384, model="test-model-x") is True
        row = storage._conn.execute(
            "SELECT dim, model FROM synapse_vectors WHERE id = ?", (nid,)
        ).fetchone()
        assert row[0] == 384
        assert row[1] == "test-model-x"
    finally:
        storage.close()
