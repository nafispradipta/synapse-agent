"""Failed writes must not leave the SQLite connection inside a transaction."""

from __future__ import annotations

import sqlite3

import pytest

from synapse.storage import SynapseStorage


def test_duplicate_uri_failure_rolls_back(tmp_path):
    db_file = tmp_path / "rollback.db"
    storage = SynapseStorage(str(db_file))
    try:
        storage.add_node(uri="synapse://dup/1", l0_abstract="first")
        with pytest.raises(sqlite3.IntegrityError):
            storage.add_node(uri="synapse://dup/1", l0_abstract="second")
        assert storage._conn.in_transaction is False
        assert storage.get_stats()["node_count"] == 1
    finally:
        storage.close()

    second = SynapseStorage(str(db_file))
    try:
        assert second.get_stats()["node_count"] == 1
    finally:
        second.close()


def test_foreign_key_failure_rolls_back(tmp_path):
    storage = SynapseStorage(str(tmp_path / "fk.db"))
    try:
        target = storage.add_node(uri="synapse://fk/target", l0_abstract="target")
        assert storage.add_edge("missing-source", target, "related") is False
        assert storage._conn.in_transaction is False
        assert storage.get_stats()["edge_count"] == 0
    finally:
        storage.close()


def test_duplicate_episode_failure_rolls_back(tmp_path):
    storage = SynapseStorage(str(tmp_path / "episode.db"))
    try:
        storage.add_episode(title="one", episode_id="ep-1")
        with pytest.raises(sqlite3.IntegrityError):
            storage.add_episode(title="two", episode_id="ep-1")
        assert storage._conn.in_transaction is False
        assert len(storage.list_episodes()) == 1
    finally:
        storage.close()


def test_concurrent_writer_failure_does_not_poison_rollback(tmp_path):
    """A second thread inside _write must roll back at its own outermost level."""
    import threading
    import time

    db_file = tmp_path / "thread.db"
    storage = SynapseStorage(str(db_file))
    started = threading.Event()
    release = threading.Event()

    def worker() -> None:
        try:
            with storage._write():
                storage._conn.execute(
                    "INSERT INTO synapse_nodes (id, uri, l0_abstract) VALUES (?, ?, ?)",
                    ("slow-1", "synapse://slow/1", "slow"),
                )
                started.set()
                release.wait(5)
                raise RuntimeError("worker failed")
        except RuntimeError:
            pass

    t = threading.Thread(target=worker)
    t.start()
    assert started.wait(5)

    def other() -> None:
        try:
            with storage._write():
                storage._conn.execute(
                    "INSERT INTO synapse_nodes (id, uri, l0_abstract) VALUES (?, ?, ?)",
                    ("other-1", "synapse://other/1", "other"),
                )
                raise RuntimeError("other failed")
        except RuntimeError:
            pass

    t2 = threading.Thread(target=other)
    t2.start()
    time.sleep(0.05)  # let the second thread capture the depth counter
    release.set()
    t.join()
    t2.join()

    # The main thread must still treat its own _write as the outermost level.
    try:
        with storage._write():
            storage._conn.execute(
                "INSERT INTO synapse_nodes (id, uri, l0_abstract) VALUES (?, ?, ?)",
                ("main-1", "synapse://main/1", "main"),
            )
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert storage._conn.in_transaction is False
    assert storage.get_stats()["node_count"] == 0
    storage.close()
