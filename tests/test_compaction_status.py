"""Compaction status must reflect per-step failure."""

from __future__ import annotations

import sqlite3

from synapse.stores.maintenance import compaction_status


def test_ok_when_all_steps_ok() -> None:
    assert compaction_status({"wal_checkpoint": "done", "optimize": "ok", "vacuum": "ok"}) == "ok"


def test_partial_on_error() -> None:
    assert compaction_status({"wal_checkpoint": "error: boom", "optimize": "ok", "vacuum": "ok"}) == "partial"


def test_partial_on_exception() -> None:
    assert compaction_status({"wal_checkpoint": "none", "optimize": "error: no", "vacuum": "error: no"}) == "partial"
