"""
Unit tests for Synapse Cognitive Memory Core - Cross-Platform Universal Lifecycle Hook
& In-Process Timestamp Guard (Bab 6: Protokol Cross-Platform & Universal In-Process Lifecycle).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synapse.session import SessionManager
from synapse.adapter import SynapseAdapter
from synapse.storage import SynapseStorage
from synapse.worker import SynapseWorker


# ============================================================================
# 1. In-Process Timestamp Guard Tests
# ============================================================================

def test_first_run_missing_sync_file_triggers_maintenance(tmp_path: Path) -> None:
    """
    Test that when sync state file does not exist (first run):
    - Maintenance is triggered (returns True).
    - Sync state file is created with valid timestamp and metadata.
    """
    db_file = tmp_path / "synapse_test.db"
    sync_state_file = tmp_path / ".synapse" / "synapse_last_sync.json"
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir(parents=True, exist_ok=True)

    # Create sample SOP file in vault
    sop_file = vault_dir / "deployment_guide.md"
    sop_file.write_text("# Deployment Guide\n\nAlways verify environment variables before deploy.", encoding="utf-8")

    storage = SynapseStorage(str(db_file))
    assert not sync_state_file.exists()

    triggered = SynapseWorker.run_in_process_maintenance_if_needed(
        vault_path=str(vault_dir),
        interval_hours=12.0,
        non_blocking=False,
        sync_state_path=sync_state_file,
        storage=storage,
    )

    assert triggered is True
    assert sync_state_file.exists()

    data = json.loads(sync_state_file.read_text(encoding="utf-8"))
    assert "last_sync" in data
    assert "timestamp" in data
    assert "last_sync_iso" in data
    assert "vault_path" in data
    assert isinstance(data["last_sync"], (int, float))
    assert abs(data["last_sync"] - time.time()) < 10.0

    # Verify vault node was registered in storage
    nodes = storage.list_nodes(category="sop")
    assert len(nodes) == 1
    assert "deployment_guide" in nodes[0]["uri"]
    storage.close()


def test_subsequent_run_within_interval_skips(tmp_path: Path) -> None:
    """
    Test that running maintenance when last sync was within 12 hours:
    - Skips maintenance (returns False).
    - Sync state file timestamp is preserved without modifications.
    """
    db_file = tmp_path / "synapse_test.db"
    sync_state_file = tmp_path / "synapse_last_sync.json"
    storage = SynapseStorage(str(db_file))

    # Write a fresh timestamp (e.g. 1 hour ago)
    recent_ts = time.time() - 3600.0
    sync_state_file.write_text(
        json.dumps({"last_sync": recent_ts, "timestamp": recent_ts}),
        encoding="utf-8",
    )

    triggered = SynapseWorker.run_in_process_maintenance_if_needed(
        interval_hours=12.0,
        non_blocking=False,
        sync_state_path=sync_state_file,
        storage=storage,
    )

    assert triggered is False

    # Timestamp should remain untouched
    data = json.loads(sync_state_file.read_text(encoding="utf-8"))
    assert data["last_sync"] == recent_ts
    storage.close()


def test_elapsed_greater_than_interval_triggers(tmp_path: Path) -> None:
    """
    Test that when elapsed time exceeds interval_hours (e.g. 13 hours ago):
    - Maintenance is re-triggered (returns True).
    - Timestamp is updated to current time.
    """
    db_file = tmp_path / "synapse_test.db"
    sync_state_file = tmp_path / "synapse_last_sync.json"
    storage = SynapseStorage(str(db_file))

    # Write an old timestamp (13 hours ago)
    old_ts = time.time() - (13.0 * 3600.0)
    sync_state_file.write_text(
        json.dumps({"last_sync": old_ts, "timestamp": old_ts}),
        encoding="utf-8",
    )

    triggered = SynapseWorker.run_in_process_maintenance_if_needed(
        interval_hours=12.0,
        non_blocking=False,
        sync_state_path=sync_state_file,
        storage=storage,
    )

    assert triggered is True

    # Timestamp should now be updated to ~now
    data = json.loads(sync_state_file.read_text(encoding="utf-8"))
    assert data["last_sync"] > old_ts
    assert abs(data["last_sync"] - time.time()) < 5.0
    storage.close()


def test_non_blocking_execution_spawns_daemon_thread(tmp_path: Path) -> None:
    """
    Test non_blocking=True mode:
    - Returns True immediately without blocking.
    - Spawns a background daemon thread.
    - Daemon thread executes maintenance and updates state file.
    """
    db_file = tmp_path / "synapse_test.db"
    sync_state_file = tmp_path / "synapse_last_sync.json"
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir(parents=True, exist_ok=True)

    sop_file = vault_dir / "quick_sop.md"
    sop_file.write_text("# Quick SOP\n\nStandard operation details.", encoding="utf-8")

    storage = SynapseStorage(str(db_file))

    triggered = SynapseWorker.run_in_process_maintenance_if_needed(
        vault_path=str(vault_dir),
        interval_hours=12.0,
        non_blocking=True,
        sync_state_path=sync_state_file,
        storage=storage,
    )

    assert triggered is True
    thread = SynapseWorker._latest_background_thread
    assert thread is not None
    assert isinstance(thread, threading.Thread)
    assert thread.daemon is True

    # Wait for daemon thread to finish
    thread.join(timeout=3.0)

    # Verify state file was written
    assert sync_state_file.exists()
    data = json.loads(sync_state_file.read_text(encoding="utf-8"))
    assert "last_sync" in data
    assert abs(data["last_sync"] - time.time()) < 5.0

    # Verify vault node synced
    nodes = storage.list_nodes(category="sop")
    assert len(nodes) == 1
    storage.close()


def test_corrupt_or_empty_sync_file_triggers_recovery(tmp_path: Path) -> None:
    """
    Test that corrupted, empty, or malformed sync state files:
    - Are treated as requiring sync.
    - Successfully re-trigger maintenance and repair the file.
    """
    db_file = tmp_path / "synapse_test.db"
    sync_state_file = tmp_path / "corrupt_sync.json"
    storage = SynapseStorage(str(db_file))

    # Case 1: Empty file
    sync_state_file.write_text("", encoding="utf-8")
    t1 = SynapseWorker.run_in_process_maintenance_if_needed(
        non_blocking=False,
        sync_state_path=sync_state_file,
        storage=storage,
    )
    assert t1 is True
    assert sync_state_file.exists()

    # Case 2: Invalid JSON content
    sync_state_file.write_text("{not: valid json--", encoding="utf-8")
    t2 = SynapseWorker.run_in_process_maintenance_if_needed(
        non_blocking=False,
        sync_state_path=sync_state_file,
        storage=storage,
    )
    assert t2 is True
    data = json.loads(sync_state_file.read_text(encoding="utf-8"))
    assert "last_sync" in data
    storage.close()


# ============================================================================
# 2. Instance vs Class Method Invocation Tests
# ============================================================================

def test_instance_and_class_method_invocation(tmp_path: Path) -> None:
    """
    Test that run_in_process_maintenance_if_needed works identically
    when invoked on SynapseWorker class or on a worker instance.
    """
    db_file = tmp_path / "synapse_test.db"
    sync_state_file = tmp_path / "sync_state.json"
    storage = SynapseStorage(str(db_file))
    worker = SynapseWorker(storage=storage)

    # Calling on instance
    res1 = worker.run_in_process_maintenance_if_needed(
        interval_hours=12.0,
        non_blocking=False,
        sync_state_path=sync_state_file,
    )
    assert res1 is True

    # Immediate next run should skip
    res2 = worker.run_in_process_maintenance_if_needed(
        interval_hours=12.0,
        non_blocking=False,
        sync_state_path=sync_state_file,
    )
    assert res2 is False

    storage.close()


# ============================================================================
# 3. Path Resolution & Cross-Platform Portability Tests
# ============================================================================

def test_path_home_resolution_and_portability(tmp_path: Path) -> None:
    """
    Verify that Path.home() resolution cleanly handles paths on all operating systems
    (Linux, macOS, Windows) without hardcoded OS root assumptions.
    """
    fake_home = tmp_path / "user_home"
    fake_home.mkdir(parents=True, exist_ok=True)

    with patch("pathlib.Path.home", return_value=fake_home):
        # 1. SynapseStorage default path
        storage = SynapseStorage()
        expected_db = (fake_home / ".synapse" / "synapse.db").resolve()
        assert Path(storage.db_path).resolve() == expected_db
        storage.close()

        # 2. SynapseAdapter default path
        adapter = SynapseAdapter()
        assert Path(adapter.db_path).resolve() == expected_db
        if adapter.storage:
            adapter.storage.close()

        # 3. SynapseWorker default sync state path
        expected_sync_file = fake_home / ".synapse" / "synapse_last_sync.json"
        assert not expected_sync_file.exists()

        mem_storage = SynapseStorage(":memory:")
        triggered = SynapseWorker.run_in_process_maintenance_if_needed(
            storage=mem_storage,
            non_blocking=False,
        )
        assert triggered is True
        assert expected_sync_file.exists()
        mem_storage.close()


# ============================================================================
# 4. Session Lifecycle Integration Tests
# ============================================================================

def test_session_manager_lifecycle_hook_trigger(tmp_path: Path) -> None:
    """
    Verify that SessionManager initializes cleanly and triggers the in-process
    maintenance hook safely with zero crash risk even with custom or missing folders.
    """
    config_dir = tmp_path / ".synapse"
    session_mgr = SessionManager(config_dir=config_dir)

    assert session_mgr.session_id is not None
    sync_file = config_dir / "synapse_last_sync.json"

    # Wait briefly for non-blocking daemon thread if active
    if SynapseWorker._latest_background_thread:
        SynapseWorker._latest_background_thread.join(timeout=3.0)

    # Calling on_session_start explicitly should work safely
    status = session_mgr.on_session_start()
    assert isinstance(status, bool)


def test_session_lifecycle_resilience_on_exception(tmp_path: Path) -> None:
    """
    Verify that if SynapseWorker raises an exception during lifecycle hook,
    SessionManager absorbs it safely and continues normal operation.
    """
    config_dir = tmp_path / ".synapse"

    with patch.object(
        SynapseWorker,
        "run_in_process_maintenance_if_needed",
        side_effect=RuntimeError("Disk I/O error simulation"),
    ):
        # Should not raise exception
        session_mgr = SessionManager(config_dir=config_dir)
        res = session_mgr.on_session_start()
        assert res is False
        assert session_mgr.session_id is not None
