#!/usr/bin/env python3
"""Unit test suite for Synapse Consolidation Layer 1 (Deterministic Engine)."""
import os
import sqlite3
import tempfile
import pytest

def create_mock_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE synapse_nodes (
            id TEXT PRIMARY KEY,
            uri TEXT UNIQUE,
            l0_abstract TEXT,
            l1_overview TEXT,
            l2_target_path TEXT,
            category TEXT CHECK(category IN ('preference', 'decision', 'architecture', 'incident', 'sop', 'general')),
            importance REAL DEFAULT 0.5,
            strength REAL DEFAULT 1.0,
            created_at TEXT,
            last_accessed TEXT,
            status TEXT DEFAULT 'active',
            salience REAL DEFAULT 0.5,
            confidence REAL DEFAULT 1.0
        )
    """)
    conn.commit()
    conn.close()
    return path

def test_amigdala_salience_protection():
    db_path = create_mock_db()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Insert Master Directive (Salience = 1.0)
    cur.execute("""
        INSERT INTO synapse_nodes (id, uri, l0_abstract, category, salience, strength, last_accessed)
        VALUES ('test:master_directive', 'vault://master/rule1', 'Master Rule', 'decision', 1.0, 0.5, '2020-01-01T00:00:00')
    """)

    # Insert Temporary low-salience node (Salience = 0.1)
    cur.execute("""
        INSERT INTO synapse_nodes (id, uri, l0_abstract, category, salience, strength, last_accessed)
        VALUES ('test:temp_junk', 'vault://temp/junk', 'Temp Junk', 'general', 0.1, 0.04, '2020-01-01T00:00:00')
    """)
    conn.commit()

    # Run Step 2 (Amigdala hard-lock)
    cur.execute("""
        UPDATE synapse_nodes
        SET strength = 1.0, confidence = 1.0
        WHERE salience >= 0.90 OR category IN ('preference', 'decision')
    """)

    # Run Step 4 (Prune)
    cur.execute("""
        DELETE FROM synapse_nodes
        WHERE salience < 0.25
          AND strength <= 0.05
          AND category NOT IN ('preference', 'decision', 'architecture', 'sop')
    """)
    conn.commit()

    # Verify Master Directive still exists & has strength 1.0
    master_node = cur.execute("SELECT strength, salience FROM synapse_nodes WHERE id='test:master_directive'").fetchone()
    assert master_node is not None
    assert master_node[0] == 1.0
    assert master_node[1] == 1.0

    # Verify Junk node was pruned
    junk_node = cur.execute("SELECT id FROM synapse_nodes WHERE id='test:temp_junk'").fetchone()
    assert junk_node is None

    conn.close()
    os.remove(db_path)

def test_atomic_rollback_on_error():
    db_path = create_mock_db()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("INSERT INTO synapse_nodes (id, uri, l0_abstract, category) VALUES ('n1', 'u1', 'abs', 'general')")
    conn.commit()

    # Simulate failing transaction
    try:
        with conn:
            cur.execute("UPDATE synapse_nodes SET l0_abstract='MODIFIED' WHERE id='n1'")
            # Trigger deliberate error
            cur.execute("INSERT INTO synapse_nodes (id, uri, category) VALUES ('n1', 'u1', 'invalid_category')")
    except Exception:
        pass

    # Verify rollback
    cur.execute("SELECT l0_abstract FROM synapse_nodes WHERE id='n1'")
    val = cur.fetchone()[0]
    assert val == 'abs', "Transaction should have rolled back to original value"

    conn.close()
    os.remove(db_path)

def test_moc_bookkeeper_consolidation_step():
    from synapse.moc_bookkeeper import run_moc_bookkeeper_step
    import tempfile, shutil
    from pathlib import Path

    tmp_dir = tempfile.mkdtemp()
    try:
        vault_path = Path(tmp_dir)
        (vault_path / "server").mkdir()
        (vault_path / "server" / "00-INDEX-SERVER.md").write_text(
            "# Index Server\n\nList of connected documents in this folder:\n\n- [[existing-server-doc]]\n\n## Archive & Historical Documents\n",
            encoding="utf-8"
        )
        (vault_path / "server" / "existing-server-doc.md").write_text("# Existing Doc\nExisting doc.", encoding="utf-8")
        (vault_path / "server" / "new-server-doc.md").write_text("# New Server Doc\nNew server doc.", encoding="utf-8")

        stats = run_moc_bookkeeper_step(vault_dir=str(vault_path))
        assert "new-server-doc.md" in stats.get("server", [])
        updated_index = (vault_path / "server" / "00-INDEX-SERVER.md").read_text(encoding="utf-8")
        assert "[[new-server-doc]]" in updated_index
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

