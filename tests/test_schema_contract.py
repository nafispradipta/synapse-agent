"""
Schema contract for SynapseStorage.

These tests pin the observable database schema and the FTS/vector sync
behaviour so the schema initialisation can be restructured without silently
changing the contract that existing databases rely on.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synapse.storage import SynapseStorage

CORE_TABLES = {
    "_synapse_vector_meta",
    "episodes",
    "episodes_fts",
    "synapse_edges",
    "synapse_external_research",
    "synapse_fts",
    "synapse_mental_models",
    "synapse_nodes",
    "synapse_observations",
    "synapse_superseded_archive",
    "synapse_vectors",
}

EXPECTED_INDEXES = {
    "idx_episodes_last_accessed",
    "idx_episodes_salience",
    "idx_superseded_archive_cat",
    "idx_superseded_archive_rollup",
    "idx_synapse_edges_target",
    "idx_synapse_external_research_status",
    "idx_synapse_external_research_topic",
    "idx_synapse_mental_models_last_refreshed",
    "idx_synapse_nodes_access_count",
    "idx_synapse_nodes_category",
    "idx_synapse_nodes_last_accessed",
    "idx_synapse_nodes_salience",
    "idx_synapse_nodes_status",
    "idx_synapse_observations_proof_count",
    "idx_synapse_observations_status",
    "idx_synapse_vectors_updated_at",
}

EXPECTED_TRIGGERS = {
    "episodes_ad",
    "episodes_ai",
    "episodes_au",
    "synapse_nodes_ad",
    "synapse_nodes_ai",
    "synapse_nodes_au",
    "synapse_nodes_version_ad",
    "synapse_nodes_version_au_status",
    "synapse_vectors_version_ad",
    "synapse_vectors_version_ai",
    "synapse_vectors_version_au",
}

EXPECTED_COLUMNS = {
    "synapse_nodes": [
        "id", "uri", "l0_abstract", "l1_overview", "l2_target_path", "category",
        "importance", "strength", "created_at", "last_accessed", "status",
        "salience", "confidence", "access_count",
    ],
    "synapse_edges": [
        "source_id", "target_id", "relation", "weight", "last_fired", "access_count",
    ],
    "episodes": [
        "id", "title", "situation", "action_taken", "outcome", "lessons_learned",
        "salience", "confidence", "created_at", "last_accessed", "tags",
    ],
    "synapse_vectors": ["id", "vector_json", "dim", "updated_at"],
    "synapse_mental_models": [
        "id", "name", "source_query", "content", "structured_content",
        "trigger_mode", "tags", "max_tokens", "last_refreshed_at",
        "last_refreshed_source_query", "history", "created_at",
    ],
    "synapse_observations": [
        "id", "pattern", "supporting_facts", "proof_count", "confidence",
        "status", "category", "created_at", "updated_at",
    ],
    "synapse_external_research": [
        "id", "topic", "title", "source_url", "domain", "summary",
        "raw_snippet", "tier_level", "vault_path", "fetched_at", "status",
    ],
    "synapse_superseded_archive": [
        "id", "uri", "l0_abstract", "l1_overview", "l2_target_path", "category",
        "importance", "strength", "created_at", "last_accessed", "salience",
        "confidence", "access_count", "archived_at", "rollup_id",
    ],
}


@pytest.fixture
def storage():
    s = SynapseStorage(db_path=":memory:")
    yield s
    s.close()


def _names_by_type(storage, object_type):
    return {
        row[0]
        for row in storage._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = ?", (object_type,)
        ).fetchall()
    }


def test_schema_creates_expected_core_tables(storage):
    assert CORE_TABLES <= _names_by_type(storage, "table")


def test_schema_creates_expected_fts_shadow_tables(storage):
    tables = _names_by_type(storage, "table")
    for fts_table in ("synapse_fts", "episodes_fts"):
        for suffix in ("_data", "_idx", "_content", "_docsize", "_config"):
            assert f"{fts_table}{suffix}" in tables


@pytest.mark.parametrize("table,columns", EXPECTED_COLUMNS.items())
def test_table_columns_are_stable(storage, table, columns):
    actual = [row[1] for row in storage._conn.execute(f"PRAGMA table_info({table})")]
    assert actual == columns


def test_schema_creates_expected_indexes(storage):
    assert EXPECTED_INDEXES <= _names_by_type(storage, "index")


def test_schema_creates_expected_triggers(storage):
    assert _names_by_type(storage, "trigger") == EXPECTED_TRIGGERS


def test_foreign_keys_are_enforced(storage):
    assert storage._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_busy_timeout_allows_concurrent_writer_window(storage):
    assert storage._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 10_000


def test_node_fts_stays_in_sync_across_insert_update_delete(storage):
    node_id = storage.add_node(uri="synapse://test/fts", l0_abstract="Quantum widget")

    assert [r["id"] for r in storage.search_fts("quantum")] == [node_id]

    storage.update_node(node_id, l0_abstract="Classical gadget")
    assert storage.search_fts("quantum") == []
    assert [r["id"] for r in storage.search_fts("classical")] == [node_id]

    storage.delete_node(node_id)
    assert storage.search_fts("classical") == []


def test_episode_fts_stays_in_sync_across_insert_update_delete(storage):
    episode_id = storage.add_episode(
        title="Cache stampede",
        situation="Traffic spiked after cache expiry",
        lessons_learned="Add jitter to TTLs",
        salience=0.8,
    )

    assert [r["id"] for r in storage.search_episodes("stampede")] == [episode_id]

    storage.update_episode(episode_id, lessons_learned="Add jitter and warm the cache")
    assert storage.search_episodes("warm") != []

    storage.delete_episode(episode_id)
    assert storage.search_episodes("stampede") == []


def test_vector_version_metadata_advances_on_vector_writes(storage):
    def version():
        return storage._conn.execute(
            "SELECT version FROM _synapse_vector_meta WHERE id = 1"
        ).fetchone()[0]

    baseline = version()
    node_id = storage.add_node(uri="synapse://test/vector", l0_abstract="Vector node")
    storage.upsert_vector(node_id, [0.1, 0.2, 0.3])
    after_write = version()

    assert after_write > baseline

    storage.delete_node(node_id)
    assert version() > after_write


def test_legacy_nodes_table_is_rebuilt_for_custom_categories_and_flagged_status(tmp_path):
    db_file = tmp_path / "legacy.db"
    legacy = sqlite3.connect(db_file)
    legacy.execute(
        """
        CREATE TABLE synapse_nodes (
            id TEXT PRIMARY KEY,
            uri TEXT UNIQUE NOT NULL,
            l0_abstract TEXT NOT NULL,
            l1_overview TEXT,
            l2_target_path TEXT,
            category TEXT DEFAULT 'general' CHECK(category IN ('preference', 'decision', 'general')),
            importance REAL DEFAULT 0.5,
            strength REAL DEFAULT 1.0,
            created_at INTEGER,
            last_accessed INTEGER,
            status TEXT DEFAULT 'active' CHECK(status IN ('active', 'superseded', 'archived'))
        )
        """
    )
    legacy.execute(
        "INSERT INTO synapse_nodes (id, uri, l0_abstract, category) "
        "VALUES ('legacy-1', 'synapse://legacy/one', 'Legacy node', 'decision')"
    )
    legacy.commit()
    legacy.close()

    storage = SynapseStorage(db_path=str(db_file), allow_custom_categories=True)
    try:
        assert storage.get_node("legacy-1")["l0_abstract"] == "Legacy node"

        columns = [
            row[1] for row in storage._conn.execute("PRAGMA table_info(synapse_nodes)")
        ]
        assert "access_count" in columns

        storage.add_node(
            uri="synapse://legacy/two",
            l0_abstract="New node",
            category="medical",
            status="flagged",
        )
        assert storage.get_node("synapse://legacy/two")["status"] == "flagged"
    finally:
        storage.close()


def _write_legacy_db(db_file):
    legacy = sqlite3.connect(db_file)
    legacy.execute(
        """
        CREATE TABLE synapse_nodes (
            id TEXT PRIMARY KEY,
            uri TEXT UNIQUE NOT NULL,
            l0_abstract TEXT NOT NULL,
            l1_overview TEXT,
            l2_target_path TEXT,
            category TEXT DEFAULT 'general' CHECK(category IN ('preference', 'decision', 'general')),
            importance REAL DEFAULT 0.5,
            strength REAL DEFAULT 1.0,
            created_at INTEGER,
            last_accessed INTEGER,
            status TEXT DEFAULT 'active' CHECK(status IN ('active', 'superseded', 'archived'))
        )
        """
    )
    legacy.execute(
        "INSERT INTO synapse_nodes (id, uri, l0_abstract) "
        "VALUES ('legacy-1', 'synapse://legacy/one', 'Legacy node')"
    )
    legacy.commit()
    legacy.close()


def test_legacy_migration_keeps_foreign_key_enforcement(tmp_path):
    """The rebuild flips foreign_keys off; it must flip it back on.

    PRAGMA foreign_keys is a no-op inside a transaction, and the row copy opens
    one, so the enforcement flag used to stay disabled for the lifetime of the
    connection: ON DELETE CASCADE silently stopped working after a migration.
    """
    db_file = tmp_path / "legacy.db"
    _write_legacy_db(db_file)

    storage = SynapseStorage(db_path=str(db_file), allow_custom_categories=True)
    try:
        assert storage._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

        source = storage.add_node(uri="synapse://legacy/src", l0_abstract="source")
        target = storage.add_node(uri="synapse://legacy/dst", l0_abstract="target")
        storage.add_edge(source, target, relation="related")
        assert len(storage.get_edges(target)) == 1

        storage.delete_node(source)

        assert storage.get_edges(target) == []
    finally:
        storage.close()


class _FailingCopyCursor:
    """Cursor double whose row copy fails after the pragmas have been flipped."""

    def __init__(self):
        self.statements = []

    def execute(self, sql, params=()):
        head = " ".join(sql.split())
        self.statements.append(head[:40])
        if head.startswith("INSERT INTO synapse_nodes"):
            raise sqlite3.OperationalError("simulated copy failure")
        return self

    def fetchone(self):
        return (
            "CREATE TABLE synapse_nodes (category TEXT CHECK(category IN ('a', 'b')))",
        )


def test_rebuild_restores_pragmas_when_the_copy_fails(storage):
    cursor = _FailingCopyCursor()

    with pytest.raises(sqlite3.OperationalError):
        storage._rebuild_obsolete_nodes_table(cursor)

    assert "PRAGMA legacy_alter_table=ON;" in cursor.statements
    assert "PRAGMA foreign_keys=OFF;" in cursor.statements
    assert cursor.statements[-2:] == [
        "PRAGMA foreign_keys=ON;",
        "PRAGMA legacy_alter_table=OFF;",
    ]
