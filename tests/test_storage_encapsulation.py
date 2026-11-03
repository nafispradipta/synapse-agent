"""
Encapsulation contract for SynapseStorage.

Production modules must talk to storage through its public API. Reaching into
``storage._conn`` or ``storage._lock`` turns the SQLite schema and locking model
into an implicit cross-module contract, so any schema change can break callers
silently. These tests pin the named data-access methods that replaced those
direct accesses.
"""

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synapse.storage import SynapseStorage

SYNAPSE_DIR = PROJECT_ROOT / "synapse"


@pytest.fixture
def storage():
    """In-memory SynapseStorage fixture."""
    s = SynapseStorage(db_path=":memory:")
    yield s
    s.close()


def test_no_production_module_reaches_into_storage_private_connection():
    pattern = re.compile(r"storage\._(?:conn|lock)\b")
    offenders = []
    for path in sorted(SYNAPSE_DIR.rglob("*.py")):
        if path.name == "storage.py":
            continue
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            line = text[: match.start()].count("\n") + 1
            offenders.append(f"{path.relative_to(SYNAPSE_DIR).as_posix()}:{line}")
    assert offenders == []


def drop_vector_row(storage, node_id):
    """Simulate a node whose embedding never landed (import, crash, legacy row)."""
    storage._conn.execute("DELETE FROM synapse_vectors WHERE id = ?", (node_id,))
    storage._conn.commit()


class TestVectorIndexingQueries:
    def test_list_nodes_missing_vectors_returns_only_unvectorized_active_nodes(self, storage):
        indexed = storage.add_node(uri="synapse://test/indexed", l0_abstract="indexed node")
        unindexed = storage.add_node(uri="synapse://test/unindexed", l0_abstract="unindexed node")
        drop_vector_row(storage, unindexed)

        missing = storage.list_nodes_missing_vectors(limit=10)

        assert [row["id"] for row in missing] == [unindexed]
        assert {"id", "l0_abstract", "l1_overview"} <= set(missing[0])
        assert indexed != unindexed

    def test_list_nodes_missing_vectors_skips_superseded_nodes(self, storage):
        superseded = storage.add_node(
            uri="synapse://test/old", l0_abstract="old node", salience=0.1
        )
        drop_vector_row(storage, superseded)
        storage.supersede_node(superseded)

        assert storage.list_nodes_missing_vectors(limit=10) == []

    def test_list_nodes_missing_vectors_respects_limit(self, storage):
        for index in range(3):
            node_id = storage.add_node(
                uri=f"synapse://test/{index}", l0_abstract=f"node {index}"
            )
            drop_vector_row(storage, node_id)

        assert len(storage.list_nodes_missing_vectors(limit=2)) == 2


class TestWikilinkLookupQueries:
    def test_resolve_wikilink_node_matches_uri_candidate(self, storage):
        node_id = storage.add_node(uri="synapse://sop/auth-flow", l0_abstract="Auth flow")

        found = storage.resolve_wikilink_node(
            raw="auth-flow",
            clean="auth-flow",
            clean_slug="auth-flow",
            clean_stem="auth-flow",
            stem="auth-flow",
        )

        assert found is not None
        assert found["id"] == node_id

    def test_resolve_wikilink_node_matches_l2_target_path_like_pattern(self, storage):
        node_id = storage.add_node(
            uri="synapse://sop/x",
            l0_abstract="X",
            l2_target_path="vault/notes/x.md",
        )

        found = storage.resolve_wikilink_node(
            raw="x", clean="x", clean_slug="x", clean_stem="x", stem="x"
        )

        assert found is not None
        assert found["id"] == node_id

    def test_resolve_wikilink_node_returns_none_when_absent(self, storage):
        assert (
            storage.resolve_wikilink_node(
                raw="nope", clean="nope", clean_slug="nope", clean_stem="nope", stem="nope"
            )
            is None
        )

    def test_find_active_node_by_l2_paths_matches_second_candidate(self, storage):
        node_id = storage.add_node(
            uri="synapse://sop/x",
            l0_abstract="X",
            l2_target_path="C:/vault/x.md",
        )

        found = storage.find_active_node_by_l2_paths(["/not/this.md", "C:/vault/x.md"])

        assert found is not None
        assert found["id"] == node_id

    def test_list_active_vault_targets_only_returns_nodes_with_paths(self, storage):
        with_path = storage.add_node(
            uri="synapse://sop/a", l0_abstract="A", l2_target_path="vault/a.md"
        )
        storage.add_node(uri="synapse://sop/b", l0_abstract="B")
        superseded = storage.add_node(
            uri="synapse://sop/c", l0_abstract="C", l2_target_path="vault/c.md", salience=0.1
        )
        storage.supersede_node(superseded)

        rows = storage.list_active_vault_targets()

        assert [row["id"] for row in rows] == [with_path]
        assert set(rows[0]) >= {"id", "uri", "l2_target_path"}


class TestBeliefRevisionWrites:
    def test_supersede_low_salience_nodes_only_touches_nodes_below_threshold(self, storage):
        low = storage.add_node(uri="synapse://test/low", l0_abstract="low", salience=0.4)
        high = storage.add_node(uri="synapse://test/high", l0_abstract="high", salience=0.95)

        updated = storage.supersede_low_salience_nodes([low, high])

        assert updated == 1
        assert storage.get_node(low)["status"] == "superseded"
        assert storage.get_node(high)["status"] == "active"

    def test_supersede_low_salience_nodes_leaves_unknown_ids_alone(self, storage):
        assert storage.supersede_low_salience_nodes(["missing-node"]) == 0


class TestRollupQueries:
    def test_list_superseded_nodes_returns_only_superseded_rows(self, storage):
        superseded = storage.add_node(
            uri="synapse://test/old", l0_abstract="old", salience=0.1
        )
        storage.add_node(uri="synapse://test/live", l0_abstract="live", salience=0.9)
        storage.supersede_node(superseded)

        rows = storage.list_superseded_nodes()

        assert [row["id"] for row in rows] == [superseded]
        assert "access_count" in rows[0]

    def test_commit_superseded_rollup_archives_master_and_purges_superseded(self, storage):
        survivor = storage.add_node(uri="synapse://test/live", l0_abstract="live", salience=0.9)
        superseded = storage.add_node(
            uri="synapse://test/old", l0_abstract="old", salience=0.1
        )
        storage.add_edge(superseded, survivor, relation="related")
        storage.supersede_node(superseded)

        result = storage.commit_superseded_rollup(
            master={
                "id": "arch-rollup-superseded-master",
                "uri": "synapse://archive/superseded_rollup",
                "l0_abstract": "Rollup of superseded nodes",
                "l1_overview": "# Rollup",
            },
            rollup_id="rollup-1",
            archived_at=1700000000,
        )

        assert result["archived_rows"] == 1
        assert result["deleted_nodes"] == 1
        assert result["deleted_edges"] == 1
        assert storage.get_node(superseded) is None
        master = storage.get_node("synapse://archive/superseded_rollup")
        assert master is not None
        assert master["status"] == "archived"

    def test_commit_superseded_rollup_records_archive_rows(self, storage):
        superseded = storage.add_node(
            uri="synapse://test/old", l0_abstract="old", salience=0.1
        )
        storage.supersede_node(superseded)

        storage.commit_superseded_rollup(
            master={
                "id": "arch-rollup-superseded-master",
                "uri": "synapse://archive/superseded_rollup",
                "l0_abstract": "Rollup",
                "l1_overview": "# Rollup",
            },
            rollup_id="rollup-2",
            archived_at=1700000001,
        )

        archived = [
            dict(row)
            for row in storage._conn.execute(
                "SELECT id, rollup_id, archived_at FROM synapse_superseded_archive"
            ).fetchall()
        ]
        assert [row["id"] for row in archived] == [superseded]
        assert archived[0]["rollup_id"] == "rollup-2"
        assert archived[0]["archived_at"] == 1700000001


class TestEpisodeConsolidationQueries:
    def test_list_consolidatable_episodes_requires_lessons_and_salience(self, storage):
        eligible = storage.add_episode(
            title="Eligible", lessons_learned="Always do X", salience=0.9
        )
        storage.add_episode(title="No lessons", lessons_learned="", salience=0.9)
        storage.add_episode(title="Low salience", lessons_learned="Do Z", salience=0.1)

        rows = storage.list_consolidatable_episodes(min_salience=0.5)

        assert [row["id"] for row in rows] == [eligible]

    def test_find_node_by_episode_ref_matches_target_path(self, storage):
        node_id = storage.add_node(
            uri="synapse://sop/episode-node",
            l0_abstract="Episode node",
            l2_target_path="episode:abc123",
        )

        assert storage.find_node_by_episode_ref("abc123")["id"] == node_id
        assert storage.find_node_by_episode_ref("does-not-exist") is None

    def test_get_node_by_l2_target_path_returns_exact_match_only(self, storage):
        node_id = storage.add_node(
            uri="synapse://sop/exact",
            l0_abstract="Exact",
            l2_target_path="episode:exact",
        )

        assert storage.get_node_by_l2_target_path("episode:exact")["id"] == node_id
        assert storage.get_node_by_l2_target_path("episode:exa") is None
