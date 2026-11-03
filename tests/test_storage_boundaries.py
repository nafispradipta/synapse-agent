"""
Bounded-context boundaries for SynapseStorage.

SynapseStorage is a facade over several storage concerns that matured into one
class: graph/FTS, vectors, episodes, consolidated knowledge, and scheduled
maintenance plus the schema it all sits on. These tests pin the split so
methods cannot drift back into one file, cannot be duplicated across mixins,
and cannot re-introduce a circular import back into ``synapse.storage``.
"""

import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synapse.storage import SynapseStorage
from synapse.stores.episodes import EpisodeStore
from synapse.stores.graph import GraphStore
from synapse.stores.knowledge import KnowledgeStore
from synapse.stores.maintenance import MaintenanceStore
from synapse.stores.schema import SchemaMixin
from synapse.stores.vectors import VectorStore

STORES_DIR = PROJECT_ROOT / "synapse" / "stores"

PUBLIC_API = {
    "add_edge", "add_episode", "add_external_research", "add_node",
    "add_or_update_observation", "close", "commit_superseded_rollup", "delete_episode",
    "delete_node", "find_active_node_by_l2_paths", "find_node_by_episode_ref", "get_active_nodes",
    "get_edges", "get_episode", "get_mental_model", "get_node",
    "get_node_by_l2_target_path", "get_stats", "get_vector", "get_vector_matrix",
    "incremental_cache", "invalidate_vector_cache", "list_active_vault_targets", "list_consolidatable_episodes",
    "list_episodes", "list_external_research", "list_mental_models", "list_nodes",
    "list_nodes_missing_vectors", "list_observations", "list_superseded_nodes", "optimize_and_compact",
    "promote_external_research", "prune_decayed_edges", "raw_vector_matrix", "rebuild_vector_cache",
    "record_node_access", "resolve_wikilink_node", "search_episodes", "search_fts",
    "search_vectors", "set_mental_model", "supersede_low_salience_nodes", "supersede_node",
    "update_episode", "update_node", "upsert_vector", "vector_ids",
    "vector_matrix", "vector_node_ids",
}

# Each bounded context owns its methods; the facade only wires them together.
METHOD_HOME = {
    GraphStore: {
        "add_node", "get_node", "record_node_access", "update_node", "delete_node",
        "supersede_node", "get_active_nodes", "list_nodes", "add_edge", "get_edges",
        "prune_decayed_edges", "search_fts", "resolve_wikilink_node",
        "find_active_node_by_l2_paths", "list_active_vault_targets",
        "find_node_by_episode_ref", "get_node_by_l2_target_path",
    },
    VectorStore: {
        "upsert_vector", "get_vector", "incremental_cache", "vector_matrix",
        "raw_vector_matrix", "vector_node_ids", "vector_ids", "get_vector_matrix",
        "invalidate_vector_cache", "rebuild_vector_cache", "search_vectors",
        "list_nodes_missing_vectors",
    },
    EpisodeStore: {
        "add_episode", "get_episode", "update_episode", "delete_episode",
        "list_episodes", "search_episodes", "list_consolidatable_episodes",
    },
    KnowledgeStore: {
        "set_mental_model", "get_mental_model", "list_mental_models",
        "add_or_update_observation", "list_observations", "add_external_research",
        "list_external_research", "promote_external_research",
    },
    MaintenanceStore: {
        "optimize_and_compact", "supersede_low_salience_nodes",
        "list_superseded_nodes", "commit_superseded_rollup",
        "lock_critical_nodes", "apply_hebbian_decay", "prune_exhausted_nodes",
        "prune_weak_edges", "upsert_vault_node", "optimize_fts_indices",
        "verify_integrity", "wal_checkpoint",
    },
}

SCHEMA_METHODS = {
    "_configure_pragmas", "_init_schema", "_create_node_tables",
    "_rebuild_obsolete_nodes_table", "_create_edge_tables", "_create_node_fts",
    "_create_episode_tables", "_create_episode_fts", "_create_vector_tables",
    "_create_knowledge_tables",
}


@pytest.fixture
def storage():
    s = SynapseStorage(db_path=":memory:")
    yield s
    s.close()


def _defined_methods(cls):
    """Names defined directly on ``cls`` (excluding dunder helpers)."""
    return {
        name
        for name, value in vars(cls).items()
        if not name.startswith("__")
    }


def test_public_api_surface_is_unchanged(storage):
    public = {name for name in dir(SynapseStorage) if not name.startswith("_")}
    assert PUBLIC_API <= public


def test_synapse_storage_composes_every_bounded_context():
    mro = set(SynapseStorage.__mro__)
    for store in (SchemaMixin, GraphStore, VectorStore, EpisodeStore, KnowledgeStore, MaintenanceStore):
        assert store in mro, f"{store.__name__} missing from SynapseStorage MRO"


@pytest.mark.parametrize("store,methods", METHOD_HOME.items(), ids=lambda v: getattr(v, "__name__", ""))
def test_methods_live_on_their_bounded_context(store, methods):
    assert methods <= _defined_methods(store)


def test_public_methods_are_defined_exactly_once_across_mixins():
    owners = {}
    for store in METHOD_HOME:
        for name in _defined_methods(store):
            owners.setdefault(name, []).append(store.__name__)

    duplicated = {name: where for name, where in owners.items() if len(where) > 1}
    assert duplicated == {}


def test_schema_helpers_live_on_the_schema_mixin():
    assert SCHEMA_METHODS <= _defined_methods(SchemaMixin)


def test_facade_keeps_only_wiring_and_cross_context_aggregates():
    facade_methods = _defined_methods(SynapseStorage)
    assert "get_stats" in facade_methods
    assert "add_node" not in facade_methods
    assert "_init_schema" not in facade_methods


def test_store_modules_do_not_import_the_facade_module():
    """Mixins must stay importable without synapse.storage (no cycles)."""
    offenders = []
    for path in sorted(STORES_DIR.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module in {"synapse.storage", "synapse"}:
                offenders.append(f"{path.name}:{node.lineno} from {node.module}")
            elif isinstance(node, ast.Import) and any(
                alias.name == "synapse.storage" for alias in node.names
            ):
                offenders.append(f"{path.name}:{node.lineno} import synapse.storage")
    assert offenders == []


def test_mixins_are_instantiable_only_through_the_facade():
    """A mixin on its own must not be usable storage (documents the contract)."""
    for store in METHOD_HOME:
        assert not hasattr(store(), "_conn")
