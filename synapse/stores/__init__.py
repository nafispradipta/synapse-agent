"""
Bounded-context mixins that compose into :class:`synapse.storage.SynapseStorage`.

Each module owns one storage concern and talks to SQLite through ``self._conn``
and ``self._lock`` provided by the facade. Import order here is the MRO order
used by the facade, so schema helpers resolve before any consumer.
"""

from synapse.stores.episodes import EpisodeStore
from synapse.stores.graph import GraphStore
from synapse.stores.knowledge import KnowledgeStore
from synapse.stores.maintenance import MaintenanceStore
from synapse.stores.schema import SchemaMixin
from synapse.stores.vectors import VectorStore

__all__ = [
    "SchemaMixin",
    "GraphStore",
    "VectorStore",
    "EpisodeStore",
    "KnowledgeStore",
    "MaintenanceStore",
]
