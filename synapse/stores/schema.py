"""Schema bootstrap for the SQLite store: DDL, indexes, triggers, migrations."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Schema DDL
#
# Each table keeps its own statement so the DDL can be reviewed in isolation.
# _init_schema below runs them in dependency order and applies the migrations
# for databases created by older releases.
# ---------------------------------------------------------------------------

_DDL_SYNAPSE_NODES = """
    CREATE TABLE IF NOT EXISTS synapse_nodes (
        id TEXT PRIMARY KEY,
        uri TEXT UNIQUE NOT NULL,
        l0_abstract TEXT NOT NULL,
        l1_overview TEXT,
        l2_target_path TEXT,
        category TEXT DEFAULT 'general',
        importance REAL DEFAULT 0.5,
        strength REAL DEFAULT 1.0,
        created_at INTEGER,
        last_accessed INTEGER,
        status TEXT DEFAULT 'active' CHECK(status IN ('active', 'superseded', 'archived', 'flagged')),
        salience REAL DEFAULT 0.5,
        confidence REAL DEFAULT 1.0,
        access_count INTEGER DEFAULT 0
    );
"""

# Rebuild target for databases whose synapse_nodes still carries the old
# category CHECK constraint or misses the 'flagged' status.
_DDL_SYNAPSE_NODES_REBUILT = """
    CREATE TABLE synapse_nodes (
        id TEXT PRIMARY KEY,
        uri TEXT UNIQUE NOT NULL,
        l0_abstract TEXT NOT NULL,
        l1_overview TEXT,
        l2_target_path TEXT,
        category TEXT DEFAULT 'general',
        importance REAL DEFAULT 0.5,
        strength REAL DEFAULT 1.0,
        created_at INTEGER,
        last_accessed INTEGER,
        status TEXT DEFAULT 'active' CHECK(status IN ('active', 'superseded', 'archived', 'flagged')),
        salience REAL DEFAULT 0.5,
        confidence REAL DEFAULT 1.0,
        access_count INTEGER DEFAULT 0
    );
"""

_DDL_SYNAPSE_SUPERSEDED_ARCHIVE = """
    CREATE TABLE IF NOT EXISTS synapse_superseded_archive (
        id TEXT PRIMARY KEY,
        uri TEXT,
        l0_abstract TEXT,
        l1_overview TEXT,
        l2_target_path TEXT,
        category TEXT,
        importance REAL,
        strength REAL,
        created_at INTEGER,
        last_accessed INTEGER,
        salience REAL,
        confidence REAL,
        access_count INTEGER,
        archived_at INTEGER,
        rollup_id TEXT
    );
"""

_DDL_SYNAPSE_EDGES = """
    CREATE TABLE IF NOT EXISTS synapse_edges (
        source_id TEXT,
        target_id TEXT,
        relation TEXT NOT NULL,
        weight REAL DEFAULT 0.5,
        last_fired INTEGER,
        access_count INTEGER DEFAULT 0,
        PRIMARY KEY (source_id, target_id, relation),
        FOREIGN KEY(source_id) REFERENCES synapse_nodes(id) ON DELETE CASCADE,
        FOREIGN KEY(target_id) REFERENCES synapse_nodes(id) ON DELETE CASCADE
    );
"""

_DDL_SYNAPSE_FTS = """
    CREATE VIRTUAL TABLE IF NOT EXISTS synapse_fts USING fts5(
        id UNINDEXED,
        l0_abstract,
        l1_overview,
        tokenize = 'porter unicode61'
    );
"""

_DDL_EPISODES = """
    CREATE TABLE IF NOT EXISTS episodes (
        id TEXT PRIMARY KEY,
        title TEXT,
        situation TEXT,
        action_taken TEXT,
        outcome TEXT,
        lessons_learned TEXT,
        salience REAL DEFAULT 0.5,
        confidence REAL DEFAULT 1.0,
        created_at INTEGER,
        last_accessed INTEGER,
        tags TEXT
    );
"""

_DDL_EPISODES_FTS = """
    CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
        id UNINDEXED,
        title,
        situation,
        action_taken,
        outcome,
        lessons_learned,
        tags,
        tokenize = 'porter unicode61'
    );
"""

_DDL_SYNAPSE_VECTORS = """
    CREATE TABLE IF NOT EXISTS synapse_vectors (
        id TEXT PRIMARY KEY,
        vector_json TEXT NOT NULL,
        dim INTEGER NOT NULL,
        updated_at INTEGER,
        FOREIGN KEY(id) REFERENCES synapse_nodes(id) ON DELETE CASCADE
    );
"""

_DDL_VECTOR_META = """
    CREATE TABLE IF NOT EXISTS _synapse_vector_meta (
        id INTEGER PRIMARY KEY,
        version INTEGER
    );
"""

_DDL_MENTAL_MODELS = """
    CREATE TABLE IF NOT EXISTS synapse_mental_models (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        source_query TEXT NOT NULL,
        content TEXT NOT NULL,
        structured_content TEXT,
        trigger_mode TEXT DEFAULT 'delta' CHECK(trigger_mode IN ('full', 'delta')),
        tags TEXT,
        max_tokens INTEGER DEFAULT 2048,
        last_refreshed_at INTEGER,
        last_refreshed_source_query TEXT,
        history TEXT,
        created_at INTEGER
    );
"""

_DDL_OBSERVATIONS = """
    CREATE TABLE IF NOT EXISTS synapse_observations (
        id TEXT PRIMARY KEY,
        pattern TEXT NOT NULL,
        supporting_facts TEXT NOT NULL,
        proof_count INTEGER DEFAULT 1,
        confidence REAL DEFAULT 1.0,
        status TEXT DEFAULT 'active' CHECK(status IN ('active', 'contradicted', 'superseded')),
        category TEXT,
        created_at INTEGER,
        updated_at INTEGER
    );
"""

_DDL_EXTERNAL_RESEARCH = """
    CREATE TABLE IF NOT EXISTS synapse_external_research (
        id TEXT PRIMARY KEY,
        topic TEXT NOT NULL,
        title TEXT NOT NULL,
        source_url TEXT NOT NULL,
        domain TEXT NOT NULL,
        summary TEXT NOT NULL,
        raw_snippet TEXT,
        tier_level INTEGER DEFAULT 4,
        vault_path TEXT,
        fetched_at INTEGER,
        status TEXT DEFAULT 'quarantined' CHECK(status IN ('quarantined', 'verified', 'archived', 'rejected'))
    );
"""

_INDEXES_ARCHIVE = (
    "CREATE INDEX IF NOT EXISTS idx_superseded_archive_cat ON synapse_superseded_archive(category);",
    "CREATE INDEX IF NOT EXISTS idx_superseded_archive_rollup ON synapse_superseded_archive(rollup_id);",
)

_INDEXES_SYNAPSE_NODES = (
    "CREATE INDEX IF NOT EXISTS idx_synapse_nodes_category ON synapse_nodes(category);",
    "CREATE INDEX IF NOT EXISTS idx_synapse_nodes_status ON synapse_nodes(status);",
    "CREATE INDEX IF NOT EXISTS idx_synapse_nodes_last_accessed ON synapse_nodes(last_accessed);",
    "CREATE INDEX IF NOT EXISTS idx_synapse_nodes_salience ON synapse_nodes(salience);",
    "CREATE INDEX IF NOT EXISTS idx_synapse_nodes_access_count ON synapse_nodes(access_count);",
)

_INDEXES_SYNAPSE_EDGES = (
    "CREATE INDEX IF NOT EXISTS idx_synapse_edges_target ON synapse_edges(target_id);",
)

_INDEXES_EPISODES = (
    "CREATE INDEX IF NOT EXISTS idx_episodes_last_accessed ON episodes(last_accessed);",
    "CREATE INDEX IF NOT EXISTS idx_episodes_salience ON episodes(salience);",
)

_INDEXES_SYNAPSE_VECTORS = (
    "CREATE INDEX IF NOT EXISTS idx_synapse_vectors_updated_at ON synapse_vectors(updated_at);",
)

_INDEXES_MENTAL_MODELS = (
    "CREATE INDEX IF NOT EXISTS idx_synapse_mental_models_last_refreshed ON synapse_mental_models(last_refreshed_at);",
)

_INDEXES_OBSERVATIONS = (
    "CREATE INDEX IF NOT EXISTS idx_synapse_observations_status ON synapse_observations(status);",
    "CREATE INDEX IF NOT EXISTS idx_synapse_observations_proof_count ON synapse_observations(proof_count);",
)

_INDEXES_EXTERNAL_RESEARCH = (
    "CREATE INDEX IF NOT EXISTS idx_synapse_external_research_topic ON synapse_external_research(topic);",
    "CREATE INDEX IF NOT EXISTS idx_synapse_external_research_status ON synapse_external_research(status);",
)

_TRIGGERS_NODE_FTS = (
    """
    CREATE TRIGGER IF NOT EXISTS synapse_nodes_ai AFTER INSERT ON synapse_nodes BEGIN
        INSERT INTO synapse_fts (id, l0_abstract, l1_overview)
        VALUES (new.id, new.l0_abstract, coalesce(new.l1_overview, ''));
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS synapse_nodes_ad AFTER DELETE ON synapse_nodes BEGIN
        DELETE FROM synapse_fts WHERE id = old.id;
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS synapse_nodes_au AFTER UPDATE ON synapse_nodes BEGIN
        DELETE FROM synapse_fts WHERE id = old.id;
        INSERT INTO synapse_fts (id, l0_abstract, l1_overview)
        VALUES (new.id, new.l0_abstract, coalesce(new.l1_overview, ''));
    END;
    """,
)

_TRIGGERS_EPISODE_FTS = (
    """
    CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
        INSERT INTO episodes_fts (id, title, situation, action_taken, outcome, lessons_learned, tags)
        VALUES (
            new.id,
            coalesce(new.title, ''),
            coalesce(new.situation, ''),
            coalesce(new.action_taken, ''),
            coalesce(new.outcome, ''),
            coalesce(new.lessons_learned, ''),
            coalesce(new.tags, '')
        );
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
        DELETE FROM episodes_fts WHERE id = old.id;
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE ON episodes BEGIN
        DELETE FROM episodes_fts WHERE id = old.id;
        INSERT INTO episodes_fts (id, title, situation, action_taken, outcome, lessons_learned, tags)
        VALUES (
            new.id,
            coalesce(new.title, ''),
            coalesce(new.situation, ''),
            coalesce(new.action_taken, ''),
            coalesce(new.outcome, ''),
            coalesce(new.lessons_learned, ''),
            coalesce(new.tags, '')
        );
    END;
    """,
)

_TRIGGERS_VECTOR_VERSION = (
    """
    CREATE TRIGGER IF NOT EXISTS synapse_vectors_version_ai AFTER INSERT ON synapse_vectors BEGIN
        UPDATE _synapse_vector_meta SET version = version + 1 WHERE id = 1;
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS synapse_vectors_version_au AFTER UPDATE ON synapse_vectors BEGIN
        UPDATE _synapse_vector_meta SET version = version + 1 WHERE id = 1;
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS synapse_vectors_version_ad AFTER DELETE ON synapse_vectors BEGIN
        UPDATE _synapse_vector_meta SET version = version + 1 WHERE id = 1;
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS synapse_nodes_version_ad AFTER DELETE ON synapse_nodes BEGIN
        UPDATE _synapse_vector_meta SET version = version + 1 WHERE id = 1;
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS synapse_nodes_version_au_status AFTER UPDATE OF status ON synapse_nodes BEGIN
        UPDATE _synapse_vector_meta SET version = version + 1 WHERE id = 1;
    END;
    """,
)


class SchemaMixin:
    """Creates the schema and migrates databases written by older releases."""

    @contextmanager
    def _write(self) -> Iterator[Any]:
        """Run a write block under the lock and roll back if it raises.

        Rollback happens only at the outermost ``_write`` level so a nested
        failure cannot discard uncommitted DML performed by an outer scope.
        """
        depth = getattr(self, "_write_depth", 0)
        self._write_depth = depth + 1
        try:
            with self._lock:
                try:
                    yield self._conn
                except BaseException:
                    if depth == 0:
                        self._conn.rollback()
                    raise
        finally:
            self._write_depth = depth

    def _configure_pragmas(self) -> None:
        """Apply performance and integrity pragmas."""
        with self._lock:
            if not self.is_memory:
                self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            # Stay compatible with concurrent writers holding the WAL briefly.
            self._conn.execute("PRAGMA busy_timeout=10000;")

    def _init_schema(self) -> None:
        """Create tables, indexes, FTS5 virtual tables, and sync triggers."""
        with self._lock:
            cursor = self._conn.cursor()
            self._create_node_tables(cursor)
            self._create_edge_tables(cursor)
            self._create_node_fts(cursor)
            self._create_episode_tables(cursor)
            self._create_episode_fts(cursor)
            self._create_vector_tables(cursor)
            self._create_knowledge_tables(cursor)
            self._conn.commit()

    def _create_node_tables(self, cursor: Any) -> None:
        """Nodes, superseded archive, their indexes, and legacy migrations."""
        cursor.execute(_DDL_SYNAPSE_NODES)

        # Auto-migrate synapse_nodes if columns are missing in existing databases
        cursor.execute("PRAGMA table_info(synapse_nodes);")
        existing_cols = {row[1] for row in cursor.fetchall()}
        if "salience" not in existing_cols:
            cursor.execute("ALTER TABLE synapse_nodes ADD COLUMN salience REAL DEFAULT 0.5;")
        if "confidence" not in existing_cols:
            cursor.execute("ALTER TABLE synapse_nodes ADD COLUMN confidence REAL DEFAULT 1.0;")
        if "access_count" not in existing_cols:
            cursor.execute("ALTER TABLE synapse_nodes ADD COLUMN access_count INTEGER DEFAULT 0;")

        # 1b. Synapse Superseded Archive (Abstract Rollup historical storage)
        cursor.execute(_DDL_SYNAPSE_SUPERSEDED_ARCHIVE)
        for statement in _INDEXES_ARCHIVE:
            cursor.execute(statement)

        self._rebuild_obsolete_nodes_table(cursor)

        for statement in _INDEXES_SYNAPSE_NODES:
            cursor.execute(statement)

    def _rebuild_obsolete_nodes_table(self, cursor: Any) -> None:
        """Drop the legacy category CHECK constraint and accept 'flagged' status."""
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='synapse_nodes';")
        node_table_row = cursor.fetchone()
        if not node_table_row or not node_table_row[0]:
            return
        table_sql = node_table_row[0]
        if "CHECK(category IN" not in table_sql and "'flagged'" in table_sql:
            return
        cursor.execute("PRAGMA legacy_alter_table=ON;")
        cursor.execute("PRAGMA foreign_keys=OFF;")
        try:
            cursor.execute("ALTER TABLE synapse_nodes RENAME TO _synapse_nodes_old;")
            cursor.execute(_DDL_SYNAPSE_NODES_REBUILT)
            cursor.execute("""
                INSERT INTO synapse_nodes (
                    id, uri, l0_abstract, l1_overview, l2_target_path,
                    category, importance, strength, created_at, last_accessed,
                    status, salience, confidence, access_count
                )
                SELECT
                    id, uri, l0_abstract, l1_overview, l2_target_path,
                    category, importance, strength, created_at, last_accessed,
                    status, salience, confidence, access_count
                FROM _synapse_nodes_old;
            """)
            cursor.execute("DROP TABLE _synapse_nodes_old;")
        except Exception:
            self._conn.rollback()
            raise
        else:
            # The row copy opens an implicit transaction and PRAGMA foreign_keys
            # is a no-op inside one, so commit first or enforcement stays off.
            self._conn.commit()
        finally:
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.execute("PRAGMA legacy_alter_table=OFF;")

    def _create_edge_tables(self, cursor: Any) -> None:
        """Edges table and its indexes."""
        # 2. Synapse Edges table
        cursor.execute(_DDL_SYNAPSE_EDGES)

        cursor.execute("PRAGMA table_info(synapse_edges);")
        existing_edge_cols = {row[1] for row in cursor.fetchall()}
        if "access_count" not in existing_edge_cols:
            cursor.execute("ALTER TABLE synapse_edges ADD COLUMN access_count INTEGER DEFAULT 0;")

        for statement in _INDEXES_SYNAPSE_EDGES:
            cursor.execute(statement)

    def _create_node_fts(self, cursor: Any) -> None:
        """Node full-text index and its synchronisation triggers."""
        # 3. FTS5 Virtual Table for full-text search (Nodes)
        cursor.execute(_DDL_SYNAPSE_FTS)

        # 4. Triggers to keep synapse_fts synchronized with synapse_nodes
        for statement in _TRIGGERS_NODE_FTS:
            cursor.execute(statement)

    def _create_episode_tables(self, cursor: Any) -> None:
        """Episodic memory table and its indexes."""
        # 5. Episodes Table (Episodic Memory Cognitive Architecture)
        cursor.execute(_DDL_EPISODES)

        for statement in _INDEXES_EPISODES:
            cursor.execute(statement)

    def _create_episode_fts(self, cursor: Any) -> None:
        """Episode full-text index and its synchronisation triggers."""
        # 6. FTS5 Virtual Table for Episodes
        cursor.execute(_DDL_EPISODES_FTS)

        # 7. Triggers to keep episodes_fts synchronized with episodes
        for statement in _TRIGGERS_EPISODE_FTS:
            cursor.execute(statement)

    def _create_vector_tables(self, cursor: Any) -> None:
        """Vectors, cache version metadata, and version-bump triggers."""
        # 8. Synapse Vectors table
        cursor.execute(_DDL_SYNAPSE_VECTORS)

        for statement in _INDEXES_SYNAPSE_VECTORS:
            cursor.execute(statement)

        # 8b. Vector cache metadata table and change-tracking triggers
        cursor.execute(_DDL_VECTOR_META)
        cursor.execute("INSERT OR IGNORE INTO _synapse_vector_meta (id, version) VALUES (1, 0);")

        for statement in _TRIGGERS_VECTOR_VERSION:
            cursor.execute(statement)

    def _create_knowledge_tables(self, cursor: Any) -> None:
        """Mental models, observations, and sandboxed external research."""
        # 9. Synapse Mental Models table (Hierarchical Consolidated Understanding)
        cursor.execute(_DDL_MENTAL_MODELS)

        for statement in _INDEXES_MENTAL_MODELS:
            cursor.execute(statement)

        # 10. Synapse Observations table (Tier 2 Evidence-Grounded Pattern Consolidation)
        cursor.execute(_DDL_OBSERVATIONS)

        for statement in _INDEXES_OBSERVATIONS:
            cursor.execute(statement)

        # 11. Synapse External Research table (Tier 4 Sandboxed Autonomous Research)
        cursor.execute(_DDL_EXTERNAL_RESEARCH)

        for statement in _INDEXES_EXTERNAL_RESEARCH:
            cursor.execute(statement)
