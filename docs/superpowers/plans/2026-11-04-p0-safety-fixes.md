# P0 Safety Fixes Implementation Plan

> **For the executing engineer:** execute tasks in order, TDD-first. Every task
> starts with a test you watch fail. No production code before its failing test.

**Goal:** Remove the verified P0 data-loss and protocol-failure paths from
`synapse-cortex`: false-positive belief revision, writes on the read path,
open transactions after failed writes, non-atomic legacy migration, prune
deleting superseded rows before archival, and MCP crashes/misleading success
reports.

**Spec:** the audit conversation of 2026-11-04, reproduced by the probes in
`%LOCALAPPDATA%\Temp\opencode\probe_synapse.py`, `probe2.py`, `probe_mcp.py`,
`probe_unicode.py`. This plan is the authority for exact names and tests.

**Out of scope:** every P1/P2 item from the audit (vault path containment,
embedding provenance, harvester SSRF/quarantine, incremental cache, `.env`,
token estimation, NULL edge weight, CI, schema versioning, documentation
accuracy). Do not touch them here.

## Global Constraints

- `pyproject.toml` stays `dependencies = []`; only stdlib in the implementation.
- Python 3.9 compatible: no `match`, no runtime PEP 604 unions, no
  `list[str]`/`dict[str, int]` at runtime. Use `typing` names.
- English-only code and docs, no emojis.
- Never access `storage._conn`/`storage._lock` from production modules outside
  `synapse/storage.py`. Tests may inspect `storage._conn`; production mixins use
  `self._conn`.
- Add storage behaviour to the mixin that owns the concern. Facade
  (`synapse/storage.py`) gets no new feature methods.
- Full suite command: `python -m pytest`. Per-task gate: run the task's test
  file, then the full suite before the task commit.
- Every task ends with one conventional commit. Do not squash tasks.
- Redirect long pytest output to a file in the workspace and read its tail.

## Interfaces (cross-task)

- `synapse.contradiction_resolver.extract_numeric_specs(text) -> Dict[str, Tuple[float, str]]`
  keeps its signature and value shape.
- New `synapse.contradiction_resolver.extract_subject_ids(text) -> Set[str]`
  returns `"<noun>:<number>"` identifiers. Consumed only inside
  `ContradictionResolver.detect_claim_conflict`.
- New `SchemaMixin._write()` is a context manager yielding `self._conn`, taking
  `self._lock`, and rolling back on exception. Every write method in
  `graph.py`, `vectors.py`, `episodes.py`, `knowledge.py`, `maintenance.py`
  consumes it. `_recover_interrupted_node_rebuild()` is private to
  `SchemaMixin` and runs from `_rebuild_obsolete_nodes_table`.
- `synapse.mcp` exposes `get_adapter()`, `SERVER_VERSION`, and `main()`.
  `tests/test_mcp_protocol.py` drives it as a subprocess.

---

## Task 1: Bind numeric specs to the right attribute and require subject identity

**Files:** `synapse/contradiction_resolver.py`,
`tests/test_contradiction_resolver.py`

**Problem:** `extract_numeric_specs("Project Alpha service 7 uses port 8007 with
timeout 30s.")` returns `{"port": 7.0, "timeout": 8007.0}` because an
`N <word> <attr>` pattern accepts any word as a unit and an attribute pattern
jumps over arbitrary words. Twenty distinct numeric facts therefore look like
one contradicting fact and 19 of 20 are superseded.

### Step 1 (RED): add regression tests

Append to `tests/test_contradiction_resolver.py`:

```python
def test_extract_numeric_specs_binds_values_to_their_own_attribute() -> None:
    specs = extract_numeric_specs(
        "Project Alpha service 7 uses port 8007 with timeout 30s."
    )
    assert specs["port"] == (8007.0, "")
    assert specs["timeout"] == (30.0, "s")
    assert "service" not in specs

    specs2 = extract_numeric_specs("Service 3 runs on port 9003 with timeout 45s.")
    assert specs2["port"] == (9003.0, "")
    assert specs2["timeout"] == (45.0, "s")


def test_numeric_facts_for_distinct_services_are_not_contradictions() -> None:
    adapter = SynapseAdapter({"db_path": ":memory:"})
    try:
        for i in range(20):
            adapter.retain(
                f"Project Alpha service {i} uses port {8000 + i} "
                f"and has timeout {30 + i}s.",
                category="architecture",
                salience=0.5,
            )
        statuses = adapter.storage.get_stats()["statuses"]
        assert statuses.get("active") == 20
        assert statuses.get("superseded", 0) == 0
    finally:
        adapter.storage.close()


def test_same_service_spec_change_still_supersedes() -> None:
    adapter = SynapseAdapter({"db_path": ":memory:"})
    try:
        adapter.retain(
            "Service 7 uses port 8007 with timeout 30s.",
            category="architecture",
            salience=0.5,
        )
        adapter.retain(
            "Service 7 uses port 9007 with timeout 45s.",
            category="architecture",
            salience=0.5,
        )
        assert len(adapter.storage.list_nodes(status="active")) == 1
        assert len(adapter.storage.list_nodes(status="superseded")) == 1
    finally:
        adapter.storage.close()
```

**Run:** `python -m pytest tests/test_contradiction_resolver.py -q`
**Expected:** the two new numeric tests FAIL (first yields `port == 7.0` and
`timeout == 8007.0`; second yields active 1 / superseded 19). The same-service
test may already pass; keep it as the false-negative guard.

### Step 2 (GREEN): rewrite `extract_numeric_specs`

Replace the body of `extract_numeric_specs` (lines 151-218) with four ordered
passes. First match per canonical attribute wins.

1. **Explicit `key: value` / `key = value`** (highest confidence). Capture
   `([a-zA-Z_][a-zA-Z0-9_]{2,20})\s*(?::|=)\s*(\d+(?:\.\d+)?)\s*([a-zA-Z%]+)?`.
   Skip keys in `STOPWORDS` and keys in `INSTANCE_NOUNS` (they are subject
   identifiers, not specs).
2. **Attribute then number, with only a closed bridge vocabulary.** Pattern:

   ```python
   _BRIDGE = (
       r"(?:of|is|are|was|were|adalah|sebesar|sebanyak|sekitar|about|"
       r"approximately|capacity|size|limit|value|total|max|maximum|min|minimum|"
       r"api|http|grpc)"
   )
   pat = (
       r"\b" + re.escape(attr) + r"\b"
       r"(?:\s+" + _BRIDGE + r"\b){0,3}"
       r"\s*(?::|=)?\s*(?<![\w\-])(\d+(?:\.\d+)?)\s*([a-zA-Z%]+)?\b"
   )
   ```

   This accepts `port 8080`, `RAM is 4GB`, `Disk capacity of 100GB`,
   `timeout sebesar 30s`; it rejects `7 uses port` and `8007 with timeout`
   because `uses`/`with` are not bridge words.
3. **Number + known unit then attribute** (`8 GB RAM`, `4 core CPU`,
   `30s timeout`). Require `raw_unit in KNOWN_UNITS or raw_unit in
   UNIT_NORMALIZATION`; otherwise do not bind. This is the pass that rejects
   `7 uses port` and `8007 with timeout`.
4. **Technology version** (`PostgreSQL 14`, `Python 3.11`): keep the existing
   pass 4 unchanged.

Do not change `UNIT_NORMALIZATION`, `KNOWN_UNITS`, or `SPEC_ATTRIBUTES` values.

### Step 3 (GREEN): add subject-identity guard

Add near the vocabulary constants:

```python
# Countable instances whose trailing number identifies the subject, not a spec.
INSTANCE_NOUNS: Set[str] = {
    "service", "server", "node", "instance", "cluster", "shard", "pod",
    "container", "project", "tenant", "account", "environment", "env",
    "host", "vps", "machine", "device", "region", "replica", "worker",
    "thread", "process",
}

_SUBJECT_ID_RE = re.compile(
    r"\b(" + "|".join(sorted(INSTANCE_NOUNS)) + r")\s*[:#]?\s*(\d+)\b",
    re.IGNORECASE,
)


def extract_subject_ids(text: str) -> Set[str]:
    """Return identifiers such as 'service:7' that name a countable subject."""
    if not text:
        return set()
    return {
        f"{m.group(1).lower()}:{m.group(2)}"
        for m in _SUBJECT_ID_RE.finditer(text)
    }
```

At the top of `detect_claim_conflict`, after `text_a`/`text_b` are built:

```python
ids_a = extract_subject_ids(text_a)
ids_b = extract_subject_ids(text_b)
if ids_a and ids_b and ids_a.isdisjoint(ids_b):
    # Different services/servers/nodes are different subjects, so differing
    # numbers are not a revision of the same belief.
    return None
```

This must not suppress `Service 7 ... 8007` vs `Service 7 ... 9007`
(identical id set) nor version changes (`postgresql` is not an instance noun).

### Step 4: verify

```powershell
python -m pytest tests/test_contradiction_resolver.py -q
python -m pytest -q
```

**Expected:** all contradiction tests pass; full suite green (318 + new tests).

**Commit:** `fix(contradiction): bind numeric specs and require subject identity`

---

## Task 2: Make retrieval read-only for belief state

**Files:** `synapse/retriever.py`, `tests/test_synapse_retriever.py`

**Problem:** `retrieve(..., touch=False)` calls
`supersede_low_salience_nodes` at `retriever.py:1024-1032`, so a read can
permanently supersede a node using a weak heuristic.

### Step 1 (RED): invert the existing test

Replace `test_contradiction_resolution_under_storage_lock` (around line 401)
with:

```python
def test_retrieval_does_not_mutate_belief_state(self, mem_storage):
    """Retrieval may hide a contradiction in the response, never in storage."""
    n1 = mem_storage.add_node(
        uri="synapse://decision/postgres_use",
        l0_abstract="Always prefer PostgreSQL for production relational persistence.",
        category="decision",
        salience=0.85,
        confidence=0.95,
    )
    n2 = mem_storage.add_node(
        uri="synapse://decision/postgres_avoid",
        l0_abstract="Never use PostgreSQL, avoid postgres in all services.",
        category="decision",
        salience=0.50,
        confidence=0.50,
    )

    retriever = SynapseRetriever(storage=mem_storage)
    results = retriever.retrieve("PostgreSQL database preference", limit=5, touch=False)

    assert mem_storage.get_node(n1)["status"] == "active"
    assert mem_storage.get_node(n2)["status"] == "active"
    assert n2 not in [item["id"] for item in results]

    retriever.retrieve("PostgreSQL database preference", limit=5, touch=True)
    assert mem_storage.get_node(n2)["status"] == "active"
```

**Run:** `python -m pytest tests/test_synapse_retriever.py -q`
**Expected:** FAIL on `n2["status"] == "active"` because retrieval superseded it.

### Step 2 (GREEN): remove the storage write

In `retriever.py` Step 6.5, delete only the `supersede_low_salience_nodes`
call (and its nested `try/except`). Keep the in-memory `loser_ids` filter and
keep the outer `try/except` so a resolver failure still degrades to an
unfiltered result set. `retrieve()` must not call any storage method that
changes node status; `touch` keeps its current reinforcement behaviour.

### Step 3: verify

```powershell
python -m pytest tests/test_synapse_retriever.py -q
python -m pytest -q
```

**Expected:** all pass; no test asserts read-path supersede afterwards.

**Commit:** `fix(retriever): stop superseding nodes during retrieval`

---

## Task 3: Add a rollback-on-failure write helper and use it everywhere

**Files:** `synapse/stores/schema.py`, `synapse/stores/graph.py`,
`synapse/stores/vectors.py`, `synapse/stores/episodes.py`,
`synapse/stores/knowledge.py`, `synapse/stores/maintenance.py`,
`synapse/storage.py`, `tests/test_storage_transactions.py` (new)

**Problem:** a failed `INSERT` (duplicate URI, FK violation) raises while the
implicit transaction stays open. A second connection then blocks for the full
`busy_timeout` (measured 11.27 s) and fails with `database is locked`.

### Step 1 (RED): create `tests/test_storage_transactions.py`

```python
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
```

**Run:** `python -m pytest tests/test_storage_transactions.py -q`
**Expected:** FAIL with `assert True is False` (`in_transaction` is True);
the duplicate-episode test additionally proves the same leak outside
`add_node`.

### Step 2 (GREEN): add `_write` to `SchemaMixin`

In `synapse/stores/schema.py` add `from contextlib import contextmanager` and
`Iterator` to the typing import, then add the helper below. Rollback happens
only at the outermost `_write` level: an inner failure propagates through
nested calls, and an unguarded inner rollback would discard uncommitted DML
performed by an outer scope. A depth counter tracks nesting:

```python
    @contextmanager
    def _write(self) -> Iterator[Any]:
        """Run a write block under the lock and roll back if it raises."""
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
```

### Step 3 (GREEN): consume `_write` in every write method

Replace `with self._lock:` with `with self._write():` in exactly these
methods (commit calls stay where they are):

- `graph.py`: `add_node`, `get_node`, `record_node_access`, `update_node`,
  `delete_node`, `add_edge`, `prune_decayed_edges`
- `episodes.py`: `add_episode`, `get_episode`, `update_episode`,
  `delete_episode`
- `vectors.py`: `upsert_vector`; also add `self._conn.rollback()` inside its
  `except sqlite3.IntegrityError` before `return False`
- `knowledge.py`: `set_mental_model`, `add_or_update_observation`,
  `add_external_research`, `promote_external_research`
- `maintenance.py`: `lock_critical_nodes`, `apply_hebbian_decay`,
  `prune_exhausted_nodes`, `prune_weak_edges`, `upsert_vault_node`,
  `supersede_low_salience_nodes`, `commit_superseded_rollup`

In `graph.py` `record_node_access` and `add_edge`, the existing
`except Exception`/`except sqlite3.IntegrityError` blocks swallow the
exception before `_write` can see it, so add `self._conn.rollback()` as the
first line of each handler. Do not change their return values.

Do **not** wrap `_init_schema`, `_configure_pragmas`,
`_rebuild_obsolete_nodes_table`, `optimize_and_compact`, or
`optimize_fts_indices`; they manage their own transactions/pragmas.

### Step 4 (GREEN): close the connection when init fails

In `synapse/storage.py.__init__`, wrap the two init calls:

```python
        try:
            self._configure_pragmas()
            self._init_schema()
        except BaseException:
            self._conn.close()
            raise
```

This prevents a leaked open connection when a migration raises.

### Step 5: verify

```powershell
python -m pytest tests/test_storage_transactions.py -q
python -m pytest -q
```

**Expected:** all pass. If `test_storage_boundaries.py` flags `_write`, the
extra private method on `SchemaMixin` is allowed (the test only pins a subset),
so investigate only duplicate-definition failures.

**Commit:** `fix(storage): roll back failed writes and close on init failure`

---

## Task 4: Make the legacy node rebuild atomic and recover broken databases

**Files:** `synapse/stores/schema.py`, `tests/test_schema_contract.py`

**Problem:** `ALTER TABLE ... RENAME` and `CREATE TABLE` auto-commit, so a
crash between them leaves an empty live table plus `_synapse_nodes_old`. The
guard then treats the schema as migrated and never repairs it.

### Step 1 (RED): add two tests to `tests/test_schema_contract.py`

```python
def test_failed_legacy_rebuild_keeps_rows_and_old_table(tmp_path, monkeypatch):
    """A failure inside the rebuild must roll back rename, create, and copy."""
    import synapse.stores.schema as schema_module

    db_file = tmp_path / "legacy-atomic.db"
    _write_legacy_db(db_file)
    monkeypatch.setattr(
        schema_module, "_DDL_SYNAPSE_NODES_REBUILT", "CREATE TABLE broken ("
    )

    with pytest.raises(sqlite3.OperationalError):
        SynapseStorage(db_path=str(db_file), allow_custom_categories=True)

    check = sqlite3.connect(str(db_file))
    try:
        assert check.execute("SELECT COUNT(*) FROM synapse_nodes").fetchone()[0] == 1
        assert check.execute(
            "SELECT name FROM sqlite_master WHERE name = '_synapse_nodes_old'"
        ).fetchone() is None
    finally:
        check.close()


def test_startup_recovers_interrupted_legacy_rebuild(tmp_path):
    """A DB left mid-rename must restore its rows instead of exposing an empty table."""
    from synapse.stores.schema import _DDL_SYNAPSE_NODES_REBUILT

    db_file = tmp_path / "legacy-interrupted.db"
    _write_legacy_db(db_file)

    broken = sqlite3.connect(str(db_file))
    try:
        broken.execute("ALTER TABLE synapse_nodes RENAME TO _synapse_nodes_old")
        broken.execute(_DDL_SYNAPSE_NODES_REBUILT)
        broken.commit()
    finally:
        broken.close()

    storage = SynapseStorage(db_path=str(db_file), allow_custom_categories=True)
    try:
        node = storage.get_node("legacy-1")
        assert node is not None
        assert node["uri"] == "synapse://legacy/one"
        assert storage.get_stats()["node_count"] == 1
        assert storage._conn.execute(
            "SELECT name FROM sqlite_master WHERE name = '_synapse_nodes_old'"
        ).fetchone() is None
    finally:
        storage.close()
```

**Run:** `python -m pytest tests/test_schema_contract.py -q`
**Expected:** both new tests FAIL: the first leaves 0 live rows or an
`_synapse_nodes_old` table; the second reports 0 nodes.

### Step 2 (GREEN): extract the column-ensure helper

In `_create_node_tables`, replace the four `PRAGMA table_info` + `ALTER TABLE`
blocks with a call to a new private helper, preserving the same columns:

```python
    def _ensure_node_columns(self, cursor: Any) -> None:
        """Add columns introduced after the original schema."""
        cursor.execute("PRAGMA table_info(synapse_nodes);")
        existing = {row[1] for row in cursor.fetchall()}
        additions = {
            "salience": "REAL DEFAULT 0.5",
            "confidence": "REAL DEFAULT 1.0",
            "access_count": "INTEGER DEFAULT 0",
        }
        for column, ddl in additions.items():
            if column not in existing:
                cursor.execute(f"ALTER TABLE synapse_nodes ADD COLUMN {column} {ddl};")
```

Keep any other legacy columns already handled there; do not drop them.
Add `_ensure_node_columns` to the helper set in
`tests/test_storage_boundaries.py` only if you prefer pinning it; the test's
subset check passes without editing.

### Step 3 (GREEN): add recovery and make the rebuild atomic

Add to `SchemaMixin`:

```python
    def _recover_interrupted_node_rebuild(self) -> None:
        """Repair a DB left mid-rebuild by an older, non-atomic migration."""
        old = self._conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='_synapse_nodes_old'"
        ).fetchone()
        if not old:
            return

        live = self._conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='synapse_nodes'"
        ).fetchone()
        live_rows = 0
        if live:
            live_rows = self._conn.execute(
                "SELECT COUNT(*) FROM synapse_nodes"
            ).fetchone()[0]
        old_rows = self._conn.execute(
            "SELECT COUNT(*) FROM _synapse_nodes_old"
        ).fetchone()[0]

        logger.warning(
            "Recovering interrupted synapse_nodes rebuild (live=%s, old=%s)",
            live_rows,
            old_rows,
        )
        self._conn.execute("PRAGMA foreign_keys=OFF;")
        if self._conn.in_transaction:
            self._conn.commit()
        self._conn.execute("BEGIN IMMEDIATE")
        restored = False
        try:
            if live and live_rows == 0 and old_rows > 0:
                self._conn.execute("DROP TABLE synapse_nodes;")
                live = None
            if not live:
                self._conn.execute(
                    "ALTER TABLE _synapse_nodes_old RENAME TO synapse_nodes;"
                )
                restored = True
            else:
                live_cols = {
                    row[1]
                    for row in self._conn.execute("PRAGMA table_info(synapse_nodes)")
                }
                old_cols = {
                    row[1]
                    for row in self._conn.execute(
                        "PRAGMA table_info(_synapse_nodes_old)"
                    )
                }
                shared = [
                    column
                    for column in (
                        "id", "uri", "l0_abstract", "l1_overview",
                        "l2_target_path", "category", "importance", "strength",
                        "created_at", "last_accessed", "status", "salience",
                        "confidence", "access_count",
                    )
                    if column in live_cols and column in old_cols
                ]
                columns = ", ".join(shared)
                self._conn.execute(
                    f"INSERT OR IGNORE INTO synapse_nodes ({columns}) "
                    f"SELECT {columns} FROM _synapse_nodes_old;"
                )
                self._conn.execute("DROP TABLE _synapse_nodes_old;")
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        finally:
            self._conn.execute("PRAGMA foreign_keys=ON;")

        if restored:
            self._ensure_node_columns(self._conn.cursor())
```

Add `import logging` and `logger = logging.getLogger(__name__)` to schema.py
if not already present.

Change `_rebuild_obsolete_nodes_table` to:

```python
    def _rebuild_obsolete_nodes_table(self, cursor: Any) -> None:
        """Drop the legacy category CHECK constraint and accept 'flagged' status."""
        self._recover_interrupted_node_rebuild()

        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='synapse_nodes';"
        )
        node_table_row = cursor.fetchone()
        if not node_table_row or not node_table_row[0]:
            return
        table_sql = node_table_row[0]
        if "CHECK(category IN" not in table_sql:
            return
        cursor.execute("PRAGMA legacy_alter_table=ON;")
        cursor.execute("PRAGMA foreign_keys=OFF;")
        if self._conn.in_transaction:
            self._conn.commit()
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            cursor.execute("ALTER TABLE synapse_nodes RENAME TO _synapse_nodes_old;")
            cursor.execute(_DDL_SYNAPSE_NODES_REBUILT)
            cursor.execute(
                """
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
                """
            )
            cursor.execute("DROP TABLE _synapse_nodes_old;")
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        finally:
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.execute("PRAGMA legacy_alter_table=OFF;")
```

Note: dropping the `and "'flagged'" in table_sql` half of the old guard is
intentional; a table that lacks the CHECK cannot be the legacy shape. Keep the
existing `test_rebuild_restores_pragmas_when_the_copy_fails` passing: recovery
uses `self._conn`, so the fake cursor is untouched, and the pragma sequence at
the end is unchanged.

### Step 4: verify

```powershell
python -m pytest tests/test_schema_contract.py -q
python -m pytest -q
```

**Expected:** all schema tests pass; full suite green.

**Commit:** `fix(schema): make legacy node migration atomic and recoverable`

---

## Task 5: Stop pruning superseded nodes before archival

**Files:** `synapse/stores/maintenance.py`, `tests/test_prune_safety.py` (new)

**Problem:** `prune_exhausted_nodes` deletes any node with low salience and
floor strength regardless of status. A superseded node with salience 0.2 is
deleted before `rollup_superseded_nodes` can archive it.

### Step 1 (RED): create `tests/test_prune_safety.py`

```python
"""Prune must never delete a superseded row before the rollup archives it."""

from __future__ import annotations

from synapse.storage import SynapseStorage


def test_prune_leaves_superseded_rows_for_the_archive():
    storage = SynapseStorage(":memory:")
    try:
        superseded = storage.add_node(
            uri="synapse://prune/old",
            l0_abstract="Old low-salience fact",
            category="general",
            salience=0.2,
        )
        storage.supersede_node(superseded)
        assert len(storage.list_superseded_nodes()) == 1

        assert storage.prune_exhausted_nodes() == 0
        assert len(storage.list_superseded_nodes()) == 1
        assert storage._conn.execute(
            "SELECT COUNT(*) FROM synapse_superseded_archive"
        ).fetchone()[0] == 0
    finally:
        storage.close()


def test_prune_still_deletes_exhausted_active_rows():
    storage = SynapseStorage(":memory:")
    try:
        exhausted = storage.add_node(
            uri="synapse://prune/active",
            l0_abstract="Exhausted active fact",
            category="general",
            salience=0.2,
        )
        storage.update_node(exhausted, strength=0.05)
        assert storage.prune_exhausted_nodes() == 1
        assert storage.get_node(exhausted) is None
    finally:
        storage.close()
```

**Run:** `python -m pytest tests/test_prune_safety.py -q`
**Expected:** first test FAILS (`prune_exhausted_nodes() == 1` and the
superseded row disappears); second passes.

### Step 2 (GREEN): filter by status

Add `AND status = 'active'` to the `prune_exhausted_nodes` `DELETE`. Do not
change the salience/strength thresholds or protected categories.

### Step 3: verify

```powershell
python -m pytest tests/test_prune_safety.py -q
python -m pytest -q
```

**Expected:** all pass, including the existing nightly-consolidation and
rollup tests.

**Commit:** `fix(maintenance): keep superseded nodes until archival`

---

## Task 6: Harden the MCP server and add protocol tests

**Files:** `synapse/mcp.py`, `tests/test_mcp_protocol.py` (new),
`pyproject.toml`, `CHANGELOG.md`, `AGENTS.md`

**Problems:** array payload kills the process; non-ASCII output raises
`UnicodeEncodeError` under a cp1252 stdout; malformed JSON gets no `-32700`;
notifications are answered with `id: null`; a failed retain reports
`"Successfully retained ... False"` with no `isError`; import creates a DB and
log directory; `protocolVersion`/version/category enum drift.

### Step 1 (RED): create `tests/test_mcp_protocol.py`

```python
"""End-to-end JSON-RPC protocol tests for `python -m synapse.mcp`."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _env(tmp_path, db_path=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["SYNAPSE_DB_PATH"] = str(db_path or (tmp_path / "mcp.db"))
    env["SYNAPSE_LOGS_DIR"] = str(tmp_path / "logs")
    env.pop("PYTHONIOENCODING", None)
    env.pop("PYTHONUTF8", None)
    return env


def _run(tmp_path, lines, db_path=None):
    payload = "".join(
        (line if isinstance(line, str) else json.dumps(line)) + "\n"
        for line in lines
    )
    completed = subprocess.run(
        [sys.executable, "-m", "synapse.mcp"],
        input=payload.encode("utf-8"),
        capture_output=True,
        cwd=str(tmp_path),
        env=_env(tmp_path, db_path),
        timeout=120,
    )
    stdout = completed.stdout.decode("utf-8")
    responses = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    return completed, responses


def _request(req_id, method, params=None):
    body = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        body["params"] = params
    return body


def _call(req_id, name, arguments):
    return _request(req_id, "tools/call", {"name": name, "arguments": arguments})


def test_initialize_tools_and_metadata(tmp_path):
    completed, responses = _run(
        tmp_path,
        [
            _request(1, "initialize"),
            _request(2, "tools/list"),
            _request(3, "ping"),
        ],
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert [r["id"] for r in responses] == [1, 2, 3]

    init = responses[0]["result"]
    assert init["protocolVersion"] == "2024-11-05"
    assert init["serverInfo"]["name"] == "synapse-cortex"

    version_line = re.search(
        r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(encoding="utf-8"), re.M
    )
    assert init["serverInfo"]["version"] == version_line.group(1)

    tools = {tool["name"]: tool for tool in responses[1]["result"]["tools"]}
    assert {"synapse_recall", "synapse_retain", "synapse_get_stats"} <= set(tools)
    category = tools["synapse_retain"]["inputSchema"]["properties"]["category"]
    assert "general" in category["enum"]


def test_non_ascii_round_trip(tmp_path):
    db_path = tmp_path / "unicode.db"
    content = "caf\u00e9 \u2615 \u65e5\u672c\u8a9e"
    completed, responses = _run(
        tmp_path,
        [
            _call(1, "synapse_retain", {"content": content, "category": "general"}),
            _call(2, "synapse_recall", {"query": "caf\u00e9"}),
        ],
        db_path=db_path,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert "isError" not in responses[0]["result"]
    assert "caf\u00e9" in responses[1]["result"]["content"][0]["text"]

    connection = sqlite3.connect(str(db_path))
    try:
        stored = connection.execute(
            "SELECT l0_abstract FROM synapse_nodes"
        ).fetchone()[0]
    finally:
        connection.close()
    assert stored == content


def test_invalid_framing_notifications_and_recovery(tmp_path):
    completed, responses = _run(
        tmp_path,
        [
            "[1, 2, 3]",
            "{not json",
            {"jsonrpc": "2.0", "method": "tools/list"},
            _request(9, "ping"),
        ],
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert [r["id"] for r in responses] == [None, None, 9]
    assert responses[0]["error"]["code"] == -32600
    assert responses[1]["error"]["code"] == -32700
    assert responses[2]["result"] == {}


def test_failed_write_is_reported_as_error(tmp_path):
    completed, responses = _run(
        tmp_path,
        [_call(1, "synapse_retain", {"content": "hello", "category": "general"})],
        db_path=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    result = responses[0]["result"]
    assert result.get("isError") is True
    assert "Successfully" not in result["content"][0]["text"]


def test_import_has_no_filesystem_side_effects(tmp_path):
    home = tmp_path / "home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["USERPROFILE"] = str(home)
    env["HOME"] = str(home)
    env.pop("SYNAPSE_DB_PATH", None)
    env.pop("SYNAPSE_LOGS_DIR", None)
    completed = subprocess.run(
        [sys.executable, "-c", "import synapse.mcp"],
        capture_output=True,
        cwd=str(work),
        env=env,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert [p for p in home.rglob("*") if p.is_file()] == []
    assert [p for p in work.rglob("*") if p.is_file()] == []
```

**Run:** `python -m pytest tests/test_mcp_protocol.py -q`
**Expected:** all five FAIL in the documented ways (array exits 1, no `-32700`
response, `id: null` notification reply, `"Successfully ... False"`,
import side effects) plus the version/enum assertions.

### Step 2 (GREEN): rewrite `synapse/mcp.py` transport

Keep the tool schemas and `handle_call_tool` shape, with these changes.

1. Add `from synapse.stores.constants import VALID_CATEGORIES` and set
   `SERVER_VERSION = "1.0.5"`. Category enum becomes
   `sorted(VALID_CATEGORIES)`.
2. Replace import-time logging with:

   ```python
   def _configure_logging() -> None:
       log_dir = Path(
           os.environ.get("SYNAPSE_LOGS_DIR") or (Path.home() / ".synapse" / "logs")
       ).resolve()
       log_dir.mkdir(parents=True, exist_ok=True)
       logging.basicConfig(
           filename=str(log_dir / "mcp_server.log"),
           level=logging.INFO,
           format="%(asctime)s [%(levelname)s] %(message)s",
       )
   ```

   Keep `logger = logging.getLogger("synapse.mcp")` at module level.
3. Replace the import-time adapter with a lazy singleton:

   ```python
   _adapter: Optional[SynapseAdapter] = None


   def _resolve_db_path() -> str:
       local_db = Path("./data/synapse.db").resolve()
       default_db = (
           str(local_db)
           if local_db.exists()
           else str((Path.home() / ".synapse" / "synapse.db").resolve())
       )
       return os.environ.get("SYNAPSE_DB_PATH", default_db)


   def get_adapter() -> SynapseAdapter:
       global _adapter
       if _adapter is None:
           _adapter = SynapseAdapter({"db_path": _resolve_db_path()})
       return _adapter
   ```

   `handle_call_tool` starts with `adapter = get_adapter()`.
4. In `handle_call_tool`, validate and report failures instead of claiming
   success:
   - `synapse_recall`: empty/non-string query -> `isError`.
   - `synapse_retain`: empty content -> `isError`; if `res` is not `True` ->
     `isError` with `"Failed to retain..."`.
   - `synapse_record_episode`: empty title -> `isError`; if `res` is not
     truthy -> `isError`.
   - `synapse_get_stats`: if `adapter.storage is None` -> `isError`.
5. Add UTF-8 stdio configuration and JSON-RPC dispatch:

   ```python
   def _configure_stdio() -> None:
       for stream in (sys.stdin, sys.stdout):
           reconfigure = getattr(stream, "reconfigure", None)
           if reconfigure is not None:
               try:
                   reconfigure(encoding="utf-8", errors="strict")
               except Exception:
                   pass


   def _error(req_id, code, message):
       return {
           "jsonrpc": "2.0",
           "id": req_id,
           "error": {"code": code, "message": message},
       }


   def dispatch(req):
       if not isinstance(req, dict):
           return _error(None, -32600, "Invalid Request")
       req_id = req.get("id")
       is_notification = "id" not in req
       method = req.get("method")
       params = req.get("params") or {}
       if not method or not isinstance(params, dict):
           return _error(req_id, -32600, "Invalid Request")

       if method == "initialize":
           response = {
               "jsonrpc": "2.0",
               "id": req_id,
               "result": {
                   "protocolVersion": "2024-11-05",
                   "capabilities": {"tools": {}},
                   "serverInfo": {
                       "name": "synapse-cortex",
                       "version": SERVER_VERSION,
                   },
               },
           }
       elif method == "notifications/initialized":
           return None
       elif method == "tools/list":
           response = {"jsonrpc": "2.0", "id": req_id,
                       "result": {"tools": TOOLS}}
       elif method == "tools/call":
           tool_name = params.get("name", "")
           tool_args = params.get("arguments") or {}
           if not isinstance(tool_args, dict):
               return _error(req_id, -32602, "Invalid params")
           response = {"jsonrpc": "2.0", "id": req_id,
                       "result": handle_call_tool(tool_name, tool_args)}
       elif method == "ping":
           response = {"jsonrpc": "2.0", "id": req_id, "result": {}}
       else:
           if is_notification:
               return None
           return _error(req_id, -32601, f"Method not found: {method}")

       return None if is_notification else response
   ```

   `main()` calls `_configure_logging()` and `_configure_stdio()`, then:

   ```python
   for raw_line in sys.stdin:
       ...
       try:
           req = json.loads(line)
       except Exception as exc:
           logger.error("JSON parse error: %s", exc)
           send_response(_error(None, -32700, "Parse error"))
           continue
       try:
           response = dispatch(req)
       except Exception as exc:
           logger.error("Unhandled request error: %s", exc, exc_info=True)
           req_id = req.get("id") if isinstance(req, dict) else None
           response = _error(req_id, -32603, "Internal error")
       if response is not None:
           send_response(response)
   ```

   `send_response` keeps `ensure_ascii=False`; UTF-8 is now forced on stdout.

### Step 3 (GREEN): version and docs

- Bump `pyproject.toml` to `version = "1.0.5"`.
- Add a `CHANGELOG.md` entry for 1.0.5 covering the six P0 fixes.
- Update the `synapse/mcp.py` gotcha in `AGENTS.md`: the adapter and logging
  are now lazy, so importing the module writes nothing; tests live in
  `tests/test_mcp_protocol.py`.

### Step 4: verify

```powershell
python -m pytest tests/test_mcp_protocol.py -q
python -m pytest -q
```

**Expected:** all protocol tests pass; full suite green.

**Commit:** `fix(mcp): harden JSON-RPC transport and report real failures`

---

## Final verification (before review)

```powershell
python -m pytest -q
python examples/quickstart.py
```

Expected: full suite green; quickstart prints a recall result without error.

## Review Focus

Give this list to the final reviewer verbatim.

- **Transaction helper:** `_write` must not deadlock with `RLock`, must not
  swallow `KeyboardInterrupt` (it re-raises after rollback), and must not wrap
  methods that manage explicit `BEGIN`/`VACUUM`.
- **Migration recovery:** the merge branch must not drop rows when both tables
  contain data; the restore branch must re-add post-legacy columns; a second
  startup after recovery must be a no-op.
- **Numeric extraction:** check over-matching (`port 8080` inside a larger
  token), under-matching (units outside `KNOWN_UNITS`), and catastrophic
  backtracking on long digit runs.
- **Subject guard:** ensure it does not suppress legitimate revisions of the
  same subject, nor same-technology version upgrades.
- **MCP:** notification semantics (no response, including unknown methods and
  `tools/call`), error codes (`-32700`, `-32600`, `-32602`, `-32601`,
  `-32603`), stdout UTF-8 under cp1252, and the lazy adapter under a read-only
  or invalid DB path.
- **Read path:** `retrieve()` must not change `status`, `strength`, or
  `salience` when `touch=False`; with `touch=True` it may only reinforce.
