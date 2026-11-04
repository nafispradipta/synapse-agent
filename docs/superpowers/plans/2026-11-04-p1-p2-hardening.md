# P1/P2 Hardening Implementation Plan

> **For the executing engineer:** execute tasks in order, TDD-first. Every task
> starts with a test you watch fail. No production code before its failing test.

**Goal:** Close the remaining verified bugs from the 2026-11-04 audit: vault
path traversal and frontmatter injection, harvester SSRF and unbounded reads,
active recall of unvetted web content, embedding staleness, cross-test
filesystem pollution, dead/misleading configuration, and the P2 correctness
items (incremental cache, Unicode token estimates, NULL edge weights, compact
status, schema versioning, CI, version drift).

**Spec:** the audit conversation and the P0 plan
(`docs/superpowers/plans/2026-11-04-p0-safety-fixes.md`). This plan is the
authority for exact names and tests.

**Out of scope:** anything not listed here. Do not add `user_version` cleanups
beyond the trigger migration, do not refactor, do not bump to a feature epoch.

## Global Constraints

- `pyproject.toml` stays `dependencies = []`; only stdlib in the implementation.
- Python 3.9 compatible: no `match`, no runtime PEP 604 unions, no
  builtin generics evaluated at runtime.
- English-only code and docs, no emojis.
- Never access `storage._conn`/`storage._lock` from production modules outside
  `synapse/storage.py`. Tests may inspect `storage._conn`.
- Add storage behaviour to the mixin that owns the concern; facade stays thin.
- Full suite command: `python -m pytest`. Per-task gate: task's test file,
  then the full suite before the task commit.
- Every task ends with one conventional commit. Do not squash tasks.
- Read `AGENTS.md` conventions: keep `dependencies = []`, register user-visible
  changes in `CHANGELOG.md`, bump `pyproject.toml` version at the end (Task 13).

## Interfaces (cross-task)

- Task 4 adds a `model TEXT` column to `synapse_vectors` and optional
  `model: Optional[str] = None` parameter to `VectorStore.upsert_vector`.
  Tasks 5-13 do not depend on it.
- Task 8 sets `PRAGMA user_version` and must leave every existing database
  open for normal reads/writes afterwards.
- Task 9 touches `VectorStore._ensure_vector_matrix_cache`; Task 4 touches
  `VectorStore.upsert_vector`. Same file, separate methods.

---

## Task 1: Contain vault writeback paths and escape frontmatter values

**Files:** `synapse/adapter.py`, `tests/test_vault_containment.py` (new)

**Problem:** `metadata={"slug": "../../escape"}` writes `escape.md` outside
`vault_path` (verified). Frontmatter values are interpolated raw, so a title
with quotes or newlines breaks the YAML block (and `tags` come from arbitrary
input).

### Step 1 (RED): create `tests/test_vault_containment.py`

```python
"""Vault writeback must stay inside the vault and emit valid frontmatter."""

from __future__ import annotations

from pathlib import Path

from synapse.adapter import SynapseAdapter


def _retain_with_metadata(tmp_path: Path, **metadata_extra) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    adapter = SynapseAdapter(
        {"db_path": ":memory:", "vault_path": str(vault)}
    )
    metadata = {"category": "architecture", "salience": 0.95}
    metadata.update(metadata_extra)
    adapter.retain(
        "Architecture rule about vault containment",
        category="architecture",
        salience=0.95,
        metadata=metadata,
    )
    adapter.storage.close()
    return vault


def test_slug_traversal_is_confined_to_vault(tmp_path):
    vault = _retain_with_metadata(tmp_path, slug="../../escape")
    all_md = list(Path(tmp_path).rglob("*.md"))
    assert all(p.is_relative_to(vault) for p in all_md)
    assert (Path(tmp_path) / "escape.md").exists() is False


def test_slug_with_separators_is_sanitized(tmp_path):
    vault = _retain_with_metadata(tmp_path, slug="a/b/../../c")
    assert list(vault.rglob("*.md"))
    assert (Path(tmp_path) / "escape.md").exists() is False


def test_frontmatter_quotes_and_newlines_are_escaped(tmp_path):
    vault = _retain_with_metadata(
        tmp_path,
        title='Bad "title: with \\"quotes\\" and\nnewlines',
        slug="quote-test",
    )
    candidates = list(vault.rglob("quote-test.md"))
    assert candidates, "writeback produced no note"
    text = candidates[0].read_text(encoding="utf-8")
    assert "title:" in text
    frontmatter = text.split("---", 2)[1]
    assert "\\n" not in frontmatter.splitlines()[0]
    for line in frontmatter.splitlines():
        assert line.count('"') % 2 == 0, f"unbalanced quotes: {line}"
```

**Run:** `python -m pytest tests/test_vault_containment.py -q`
**Expected:** FAIL — `test_slug_traversal...` finds `tmp_path/escape.md` and
frontmatter quote test sees raw embedded quotes.

### Step 2 (GREEN): confined slug + YAML-safe values

In `synapse/adapter.py` near `_auto_writeback_vault` add:

```python
    @staticmethod
    def _safe_vault_slug(raw: Any) -> str:
        """Normalize a user-supplied slug to a single safe file stem."""
        text = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(raw)).strip("-") or "note"
        return text.strip(".") or "note"

    @staticmethod
    def _yaml_quote(value: Any) -> str:
        """Render a YAML double-quoted scalar safe for arbitrary content."""
        return json.dumps(str(value), ensure_ascii=False)
```

In `_auto_writeback_vault`, replace the slug computation block:

```python
            slug = meta.get("slug")
            if slug:
                slug = self._safe_vault_slug(slug)
            else:
                clean_slug = re.sub(r"[^a-zA-Z0-9\s_-]", "", title.lower()).strip()
                slug_words = clean_slug.split()[:8]
                slug = "-".join(slug_words).strip("-") or f"{category}-{uuid.uuid4().hex[:6]}"
            if not slug:
                slug = f"{category}-{uuid.uuid4().hex[:6]}"
```

Then confine the final file to the vault before writing:

```python
            target_file = (target_dir / f"{slug}.md").resolve()
            try:
                target_file.relative_to(vault_base)
            except ValueError:
                logger.warning("Vault slug %r escapes vault; skipping writeback", slug)
                return None
```

Escape the frontmatter values (keep `salience`/`date` numeric):

```python
            escaped_title = self._yaml_quote(title)
            tags_str = ", ".join(self._yaml_quote(tag) for tag in final_tags)
            frontmatter = (
                "---\n"
                f"title: {escaped_title}\n"
                f"tags: [{tags_str}]\n"
                f"date: {date_str}\n"
                f"salience: {salience:.2f}\n"
                f"uri: {self._yaml_quote(uri)}\n"
                "status: active\n"
                "---\n\n"
            )
```

### Step 3: verify

```powershell
python -m pytest tests/test_vault_containment.py -q
python -m pytest -q
```

**Expected:** containment tests pass; full suite green.

**Commit:** `fix(vault): confine writeback slugs and escape frontmatter`

---

## Task 2: Make the harvester redirect-safe and size-bounded

**Files:** `synapse/harvester.py`, `tests/test_harvester_ssrf.py` (new)

**Problem:** `fetch_url_text` validates only the initial URL; 301/302 redirects
can reach any host, and `resp.read()` is unbounded.

### Step 1 (RED): create `tests/test_harvester_ssrf.py`

```python
"""Harvester redirects and body sizes must be bounded and whitelisted."""

from __future__ import annotations

import gzip
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from synapse.harvester import SynapseWebHarvester


class _Handler(BaseHTTPRequestHandler):
    body_bytes: bytes = b""
    redirect_to: str | None = None

    def do_GET(self):  # noqa: N802 (http.server contract)
        if self.redirect_to is not None:
            self.send_response(301)
            self.send_header("Location", self.redirect_to)
            self.end_headers()
            return
        content = self.body_bytes
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):  # silence
        return


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    thread.join(timeout=5)


def _harvester(allowed):
    return SynapseWebHarvester(
        storage=None,
        vault_research_dir="",
        domains=allowed,
        llm_config={"base_url": "http://127.0.0.1:1/v1", "api_key": "k", "model": "m"},
    )


def test_redirect_to_non_whitelisted_host_is_blocked(server):
    harvester = _harvester({"localhost"})
    _Handler.redirect_to = f"http://127.0.0.1:{server.server_address[1]}/evil"
    _Handler.body_bytes = b"secret"
    assert harvester.fetch_url_text(
        f"http://127.0.0.1:{server.server_address[1]}/start", timeout=5
    ) is None


def test_redirect_to_whitelisted_host_is_allowed(server):
    harvester = _harvester({"127.0.0.1", "localhost", "example.com"})
    _Handler.redirect_to = None
    _Handler.body_bytes = b"public"
    text = harvester.fetch_url_text(
        f"http://127.0.0.1:{server.server_address[1]}/ok", timeout=5
    )
    assert text is not None


def test_oversized_body_is_rejected(server):
    harvester = _harvester({"127.0.0.1", "localhost"})
    _Handler.redirect_to = None
    _Handler.body_bytes = b"a" * (3 * 1024 * 1024)
    assert (
        harvester.fetch_url_text(f"http://127.0.0.1:{server.server_address[1]}/big", timeout=10)
        is None
    )
```

Check the `SynapseWebHarvester.__init__` signature before assuming
`domains=` and `storage=None` work; adapt the fixture to the real signature
(the important part is the whitelist can be driven to loopback).

**Run:** `python -m pytest tests/test_harvester_ssrf.py -q`
**Expected:** all three FAIL (redirect followed to loopback, unbounded body
accepted, oversized body returned).

### Step 2 (GREEN): validated redirects and a byte cap

```python
MAX_FETCH_BYTES = 2 * 1024 * 1024  # 2 MiB hard cap on response bodies
_MAX_REDIRECTS = 5


class _WhitelistRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, check_url: Any) -> None:
        super().__init__()
        self._check_url = check_url
        self._redirects = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if self._redirects >= _MAX_REDIRECTS:
            raise urllib.error.HTTPError(newurl, code, "too many redirects", headers, fp)
        self._redirects += 1
        if not self._check_url(newurl):
            raise urllib.error.HTTPError(newurl, code, f"redirect blocked: {newurl}", headers, fp)
        return urllib.request.HTTPRedirectHandler.redirect_request(
            self, req, fp, code, msg, headers, newurl
        )
```

In `fetch_url_text`, after the initial whitelist check:

```python
        opener = urllib.request.build_opener(
            _WhitelistRedirectHandler(self.is_domain_allowed)
        )
        req = urllib.request.Request(url, headers={...})
        with opener.open(req, timeout=timeout) as resp:
            length = resp.headers.get("Content-Length")
            if length and length.isdigit() and int(length) > MAX_FETCH_BYTES:
                logger.warning("URL body too large (%s bytes): %s", length, url)
                return None
            raw = resp.read(MAX_FETCH_BYTES + 1)
            if len(raw) > MAX_FETCH_BYTES:
                logger.warning("URL body exceeded cap: %s", url)
                return None
            content = raw.decode("utf-8", errors="ignore")
            ...
```

Keep the existing script/style stripping and 15000-char return
(`clean_text[:15000]` is fine; the cap applies to the raw body).

### Step 3: verify

```powershell
python -m pytest tests/test_harvester_ssrf.py -q
python -m pytest -q
```

**Expected:** green; existing harvester tests stay green.

**Commit:** `fix(harvester): bound redirects and response sizes`

---

## Task 3: Quarantine harvested web content

**Files:** `synapse/harvester.py`, `tests/test_quarantine.py` (new)

**Problem:** harvested research is stored as `status="active"` `category="sop"`
so unvetted web content enters `recall()`. Tier 4 must stay out of recall
until promoted.

### Step 1 (RED): create `tests/test_quarantine.py`

```python
"""Tier 4 harvest output must not enter active recall."""

from __future__ import annotations

from synapse.adapter import SynapseAdapter


def test_harvested_node_is_not_active_and_not_recallable(tmp_path):
    adapter = SynapseAdapter({"db_path": ":memory:"})
    try:
        adapter.harvest_research(
            topic="kubernetes",
            title="Kubernetes architecture notes",
            source_url="https://example.com/k8s",
            raw_content="Kubernetes clusters are composed of control planes " * 100,
        )
        nodes = adapter.storage.list_nodes(status="flagged")
        assert len(nodes) == 1
        assert nodes[0]["category"] == "sop"
        assert adapter.storage.list_nodes(status="active") == []
        recalled = adapter.recall("kubernetes control plane", limit=5)
        assert all("Kubernetes architecture notes" not in str(item) for item in recalled)
    finally:
        adapter.storage.close()
```

Check `harvest_research`'s actual signature and the return channel; if it
returns only a bool/None, adapt assertions to the storage outcome. Use the
local HTTP-free path (`raw_content` provided, LLM stubbed out or absent).

**Run:** `python -m pytest tests/test_quarantine.py -q`
**Expected:** FAIL — node is active and reachable in recall.

### Step 2 (GREEN): flag harvested nodes

In `synapse/harvester.py`, at the `self.storage.add_node(...)` call during
harvest ingestion, change `status="active"` to `status="flagged"` with a
comment tying it to the Tier 4 promotion gate. Do NOT change
`add_external_research` (the quarantine record) or `promote_external_research`.

### Step 3: verify

```powershell
python -m pytest tests/test_quarantine.py tests/test_synapse_web_harvester.py -q
python -m pytest -q
```

**Expected:** green; existing harvester tests updated only if they asserted
`status="active"` on the ingested node (change those assertions to `flagged`,
never weaken a real check).

**Commit:** `fix(harvester): flag harvested nodes until promotion`

---

## Task 4: Embedding provenance and re-embedding stale vectors

**Files:** `synapse/stores/schema.py`, `synapse/stores/graph.py`,
`synapse/stores/vectors.py`, `tests/test_embedding_provenance.py` (new),
`tests/test_schema_contract.py`

**Problem:** `update_node` changing text leaves the old 384-dim vector in
place, so recall ranks the node by content it no longer holds. Vectors also
carry no provenance (`model`, `dim`), which is required before trusting a
matrix across model switches.

### Step 1 (RED): create `tests/test_embedding_provenance.py`

```python
"""Vectors follow the text that produced them."""

from __future__ import annotations

from synapse.storage import SynapseStorage


def test_update_node_text_change_invalidates_vector():
    storage = SynapseStorage(":memory:")
    try:
        nid = storage.add_node(
            uri="synapse://prov/1",
            l0_abstract="sqlite wal mode concurrency guidance",
            category="architecture",
            salience=0.5,
        )
        before = storage.get_vector(nid)
        assert before is not None
        storage.update_node(nid, l0_abstract="completely unrelated banana recipe text")
        after = storage.get_vector(nid)
        assert after is None or after != before
    finally:
        storage.close()


def test_upsert_vector_records_model_and_dim():
    storage = SynapseStorage(":memory:")
    try:
        nid = storage.add_node(uri="synapse://prov/2", l0_abstract="provenance check")
        assert storage.upsert_vector(nid, [0.1] * 384, model="test-model-x") is True
        row = storage._conn.execute(
            "SELECT dim, model FROM synapse_vectors WHERE id = ?", (nid,)
        ).fetchone()
        assert row[0] == 384
        assert row[1] == "test-model-x"
    finally:
        storage.close()
```

**Run:** `python -m pytest tests/test_embedding_provenance.py -q`
**Expected:** first fails (vector unchanged), second fails (no `model` column).

### Step 2 (GREEN): schema + upsert + invalidation

- In `_DDL_SYNAPSE_VECTORS` add `model TEXT` after `dim INTEGER`. In
  `_create_vector_tables`, after the DDL, add a guarded migration:
  ```python
  table_info = cursor.execute("PRAGMA table_info(synapse_vectors)").fetchall()
  if not any(row[1] == "model" for row in table_info):
      cursor.execute("ALTER TABLE synapse_vectors ADD COLUMN model TEXT;")
  ```
- Update `tests/test_schema_contract.py` exact column list for
  `synapse_vectors` to include `model` (and check whether a vector index
  constraint test needs the extra column; keep the trigger set unchanged).
- `VectorStore.upsert_vector(node_id, vector, model: Optional[str] = None)`:
  store `model`; keep `ON CONFLICT(id)` (it must now also update `model` and
  `dim`).
- In `GraphStore.update_node`, after a successful update, if any of
  `{"l0_abstract", "l1_overview"} & set(updates)`, call
  `self._refresh_or_drop_vector(node_id, updates)`:
  ```python
  def _refresh_or_drop_vector(self, node_id: str, updates: Dict[str, Any]) -> None:
      """Re-embed when text changed; otherwise drop the stale vector."""
      node = self.get_node(node_id_or_uri=node_id, touch=False)
      ...
      embedder = self._get_embedder()
      vector = None
      if embedder is not None:
          try:
              vector = embedder.embed(f"{node['l0_abstract']} {node['l1_overview']}")
          except Exception:
              vector = None
      if vector:
          self.upsert_vector(str(node["id"]), vector, model=getattr(embedder, "model_name", None))
      else:
          self._conn.execute(
              "DELETE FROM synapse_vectors WHERE id = ?", (str(node["id"]),)
          )
          self._conn.commit()
          self._invalidate_vector_cache()
  ```
  Note ordering: run the re-embed/drop BEFORE `self._conn.commit()` of the
  outer `update_node` is not possible (update already commits). Run it
  after the commit; the vector may briefly be stale, which is acceptable and
  far better than persistent staleness.
- Auto-vectorization in `GraphStore.add_node` passes
  `model=getattr(embedder, "model_name", None)`.
- `SynapseEmbedder.embed` result for the fallback CRC path: expose a stable
  model label (e.g. `"synapse-fallback-crc32"`) if `model_name` is unset;
  do this inside `_get_embedder` consumers, not by changing embed().

### Step 3: verify

```powershell
python -m pytest tests/test_embedding_provenance.py tests/test_schema_contract.py -q
python -m pytest -q
```

**Expected:** green; full suite green (update any other assertions that pin
the exact `synapse_vectors` column list).

**Commit:** `feat(vectors): record model provenance and re-embed on text change`

---

## Task 5: Isolate the test environment

**Files:** `tests/conftest.py` (new), `synapse/adapter.py`,
`tests/test_synapse_cognitive_architecture.py` (if it needs an explicit
`config_dir`), `README.md`/`CONTRIBUTING.md` test commands if wrong

**Problem:** tests set no global safeguards and the adapter defaults its
harvester `research_dir` to `./data/research/auto_harvest` relative to cwd, so
test runs write into the repo (observed: `data/research/auto_harvest/*.md`).

### Step 1 (RED): demonstrate the repo is polluted by the suite

```powershell
python -m pytest -q
Get-ChildItem data -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First 5
```

**Expected:** shows files under `data/research/` (they exist in this worktree
right now, created by the harvester adapter tests).

### Step 2 (GREEN): global env isolation

Create `tests/conftest.py`:

```python
"""Global test isolation: no test may touch the shared user environment."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_ENV_VARS = (
    "SYNAPSE_DB_PATH", "SYNAPSE_VAULT_PATH", "SYNAPSE_RESEARCH_DIR",
    "SYNAPSE_LOGS_DIR", "SYNAPSE_DATA_DIR", "SYNAPSE_SYNC_STATE_PATH",
    "SYNAPSE_TOKEN_BUDGET", "SYNAPSE_L0_BUDGET", "SYNAPSE_L1_BUDGET",
    "SYNAPSE_BASE_URL", "SYNAPSE_API_KEY", "SYNAPSE_MODEL",
    "SYNAPSE_EMBEDDING_MODEL", "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL",
)


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch, tmp_path: Path):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    if os.name == "nt":
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
    yield
```

### Step 3 (GREEN): stop the repo-poisoning default

In `synapse/adapter.py` `__init__`, replace

```python
(str(Path("./data/research/auto_harvest").resolve()))
```

with

```python
(os.environ.get("SYNAPSE_RESEARCH_DIR") or str(Path.home() / ".synapse" / "research" / "auto_harvest"))
```

and document `SYNAPSE_RESEARCH_DIR` in `.env.example`.

### Step 4: verify

```powershell
python -m pytest -q
Get-ChildItem data -Recurse -File -ErrorAction SilentlyContinue
```

**Expected:** suite green; `data/` is empty or absent (delete the two stale
files this worktree accumulated: they are regenerable test artifacts, not
source).

**Commit:** `test: isolate environment and stop repo data pollution`

---

## Task 6: Remove dead documented configuration

**Files:** `.env.example`, `README.md`, `CONTRIBUTING.md`

**Problem:** `.env.example` and docs advertise `SYNAPSE_TOKEN_BUDGET`,
`SYNAPSE_L0_BUDGET`, `SYNAPSE_L1_BUDGET`, `SYNAPSE_SYNC_STATE_PATH`, none of
which the code reads; the default DB comment claims `./data/synapse.db` while
the real default is `~/.synapse/synapse.db`.

### Step 1 (RED): grep documentation

```powershell
Select-String -Path README.md,CONTRIBUTING.md,.env.example -Pattern 'SYNAPSE_TOKEN_BUDGET|SYNAPSE_L0_BUDGET|SYNAPSE_L1_BUDGET|SYNAPSE_SYNC_STATE_PATH|\./data/synapse\.db'
```

**Expected:** matches found (they are the bug).

### Step 2 (GREEN): delete or fix

- `.env.example`: drop the three token-budget vars and the
  `SYNAPSE_SYNC_STATE_PATH` block; fix the DB default comment to the real
  default (`~/.synapse/synapse.db`, overridable by `SYNAPSE_DB_PATH`).
- `README.md` / `CONTRIBUTING.md`: remove the same env vars and replace any
  bare `pytest` instruction with `python -m pytest` (AGENTS.md already
  documents this); keep `python -m pytest` and the `.[dev]` install step.

### Step 3: verify

```powershell
Select-String -Path README.md,CONTRIBUTING.md,.env.example -Pattern 'SYNAPSE_TOKEN_BUDGET|SYNAPSE_L0_BUDGET|SYNAPSE_L1_BUDGET|SYNAPSE_SYNC_STATE_PATH|\./data/synapse\.db'
python -m pytest -q
```

**Expected:** no matches; suite green.

**Commit:** `docs: remove dead configuration from docs and env template`

---

## Task 7: CI workflow

**Files:** `.github/workflows/ci.yml` (new)

**Problem:** no CI exists; 342 tests only protect whoever remembers to run
them. Infrastructure is the gap, not assertions.

### Step 1 (GREEN): add the workflow

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.9", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -e ".[dev]"
      - run: python -m pytest -q
```

Dev extras already declare `pytest` and `pytest-cov`; keep the job to install
+ `pytest`. Do NOT add lint/typecheck steps: the repo has no lint/typecheck
configuration and adding one was not requested.

### Step 2: verify

```powershell
python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))" 2>$null
```

If PyYAML is unavailable (core runtimes lack it), validate by eye against the
template above; `actions/setup-python@v5` accepts quoted `"3.9"`.

**Commit:** `ci: run the test suite on push and pull request`

---

## Task 8: Schema versioning (`PRAGMA user_version`) and narrow triggers

**Files:** `synapse/stores/schema.py`, `tests/test_schema_contract.py`

**Problem:** every trigger is `CREATE TRIGGER IF NOT EXISTS` and there is no
schema version, so an existing database never receives changed trigger bodies
— e.g. the `AFTER UPDATE` nodes FTS trigger fires on `last_accessed` bumps,
writing FTS rows on every read-with-touch.

### Step 1 (RED): tests for versioning and the narrow trigger

```python
def test_upgrade_from_user_version_zero_migrates_triggers(tmp_path):
    db_file = tmp_path / "legacy-triggers.db"
    _write_legacy_db(db_file)

    conn = sqlite3.connect(str(db_file))
    try:
        conn.execute(
            "CREATE TRIGGER synapse_nodes_au AFTER UPDATE ON synapse_nodes BEGIN "
            "  DELETE FROM synapse_fts WHERE id = old.id; "
            "END"
        )
        conn.execute("PRAGMA user_version=0")
        conn.commit()
    finally:
        conn.close()

    storage = SynapseStorage(db_path=str(db_file), allow_custom_categories=True)
    try:
        row = storage._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND name='synapse_nodes_au'"
        ).fetchone()
        assert "AFTER UPDATE OF" in row[0]
        assert storage._conn.execute("PRAGMA user_version").fetchone()[0] == 1
        nid = storage.add_node(uri="synapse://ver/1", l0_abstract="original text")
        storage.update_node(nid, last_accessed=1234567890)
        assert storage.search_fts("original", limit=1)
        storage.update_node(nid, l0_abstract="completely different content")
        hits = storage.search_fts("completely", limit=1)
        assert [row["id"] for row in hits] == [nid]
    finally:
        storage.close()
```

And a schema-contract assertion that `user_version` is 1 after a fresh
`:memory:` open.

**Run:** `python -m pytest tests/test_schema_contract.py -q`
**Expected:** both new tests FAIL.

### Step 2 (GREEN): migration + version stamp

- Change `_TRIGGERS_NODE_FTS` third trigger to:
  `CREATE TRIGGER synapse_nodes_au AFTER UPDATE OF l0_abstract, l1_overview ON synapse_nodes BEGIN ... END;`
  (drop `IF NOT EXISTS` in the migration copy; fresh DDL may keep it or not).
- Change `_TRIGGERS_EPISODE_FTS` to `AFTER UPDATE OF title, situation, action_taken, outcome, lessons_learned, tags`.
- In `_init_schema` (end of it), after all tables/triggers are created:
  ```python
  current = self._conn.execute("PRAGMA user_version").fetchone()[0]
  if current < 1:
      self._migrate_triggers_v1(cursor)
      self._conn.execute("PRAGMA user_version=1")
  ```
  `_migrate_triggers_v1` drops the old `synapse_nodes_au`, `episodes_au`
  triggers (if present) and recreates the `AFTER UPDATE OF ...` ones, both for
  fresh and pre-existing databases.
- Keep `CREATE TRIGGER IF NOT EXISTS` in the constants for fresh DBs but make
  sure the re-created trigger names are the same ones `_TRIGGERS_*` reference;
  simplest: migration drops and re-creates exactly those two triggers.
- Update the exact trigger list in `tests/test_schema_contract.py` to the new
  `AFTER UPDATE OF` form.

### Step 3: verify

```powershell
python -m pytest tests/test_schema_contract.py -q
python -m pytest -q
```

**Expected:** green; reviewers must see the trigger set table in the contract
tests updated with the migration test.

**Commit:** `feat(schema): stamp user_version and narrow FTS update triggers`

---

## Task 9: Make `incremental_cache` actually incremental

**Files:** `synapse/stores/vectors.py`, `tests/test_vector_matrix_cache.py`

**Problem:** with `incremental_cache = True`, every vector write still causes a
full matrix rebuild on next search (measured 5 writes + 5 searches → 5
rebuilds), because the version trigger bumps the counter and
`_ensure_vector_matrix_cache` then discards the whole matrix.

### Step 1 (RED): extend `tests/test_vector_matrix_cache.py`

```python
def test_incremental_cache_avoids_rebuild(tmp_path, monkeypatch):
    storage = SynapseStorage(":memory:")
    try:
        storage.incremental_cache = True
        calls = {"n": 0}
        original = storage._build_vector_matrix_cache
        def counted():
            calls["n"] += 1
            return original()
        monkeypatch.setattr(storage, "_build_vector_matrix_cache", counted)

        ids = [storage.add_node(uri=f"synapse://incr/{i}", l0_abstract=f"node {i}") for i in range(3)]
        storage.search_vectors([0.1] * 384, top_k=2)  # initial build
        baseline = calls["n"]

        for i, nid in enumerate(ids):
            storage.upsert_vector(nid, [float(i % 3) + 0.1] * 384)
            storage.search_vectors([0.1] * 384, top_k=2)
        assert calls["n"] == baseline, f"incremental writes triggered rebuilds: {calls['n']}"
    finally:
        storage.close()
```

**Run:** `python -m pytest tests/test_vector_matrix_cache.py -q`
**Expected:** FAIL — rebuilds happen on every write/search cycle.

### Step 2 (GREEN): sync the cached version after incremental ops

In `VectorStore.upsert_vector` and `delete_vector`/incremental delete path,
when `self._incremental_cache` is truthy and `_incremental_upsert` (or its
delete counterpart) ran, set
`self._cached_vector_version = self._get_vector_data_version()` and
`self._vector_cache_valid = True` instead of leaving the counter stale. The
vector contract tests (`tests/test_vector_matrix_cache.py:203`) pin the
invalidate-on-status path behavior; keep it.

### Step 3: verify

```powershell
python -m pytest tests/test_vector_matrix_cache.py -q
python -m pytest -q
```

**Expected:** green.

**Commit:** `fix(vectors): keep matrix cache warm across incremental writes`

---

## Task 10: Unicode-aware token estimation

**Files:** `synapse/loader.py`, `tests/test_loader_token_budget.py` (new)

**Problem:** `estimate_tokens` is `ceil(len(text)/4)` for everything; CJK text
is under-counted (Japanese "10 chars → 3 tokens"), wasting budget headroom.

### Step 1 (RED): create `tests/test_loader_token_budget.py`

```python
"""Token estimates must not undercount CJK text."""

from __future__ import annotations

from synapse.loader import estimate_tokens


def test_cjk_scales_per_character():
    assert estimate_tokens("\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3067\u3059") >= 8


def test_latin_keeps_quarter_rate():
    assert estimate_tokens("hello world this is english text") == 8


def test_emoji_counts_as_tokens():
    assert estimate_tokens("\U0001f642\U0001f642\U0001f642\U0001f642") >= 4


def test_empty():
    assert estimate_tokens("") == 0
```

**Run:** `python -m pytest tests/test_loader_token_budget.py -q`
**Expected:** CJK test FAILS (returns 3).

### Step 2 (GREEN): byte/char-aware estimator

```python
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    count = 0.0
    for ch in text:
        cp = ord(ch)
        if (
            0x3400 <= cp <= 0x4DBF          # CJK ext A
            or 0x4E00 <= cp <= 0x9FFF       # CJK unified
            or 0x3040 <= cp <= 0x30FF       # hiragana/katakana
            or 0xAC00 <= cp <= 0xD7AF       # hangul syllables
        ):
            count += 1.0
        elif cp >= 0x1F000:                 # emoji and astral planes
            count += 1.0
        elif cp < 0x7F:                     # ascii: ~4 chars/token
            count += 0.25
        else:
            count += 0.5                    # other non-ascii: ~2 chars/token
    return max(1, math.ceil(count))
```

Rename nothing; `split_sentences` is untouched.

### Step 3: verify

```powershell
python -m pytest tests/test_loader_token_budget.py -q
python -m pytest -q
```

**Expected:** green. If any existing assertion pins a Latin-shaped estimate
via a mixed string, adjust it to the new conservative value only when clearly
intended (do not loosen a budget contract).

**Commit:** `fix(loader): estimate tokens per Unicode block`

---

## Task 11: Tolerate NULL edge weights

**Files:** `synapse/stores/graph.py`, `tests/test_null_edge_weight.py` (new)

**Problem:** `prune_decayed_edges` does `float(row[3])` and `float(row[4])
(last_fired)` crashes on NULL weights (`NOT NULL` not enforced).

### Step 1 (RED): create `tests/test_null_edge_weight.py`

```python
"""Decay/prune must not crash on NULL weights."""

from __future__ import annotations

from synapse.storage import SynapseStorage


def test_null_weight_edge_is_decayed_and_pruned():
    storage = SynapseStorage(":memory:")
    try:
        source = storage.add_node(uri="synapse://null/a", l0_abstract="a")
        target = storage.add_node(uri="synapse://null/b", l0_abstract="b")
        storage.add_edge(source, target, relation="co_activated", weight=0.5)
        storage._conn.execute(
            "UPDATE synapse_edges SET weight = NULL, last_fired = 1"
        )
        storage._conn.commit()
        stats = storage.prune_decayed_edges(idle_days_threshold=0.0)
        assert isinstance(stats, dict)
        assert storage.get_edges(source) == []
    finally:
        storage.close()
```

**Run:** `python -m pytest tests/test_null_edge_weight.py -q`
**Expected:** FAIL with `TypeError: float() argument must be a string or real number, not 'NoneType'`.

### Step 2 (GREEN): coalesce defaults in the SELECT

Change the decay query to
`SELECT source_id, target_id, relation, COALESCE(weight, 0.5), COALESCE(last_fired, 0) ...`
and keep the Python arithmetic. Delete the now-redundant fallback.

### Step 3: verify

```powershell
python -m pytest tests/test_null_edge_weight.py -q
python -m pytest -q
```

**Expected:** green.

**Commit:** `fix(graph): coalesce NULL edge weights during decay`

---

## Task 12: Report real compaction status

**Files:** `synapse/stores/maintenance.py`, `tests/test_compaction_status.py` (new)

**Problem:** `optimize_and_compact` returns `status="ok"` even when the
checkpoint/optimize/VACUUM steps fail (they are recorded as strings, but the
result never surfaces that).

### Step 1 (RED): create `tests/test_compaction_status.py`

```python
"""Compaction status reflects per-step failure."""

from __future__ import annotations

import sqlite3

from synapse.stores.maintenance import compaction_status


def test_compaction_status_ok_when_all_steps_ok():
    assert compaction_status({"wal_checkpoint": "done", "optimize": "ok", "vacuum": "ok"}) == "ok"


def test_compaction_status_partial_on_error():
    assert compaction_status({"wal_checkpoint": "error: boom", "optimize": "ok", "vacuum": "ok"}) == "partial"


def test_compaction_status_partial_on_exception():
    assert compaction_status({"wal_checkpoint": "none", "optimize": "error: no", "vacuum": "error: no"}) == "partial"
```

### Step 2 (GREEN): helper + aggregate

Add to `maintenance.py`:

```python
def compaction_status(results: Dict[str, Any]) -> str:
    failed = any(
        isinstance(value, str) and value.startswith("error")
        for key, value in results.items()
        if key not in ("status", "timestamp")
    )
    return "partial" if failed else "ok"
```

In `optimize_and_compact`, set `results["status"] = compaction_status(results)`
at the end (after all steps ran). Keep the argument-free `Dict[str, Any]`
shape so `scripts/nightly_consolidation.py` still works unchanged.

### Step 3: verify

```powershell
python -m pytest tests/test_compaction_status.py -q
python -m pytest -q
```

**Expected:** green.

**Commit:** `fix(maintenance): report compaction status from step results`

---

## Task 13: Single source for the package version

**Files:** `synapse/mcp.py`, `pyproject.toml`, `CHANGELOG.md`, `README.md`,
`CONTRIBUTING.md`, `AGENTS.md`

**Problem:** the version is hardcoded in `pyproject.toml`; `mcp.py` carries a
pinned `SERVER_VERSION` (currently 1.0.5); README/CONTRIBUTING still say 318
tests and `pytest` bare.

### Step 1 (GREEN): resolve from package metadata

```python
def _package_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version("synapse-cortex")
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    try:
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        match = re.search(
            r'^version\s*=\s*"([^"]+)"',
            pyproject.read_text(encoding="utf-8"),
            re.M,
        )
        if match:
            return match.group(1)
    except Exception:
        pass
    return "1.1.0"


SERVER_VERSION = _package_version()
```

`importlib.metadata` is stdlib since 3.8, so Python 3.9 is fine.

### Step 2 (GREEN): bump and doc numbers

- `pyproject.toml`: `version = "1.1.0"` (behavior changes + new CI/columns).
- `CHANGELOG.md`: add `## [1.1.0] - 2026-11-04` listing all P1/P2 fixes.
- README `## Commands`/`Testing`: `python -m pytest`, mention 350+ tests.
- CONTRIBUTING: same command guidance (no bare `pytest`).
- AGENTS.md gotchas: bump test count; CI now exists (the
  "No CI workflows" line must change).

### Step 3: verify

```powershell
python -m pytest -q
$env:PYTHONPATH="."; python examples/quickstart.py 2>&1 | Select-Object -First 2
```

**Expected:** suite green; quickstart works.

**Commit:** `chore: release 1.1.0`

---

## Final verification (before review)

```powershell
python -m pytest -q
python examples/quickstart.py
```

Expected: suite green; quickstart prints a recall result.

## Review Focus

Give this list verbatim to the final reviewer.

- **Vault containment:** the `relative_to(vault_base)` check happens after
  resolving `target_file`; confirm no symlink escape (`target_dir.mkdir` may
  create directories that alias elsewhere) and that `json.dumps` output is a
  valid YAML double-quoted scalar in this codebase's frontmatter dialect.
- **Harvester:** redirect validation must see the FINAL URL; content-length
  header must not be trusted blindly; confirm loopback is unreachable through
  the whitelist unless explicitly allowed; check `MAX_REDIRECTS` semantics.
- **Quarantine:** `flagged` is a first-class non-active status everywhere
  (retriever, FTS, vector joins); promotion path must not auto-activate.
- **Embedding provenance:** deleting `synapse_vectors` rows is enough for
  `list_nodes_missing_vectors` to backfill; confirm dim mismatch cannot crash
  cache rebuild (`DATA_DIM` vs `_matrix_by_dim`).
- **Concurrency:** `upsert_vector` + `update_node` re-embed run under the
  storage lock; version counter consistency across incremental and full rebuilds.
- **Migration:** `user_version` stamping must not touch `:memory:` behavior;
  the trigger migration must not break fresh installs or user_version=0 DBs
  with partial state.
- **Token budget:** verify no budget contract was silently loosened; check CJK
  estimate against real tokenizer output if available.
- **CI:** YAML validity, correct Python versions, no secrets, no network-dependent tests.
- **Version source:** `importlib.metadata` mismatch with installed editable
  packages — will the protocol test still pass?
