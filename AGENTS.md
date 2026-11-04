# AGENTS.md

`synapse-cortex` is a Python library (plus MCP stdio server) for SQLite-backed cognitive graph memory: retain/recall/reflect over FTS5 + 384-dim vectors + graph spreading activation. Packaging name is `synapse-cortex`; import path is `synapse`.

## Commands

- Install: `pip install -e ".[dev]"`
- Full suite: `python -m pytest` (362+ tests, ~22s). `pytest` is not always on PATH, so use `python -m pytest`.
- Single test: `python -m pytest tests/test_schema_contract.py::test_schema_creates_expected_core_tables -q`
- Smoke test: `python examples/quickstart.py` (in-memory, no API key needed). It has no `sys.path` bootstrap, so the repo root must be on the path first: `$env:PYTHONPATH="."` in PowerShell or `PYTHONPATH=.`, unless you already ran `pip install -e .`.
- MCP server: `python -m synapse.mcp` (or `python scripts/mcp_server.py`)
- No CI/linter/typecheck beyond the `pytest` gate in `.github/workflows/ci.yml`. The test suite is the gate; do not invent tooling config.

## Architecture

- `synapse/adapter.py` — main public entrypoint `SynapseAdapter`; wires storage, retriever, context loader, worker, mental-model/observation engines, harvester, contradiction resolver, metacognitive scorer, and guardrail. README/MCP examples use this.
- `synapse/storage.py` — `SynapseStorage` facade only: owns `_conn`, `_lock`, pragmas, and cross-context aggregates such as `get_stats`.
- `synapse/stores/` — bounded-context mixins behind storage: `schema` (DDL + migrations), `graph` (nodes/edges/FTS), `vectors` (embeddings + matrix cache), `episodes`, `knowledge` (mental models/observations/research), `maintenance` (decay/rollup/compaction/vault upsert).
- `synapse/retriever.py` hybrid search; `synapse/loader.py` L0/L1/L2 token-budget packing; `synapse/worker.py` consolidation, Hebbian decay, rollups, episode distiller.
- Optional acceleration only: numpy for the vector matrix, `sentence-transformers` (all-MiniLM-L6-v2) for neural embeddings, OpenAI-compatible `/v1/embeddings` when configured. Fallback is a deterministic CRC32 n-gram vectorizer, so core must run stdlib-only.

## Storage rules (pinned by tests)

- Keep `dependencies = []` in `pyproject.toml`; every third-party import must be optional and guarded.
- Never access `storage._conn` or `storage._lock` outside `synapse/storage.py`. `tests/test_storage_encapsulation.py` greps production modules for it; add a named data-access method instead.
- Add storage methods to the mixin that owns the concern, never to the facade. `tests/test_storage_boundaries.py` pins method homes, forbids duplicate definitions, and forbids mixins importing `synapse.storage`.
- Schema changes require updating `tests/test_schema_contract.py`: exact column lists per table, the exact trigger set, and the expected index set. FTS tables are kept in sync by triggers; changing them without those tests will fail.
- Code must stay Python 3.9-compatible: no `match` statements, and PEP 604 unions (`X | Y`) only in annotations behind `from __future__ import annotations`.

## Gotchas

- Default DB is `~/.synapse/synapse.db` whenever `db_path` is omitted. Use `db_path=":memory:"` in tests and an explicit `SYNAPSE_DB_PATH` for real runs, or you will read/write the shared user memory.
- `SynapseAdapter.__init__` catches all initialization errors and sets `storage`/`retriever`/etc. to `None`; `retain`/`recall` then silently no-op or fall back. If memory looks dead, check construction before debugging retrieval logic.
- Setting `vault_path`/`SYNAPSE_VAULT_PATH` makes `retain` write Markdown into that vault for salience >= 0.85 and architecture/SOP/decision/preference categories. Point tests at `tmp_path` or leave vaults unset.
- `synapse/mcp.py` is a hand-rolled stdio JSON-RPC loop: stdout is protocol-only (logging goes to a file), so never print there. Importing the module is side-effect free; the adapter and logging are lazy, and `tests/test_mcp_protocol.py` covers the protocol end to end.
- `SessionManager()` triggers background maintenance and defaults to `~/.synapse`; pass `config_dir`/`sync_state_path` in tests.
- Category and operational-status heuristics are bilingual (English + Indonesian) regexes. Use `\b` boundaries: substring matching caused real bugs (`mati` inside `otomatis`/`automatic`) and is regression-tested.
- `scripts/nightly_consolidation.py` is env-configured (no CLI flags), exits 1 when the DB is missing, writes `logs/consolidation.log`, and shells out to optional scripts via hardcoded `/usr/bin/python3` (Linux-oriented). Its optional sub-scripts are not in the repo.

## Conventions

- English-only code/docs, no emojis (`CONTRIBUTING.md`).
- Conventional commits (`feat:`, `fix:`, `refactor:`), atomic changes.
- Record user-visible changes in `CHANGELOG.md` and bump `pyproject.toml` version. `synapse/mcp.py` pins its `SERVER_VERSION` to the package version and `tests/test_mcp_protocol.py` reads the two from `pyproject.toml` so they cannot drift apart.
- `graphify-out/` and `data/`/`logs/` are local, gitignored artifacts; do not commit or treat them as source.
