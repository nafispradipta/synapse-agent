# AGENTS.md

Instructions for any AI coding agent working in this repository. Read this file first, then the relevant document under `docs/`. If this file and the code disagree, the code and its tests win; update this file in the same change.

`synapse-cortex` is a Python library plus a Model Context Protocol (MCP) stdio server for SQLite-backed cognitive graph memory: retain, recall, and reflect over FTS5, dense vectors (384-dimensional by default), and graph spreading activation. Packaging name is `synapse-cortex`; import path is `synapse`. License is MIT.

## Non-negotiables

- Keep `dependencies = []` in `pyproject.toml`. Third-party imports must be optional and guarded so the core runs on the standard library alone.
- Stay Python 3.9-compatible: no `match` statements, and PEP 604 unions (`X | Y`) only in annotations behind `from __future__ import annotations`.
- English only in code, comments, docstrings, and technical documentation. No emojis. `docs/Dokumentasi_user.md` is the single Indonesian document, written for non-technical readers.
- No hardcoded values that belong in configuration or environment variables. No duplicated logic that should be reused.
- Never access `storage._conn` or `storage._lock` outside `synapse/storage.py`.
- Never print to stdout in `synapse/mcp.py`; stdout is protocol-only. Logs go to the log file.
- Do not commit `data/`, `logs/`, `graphify-out/`, `.superpowers/`, `docs/superpowers/`, `.env`, or any `*.db` file.
- Never run `git push`. Leave integration decisions to the repository owner.

## Commands

- Install: `pip install -e ".[dev]"`. The published PyPI release may lag the repository; see [docs/Operations.md](docs/Operations.md) before documenting the install path.
- Full suite: `python -m pytest` (399 tests, about 30 to 35 seconds on 2026-11-10; re-run for the current count). `pytest` is not always on `PATH`, so use `python -m pytest`.
- Single test: `python -m pytest tests/test_schema_contract.py::test_schema_creates_expected_core_tables -q`
- Smoke test: `python examples/quickstart.py` (in-memory, no API key). The example has no `sys.path` bootstrap, so run `$env:PYTHONPATH="."` in PowerShell or `PYTHONPATH=.` in POSIX shells unless the package is installed.
- MCP server: `python -m synapse.mcp`, the `synapse-mcp` entry point, or `python scripts/mcp_server.py`
- Nightly consolidation: `python scripts/nightly_consolidation.py` (environment-configured, no CLI flags)
- WAL checkpoint: `python scripts/wal_checkpoint.py`
- CI is the pytest gate in `.github/workflows/ci.yml` (Ubuntu, Python 3.9 and 3.12). There is no linter or type checker; do not invent tooling configuration.

## Architecture map

- `synapse/adapter.py`: public `SynapseAdapter` and adapter fallbacks. Wires storage, retriever, context loader, worker, mental-model and observation engines, harvester, contradiction resolver, metacognitive scorer, and counterfactual guardrail.
- `synapse/storage.py`: `SynapseStorage` facade only. Owns the connection, the lock, pragmas, and cross-context aggregates such as `get_stats`.
- `synapse/stores/`: bounded-context mixins behind storage. `schema` (DDL, indexes, triggers, migrations), `graph` (nodes, edges, FTS), `vectors` (embeddings and matrix cache), `episodes`, `knowledge` (mental models, observations, research), `maintenance` (decay, rollup, compaction, vault upsert), `constants`.
- `synapse/retriever.py`: hybrid BM25, vector cosine similarity, spreading activation, recency, and vault wikilinks.
- `synapse/loader.py`: progressive L0/L1/L2 context packing under a token budget.
- `synapse/worker.py`: consolidation, Hebbian decay and reinforcement, rollups, vault sync, episode distillation, maintenance lifecycle.
- `synapse/embedder.py`: neural embeddings when `sentence-transformers` is available, deterministic CRC32 n-gram fallback otherwise.
- `synapse/mcp.py`: hand-rolled stdio JSON-RPC server. Importing it must stay side-effect free.
- `synapse/session.py`: `SessionManager` lifecycle helper and system prompt assembly.
- Cognitive engines: `contradiction_resolver.py`, `counterfactual_guardrail.py`, `metacognitive_scorer.py`, `mental_model_engine.py`, `mental_model_spawner.py`, `observation_engine.py`, `harvester.py`, `moc_bookkeeper.py`, `config.py`.
- `scripts/`: operational entry points. `examples/`: runnable examples. `tests/`: the behavioural gate.

Full component and data-flow documentation lives in [docs/Architecture.md](docs/Architecture.md).

## Storage rules (pinned by tests)

- Add a storage method to the mixin that owns the concern, never to the facade. `tests/test_storage_boundaries.py` pins method homes, forbids duplicate definitions, and forbids mixins importing `synapse.storage`.
- Add a named data-access method instead of reaching into `_conn`. `tests/test_storage_encapsulation.py` greps production modules for direct access.
- Schema changes require updating `tests/test_schema_contract.py`: exact column lists per table, the exact trigger set, and the expected index set. FTS tables are kept in sync by triggers.
- Writes go through the storage write path so failed writes roll back and non-finite numbers are rejected at the boundary.
- FTS triggers fire only on text-column changes. Update `PRAGMA user_version` and the migration path when the schema changes.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `SYNAPSE_DB_PATH` | SQLite database. Library default `~/.synapse/synapse.db`; the MCP server prefers an existing `./data/synapse.db` when unset. Use `:memory:` in tests. |
| `SYNAPSE_VAULT_PATH` | Markdown vault for writeback and wikilink sync. Also accepts `OBSIDIAN_VAULT_DIR` as a fallback. |
| `SYNAPSE_RESEARCH_DIR` | Harvested research notes. Default `~/.synapse/research/auto_harvest`. |
| `SYNAPSE_CONFIG_DIR` | Session configuration and sync state. Default `~/.synapse`. |
| `SYNAPSE_LOGS_DIR` | Log directory for the MCP server and scripts. |
| `SYNAPSE_EMBEDDING_MODEL` | `sentence-transformers` model. Default `all-MiniLM-L6-v2`. |
| `SYNAPSE_BASE_URL`, `SYNAPSE_API_KEY`, `SYNAPSE_MODEL` | Dedicated LLM overrides. |
| `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` | Standard OpenAI-compatible provider settings and fallbacks. |
| `OPENAI_EMBEDDING_BASE_URL`, `OPENAI_EMBEDDING_API_KEY`, `OPENAI_EMBEDDING_MODEL` | Remote embedding endpoint override. |
| `BASE_URL`, `API_KEY`, `MODEL`, `MODEL_NAME`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | Lower-priority compatibility fallbacks. |
| `MEMORY_BACKEND`, `MEMORY_TYPE` | Adapter selection in `resolve_memory_adapter`. |
| `SYNAPSE_DATA_DIR`, `SYNAPSE_MOC_EXCLUDE_DIRS` | Nightly consolidation data directory and MOC scan exclusions. |

Precedence and full setup instructions are in [docs/Operations.md](docs/Operations.md).

## Documentation map (single source of truth)

| Document | Owns |
| --- | --- |
| `AGENTS.md` | Agent rules, commands, architecture map, workflow |
| `docs/README.md` | Documentation index and update rules |
| `docs/Architecture.md` | System design, modules, data flow, schema |
| `docs/Product.md` | Product definition, capabilities, non-goals |
| `docs/Operations.md` | Install, configuration, MCP, maintenance, troubleshooting |
| `docs/Security_Privacy.md` | Threat model, data protection, credentials, reporting |
| `docs/Status.md` | Current state, blockers, next actions |
| `docs/Roadmap.md` | Candidate work, explicitly not commitments |
| `docs/Changelog.md` | Changelog policy; canonical history is root `CHANGELOG.md` |
| `docs/Dokumentasi_user.md` | Non-technical Indonesian project summary |
| `docs/archive/` | Superseded documents kept for reference |

Update the owning document in the same change that alters the behaviour it describes. A change that skips this step is incomplete.

## Workflow

Every task goes through the superpowers skills; "it is simple" is not a reason to skip them.

1. Plan. New feature, behaviour change, or design decision: `brainstorming` first, then `writing-plans`. Plans live in `docs/superpowers/plans/YYYY-MM-DD-<slug>.md` and are git-ignored.
2. Execute. Any implementation work follows `test-driven-development`, failing test first. Executing a written plan uses `executing-plans` inline, or `subagent-driven-development` for independent tasks.
3. Debug. Any bug, test failure, or unexpected behaviour uses `systematic-debugging` before a fix is proposed.
4. Review. Dispatch a fresh read-only sub-agent to validate the change against the code, tests, and current documentation. Give it no conclusions from the developer session. Fix every finding and repeat until it signs off. Typo and link-only changes are exempt.
5. Verify. Before claiming completion or committing, run `verification-before-completion` with real command output. The pytest suite, the smoke test, and the documentation link check must pass.
6. Document. Update `docs/Status.md`, root `CHANGELOG.md`, and any owning document affected by the change.
7. Commit. Conventional commit (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`), atomic scope. Never `git push`.

OpenCode invokes skills by bare id (`brainstorming`, not `superpowers:brainstorming`).

## Definition of done

- The failing test was observed before the fix, and the full suite passes.
- Behaviour claims in docs match the code and the test output.
- New configuration is an environment variable or adapter config key, not a literal.
- `docs/Status.md` and `CHANGELOG.md` reflect the change.
- The independent review signed off.
- The working tree contains only intended files, and the commit exists locally.

## Gotchas

- The library default database is `~/.synapse/synapse.db` whenever `db_path` is omitted. The MCP server additionally prefers an existing `./data/synapse.db` when `SYNAPSE_DB_PATH` is unset. Use `db_path=":memory:"` in tests and an explicit `SYNAPSE_DB_PATH` for real runs, or you will read and write shared user memory.
- `SynapseAdapter.__init__` catches initialization errors and sets `storage`, `retriever`, and related attributes to `None`; `retain` and `recall` then no-op or fall back. If memory looks dead, check construction before debugging retrieval.
- Setting `vault_path` or `SYNAPSE_VAULT_PATH` makes `retain` write Markdown for salience >= 0.85 and architecture, SOP, decision, or preference categories. Point tests at `tmp_path` or leave vaults unset.
- `SessionManager()` triggers background maintenance and defaults to `~/.synapse`; pass `config_dir` in tests.
- Category and operational-status heuristics are bilingual English and Indonesian regexes. Keep `\b` boundaries; substring matching caused real bugs (`mati` inside `otomatis`) and is regression-tested.
- Harvested research is quarantined: the research record and vault note use `status="quarantined"` and the derived graph node uses `status="flagged"`. None of them appear in recall until promoted.
- `scripts/nightly_consolidation.py` is environment-configured, exits 1 when the database is missing, writes `logs/consolidation.log`, and shells out to optional scripts through a hardcoded `/usr/bin/python3` (Linux-oriented). Its optional sub-scripts are not in the repository.
- `graphify-out/`, `data/`, `logs/`, and `docs/superpowers/` are local artifacts. Do not treat them as source.
