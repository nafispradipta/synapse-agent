# Status

Last updated: 2026-11-10

## Current state

| Item | Value |
| --- | --- |
| Package version (repository) | `1.2.0` |
| Latest published PyPI release | `1.0.4` at review time (2026-11-10); PyPI may lag the repository |
| Python support | `3.9` to `3.13` declared; CI runs `3.9` and `3.12` |
| Test baseline | 406 tests, about 35 seconds on the maintainer machine (2026-11-10) |
| Default database | `~/.synapse/synapse.db`; the MCP server prefers an existing `./data/synapse.db` when `SYNAPSE_DB_PATH` is unset |
| MCP tools | `synapse_recall`, `synapse_retain`, `synapse_record_episode`, `synapse_get_stats` |
| Storage schema version | `PRAGMA user_version = 1` |
| Documentation set | Established in this change; see [README.md](README.md) |

## Completed

- Storage split into bounded-context mixins with a facade that owns the connection.
- Storage encapsulation guards and schema-contract tests.
- MCP protocol negotiation, annotations, structured content, and input validation.
- Vault path containment, harvester redirect limits, and research quarantine.
- Numeric boundary validation and rollback on failed writes.
- Root `AGENTS.md` rewritten and `docs/` established as the documentation source of truth.
- Stale test counts, `SECURITY.md` version support, and changelog ordering corrected.
- Independent reviewer findings 1 to 19 addressed. A follow-up independent review (2026-11-10) completed: no blockers; storage, schema, and boundary rules intact; one flaky environment-dependent performance test confirmed passing on re-run; `.gitignore` change reverted as flagged.
- Seamless harness setup: `synapse` CLI (`setup`, `doctor`), harness registry (Claude Desktop, Claude Code, Codex, OpenCode), per-harness config templates, and `docs/Harness_Setup.md`. Package version bumped to 1.2.0.

## Open items

- The documentation link check is a manual procedure; no script is tracked in the repository.
- One-line installer (curl|bash / irm|iex) deferred; CLI setup is the MVP. Additional harnesses (Cursor, Windsurf, Cline, Continue, Zed) can be added to the registry later.
- (Fixed) `VectorStore.upsert_vector` now receives the embedder's `model_name` on adapter, retriever, and nightly-consolidation paths, so vector model provenance is recorded instead of `NULL`.
- `scripts/nightly_consolidation.py` remains Linux-oriented because it shells out through `/usr/bin/python3`.
- The `[Unreleased]` changelog section is not yet released as a version.

## Blockers

None known.

## Next actions

1. (Done) Fixed `VectorStore.upsert_vector` model provenance across adapter, retriever, and nightly consolidation.
2. Add a tracked documentation link-check script if desired.
3. Release the `[Unreleased]` changelog section as a version when ready.
4. Do not push; integration decisions belong to the repository owner.

## Verification commands

```bash
python -m pytest -q
PYTHONPATH=. python examples/quickstart.py
python -c "import synapse, synapse.mcp"
```

Documentation check: assert that every required document exists and every relative Markdown link resolves. A broken link or missing required file must fail the check.

## Update protocol

Update this file whenever work completes, a blocker appears, the focus changes, or the test baseline moves. Keep it short. Historical detail belongs in the root `CHANGELOG.md`.
