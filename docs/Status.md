# Status

Last updated: 2026-11-10

## Current state

| Item | Value |
| --- | --- |
| Package version (repository) | `1.1.1` |
| Latest published PyPI release | `1.0.4` at review time (2026-11-10); PyPI may lag the repository |
| Python support | `3.9` to `3.13` declared; CI runs `3.9` and `3.12` |
| Test baseline | 399 tests, about 34 seconds on the maintainer machine (2026-11-10) |
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
- Independent reviewer findings 1 to 19 addressed. Because the subagent tool hit a usage limit during the follow-up, the final pass was a self-review by the author; an independent review is still recommended before release.

## Open items

- Independent review was requested but the subagent tool hit a usage limit. The final pass was a self-review; an independent review is still recommended before release.
- The documentation link check is a manual procedure; no script is tracked in the repository.
- `VectorStore.upsert_vector` is called without a `model` argument on the adapter retain path, so vector model provenance can be overwritten with `NULL`. This is a code fix candidate, not part of the documentation change.
- `scripts/nightly_consolidation.py` remains Linux-oriented because it shells out through `/usr/bin/python3`.
- The `[Unreleased]` changelog section is not yet released as a version.

## Blockers

None known.

## Next actions

1. Run an independent review if possible; otherwise proceed with the self-review record above.
2. Re-run `python -m pytest -q`, the quickstart smoke test, and the documentation link check.
3. Record the outcome here and in the root `CHANGELOG.md`.
4. Commit the documentation change locally. Do not push.

## Verification commands

```bash
python -m pytest -q
PYTHONPATH=. python examples/quickstart.py
python -c "import synapse, synapse.mcp"
```

Documentation check: assert that every required document exists and every relative Markdown link resolves. A broken link or missing required file must fail the check.

## Update protocol

Update this file whenever work completes, a blocker appears, the focus changes, or the test baseline moves. Keep it short. Historical detail belongs in the root `CHANGELOG.md`.
