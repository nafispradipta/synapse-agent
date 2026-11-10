# Roadmap

## Status of this document

Everything here is a candidate derived from documented limits, known gaps, and trade-offs already stated in [Architecture.md](Architecture.md) and the root `CHANGELOG.md`. Nothing here is committed, scheduled, or promised. The repository owner decides what enters active work, and accepted work goes through the planning workflow in `AGENTS.md`.

No dates are listed on purpose. A date without an accepted plan would be a fabricated commitment.

## Now

### Documentation source of truth

- Goal: `AGENTS.md` and `docs/` describe the system accurately and stay current.
- Why now: documentation was scattered across the root files and the `docs/` tree was empty.
- Done when: every required document exists, links resolve, and the status page is updated with each change.

### Automated documentation check

- Goal: catch broken links and stale required files before a commit.
- Why: link drift is the most common documentation failure.
- Candidate implementation: a small test under `tests/` or a CI step, decided by the owner.
- Done when: a broken relative link fails the check.

## Next

### Optional dependency extras

- Goal: make neural embeddings and accelerated vector search installable with one command.
- Why: users currently install `sentence-transformers` and `numpy` manually.
- Candidate: extras such as `neural` and `accelerated` in `pyproject.toml`, keeping `dependencies = []`.
- Done when: the documented install command installs the optional stack and the fallback still works without it.

### MCP surface review

- Goal: expose capabilities that users actually need from an MCP client.
- Why: the server exposes recall, retain, episode, and stats. Reflection and uncertainty are available in Python but not over MCP.
- Candidate: design review before adding tools; every new tool needs validation, limits, and protocol tests.
- Done when: the owner accepts a tool set and `tests/test_mcp_protocol.py` covers each addition.

### Backup and restore helper

- Goal: a single supported command for checkpoint, copy, verify, and restore.
- Why: the current guidance is manual and easy to get wrong while a writer is active.
- Done when: the helper verifies integrity after restore and refuses to overwrite an open database.

### Log secret redaction

- Goal: guarantee that credentials cannot appear in log files even when a URL or error message contains them.
- Why: logs are plaintext and may be shared during support.
- Done when: a test feeds credential-shaped input and asserts it is redacted.

### Cross-language retrieval baseline

- Goal: document measured recall behaviour for English and Indonesian queries with the fallback vectorizer and with neural embeddings.
- Why: the current claim is qualitative. A small reproducible evaluation would make the trade-off concrete.
- Done when: the evaluation is reproducible from the repository and its results are in the docs.

## Later

### Pluggable vector index

- Goal: support graphs well beyond the current target of roughly 10,000 active nodes.
- Why: cosine similarity currently runs in process over candidate nodes.
- Constraint: the SQLite graph remains the system of record; an external index is an accelerator, not a second source of truth.

### Optional encryption at rest

- Goal: protect the database and vault on shared or portable machines.
- Why: storage is plaintext today.
- Constraint: must not break the standard-library core or the zero-dependency default.

### Vector model provenance

- Goal: preserve the embedding model name on every retain path.
- Why: the adapter re-embeds and calls `upsert_vector` without `model`, which can overwrite provenance with `NULL`.
- Done when: a test asserts the stored `synapse_vectors.model` matches the embedder after a normal `retain`.

### Research promotion workflow

- Goal: a deliberate, auditable path from quarantined research to active memory.
- Why: quarantine exists, but promotion is manual.
- Constraint: promotion must preserve provenance and never auto-activate harvested content.

### Write concurrency profiling

- Goal: document and improve behaviour under many concurrent writers.
- Why: SQLite allows one writer at a time and the current answer is a busy timeout.
- Constraint: stay compatible with Python 3.9 and the single-file deployment model.

## How an item moves forward

1. The owner selects a candidate and states the intended outcome.
2. The work goes through brainstorming and a written plan.
3. The plan is approved before implementation.
4. On completion, the item moves to the root `CHANGELOG.md`, and this document is updated to remove or replace it.
