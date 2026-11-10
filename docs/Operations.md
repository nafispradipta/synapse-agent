# Operations

## Requirements

- Python 3.9 or newer.
- `pip` and a virtual environment.
- SQLite compiled with FTS5. The SQLite bundled with standard CPython builds includes it; a custom build may not.
- Optional: `numpy` for the vector matrix, `sentence-transformers` for neural embeddings, and any OpenAI-compatible endpoint for LLM reflection.

## Install

From PyPI:

```bash
pip install synapse-cortex
```

The published PyPI release may lag the repository. Check the PyPI page or `pip index versions synapse-cortex` before relying on a specific version. For the current source version, use the editable install below.

Editable development install:

```bash
git clone https://github.com/nafispradipta/synapse-agent.git
cd synapse-agent
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Linux and macOS
source .venv/bin/activate
```

Then install:

```bash
pip install -e ".[dev]"
```

Verify:

```bash
python -m pytest -q
python examples/quickstart.py
```

## Configuration

Copy `.env.example` to `.env` as a reference template. The library reads process environment variables directly and does not load `.env` automatically; export the file in your shell, or set the variables in your MCP client's `env` block.

### Database and storage

| Variable | Default | Effect |
| --- | --- | --- |
| `SYNAPSE_DB_PATH` | `~/.synapse/synapse.db` | SQLite database file. `:memory:` gives an isolated in-memory store. The MCP server prefers an existing `./data/synapse.db` when this variable is unset. |
| `SYNAPSE_VAULT_PATH` | unset | Markdown vault for writeback and wikilink sync. `OBSIDIAN_VAULT_DIR` is a fallback. |
| `SYNAPSE_RESEARCH_DIR` | `~/.synapse/research/auto_harvest` | Harvested research notes. |
| `SYNAPSE_CONFIG_DIR` | `~/.synapse` | Session sync state. |
| `SYNAPSE_LOGS_DIR` | `~/.synapse/logs` for the MCP server; `./logs` for scripts | Log output. |

Do not rely on the default database for project isolation. Set `SYNAPSE_DB_PATH` explicitly, or pass `db_path` to `SynapseAdapter`.

### LLM endpoints

Resolution order:

1. `SYNAPSE_BASE_URL`, `SYNAPSE_API_KEY`, `SYNAPSE_MODEL`
2. `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`
3. `BASE_URL`, `API_KEY`, `MODEL`, `MODEL_NAME`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`

A blank endpoint disables LLM calls. The library then uses deterministic heuristics and core memory keeps working.

Local runner example:

```bash
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=
OPENAI_MODEL=
```

### Embeddings

| Variable | Default | Effect |
| --- | --- | --- |
| `SYNAPSE_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | `sentence-transformers` model |
| `OPENAI_EMBEDDING_BASE_URL` | unset | Remote embedding endpoint |
| `OPENAI_EMBEDDING_API_KEY` | unset | Remote embedding credential |
| `OPENAI_EMBEDDING_MODEL` | unset | Remote embedding model |

Install local neural embeddings with `pip install sentence-transformers`. Without it, the deterministic 384-dimension CRC32 n-gram vectorizer is used.

## MCP server setup

Add an entry to the client configuration:

```json
{
  "mcpServers": {
    "synapse": {
      "command": "synapse-mcp",
      "env": {
        "SYNAPSE_DB_PATH": "/absolute/path/to/synapse.db"
      }
    }
  }
}
```

Run the server directly for inspection:

```bash
synapse-mcp
```

Alternatives: `python -m synapse.mcp` or `python scripts/mcp_server.py`.

If `SYNAPSE_DB_PATH` is unset, the server uses an existing `./data/synapse.db` relative to its working directory, otherwise `~/.synapse/synapse.db`.

The server echoes the client's requested MCP protocol version when it is one of `2024-11-05`, `2025-03-26`, or `2025-06-18`; an absent or unknown version defaults to `2025-06-18`. It exposes `synapse_recall`, `synapse_retain`, `synapse_record_episode`, and `synapse_get_stats`.

Stdout is protocol-only. If the client reports a parse error, check that nothing else writes to stdout in the same process.

## Maintenance

### In-process

`SessionManager` triggers non-blocking maintenance when the sync state says it is due. The state file lives under `SYNAPSE_CONFIG_DIR`.

### Nightly consolidation

```bash
python scripts/nightly_consolidation.py
```

It is configured through environment variables only, exits `1` when the database is missing, and appends to `logs/consolidation.log`. It uses `SYNAPSE_DATA_DIR` and `SYNAPSE_DB_PATH`; when unset, it targets `./data/synapse.db`, not the library default. It shells out to optional helper scripts through `/usr/bin/python3`, so treat it as Linux-oriented.

### WAL checkpoint

```bash
python scripts/wal_checkpoint.py
```

Folds the write-ahead log into the main database with `PRAGMA wal_checkpoint(TRUNCATE)`. When `SYNAPSE_DB_PATH` is unset, it targets `./data/synapse.db`.

### Backup and restore

1. Stop writers, or run the checkpoint script.
2. Copy the `.db` file. If the process cannot be stopped, copy `.db`, `.db-wal`, and `.db-shm` together.
3. Restore by replacing the files while no process has the database open.
4. Verify with a read-only recall and `synapse_get_stats`.

The database is plaintext. Protect backups with operating-system permissions and disk encryption.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `retain` or `recall` silently does nothing | Adapter construction failed and set `storage` to `None`. Inspect the initialization error. |
| Recall returns nothing after a restart | Confirm `db_path` and `SYNAPSE_DB_PATH` point to the same file. |
| `database is locked` | Another writer holds the lock longer than `busy_timeout` (10 seconds). Reduce concurrent writers or retry. |
| `no such module: fts5` | The Python SQLite build lacks FTS5. Use a standard CPython build or rebuild SQLite with FTS5. |
| Recall quality is poor on paraphrases | Install `sentence-transformers` or configure a remote embedding endpoint. The built-in vectorizer is lexical. |
| MCP client reports invalid JSON | Something wrote to stdout. Logs must go to `SYNAPSE_LOGS_DIR`. |
| Vault writes fail | The vault path must exist or be creatable, and the process needs write permission. Paths outside the vault are rejected by design. |
| Harvested note never appears in recall | The research record is `quarantined` and the derived graph node is `flagged` until promoted. |
| Version reported by MCP is wrong | `SERVER_VERSION` resolves from package metadata with a `pyproject.toml` fallback. Reinstall the package after changing the version. |

## Verification checklist

- `python -m pytest -q` passes.
- `PYTHONPATH=. python examples/quickstart.py` stores and recalls an in-memory fact.
- `python -c "import synapse, synapse.mcp"` succeeds without side effects.
- A retained fact is recalled in a fresh process using the same explicit database path.
- Documentation links resolve. See [Status.md](Status.md) for the current check.
