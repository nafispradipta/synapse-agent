# Security and Privacy

## Scope

This document covers the library, the MCP stdio server, the Markdown vault integration, and the operational scripts. It describes controls that exist in the current code. It does not claim certifications or guarantees the project does not have.

## Trust boundaries

| Boundary | Trust level | Notes |
| --- | --- | --- |
| Local process and environment variables | Trusted | Anyone who can set environment variables or read process memory can read the database. |
| SQLite database file | Sensitive | Plaintext. Protect with filesystem permissions and disk encryption. |
| MCP client over stdio | Local caller | No authentication layer. The parent process controls access. |
| Markdown vault | Sensitive | Plaintext notes, written only inside the configured vault. |
| Harvested web content | Untrusted | Research records are `quarantined`; derived nodes are `flagged`; both are excluded from recall until promoted. |
| Remote LLM or embedding endpoint | External | Content sent for inference leaves the machine. |

## Data at rest

- The database stores memory text, embeddings, episodes, mental models, observations, and research records in plaintext SQLite.
- Markdown writeback stores note content in plaintext.
- There is no built-in encryption at rest. Use OS permissions, an encrypted volume, or full-disk encryption.
- Logs may contain database paths, endpoint URLs, and error messages. They do not log API keys by design, but operators must still protect log files.
- Deleting a database file is the supported way to erase the memory. Backups and vault copies must be removed separately.

## Credentials

- Credentials come from the adapter/config dictionary or environment variables. The library does not load `.env` by itself.
- Precedence for LLM credentials is the explicit config dictionary first (`api_key`, `synapse_api_key`, `openai_api_key`), then `SYNAPSE_API_KEY`, then `OPENAI_API_KEY`, then `API_KEY`, then `LLM_API_KEY`.
- The MCP configuration passes credentials through the client's environment block. Do not paste keys into shared configuration repositories.
- A blank key is valid for local runners that do not require authentication.

## Remote calls

- LLM completion is sent to the configured OpenAI-compatible `/chat/completions` endpoint.
- Embeddings are sent to the configured embedding endpoint when remote embeddings are enabled. Remote vectors use the provider's dimension, which may not be 384; local MiniLM and fallback vectors are 384-dimensional.
- No endpoint is contacted when none is configured. Core memory operations remain local.
- Choosing a local runner such as Ollama or vLLM keeps inference on the machine.

## Web harvesting controls

- Only `http` and `https` URLs are accepted.
- The host must match the whitelist exactly or be a subdomain of a whitelisted domain.
- Userinfo-spoofed hosts are rejected.
- Every redirect hop is validated against the whitelist, with a maximum of 5 redirects.
- `Content-Length` is checked before download and the body is capped at 2 MiB.
- Harvested research records and vault notes are stored with `status="quarantined"`; the derived graph node uses `status="flagged"`. Neither appears in recall until promoted.

These controls reduce server-side request forgery and unbounded download risk. They do not make arbitrary web content trustworthy.

## Vault write controls

- The target path is resolved and must remain inside the configured vault directory.
- Metadata slugs are normalized to a single safe file stem.
- Frontmatter values are escaped so quotes, newlines, and YAML special characters cannot inject new fields.
- Target directories are created only after the path is contained.

## Input and data integrity

- MCP tools enforce content, query, episode, and budget limits, and validate salience and category.
- Numeric writes reject NaN and infinity and clamp fields to their valid ranges.
- SQL uses parameterized queries.
- Failed writes roll back at the outermost write level; the rollback depth is thread-local.
- MCP errors return a generic message instead of raw exception text.

## Protocol safety

`synapse/mcp.py` is a hand-rolled stdio JSON-RPC loop. Stdout carries protocol messages only; logging goes to a file. Malformed input receives standard JSON-RPC error codes. Importing the module creates no database or log directory.

## Privacy notes

- The default library database is `~/.synapse/synapse.db`. The MCP server uses `SYNAPSE_DB_PATH` when set, then an existing `./data/synapse.db`, then `~/.synapse/synapse.db`. Assistants on the same machine can share one database. Use an explicit database path when memories must not mix.
- Memory content can contain personal or confidential information supplied by the user or the agent. Retain only what is appropriate for the configured storage location.
- There is no tenant isolation or access control inside one database.
- Prompt injection is a residual risk: untrusted harvested content, once promoted, can influence later agent context. Keep quarantine promotion deliberate.

## Residual risks

- Plaintext storage and backups.
- No MCP authentication; security depends on local process boundaries.
- Trust in the configured LLM or embedding provider.
- Memory poisoning through retained malicious or incorrect content.
- No automatic secret scanning of retained text.

## Reporting a vulnerability

Follow [SECURITY.md](../SECURITY.md). Do not open a public issue for an undisclosed vulnerability. Reports go to `halo@nafispradipta.id`, with acknowledgement targeted within 48 hours.

## Supported versions

Security fixes target the `1.1.x` line in the repository. The latest published PyPI release may lag the source version; check PyPI before relying on a release. See [Status.md](Status.md).
