# Harness Setup

How to connect Synapse Agent's memory MCP server to your AI harness.

## Quick start (recommended)

```bash
pip install -e ".[dev]"   # or: pip install synapse-cortex
synapse setup             # interactive: detects harnesses, writes config
synapse doctor            # verify
```

`synapse setup` auto-detects installed harnesses (Claude Desktop, Claude
Code, Codex, OpenCode) and writes the correct config file for your OS.
Use `--client <key>` for non-interactive mode, e.g.
`synapse setup --client codex --db-path /path/to/db --yes`.

## Manual configuration

If you prefer, copy the template from `examples/mcp-clients/` into the
harness config location. Standard block:

```json
{
  "mcpServers": {
    "synapse": {
      "command": "synapse-mcp",
      "env": { "SYNAPSE_DB_PATH": "/absolute/path/to/synapse.db" }
    }
  }
}
```

### Claude Desktop
File: `%APPDATA%\Claude\claude_desktop_config.json` (Windows) /
`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) /
`~/.config/Claude/claude_desktop_config.json` (Linux).

### Claude Code
File: `~/.claude.json` (user scope) or `.mcp.json` (project scope).
Or run: `claude mcp add synapse -- synapse-mcp`.

### Codex (OpenAI CLI)
File: `~/.codex/config.toml`:
```toml
[mcp_servers.synapse]
command = "synapse-mcp"
env = { SYNAPSE_DB_PATH = "/absolute/path/to/synapse.db" }
```

### OpenCode
File: `~/.config/opencode/opencode.json` (or `.jsonc`):
```json
{
  "mcp": {
    "servers": {
      "synapse": {
        "type": "local",
        "command": ["synapse-mcp"],
        "environment": { "SYNAPSE_DB_PATH": "/absolute/path/to/synapse.db" },
        "enabled": true
      }
    }
  }
}
```

## Environment variables

Set `SYNAPSE_DB_PATH` to keep per-project memory isolated. Never rely on
the default `~/.synapse/synapse.db` across projects.

## Troubleshooting

- `synapse doctor` reports missing synapse entry -> re-run `synapse setup`.
- MCP client reports invalid JSON -> ensure nothing else writes to stdout.
- Storage open fails -> check file permissions and SQLite FTS5 support.
