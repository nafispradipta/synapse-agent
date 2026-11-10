"""Registry of AI harnesses and their MCP configuration locations.

Stdlib only. All paths are resolved lazily so importing this module has no
side effects.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

HOME = Path.home()


def _env(name: str) -> Optional[str]:
    v = os.environ.get(name)
    return v if v else None


def _appdata() -> Path:
    return Path(_env("APPDATA") or (HOME / "AppData" / "Roaming"))


def _config_home() -> Path:
    return Path(_env("XDG_CONFIG_HOME") or (HOME / ".config"))


@dataclass
class Harness:
    key: str
    display: str
    format: str  # "json" or "toml"
    # Returns (path, detected_bool)
    detect: Callable[[], bool]
    path: Callable[[], Path]
    # Merge synapse entry into parsed config dict (for json) or text (toml handled separately)
    merge: Callable[[Dict[str, Any], str], Dict[str, Any]] = field(default=None)


def _detect_path(p: Path) -> bool:
    return p.exists() or p.parent.exists()


def _json_merge_mcp_servers(cfg: Dict[str, Any], db_path: str) -> Dict[str, Any]:
    cfg.setdefault("mcpServers", {})
    cfg["mcpServers"]["synapse"] = {
        "command": "synapse-mcp",
        "env": {"SYNAPSE_DB_PATH": db_path},
    }
    return cfg


def _json_merge_opencode(cfg: Dict[str, Any], db_path: str) -> Dict[str, Any]:
    mcp = cfg.setdefault("mcp", {})
    servers = mcp.setdefault("servers", {})
    servers["synapse"] = {
        "type": "local",
        "command": ["synapse-mcp"],
        "environment": {"SYNAPSE_DB_PATH": db_path},
        "enabled": True,
    }
    return cfg


def sys_platform() -> str:
    import sys

    return sys.platform


def _detect_claude_desktop() -> bool:
    if os.name == "nt":
        return (_appdata() / "Claude").exists()
    if sys_platform() == "darwin":
        return (HOME / "Library" / "Application Support" / "Claude").exists()
    return (_config_home() / "Claude").exists()


def _path_claude_desktop() -> Path:
    if os.name == "nt":
        return _appdata() / "Claude" / "claude_desktop_config.json"
    if sys_platform() == "darwin":
        return HOME / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    return _config_home() / "Claude" / "claude_desktop_config.json"


def _detect_claude_code() -> bool:
    return (HOME / ".claude").exists() or shutil.which("claude") is not None


def _path_claude_code() -> Path:
    return HOME / ".claude.json"


def _detect_codex() -> bool:
    return (HOME / ".codex").exists() or shutil.which("codex") is not None


def _path_codex() -> Path:
    return HOME / ".codex" / "config.toml"


def _detect_opencode() -> bool:
    return (_config_home() / "opencode").exists() or shutil.which("opencode") is not None


def _path_opencode() -> Path:
    d = _config_home() / "opencode"
    if (d / "opencode.json").exists():
        return d / "opencode.json"
    if (d / "opencode.jsonc").exists():
        return d / "opencode.jsonc"
    return d / "opencode.json"


REGISTRY: List[Harness] = [
    Harness(
        key="claude-desktop",
        display="Claude Desktop",
        format="json",
        detect=_detect_claude_desktop,
        path=_path_claude_desktop,
        merge=_json_merge_mcp_servers,
    ),
    Harness(
        key="claude-code",
        display="Claude Code",
        format="json",
        detect=_detect_claude_code,
        path=_path_claude_code,
        merge=_json_merge_mcp_servers,
    ),
    Harness(
        key="codex",
        display="Codex (OpenAI CLI)",
        format="toml",
        detect=_detect_codex,
        path=_path_codex,
        merge=None,
    ),
    Harness(
        key="opencode",
        display="OpenCode",
        format="json",
        detect=_detect_opencode,
        path=_path_opencode,
        merge=_json_merge_opencode,
    ),
]


def get_harness(key: str) -> Optional[Harness]:
    for h in REGISTRY:
        if h.key == key:
            return h
    return None


def detected_harnesses() -> List[Harness]:
    return [h for h in REGISTRY if h.detect()]


def all_harnesses() -> List[Harness]:
    return list(REGISTRY)
