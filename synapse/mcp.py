#!/usr/bin/env python3
"""
Synapse Memory Engine - Model Context Protocol (MCP) Server.
Enables external AI coding agents and Model Context Protocol (MCP) clients
to share the exact same cognitive memory cortex, SOPs, and episodic learning.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup logging (configured only when the server starts, never at import time)
logger = logging.getLogger("synapse.mcp")


def _package_version() -> str:
    """Single source of truth: installed metadata first, pyproject.toml second."""
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

from synapse.adapter import SynapseAdapter
from synapse.stores.constants import VALID_CATEGORIES

_adapter: Optional[SynapseAdapter] = None

MAX_CONTENT_CHARS = 100_000
MAX_QUERY_CHARS = 4_000
MAX_EPISODE_FIELD_CHARS = 20_000
MIN_BUDGET = 1
MAX_BUDGET = 8_000


def _validate_salience(value: Any) -> float:
    """Accept only finite salience scores inside the documented [0, 1] range."""
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("salience must be a finite number between 0.0 and 1.0")
    return number


def _resolve_db_path() -> str:
    local_db = Path("./data/synapse.db").resolve()
    default_db = str(local_db) if local_db.exists() else str((Path.home() / ".synapse" / "synapse.db").resolve())
    return os.environ.get("SYNAPSE_DB_PATH", default_db)


def get_adapter() -> SynapseAdapter:
    """Lazily construct the shared adapter so importing this module is side-effect free."""
    global _adapter
    if _adapter is None:
        _adapter = SynapseAdapter({"db_path": _resolve_db_path()})
    return _adapter


def _configure_logging() -> None:
    log_dir = Path(os.environ.get("SYNAPSE_LOGS_DIR") or (Path.home() / ".synapse" / "logs")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_dir / "mcp_server.log"),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

TOOLS = [
    {
        "name": "synapse_recall",
        "description": "Recall memories, SOPs, and past architectural decisions from Synapse Memory Engine using hybrid semantic vector and BM25 search.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, topic, concept, or technical question."
                },
                "budget": {
                    "type": "integer",
                    "description": "Max token budget for recalled context (default 300).",
                    "default": 300
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "synapse_retain",
        "description": "Save an atomic fact, architectural decision, permanent directive, or SOP into Synapse Memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The atomic fact, rule, or decision content to remember."
                },
                "category": {
                    "type": "string",
                    "enum": sorted(VALID_CATEGORIES),
                    "description": "Knowledge category (default 'decision').",
                    "default": "decision"
                },
                "salience": {
                    "type": "number",
                    "description": "Importance score 0.0 to 1.0 (0.95+ for critical permanent invariants).",
                    "default": 0.8
                }
            },
            "required": ["content"]
        }
    },
    {
        "name": "synapse_record_episode",
        "description": "Record an episodic learning memory (Situation -> Action -> Outcome -> Lesson) so agents never repeat past mistakes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short concise episode title."
                },
                "situation": {
                    "type": "string",
                    "description": "Context or problem description."
                },
                "action_taken": {
                    "type": "string",
                    "description": "Action executed to address or resolve it."
                },
                "outcome": {
                    "type": "string",
                    "description": "Final result of the action."
                },
                "lessons_learned": {
                    "type": "string",
                    "description": "Key takeaways or rules discovered."
                },
                "salience": {
                    "type": "number",
                    "description": "Impact factor (default 0.9).",
                    "default": 0.9
                }
            },
            "required": ["title", "situation", "action_taken", "outcome", "lessons_learned"]
        }
    },
    {
        "name": "synapse_get_stats",
        "description": "Get real-time statistics and node counts of the Synapse memory cortex.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
]

_TOOL_NAMES = {tool["name"] for tool in TOOLS}


def handle_call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if name not in _TOOL_NAMES:
        return {
            "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
            "isError": True
        }

    adapter = get_adapter()
    try:
        if name == "synapse_recall":
            query = arguments.get("query", "")
            if not isinstance(query, str) or not query.strip():
                return {
                    "content": [{"type": "text", "text": "'query' must be a non-empty string."}],
                    "isError": True
                }
            if len(query) > MAX_QUERY_CHARS:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"'query' exceeds the maximum length of {MAX_QUERY_CHARS} characters."
                    }],
                    "isError": True
                }
            try:
                budget = int(arguments.get("budget", 300))
            except (TypeError, ValueError):
                return {
                    "content": [{"type": "text", "text": "'budget' must be an integer."}],
                    "isError": True
                }
            budget = max(MIN_BUDGET, min(MAX_BUDGET, budget))
            results = adapter.recall(query, budget=budget)
            if not results:
                return {
                    "content": [{"type": "text", "text": f"No relevant memories found for: '{query}'"}]
                }
            if isinstance(results, list):
                text_out = "\n\n".join(str(r) for r in results)
            else:
                text_out = str(results)
            return {
                "content": [{"type": "text", "text": text_out}]
            }

        elif name == "synapse_retain":
            content = arguments.get("content", "")
            if not isinstance(content, str) or not content.strip():
                return {
                    "content": [{"type": "text", "text": "'content' must be a non-empty string."}],
                    "isError": True
                }
            if len(content) > MAX_CONTENT_CHARS:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"'content' exceeds the maximum length of {MAX_CONTENT_CHARS} characters."
                    }],
                    "isError": True
                }
            category = arguments.get("category", "decision")
            if category not in VALID_CATEGORIES:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"'category' must be one of: {sorted(VALID_CATEGORIES)}"
                    }],
                    "isError": True
                }
            try:
                salience = _validate_salience(arguments.get("salience", 0.8))
            except (TypeError, ValueError):
                return {
                    "content": [{
                        "type": "text",
                        "text": "'salience' must be a finite number between 0.0 and 1.0."
                    }],
                    "isError": True
                }
            res = adapter.retain(content, category=category, salience=salience)
            if res is not True:
                return {
                    "content": [{"type": "text", "text": "Failed to retain in Synapse Memory (storage unavailable or write rejected)."}],
                    "isError": True
                }
            return {
                "content": [{"type": "text", "text": "Successfully retained in Synapse Memory: True"}]
            }

        elif name == "synapse_record_episode":
            title = arguments.get("title", "")
            if not isinstance(title, str) or not title.strip():
                return {
                    "content": [{"type": "text", "text": "'title' must be a non-empty string."}],
                    "isError": True
                }
            situation = arguments.get("situation", "")
            action_taken = arguments.get("action_taken", "")
            outcome = arguments.get("outcome", "")
            lessons = arguments.get("lessons_learned", "")
            for field_name, field_value in (
                ("situation", situation),
                ("action_taken", action_taken),
                ("outcome", outcome),
                ("lessons_learned", lessons),
            ):
                if not isinstance(field_value, str) or len(field_value) > MAX_EPISODE_FIELD_CHARS:
                    return {
                        "content": [{
                            "type": "text",
                            "text": (
                                f"'{field_name}' must be a string of at most "
                                f"{MAX_EPISODE_FIELD_CHARS} characters."
                            )
                        }],
                        "isError": True
                    }
            try:
                salience = _validate_salience(arguments.get("salience", 0.9))
            except (TypeError, ValueError):
                return {
                    "content": [{
                        "type": "text",
                        "text": "'salience' must be a finite number between 0.0 and 1.0."
                    }],
                    "isError": True
                }
            res = adapter.record_episode(
                title=title,
                situation=situation,
                action_taken=action_taken,
                outcome=outcome,
                lessons_learned=lessons,
                salience=salience
            )
            if not res:
                return {
                    "content": [{"type": "text", "text": "Failed to record episode in Synapse."}],
                    "isError": True
                }
            return {
                "content": [{"type": "text", "text": f"Successfully recorded episode in Synapse: {res}"}]
            }

        elif name == "synapse_get_stats":
            if adapter.storage is None:
                return {
                    "content": [{"type": "text", "text": "Synapse storage is unavailable."}],
                    "isError": True
                }
            stats = adapter.storage.get_stats()
            return {
                "content": [{"type": "text", "text": json.dumps(stats, indent=2)}]
            }

        else:
            return {
                "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                "isError": True
            }

    except Exception as e:
        logger.error("Error executing tool %s: %s", name, e, exc_info=True)
        return {
            "content": [{"type": "text", "text": f"Error executing {name}: internal error"}],
            "isError": True
        }


def _configure_stdio() -> None:
    """Force UTF-8 on stdin/stdout; the host may otherwise use a locale codec."""
    for stream in (sys.stdin, sys.stdout):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="strict")
            except Exception:
                pass


def send_response(response: Dict[str, Any]) -> None:
    line = json.dumps(response, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _error(req_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {
            "code": code,
            "message": message
        }
    }


def dispatch(req: Any) -> Optional[Dict[str, Any]]:
    """Handle one parsed request. Returns the response dict, or None for notifications."""
    if not isinstance(req, dict):
        return _error(None, -32600, "Invalid Request")

    req_id = req.get("id")
    is_notification = "id" not in req
    method = req.get("method")
    params = req.get("params")
    if params is None:
        params = {}

    if not isinstance(method, str) or not method:
        if is_notification:
            return None
        return _error(req_id, -32600, "Invalid Request")
    if not isinstance(params, dict):
        if is_notification:
            return None
        return _error(req_id, -32602, "Invalid params")

    if method == "initialize":
        response = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "synapse-cortex",
                    "version": SERVER_VERSION
                }
            }
        }

    elif method == "notifications/initialized":
        return None

    elif method == "tools/list":
        response = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": TOOLS
            }
        }

    elif method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments")
        if tool_args is None:
            tool_args = {}
        if not isinstance(tool_args, dict):
            if is_notification:
                return None
            return _error(req_id, -32602, "Invalid params")
        response = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": handle_call_tool(tool_name, tool_args)
        }

    elif method == "ping":
        response = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {}
        }

    else:
        if is_notification:
            return None
        return _error(req_id, -32601, f"Method not found: {method}")

    return None if is_notification else response


def main():
    _configure_logging()
    _configure_stdio()
    logger.info("Synapse MCP Server started.")
    stdin = getattr(sys.stdin, "buffer", sys.stdin)
    for raw_line in stdin:
        if isinstance(raw_line, bytes):
            try:
                line = raw_line.decode("utf-8").strip()
            except UnicodeDecodeError:
                logger.error("Invalid UTF-8 input line")
                send_response(_error(None, -32700, "Parse error"))
                continue
        else:
            line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            logger.error("JSON parse error: %s", e)
            send_response(_error(None, -32700, "Parse error"))
            continue

        try:
            response = dispatch(req)
        except Exception as e:
            logger.error("Unhandled request error: %s", e, exc_info=True)
            req_id = req.get("id") if isinstance(req, dict) else None
            response = _error(req_id, -32603, "Internal error")

        if response is not None:
            send_response(response)


if __name__ == "__main__":
    main()
