#!/usr/bin/env python3
"""
Synapse Memory Engine - Model Context Protocol (MCP) Server.
Enables external AI coding agents and Model Context Protocol (MCP) clients
to share the exact same cognitive memory cortex, SOPs, and episodic learning.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup logging
log_dir = Path(os.environ.get("SYNAPSE_LOGS_DIR") or (Path.home() / ".synapse" / "logs")).resolve()
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(log_dir / "mcp_server.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("synapse.mcp")

from synapse.adapter import SynapseAdapter

# Resolve database path: env var -> local project ./data/synapse.db -> global ~/.synapse/synapse.db
local_db = Path("./data/synapse.db").resolve()
default_db = str(local_db) if local_db.exists() else str((Path.home() / ".synapse" / "synapse.db").resolve())
DB_PATH = os.environ.get("SYNAPSE_DB_PATH", default_db)
adapter = SynapseAdapter({"db_path": DB_PATH})

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
                    "enum": ["preference", "decision", "architecture", "incident", "sop"],
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


def handle_call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if name == "synapse_recall":
            query = arguments.get("query", "")
            budget = int(arguments.get("budget", 300))
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
            category = arguments.get("category", "decision")
            salience = float(arguments.get("salience", 0.8))
            res = adapter.retain(content, category=category, salience=salience)
            return {
                "content": [{"type": "text", "text": f"Successfully retained in Synapse Memory: {res}"}]
            }

        elif name == "synapse_record_episode":
            title = arguments.get("title", "")
            situation = arguments.get("situation", "")
            action_taken = arguments.get("action_taken", "")
            outcome = arguments.get("outcome", "")
            lessons = arguments.get("lessons_learned", "")
            salience = float(arguments.get("salience", 0.9))
            res = adapter.record_episode(
                title=title,
                situation=situation,
                action_taken=action_taken,
                outcome=outcome,
                lessons_learned=lessons,
                salience=salience
            )
            return {
                "content": [{"type": "text", "text": f"Successfully recorded episode in Synapse: {res}"}]
            }

        elif name == "synapse_get_stats":
            stats = adapter.storage.get_stats() if adapter.storage else {}
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
            "content": [{"type": "text", "text": f"Error executing {name}: {str(e)}"}],
            "isError": True
        }


def send_response(response: Dict[str, Any]) -> None:
    line = json.dumps(response, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def main():
    logger.info("Synapse MCP Server started.")
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            logger.error("JSON parse error: %s", e)
            continue

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "initialize":
            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "synapse-cortex",
                        "version": "1.0.0"
                    }
                }
            })

        elif method == "notifications/initialized":
            pass

        elif method == "tools/list":
            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": TOOLS
                }
            })

        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            result = handle_call_tool(tool_name, tool_args)
            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result
            })

        elif method == "ping":
            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {}
            })

        else:
            if req_id is not None:
                send_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                })


if __name__ == "__main__":
    main()
