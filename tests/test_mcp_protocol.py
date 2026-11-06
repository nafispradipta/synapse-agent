"""End-to-end JSON-RPC protocol tests for `python -m synapse.mcp`."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _env(tmp_path, db_path=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["SYNAPSE_DB_PATH"] = str(db_path or (tmp_path / "mcp.db"))
    env["SYNAPSE_LOGS_DIR"] = str(tmp_path / "logs")
    env.pop("PYTHONIOENCODING", None)
    env.pop("PYTHONUTF8", None)
    return env


def _run(tmp_path, lines, db_path=None):
    payload = "".join(
        (line if isinstance(line, str) else json.dumps(line)) + "\n"
        for line in lines
    )
    completed = subprocess.run(
        [sys.executable, "-m", "synapse.mcp"],
        input=payload.encode("utf-8"),
        capture_output=True,
        cwd=str(tmp_path),
        env=_env(tmp_path, db_path),
        timeout=120,
    )
    stdout = completed.stdout.decode("utf-8")
    responses = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    return completed, responses


def _request(req_id, method, params=None):
    body = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        body["params"] = params
    return body


def _call(req_id, name, arguments):
    return _request(req_id, "tools/call", {"name": name, "arguments": arguments})


def test_initialize_tools_and_metadata(tmp_path):
    completed, responses = _run(
        tmp_path,
        [
            _request(1, "initialize"),
            _request(2, "tools/list"),
            _request(3, "ping"),
        ],
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert [r["id"] for r in responses] == [1, 2, 3]

    init = responses[0]["result"]
    assert init["protocolVersion"] == "2024-11-05"
    assert init["serverInfo"]["name"] == "synapse-cortex"

    version_line = re.search(
        r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(encoding="utf-8"), re.M
    )
    assert init["serverInfo"]["version"] == version_line.group(1)

    tools = {tool["name"]: tool for tool in responses[1]["result"]["tools"]}
    assert {"synapse_recall", "synapse_retain", "synapse_get_stats"} <= set(tools)
    category = tools["synapse_retain"]["inputSchema"]["properties"]["category"]
    assert "general" in category["enum"]


def test_non_ascii_round_trip(tmp_path):
    db_path = tmp_path / "unicode.db"
    content = "caf\u00e9 \u2615 \u65e5\u672c\u8a9e"
    completed, responses = _run(
        tmp_path,
        [
            _call(1, "synapse_retain", {"content": content, "category": "general"}),
            _call(2, "synapse_recall", {"query": "caf\u00e9"}),
        ],
        db_path=db_path,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert "isError" not in responses[0]["result"]
    assert "caf\u00e9" in responses[1]["result"]["content"][0]["text"]

    connection = sqlite3.connect(str(db_path))
    try:
        stored = connection.execute(
            "SELECT l0_abstract FROM synapse_nodes"
        ).fetchone()[0]
    finally:
        connection.close()
    assert stored == content


def test_invalid_framing_notifications_and_recovery(tmp_path):
    completed, responses = _run(
        tmp_path,
        [
            "[1, 2, 3]",
            "{not json",
            {"jsonrpc": "2.0", "method": "tools/list"},
            _request(9, "ping"),
        ],
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert [r["id"] for r in responses] == [None, None, 9]
    assert responses[0]["error"]["code"] == -32600
    assert responses[1]["error"]["code"] == -32700
    assert responses[2]["result"] == {}


def test_failed_write_is_reported_as_error(tmp_path):
    completed, responses = _run(
        tmp_path,
        [_call(1, "synapse_retain", {"content": "hello", "category": "general"})],
        db_path=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    result = responses[0]["result"]
    assert result.get("isError") is True
    assert "Successfully" not in result["content"][0]["text"]


def test_import_has_no_filesystem_side_effects(tmp_path):
    home = tmp_path / "home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["USERPROFILE"] = str(home)
    env["HOME"] = str(home)
    env.pop("SYNAPSE_DB_PATH", None)
    env.pop("SYNAPSE_LOGS_DIR", None)
    completed = subprocess.run(
        [sys.executable, "-c", "import synapse.mcp"],
        capture_output=True,
        cwd=str(work),
        env=env,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert [p for p in home.rglob("*") if p.is_file()] == []
    assert [p for p in work.rglob("*") if p.is_file()] == []


def test_invalid_utf8_bytes_get_parse_error_not_crash(tmp_path):
    env = _env(tmp_path)
    payload = (
        b"\xff\xfe not valid utf-8\n"
        + json.dumps(_request(9, "ping")).encode("utf-8")
        + b"\n"
    )
    completed = subprocess.run(
        [sys.executable, "-m", "synapse.mcp"],
        input=payload,
        capture_output=True,
        cwd=str(tmp_path),
        env=env,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    responses = [
        json.loads(line)
        for line in completed.stdout.decode("utf-8").splitlines()
        if line.strip()
    ]
    assert responses[0]["error"]["code"] == -32700
    assert responses[0]["id"] is None
    assert responses[-1]["result"] == {}


def test_params_array_is_invalid_params(tmp_path):
    completed, responses = _run(tmp_path, [_request(11, "initialize", [])])
    assert responses[0]["error"]["code"] == -32602


def test_error_code_matrix(tmp_path):
    completed, responses = _run(
        tmp_path,
        [
            _request(1, "bogus/method"),
            {"jsonrpc": "2.0", "method": "bogus/method"},
            _call(3, "synapse_retain", []),
            _request(4, "ping"),
        ],
    )
    assert responses[0]["error"]["code"] == -32601
    assert [r["id"] for r in responses] == [1, 3, 4]
    assert responses[1]["error"]["code"] == -32602
    assert responses[2]["result"] == {}


def test_nan_salience_is_rejected(tmp_path):
    completed, responses = _run(
        tmp_path,
        [
            _request(1, "initialize"),
            _call(2, "synapse_retain", {"content": "probe", "salience": float("nan")}),
            _call(3, "synapse_recall", {"query": "probe"}),
        ],
    )
    assert completed.returncode == 0
    assert responses[1]["result"]["isError"] is True
    # The rejected write must not land: the probe is absent from recall.
    assert "No relevant memories" in json.dumps(responses[2]["result"])


def test_content_at_the_cap_is_accepted(tmp_path):
    completed, responses = _run(
        tmp_path,
        [
            _request(1, "initialize"),
            _call(2, "synapse_retain", {"content": "x" * 100_000}),
        ],
    )
    assert "isError" not in responses[1].get("result", {})


def test_oversized_content_is_rejected(tmp_path):
    big = "x" * 100_001
    completed, responses = _run(
        tmp_path,
        [
            _request(1, "initialize"),
            _call(2, "synapse_retain", {"content": big}),
        ],
    )
    assert responses[1]["result"]["isError"] is True
    assert "100000" in responses[1]["result"]["content"][0]["text"]


@pytest.mark.parametrize("budget", [0, -5, 10**12])
def test_budget_is_clamped_not_crashing(tmp_path, budget):
    completed, responses = _run(
        tmp_path,
        [
            _request(1, "initialize"),
            _call(2, "synapse_recall", {"query": "x", "budget": budget}),
        ],
    )
    assert completed.returncode == 0
    assert "isError" not in responses[1].get("result", {})

