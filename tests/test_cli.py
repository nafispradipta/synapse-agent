import json
from pathlib import Path

import pytest

from synapse import cli, harnesses


def test_resolve_db_path_explicit(tmp_path):
    p = cli._resolve_db_path(str(tmp_path / "x.db"))
    assert p == (tmp_path / "x.db").resolve()


def test_read_write_json_roundtrip(tmp_path):
    p = tmp_path / "cfg.json"
    cli._write_json(p, {"a": 1})
    assert cli._read_json(p) == {"a": 1}
    # backup created on overwrite
    cli._write_json(p, {"a": 2})
    assert (tmp_path / "cfg.json.bak").exists()


def test_json_merge_mcp_servers():
    cfg = {}
    out = harnesses._json_merge_mcp_servers(cfg, "/db/x.db")
    assert out["mcpServers"]["synapse"]["env"]["SYNAPSE_DB_PATH"] == "/db/x.db"


def test_opencode_merge():
    cfg = {}
    out = harnesses._json_merge_opencode(cfg, "/db/x.db")
    assert out["mcp"]["servers"]["synapse"]["type"] == "local"
    assert out["mcp"]["servers"]["synapse"]["environment"]["SYNAPSE_DB_PATH"] == "/db/x.db"


def test_codex_merge_insert_and_replace(tmp_path):
    p = tmp_path / "config.toml"
    text = ""
    new = cli._merge_codex(text, "/db/x.db")
    assert "[mcp_servers.synapse]" in new
    # replace
    new2 = cli._merge_codex(new, "/db/y.db")
    assert new2.count("[mcp_servers.synapse]") == 1
    assert "/db/y.db" in new2
    assert "/db/x.db" not in new2


def test_setup_noninteractive_tmp(monkeypatch, tmp_path):
    # redirect all harness paths to tmp
    fake = tmp_path / "cfg.json"

    class H:
        key = "t"
        display = "T"
        format = "json"
        detect = staticmethod(lambda: True)
        path = staticmethod(lambda: fake)
        merge = staticmethod(harnesses._json_merge_mcp_servers)

    monkeypatch.setattr(harnesses, "all_harnesses", lambda: [H])
    monkeypatch.setattr(harnesses, "get_harness", lambda k: H if k == "t" else None)
    monkeypatch.setattr(cli, "get_harness", lambda k: H if k == "t" else None)
    monkeypatch.setattr(cli, "all_harnesses", lambda: [H])

    db = tmp_path / "data" / "synapse.db"
    rc = cli.cmd_setup(
        __import__("argparse").Namespace(
            client=["t"], db_path=str(db), dry_run=False, yes=True
        )
    )
    assert rc == 0
    cfg = json.loads(fake.read_text())
    assert "synapse" in cfg["mcpServers"]
    assert db.exists()


def test_doctor_returns_int(monkeypatch, tmp_path):
    db = tmp_path / "d.db"
    rc = cli.cmd_doctor(__import__("argparse").Namespace(db_path=str(db)))
    assert rc in (0, 1)
