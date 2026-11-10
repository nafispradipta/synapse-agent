"""Command-line interface for Synapse Agent.

Stdlib only. Provides `synapse setup` and `synapse doctor`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional

from synapse.harnesses import (
    REGISTRY,
    Harness,
    all_harnesses,
    detected_harnesses,
    get_harness,
)


def _strip_json_comments(text: str) -> str:
    lines = text.splitlines()
    out = []
    for line in lines:
        # Remove trailing // comments (naive but adequate for config files)
        idx = line.find("//")
        if idx != -1:
            # keep if inside string? Heuristic: only strip if // not preceded by :
            # Simple approach: strip trailing comment
            line = line[:idx]
        out.append(line)
    return "\n".join(out)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return json.loads(_strip_json_comments(raw))
        except json.JSONDecodeError:
            return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        try:
            shutil.copy2(path, backup)
        except OSError:
            pass
    os.replace(tmp, path)


def _merge_codex(text: str, db_path: str) -> str:
    section = "[mcp_servers.synapse]"
    block = (
        "\n" + section + "\n"
        'command = "synapse-mcp"\n'
        "env = { SYNAPSE_DB_PATH = \"" + db_path.replace("\\", "\\\\") + "\" }\n"
    )
    if section in text:
        # Replace existing section until next top-level section or EOF
        lines = text.splitlines(keepends=True)
        start = None
        for i, ln in enumerate(lines):
            if ln.strip() == section:
                start = i
                break
        if start is not None:
            end = len(lines)
            for j in range(start + 1, len(lines)):
                if lines[j].startswith("[") and lines[j].strip().endswith("]"):
                    end = j
                    break
            lines = lines[:start] + [block.lstrip("\n")] + lines[end:]
            return "".join(lines)
    # append
    if text and not text.endswith("\n"):
        text += "\n"
    return text + block


def _resolve_db_path(db_path: Optional[str]) -> Path:
    if db_path:
        return Path(db_path).expanduser().resolve()
    # mirror mcp resolution: existing ./data/synapse.db else ~/.synapse/synapse.db
    local = Path("data") / "synapse.db"
    if local.exists():
        return local.resolve()
    return (Path.home() / ".synapse" / "synapse.db").resolve()


def _detect_installed() -> List[Harness]:
    return [h for h in all_harnesses() if h.detect()]


def _prompt_choice(items: List[str], prompt: str) -> List[int]:
    print(prompt)
    for i, it in enumerate(items, 1):
        print(f"  {i}. {it}")
    print("  (comma-separated numbers, or 'all')")
    raw = input("> ").strip().lower()
    if raw == "all":
        return list(range(len(items)))
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(items):
                out.append(idx)
    return out


def cmd_setup(args: argparse.Namespace) -> int:
    harnesses = all_harnesses()
    if args.client:
        selected = []
        for c in args.client:
            h = get_harness(c)
            if h is None:
                print(f"Unknown client: {c}")
                return 2
            selected.append(h)
    else:
        detected = _detect_installed()
        print("Synapse Agent setup")
        print("====================")
        if detected:
            print("Detected harnesses:")
            for h in detected:
                print(f"  - {h.display} ({h.key})")
        names = [f"{h.display} ({h.key})" for h in harnesses]
        idxs = _prompt_choice(names, "Select harnesses to configure:")
        if not idxs:
            print("No harness selected. Aborting.")
            return 1
        selected = [harnesses[i] for i in idxs]

    db_path = _resolve_db_path(args.db_path)
    db_str = str(db_path)

    if args.dry_run:
        print(f"[dry-run] DB path: {db_str}")
        for h in selected:
            print(f"[dry-run] Would update: {h.path()}")
        return 0

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        db_path.touch()

    for h in selected:
        p = h.path()
        print(f"Configuring {h.display} -> {p}")
        if h.format == "json":
            cfg = _read_json(p)
            cfg = h.merge(cfg, db_str)
            _write_json(p, cfg)
        elif h.format == "toml":
            text = p.read_text(encoding="utf-8") if p.exists() else ""
            p.parent.mkdir(parents=True, exist_ok=True)
            new_text = _merge_codex(text, db_str)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(new_text, encoding="utf-8")
            if p.exists():
                try:
                    shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))
                except OSError:
                    pass
            os.replace(tmp, p)
        else:
            print(f"  Unsupported format: {h.format}")
            continue
        print("  OK")

    print()
    print(f"Database: {db_str}")
    print("Run `synapse doctor` to verify.")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    print("Synapse Agent doctor")
    print("====================")
    ok = True

    # Python version
    if sys.version_info >= (3, 9):
        print(f"[OK] Python {sys.version.split()[0]}")
    else:
        print(f"[FAIL] Python {sys.version.split()[0]} (need >=3.9)")
        ok = False

    # Import
    try:
        import synapse
        import synapse.mcp

        try:
            from importlib.metadata import version as _v

            ver = _v("synapse-cortex")
        except Exception:
            ver = getattr(synapse, "__version__", "unknown")
        print(f"[OK] import synapse, synapse.mcp (version {ver})")
    except Exception as e:
        print(f"[FAIL] import failed: {e}")
        ok = False

    # Optional deps
    try:
        import sentence_transformers  # noqa: F401

        print("[OK] sentence-transformers available (neural embeddings)")
    except ImportError:
        print("[INFO] sentence-transformers not installed (CRC32 fallback will be used)")

    # LLM env
    llm = os.environ.get("SYNAPSE_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")
    if llm:
        print(f"[OK] LLM endpoint configured: {llm}")
    else:
        print("[INFO] No LLM endpoint configured (optional; core memory works without it)")

    # DB
    db_path = _resolve_db_path(args.db_path)
    print(f"DB path: {db_path}")
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if not db_path.exists():
            db_path.touch()
        # open storage to verify + init schema
        from synapse.storage import SynapseStorage

        st = SynapseStorage(str(db_path))
        stats = st.get_stats()
        print(f"[OK] Storage opened (nodes={stats.get('node_count', '?')})")
        st.close() if hasattr(st, "close") else None
    except Exception as e:
        print(f"[FAIL] Storage open failed: {e}")
        ok = False

    # Harness configs
    for h in all_harnesses():
        p = h.path()
        if not p.exists():
            print(f"[INFO] {h.display}: not configured ({p})")
            continue
        try:
            if h.format == "json":
                cfg = _read_json(p)
                has = False
                if h.key == "opencode":
                    has = "synapse" in (cfg.get("mcp", {}).get("servers", {}) or {})
                else:
                    has = "synapse" in (cfg.get("mcpServers", {}) or {})
                if has:
                    print(f"[OK] {h.display}: synapse entry present")
                else:
                    print(f"[WARN] {h.display}: config exists but no synapse entry")
            elif h.format == "toml":
                text = p.read_text(encoding="utf-8")
                if "[mcp_servers.synapse]" in text:
                    print(f"[OK] {h.display}: synapse entry present")
                else:
                    print(f"[WARN] {h.display}: config exists but no synapse entry")
        except Exception as e:
            print(f"[FAIL] {h.display}: read error {e}")
            ok = False

    print()
    if ok:
        print("All checks passed.")
        return 0
    print("Some checks failed. See above.")
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="synapse", description="Synapse Agent CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_setup = sub.add_parser("setup", help="Configure MCP for AI harnesses")
    p_setup.add_argument("--client", action="append", help="Harness key (repeatable)")
    p_setup.add_argument("--db-path", help="SQLite database path")
    p_setup.add_argument("--yes", action="store_true", help="Non-interactive (use defaults)")
    p_setup.add_argument("--dry-run", action="store_true", help="Show actions without writing")
    p_setup.set_defaults(func=cmd_setup)

    p_doctor = sub.add_parser("doctor", help="Verify installation and configuration")
    p_doctor.add_argument("--db-path", help="SQLite database path override")
    p_doctor.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())