"""
Unit tests for Synapse Memory to Obsidian Vault Auto-Writeback feature in synapse/adapter.py.
"""

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synapse.adapter import SynapseAdapter


@pytest.fixture
def mock_vault(tmp_path: Path) -> Path:
    """Create a temporary mock obsidian vault with tags.md and index files."""
    vault = tmp_path / "vault"
    vault.mkdir()

    # Create official tags.md
    tags_file = vault / "tags.md"
    tags_file.write_text(
        "# Official Tag Dictionary\n"
        "List of official tags:\n\n"
        "- #idea\n"
        "- #meeting\n"
        "- #todo\n"
        "- #devops\n"
        "- #ai\n"
        "- #marketing\n"
        "- #finance\n"
        "- #personal\n"
        "- #server\n"
        "- #sop\n",
        encoding="utf-8",
    )

    # Create server index
    server_dir = vault / "server"
    server_dir.mkdir()
    (server_dir / "00-INDEX-SERVER.md").write_text(
        "# Server Index\n\n"
        "List of connected documents in this folder:\n\n"
        "- [[sop-server-network]]\n"
        "- [[server-folder-structure]]\n\n"
        "## Archive & Historical Documents\n"
        "- [[archive-legacy-system-migration]]\n",
        encoding="utf-8",
    )

    # Create projek index
    projek_dir = vault / "projek"
    projek_dir.mkdir()
    (projek_dir / "00-INDEX-PROJEK.md").write_text(
        "# Master Project Index\n\n"
        "- [[synapse-agent-architecture]] — Master Blueprint\n",
        encoding="utf-8",
    )

    return vault


def test_autowriteback_server_devops_sop(mock_vault: Path, tmp_path: Path) -> None:
    """Test retain with salience >= 0.85 and server/devops SOP writes to ~/vault/server/."""
    db_file = str(tmp_path / "synapse_wb.db")
    adapter = SynapseAdapter({"db_path": db_file, "vault_path": str(mock_vault)})

    ok = adapter.retain(
        content="Standard operating procedure for reverse proxy service restart and internal zero-trust bind.",
        context="Server DevOps deployment",
        category="sop",
        salience=0.92,
        title="SOP Internal Reverse Proxy",
    )
    assert ok is True

    # Check file exists in server/
    server_dir = mock_vault / "server"
    written_files = list(server_dir.glob("*.md"))
    # Exclude 00-INDEX-SERVER.md
    notes = [f for f in written_files if f.name != "00-INDEX-SERVER.md"]
    assert len(notes) == 1
    note_path = notes[0]

    # Verify content and frontmatter
    text = note_path.read_text(encoding="utf-8")
    assert "status: active" in text
    assert "salience: 0.92" in text
    assert f"date: {datetime.date.today().isoformat()}" in text
    assert 'title: "SOP Internal Reverse Proxy"' in text
    assert "[[00-INDEX-SERVER]]" in text
    assert "## Summary" in text
    assert "## Details & Context" in text

    # Verify tags comply with tags.md
    assert "server" in text or "sop" in text or "devops" in text

    # Verify bidirectional link in 00-INDEX-SERVER.md
    index_text = (server_dir / "00-INDEX-SERVER.md").read_text(encoding="utf-8")
    assert f"[[{note_path.stem}]]" in index_text

    # Verify l2_target_path in node storage
    nodes = adapter.storage.list_nodes(limit=1)
    assert len(nodes) == 1
    assert nodes[0]["l2_target_path"] == str(note_path)


def test_autowriteback_architecture_projek(mock_vault: Path, tmp_path: Path) -> None:
    """Test retain with salience >= 0.85 and category architecture writes to ~/vault/projek/."""
    db_file = str(tmp_path / "synapse_arch.db")
    adapter = SynapseAdapter({"db_path": db_file, "vault_path": str(mock_vault)})

    ok = adapter.retain(
        content="Hybrid dense vector and BM25 memory graph architecture with spreading activation learning.",
        context="Cognitive architecture design",
        category="architecture",
        salience=0.88,
        title="Cognitive Synapse Graph Architecture",
    )
    assert ok is True

    projek_dir = mock_vault / "projek"
    notes = [f for f in projek_dir.glob("*.md") if f.name != "00-INDEX-PROJEK.md"]
    assert len(notes) == 1
    note_path = notes[0]

    text = note_path.read_text(encoding="utf-8")
    assert "status: active" in text
    assert "salience: 0.88" in text
    assert "[[00-INDEX-PROJEK]]" in text
    assert "tags: [" in text

    # Verify bidirectional index
    index_text = (projek_dir / "00-INDEX-PROJEK.md").read_text(encoding="utf-8")
    assert f"[[{note_path.stem}]]" in index_text


def test_autowriteback_preference_and_decision_synapse_knowledge(mock_vault: Path, tmp_path: Path) -> None:
    """Test retain with decision and preference writes to ~/vault/projek/synapse-knowledge/."""
    db_file = str(tmp_path / "synapse_pref.db")
    adapter = SynapseAdapter({"db_path": db_file, "vault_path": str(mock_vault)})

    # Preference writeback
    ok1 = adapter.retain(
        content="System architect prefers deep modular architecture with concise interfaces rather than shallow interfaces.",
        category="preference",
        salience=0.85,
        title="Deep Modules Design Preference",
    )
    assert ok1 is True

    # Decision writeback
    ok2 = adapter.retain(
        content="Architectural decision adopting SQLite WAL mode as ground truth graph database for autonomous agents.",
        category="decision",
        salience=0.90,
        title="SQLite WAL Mode Decision",
    )
    assert ok2 is True

    sk_dir = mock_vault / "projek" / "synapse-knowledge"
    assert sk_dir.exists()

    notes = [f for f in sk_dir.glob("*.md") if f.name != "00-INDEX-SYNAPSE-KNOWLEDGE.md"]
    assert len(notes) == 2

    # Check local MOC index
    sk_index = sk_dir / "00-INDEX-SYNAPSE-KNOWLEDGE.md"
    assert sk_index.exists()
    sk_index_text = sk_index.read_text(encoding="utf-8")
    assert "[[00-INDEX-PROJEK]]" in sk_index_text
    for n in notes:
        assert f"[[{n.stem}]]" in sk_index_text


def test_no_writeback_when_salience_below_threshold(mock_vault: Path, tmp_path: Path) -> None:
    """Test retain with salience < 0.85 does not perform writeback."""
    db_file = str(tmp_path / "synapse_low_salience.db")
    adapter = SynapseAdapter({"db_path": db_file, "vault_path": str(mock_vault)})

    ok = adapter.retain(
        content="Minor server restart note for troubleshooting.",
        category="sop",
        salience=0.84,  # Below 0.85
    )
    assert ok is True

    server_dir = mock_vault / "server"
    notes = [f for f in server_dir.glob("*.md") if f.name != "00-INDEX-SERVER.md"]
    assert len(notes) == 0


def test_no_writeback_for_unsupported_categories(mock_vault: Path, tmp_path: Path) -> None:
    """Test retain with category outside ('architecture', 'sop', 'decision', 'preference') does not write back."""
    db_file = str(tmp_path / "synapse_cats.db")
    adapter = SynapseAdapter({"db_path": db_file, "vault_path": str(mock_vault)})

    # Incident with high salience should NOT writeback
    ok1 = adapter.retain(
        content="Incident: Caddy gateway failed due to missing port bind permissions.",
        category="incident",
        salience=0.95,
    )
    assert ok1 is True

    # General with high salience should NOT writeback
    ok2 = adapter.retain(
        content="General observation on user interaction timing patterns.",
        category="general",
        salience=0.99,
    )
    assert ok2 is True

    server_dir = mock_vault / "server"
    server_notes = [f for f in server_dir.glob("*.md") if f.name != "00-INDEX-SERVER.md"]
    assert len(server_notes) == 0

    projek_dir = mock_vault / "projek"
    projek_notes = [f for f in projek_dir.glob("*.md") if f.name != "00-INDEX-PROJEK.md"]
    assert len(projek_notes) == 0


def test_non_blocking_error_handling_on_permission_error(mock_vault: Path, tmp_path: Path) -> None:
    """Test retain succeeds even if writing to vault encounters an unhandled exception or PermissionError."""
    db_file = str(tmp_path / "synapse_err.db")
    adapter = SynapseAdapter({"db_path": db_file, "vault_path": str(mock_vault)})

    # Mock open to raise PermissionError during auto-writeback
    with patch("builtins.open", side_effect=PermissionError("Mock Permission Denied")):
        # Retain should not fail!
        ok = adapter.retain(
            content="High salience SOP that fails disk write due to mock permission error.",
            category="sop",
            salience=0.95,
        )
        assert ok is True

    # Node is still stored in SQLite database
    nodes = adapter.storage.list_nodes(limit=5)
    assert len(nodes) == 1
    assert nodes[0]["category"] == "sop"
    assert nodes[0]["salience"] == 0.95


def test_tags_strict_compliance_with_tags_md(mock_vault: Path, tmp_path: Path) -> None:
    """Test all tags generated strictly conform to tags.md."""
    db_file = str(tmp_path / "synapse_tags.db")
    adapter = SynapseAdapter({"db_path": db_file, "vault_path": str(mock_vault)})

    allowed_tags = {"idea", "meeting", "todo", "devops", "ai", "marketing", "finance", "personal", "server", "sop"}

    ok = adapter.retain(
        content="Fintech trading system architecture with automated market data feeds and AI algorithms.",
        category="architecture",
        salience=0.95,
        metadata={"tags": ["#finance", "#ai", "#invalid_tag_123", "nonexistent"]},
    )
    assert ok is True

    projek_dir = mock_vault / "projek"
    notes = [f for f in projek_dir.glob("*.md") if f.name != "00-INDEX-PROJEK.md"]
    assert len(notes) == 1
    content = notes[0].read_text(encoding="utf-8")

    # Extract tags line from frontmatter
    for line in content.splitlines():
        if line.startswith("tags:"):
            # e.g. tags: [ai, finance, devops]
            raw_tags = line.replace("tags:", "").strip(" []")
            note_tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
            for t in note_tags:
                assert t in allowed_tags, f"Tag '{t}' is not in tags.md"
            assert "invalid_tag_123" not in note_tags
            assert "nonexistent" not in note_tags
            break
