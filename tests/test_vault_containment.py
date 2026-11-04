"""Vault writeback must stay inside the vault and emit valid frontmatter."""

from __future__ import annotations

from pathlib import Path

from synapse.adapter import SynapseAdapter


def _retain_with_metadata(tmp_path: Path, **metadata_extra) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    adapter = SynapseAdapter(
        {"db_path": ":memory:", "vault_path": str(vault)}
    )
    metadata = {"category": "architecture", "salience": 0.95}
    metadata.update(metadata_extra)
    adapter.retain(
        "Architecture rule about vault containment",
        category="architecture",
        salience=0.95,
        metadata=metadata,
    )
    adapter.storage.close()
    return vault


def test_slug_traversal_is_confined_to_vault(tmp_path):
    vault = _retain_with_metadata(tmp_path, slug="../../escape")
    all_md = list(Path(tmp_path).rglob("*.md"))
    assert all(p.is_relative_to(vault) for p in all_md)
    assert (Path(tmp_path) / "escape.md").exists() is False


def test_slug_with_separators_is_sanitized(tmp_path):
    vault = _retain_with_metadata(tmp_path, slug="a/b/../../c")
    assert list(vault.rglob("*.md"))
    assert (Path(tmp_path) / "escape.md").exists() is False


def test_frontmatter_quotes_and_newlines_are_escaped(tmp_path):
    vault = _retain_with_metadata(
        tmp_path,
        title='Bad "title: with \\"quotes\\" and\nnewlines',
        slug="quote-test",
    )
    candidates = list(vault.rglob("quote-test.md"))
    assert candidates, "writeback produced no note"
    text = candidates[0].read_text(encoding="utf-8")
    assert "title:" in text
    frontmatter = text.split("---", 2)[1]
    title_line = next(
        line for line in frontmatter.splitlines() if line.startswith("title:")
    )
    # Unbalanced unescaped quotes would break the YAML scalar: every inner
    # quote must be escaped as \\", leaving exactly the outer pair unescaped.
    assert title_line.count('"') - title_line.count('\\"') == 2
