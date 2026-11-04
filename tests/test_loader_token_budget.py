"""Token estimates must not undercount CJK and emoji text."""

from __future__ import annotations

from synapse.loader import estimate_tokens


def test_cjk_scales_per_character() -> None:
    assert estimate_tokens("日本語のテキストです") >= 8


def test_latin_keeps_quarter_rate() -> None:
    assert estimate_tokens("hello world this is english text") == 8


def test_emoji_counts_as_tokens() -> None:
    assert estimate_tokens("🙂🙂🙂🙂") >= 4


def test_empty() -> None:
    assert estimate_tokens("") == 0
