"""
SynapseContextLoader - Biomimetic Progressive Virtual Context Loader & Token-Budget Packer.
"""

from __future__ import annotations

import math
import re
import urllib.parse
from typing import Any, Dict, List, Optional

from synapse.storage import SynapseStorage


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string.

    CJK ideographs, kana, hangul syllables and astral-plane glyphs (emoji) count
    as one token each. ASCII costs ~4 chars per token, other non-ASCII ~2.
    Uses ceiling to ensure conservative budget bounds.
    """
    if not text:
        return 0

    count = 0.0
    for ch in text:
        cp = ord(ch)
        if (
            0x3400 <= cp <= 0x4DBF   # CJK ext A
            or 0x4E00 <= cp <= 0x9FFF   # CJK unified ideographs
            or 0x3040 <= cp <= 0x30FF   # hiragana/katakana
            or 0xAC00 <= cp <= 0xD7AF   # hangul syllables
            or cp >= 0x1F000            # emoji and astral planes
        ):
            count += 1.0
        elif cp < 0x7F:
            count += 0.25
        else:
            count += 0.5
    return max(1, math.ceil(count))


def split_sentences(text: str) -> List[str]:
    """
    Split text into clean, complete sentences without truncating punctuation.
    """
    if not text:
        return []
    # Match sentences ending with punctuation [.!?] or clean newlines
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw_sentences if s.strip()]


class SynapseContextLoader:
    """
    Progressive tiered virtual context loader and token-budget packer.
    Formats retrieved synapse nodes into tiered representations (L0, L1, L2)
    and packs them into LLM system prompts strictly within token budgets.
    """

    DEFAULT_HEADER = "### Active Synapse Context:"

    def __init__(self, storage: Optional[SynapseStorage] = None) -> None:
        self.storage = storage

    def parse_synapse_uri(self, uri: str) -> Dict[str, str]:
        """
        Parse a Synapse URI into structured components.
        Example: 'synapse://preference/editor_theme#v1' ->
        {'scheme': 'synapse', 'category': 'preference', 'path': 'editor_theme', 'fragment': 'v1', 'raw': '...'}
        """
        if not uri:
            return {"scheme": "", "category": "", "path": "", "fragment": "", "raw": uri}

        parsed = urllib.parse.urlparse(uri)
        scheme = parsed.scheme or "synapse"
        
        # In URIs like synapse://preference/editor_theme, netloc is 'preference' and path is '/editor_theme'
        category = parsed.netloc or ""
        path = parsed.path.lstrip("/") if parsed.path else ""

        # If URI was formatted as synapse:/preference/editor_theme or simple path
        if not category and path and "/" in path:
            parts = path.split("/", 1)
            category = parts[0]
            path = parts[1]

        return {
            "scheme": scheme,
            "category": category,
            "path": path,
            "fragment": parsed.fragment or "",
            "raw": uri,
        }

    def resolve_synapse_uri(self, uri: str) -> Optional[Dict[str, Any]]:
        """
        Resolve a Synapse URI against storage.
        Returns the node dictionary if found, or None.
        """
        if not uri:
            return None

        if self.storage is None:
            return None

        node = self.storage.get_node(uri)
        if node is not None:
            return dict(node)

        # Fallback: check parsed path
        parsed = self.parse_synapse_uri(uri)
        if parsed["path"]:
            alt_uri = f"{parsed['scheme']}://{parsed['category']}/{parsed['path']}"
            alt_node = self.storage.get_node(alt_uri)
            if alt_node is not None:
                return dict(alt_node)

        return None

    def format_episode_l0(self, episode: Dict[str, Any]) -> str:
        """
        Format Episode L0 Abstract (<40 tokens) - 1-line essential statement.
        """
        title = episode.get("title") or "Episode"
        lesson = episode.get("lessons_learned") or episode.get("outcome") or episode.get("situation") or ""
        salience = float(episode.get("salience", 0.5))
        sal_tag = " [CRITICAL]" if salience >= 0.9 else ""
        return f"• [EPISODE{sal_tag}] {title}: {lesson.strip()}"

    def format_episode_l1(self, episode: Dict[str, Any]) -> str:
        """
        Format Episode L1 Overview (<250 tokens) - structured cognitive breakdown.
        """
        parts = []
        if episode.get("situation"):
            parts.append(f"Situation: {episode['situation'].strip()}")
        if episode.get("action_taken"):
            parts.append(f"Action: {episode['action_taken'].strip()}")
        if episode.get("outcome"):
            parts.append(f"Outcome: {episode['outcome'].strip()}")
        if episode.get("lessons_learned"):
            parts.append(f"Lesson: {episode['lessons_learned'].strip()}")

        if not parts:
            return ""
        return "  - " + " | ".join(parts)

    def format_l0_abstract(self, node: Dict[str, Any]) -> str:
        """
        Format L0 Abstract (<40 tokens) - 1-line essential statement.
        Supports both Declarative/Semantic nodes and Cognitive Episodic memories.
        """
        if node.get("type") == "episodic" or node.get("category") == "episode" or ("situation" in node and "action_taken" in node):
            return self.format_episode_l0(node)

        category = node.get("category", "general")
        uri = node.get("uri", "")
        abstract = node.get("l0_abstract", "").strip()
        l2_ref = node.get("l2_target_path") or uri

        ref_suffix = f" [ref: {l2_ref}]" if l2_ref else ""
        return f"• [{category.upper()}] {uri}: {abstract}{ref_suffix}"

    def format_l1_overview(self, node: Dict[str, Any]) -> str:
        """
        Format L1 Overview (<300 tokens) - bullet points or detailed overview.
        """
        if node.get("type") == "episodic" or node.get("category") == "episode" or ("situation" in node and "action_taken" in node):
            return self.format_episode_l1(node)

        overview = node.get("l1_overview", "").strip()
        if not overview:
            return ""
        return f"  - Overview: {overview}"

    def pack_episodic_context(
        self,
        episodes: List[Dict[str, Any]],
        token_budget: int = 250,
        include_l1: bool = True,
        header: Optional[str] = "### Relevant Episodic Experiences:",
    ) -> str:
        """
        Pack recalled episodic memories into concise L0/L1 format under strict token budget (<250 tokens).
        """
        return self.pack_context(
            episodes,
            token_budget=token_budget,
            include_l1=include_l1,
            header=header,
        )

    def pack_context(
        self,
        nodes: List[Dict[str, Any]],
        token_budget: int = 250,
        include_l1: bool = True,
        header: Optional[str] = None,
    ) -> str:
        """
        Pack retrieved nodes into an ultra-compact context block strictly within token_budget.
        Items are packed greedily in descending rank order.
        Never cuts off mid-sentence: if L1 doesn't fully fit, complete sentences are packed,
        or L1 is omitted to preserve complete L0 representation.
        """
        if not nodes or token_budget <= 0:
            return ""

        active_header = header if header is not None else self.DEFAULT_HEADER
        header_tokens = estimate_tokens(active_header)

        # If even the header doesn't fit, we cannot pack anything
        if header_tokens >= token_budget:
            return ""

        packed_entries: List[str] = []
        current_text = active_header

        for node in nodes:
            # Check L0 block
            l0_line = self.format_l0_abstract(node)
            candidate_text = f"{current_text}\n{l0_line}"
            candidate_tokens = estimate_tokens(candidate_text)

            if candidate_tokens > token_budget:
                # Cannot fit even L0 for this node, stop packing
                break

            entry_lines = [l0_line]
            current_text = candidate_text

            # Check L1 block if requested and available
            if include_l1:
                overview_text = node.get("l1_overview", "").strip()
                if overview_text:
                    full_l1 = f"  - Overview: {overview_text}"
                    with_full_l1 = f"{current_text}\n{full_l1}"
                    if estimate_tokens(with_full_l1) <= token_budget:
                        entry_lines.append(full_l1)
                        current_text = with_full_l1
                    else:
                        # Try packing complete sentences without breaking mid-sentence
                        sentences = split_sentences(overview_text)
                        accepted_sentences = []
                        for s in sentences:
                            partial_overview = "  - Overview: " + " ".join(accepted_sentences + [s])
                            with_partial = f"{current_text}\n{partial_overview}"
                            if estimate_tokens(with_partial) <= token_budget:
                                accepted_sentences.append(s)
                            else:
                                break

                        if accepted_sentences:
                            partial_l1 = "  - Overview: " + " ".join(accepted_sentences)
                            entry_lines.append(partial_l1)
                            current_text = f"{current_text}\n{partial_l1}"

            packed_entries.append("\n".join(entry_lines))

        if not packed_entries:
            return ""

        result = f"{active_header}\n" + "\n".join(packed_entries)
        return result.strip()
