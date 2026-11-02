"""
Synapse Observation Engine.
Implements Tier 2 Deduplicated, Evidence-Grounded Pattern Consolidation.
Tracks supporting memories with exact quotes and proof counts.
Refined when new evidence supports or contradicts observations.

Uses semantic similarity against existing observations instead of hardcoded
keyword patterns to ensure dynamic, semantic memory clustering.
"""

from __future__ import annotations

import logging
import math
import re
import zlib
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def _obs_vectorize(text: str, dim: int = 64) -> List[float]:
    """Lightweight deterministic vectorizer for observation matching."""
    if not text or not text.strip():
        return [0.0] * dim
    clean = text.lower().strip()
    words = re.findall(r"[a-z0-9_\-]+", clean)
    if not words:
        return [0.0] * dim
    vec = [0.0] * dim
    for w in words:
        h = zlib.crc32(w.encode("utf-8"))
        bucket = h % dim
        sign = 1.0 if ((h >> 16) & 1) else -1.0
        vec[bucket] += sign * 1.0
        if len(w) >= 3:
            for n in (3, 4):
                for i in range(len(w) - n + 1):
                    sub = w[i : i + n]
                    sh = zlib.crc32(sub.encode("utf-8"))
                    vec[sh % dim] += (1.0 if ((sh >> 16) & 1) else -1.0) * 0.5
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0.0 else vec


def _obs_cosine(v1: List[float], v2: List[float]) -> float:
    """Cosine similarity between two vectors."""
    if not v1 or not v2:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 * n2 == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (n1 * n2)))


class ObservationEngine:
    """
    Manages Tier 2 Evidence-Grounded Observations with proof counts and quotation tracking.
    Uses semantic similarity to match incoming text against existing observation patterns,
    rather than hardcoded keyword lists.
    """

    # Similarity threshold for matching an incoming turn to an existing observation
    MATCH_THRESHOLD: float = 0.40

    # Minimum text length to consider for observation ingestion
    MIN_TEXT_LENGTH: int = 10

    def __init__(self, storage: Any, match_threshold: float = 0.40) -> None:
        self.storage = storage
        self.MATCH_THRESHOLD = match_threshold

    def ingest_turn_observation(
        self,
        text: str,
        category: str = "general",
        proof_delta: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a text snippet for recurring patterns by semantically matching
        against all existing observations. If a match is found above threshold,
        reinforce it (increment proof_count). Otherwise, create a new observation
        pattern from the text.

        This approach is 100% dynamic and semantic -- no hardcoded keyword lists.
        """
        clean = text.strip()
        if not clean or len(clean) < self.MIN_TEXT_LENGTH:
            return None

        # Filter out system orchestration banners & metadata wrappers
        system_noise_markers = [
            "ASYNC DELEGATION",
            "<memory-context>",
            "</memory-context>",
            "[System note:",
            "[OUT-OF-BAND USER MESSAGE",
        ]
        if any(marker in clean for marker in system_noise_markers):
            return None

        # Vectorize incoming text
        incoming_vec = _obs_vectorize(clean)

        # Load all existing observations and find best semantic match
        all_obs = self.storage.list_observations(limit=200)
        best_match: Optional[Dict[str, Any]] = None
        best_sim: float = 0.0

        for obs in all_obs:
            obs_vec = _obs_vectorize(obs["pattern"])
            sim = _obs_cosine(incoming_vec, obs_vec)
            if sim > best_sim:
                best_sim = sim
                best_match = obs

        if best_match and best_sim >= self.MATCH_THRESHOLD:
            # Reinforce existing observation with new supporting evidence
            return self.storage.add_or_update_observation(
                pattern=best_match["pattern"],
                supporting_quote=clean[:250],
                category=best_match.get("category", category),
                proof_delta=proof_delta,
            )

        # No strong semantic match found -- create a new observation pattern
        # Derive a concise pattern statement from the input text
        pattern = self._derive_pattern(clean)
        if pattern:
            return self.storage.add_or_update_observation(
                pattern=pattern,
                supporting_quote=clean[:250],
                category=category,
                proof_delta=proof_delta,
            )

        return None

    def _derive_pattern(self, text: str) -> Optional[str]:
        """
        Derive a concise observation pattern from raw text.
        Extracts the first meaningful sentence or truncates to a clean statement.
        """
        # Split into sentences and take the most informative one
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        # Filter out very short sentences
        meaningful = [s for s in sentences if len(s.strip()) >= 15]

        if meaningful:
            candidate = meaningful[0].strip()
            # Cap at ~200 chars for pattern conciseness
            if len(candidate) > 200:
                words = candidate.split()
                candidate = " ".join(words[:30]) + "..."
            return candidate

        # Fallback: use truncated original text
        if len(text) > 200:
            return text[:197] + "..."
        return text if len(text) >= 15 else None

    def search_observations(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search observations using hybrid lexical overlap + semantic vector similarity.
        """
        all_obs = self.storage.list_observations(limit=200)
        if not all_obs:
            return []

        query_vec = _obs_vectorize(query)
        q_words = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = []

        for obs in all_obs:
            # Lexical overlap score
            p_words = set(re.findall(r"[a-z0-9]+", obs["pattern"].lower()))
            lexical_overlap = len(q_words & p_words)

            # Semantic vector similarity
            obs_vec = _obs_vectorize(obs["pattern"])
            semantic_sim = _obs_cosine(query_vec, obs_vec)

            # Proof count boost (capped)
            proof_boost = min(2.0, 1.0 + (obs.get("proof_count", 1) * 0.1))

            # Combined score: lexical + semantic, boosted by proof count
            combined_score = (lexical_overlap * 0.5 + semantic_sim * 2.0) * proof_boost

            if combined_score > 0.1:
                scored.append((combined_score, obs))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:limit]]
