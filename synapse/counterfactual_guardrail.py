"""
Episodic Counterfactual Guardrail for Cognitive Synapse Core.

Inspects proposed actions against historical episodic memory to detect actions
that caused failures, outages, data loss, or security incidents in the past,
preventing the agent from repeating costly mistakes by offering counterfactual
warnings and actionable mitigations extracted from past lessons learned.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from synapse.embedder import SynapseEmbedder, fallback_vectorize
from synapse.storage import SynapseStorage

logger = logging.getLogger(__name__)

# Common stopwords in English and Indonesian for token overlap calculation
STOPWORDS: Set[str] = {
    # English
    "a", "an", "the", "in", "on", "at", "for", "to", "with", "by", "of", "and", "or",
    "is", "are", "was", "were", "be", "been", "this", "that", "these", "those", "it",
    "its", "as", "from", "into", "we", "our", "you", "your", "they", "their", "he", "she",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "i", "me", "my", "myself", "what", "which", "who", "whom", "will", "would", "shall",
    "can", "could", "may", "might", "must", "should", "than", "then", "now",
}

# Tag indicators for incidents, failures, outages, bugs, and errors
FAILURE_TAG_KEYWORDS: Set[str] = {
    "incident", "failure", "failed", "fail", "error", "outage", "crash",
    "crashed", "bug", "postmortem", "anti-pattern", "antipattern",
    "vulnerability", "security_risk", "security-risk", "regression",
    "data_loss", "dataloss", "corruption", "disaster", "downtime",
    "breach", "leak", "leakage", "hazard", "warning", "critical",
    # Indonesian indicators
    "insiden", "kegagalan", "rusak", "kerusakan", "kebocoran",
    "mati", "padam", "bencana", "gangguan", "masalah",
}

# Negated failure phrases in outcome where failure/loss was explicitly avoided or resolved
NEGATED_FAILURE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(
        r"\b(0|zero|no|without|tanpa|mencegah|prevented|avoided|zero-)\s*"
        r"(downtime|data\s+loss|loss(\s+of\s+data)?|outage|failure|fails?|crash|crashed|errors?|issues?|corruption|kebocoran|gangguan|kerusakan)\b",
        re.IGNORECASE,
    ),
]

# Regex patterns in outcome indicating failure / negative consequence
NEGATIVE_OUTCOME_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\b(failed|failure|fails?)\b", re.IGNORECASE),
    re.compile(r"\b(crashed|crash|crashing)\b", re.IGNORECASE),
    re.compile(r"\b(outage|downtime)\b", re.IGNORECASE),
    re.compile(r"\b(data\s+loss|lost\s+data|loss\s+of\s+data)\b", re.IGNORECASE),
    re.compile(r"\b(corrupt|corrupted|corruption)\b", re.IGNORECASE),
    re.compile(r"\b(leaked?|leakage|breached?|security\s+incident)\b", re.IGNORECASE),
    re.compile(r"\b(fatal(\s+error)?|unhandled\s+exception|panic|deadlock)\b", re.IGNORECASE),
    re.compile(r"\b(destroyed|wiped|unrecoverable|dropped\s+table)\b", re.IGNORECASE),
    re.compile(r"\b(memory\s+exhaustion|oom|oomkilled)\b", re.IGNORECASE),
    re.compile(r"\b(cascading\s+failure|cascade\s+failure|service\s+down)\b", re.IGNORECASE),
    # Indonesian negative patterns
    re.compile(r"\b(gagal|kegagalan)\b", re.IGNORECASE),
    re.compile(r"\b(rusak|kerusakan)\b", re.IGNORECASE),
    re.compile(r"\b(kehilangan\s+data|hilang)\b", re.IGNORECASE),
    re.compile(r"\b(kebocoran|bocor)\b", re.IGNORECASE),
    re.compile(r"\b(padam|mati|hancur|bencana)\b", re.IGNORECASE),
]

# Patterns indicating positive outcome / successful remediation
POSITIVE_RECOVERY_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\b(successfully\s+recovered|recovered\s+successfully)\b", re.IGNORECASE),
    re.compile(r"\b(0\s+data\s+loss|no\s+data\s+loss|zero\s+data\s+loss)\b", re.IGNORECASE),
    re.compile(r"\b(0\s+downtime|no\s+downtime|zero\s+downtime|tanpa\s+downtime)\b", re.IGNORECASE),
    re.compile(r"\b(resolved\s+successfully|successfully\s+resolved)\b", re.IGNORECASE),
    re.compile(r"\b(sukses|berhasil\s+pulih|tanpa\s+kehilangan\s+data)\b", re.IGNORECASE),
]

# Cautionary/prohibitive directive patterns in lessons learned
NEGATIVE_DIRECTIVE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\b(never|do\s+not|don'?t|must\s+not|should\s+not|avoid|prohibit(ed)?)\b", re.IGNORECASE),
    re.compile(r"\b(jangan|dilarang|tidak\s+boleh|hindari|diharamkan)\b", re.IGNORECASE),
]

# Incident keywords in episode titles
INCIDENT_TITLE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\b(incident|outage|failure|crash|corruption|disaster|postmortem|vulnerability|leak)\b", re.IGNORECASE),
    re.compile(r"\b(insiden|kegagalan|kerusakan|kebocoran|gangguan)\b", re.IGNORECASE),
]

_shared_embedder_instance: Optional[SynapseEmbedder] = None


def _get_shared_embedder(dim: int = 64) -> SynapseEmbedder:
    """Singleton getter for SynapseEmbedder to optimize test and runtime performance."""
    global _shared_embedder_instance
    if _shared_embedder_instance is None:
        _shared_embedder_instance = SynapseEmbedder(dim=dim)
    return _shared_embedder_instance


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two numeric vectors in [-1.0, 1.0]."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 * n2 == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (n1 * n2)))


def parse_tags(raw_tags: Any) -> Set[str]:
    """Parse tags from various storage formats (JSON array, comma-separated string, or collection)."""
    if not raw_tags:
        return set()
    if isinstance(raw_tags, (list, set, tuple)):
        return {str(t).strip().lower() for t in raw_tags if t}
    if isinstance(raw_tags, str):
        raw = raw_tags.strip()
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return {str(t).strip().lower() for t in parsed if t}
            except Exception:
                pass
        return {t.strip().lower() for t in re.split(r"[,;\s]+", raw) if t.strip()}
    return set()


def extract_tokens(text: str) -> Set[str]:
    """Extract significant lowercase alphanumeric tokens without stopwords."""
    if not text:
        return set()
    words = re.findall(r"[a-z0-9_\-]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def token_overlap_ratio(source_tokens: Set[str], target_text: str) -> float:
    """Compute asymmetric token overlap ratio: |source & target| / |source|."""
    if not source_tokens or not target_text:
        return 0.0
    target_tokens = extract_tokens(target_text)
    if not target_tokens:
        return 0.0
    intersection = source_tokens & target_tokens
    return len(intersection) / len(source_tokens)


def extract_mitigation(lessons_learned: str) -> str:
    """
    Extract actionable mitigation directives from historical lessons learned.
    Prioritizes affirmative recommendations (e.g. Always..., Ensure...).
    """
    if not lessons_learned or not lessons_learned.strip():
        return "Review action proposal carefully and establish verification/rollback steps before execution."

    text = lessons_learned.strip()
    raw_sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n+|;\s*", text)
        if s.strip()
    ]

    affirmative_patterns = re.compile(
        r"\b(always|ensure|make sure|use|verify|instead|prefer|adopt|enable|configure|apply|deploy|test)\b",
        re.IGNORECASE,
    )

    mitigations = [s for s in raw_sentences if affirmative_patterns.search(s)]
    if mitigations:
        result = " ".join(mitigations)
        if not result.endswith((".", "!", "?")):
            result += "."
        return result

    return text


def generate_warning_message(
    episode: Dict[str, Any],
    similarity_score: float,
    action_proposal: str,
) -> str:
    """Generate an actionable warning message describing historical failure context."""
    title = episode.get("title") or "Historical Incident"
    outcome = (episode.get("outcome") or "").strip()
    action_taken = (episode.get("action_taken") or "").strip()

    warning_parts = [
        f"Proposed action is semantically similar to past incident '{title}' (similarity score: {similarity_score:.2f})."
    ]
    if action_taken:
        warning_parts.append(f"Past action taken: '{action_taken}'.")
    if outcome:
        warning_parts.append(f"Historical outcome: {outcome}.")
    return " ".join(warning_parts)


@dataclass
class GuardrailResult:
    """
    Evaluation result from CounterfactualGuardrail.

    Attributes:
        is_risky: True if proposed action is semantically similar to a past failure/incident.
        risk_level: "safe", "low", "medium", "high", or "critical".
        matched_episode: Dictionary of the matched historical episode, or None if safe.
        warning_message: Actionable warning message explaining the risk, or None if safe.
        suggested_mitigation: Actionable mitigation extracted from lessons learned, or None if safe.
        similarity_score: Computed semantic/lexical similarity score in [0.0, 1.0].
    """

    is_risky: bool
    risk_level: str
    matched_episode: Optional[Dict[str, Any]]
    warning_message: Optional[str]
    suggested_mitigation: Optional[str]
    similarity_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to standard dictionary."""
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style key access."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Invalid GuardrailResult key: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get with default value."""
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        """Dict-like containment check."""
        return hasattr(self, key)


class CounterfactualGuardrail:
    """
    Episodic Counterfactual Guardrail for Cognitive Synapse.

    Evaluates action proposals against historical episodic memory to prevent
    repeating past mistakes, incidents, or outages.
    """

    def __init__(
        self,
        storage: SynapseStorage,
        embedder: Optional[Any] = None,
        similarity_threshold: float = 0.55,
        medium_risk_threshold: float = 0.65,
        high_risk_threshold: float = 0.78,
        critical_risk_threshold: float = 0.88,
        vector_dim: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        cfg = config or {}
        self.storage = storage
        self.similarity_threshold = float(cfg.get("similarity_threshold", similarity_threshold))
        self.medium_risk_threshold = float(cfg.get("medium_risk_threshold", medium_risk_threshold))
        self.high_risk_threshold = float(cfg.get("high_risk_threshold", high_risk_threshold))
        self.critical_risk_threshold = float(cfg.get("critical_risk_threshold", critical_risk_threshold))
        self.vector_dim = cfg.get("vector_dim") or vector_dim or 64

        # Configure embedder
        if embedder is not None:
            self.embedder = embedder
        else:
            try:
                self.embedder = _get_shared_embedder(dim=self.vector_dim)
            except Exception as e:
                logger.warning("SynapseEmbedder initialization failed (%s); using fallback.", e)
                self.embedder = None

        # Ephemeral cache for episode vectors: (action_vec, lesson_vec, situation_vec)
        self._vector_cache: Dict[str, Tuple[List[float], List[float], List[float]]] = {}

    def vectorize(self, text: str) -> List[float]:
        """Compute dense semantic embedding vector for input text."""
        if not text or not text.strip():
            return [0.0] * self.vector_dim

        if self.embedder is not None:
            if hasattr(self.embedder, "embed") and callable(self.embedder.embed):
                return self.embedder.embed(text)
            if hasattr(self.embedder, "vectorize") and callable(self.embedder.vectorize):
                return self.embedder.vectorize(text)
            if callable(self.embedder):
                return self.embedder(text)

        return fallback_vectorize(text, dim=self.vector_dim)

    def is_failure_episode(self, episode: Dict[str, Any]) -> Tuple[bool, float, str, bool]:
        """
        Determine if an episode represents a historical failure/incident.

        Returns:
            (is_failure: bool, severity: float, reason: str, action_was_remedy: bool)
        """
        tags = parse_tags(episode.get("tags"))
        raw_outcome = str(episode.get("outcome") or "").strip()
        lessons = str(episode.get("lessons_learned") or "").strip()
        title = str(episode.get("title") or "").strip()
        salience = float(episode.get("salience") or 0.5)

        # Mask negated failure phrases in outcome (e.g. "0 downtime", "no data loss")
        masked_outcome = raw_outcome
        for neg_pat in NEGATED_FAILURE_PATTERNS:
            masked_outcome = neg_pat.sub(" ", masked_outcome)

        has_failure_tag = any(
            t in FAILURE_TAG_KEYWORDS or any(fk in t for fk in FAILURE_TAG_KEYWORDS)
            for t in tags
        )
        has_negative_outcome = any(pat.search(masked_outcome) for pat in NEGATIVE_OUTCOME_PATTERNS)
        is_positive_recovery = any(pat.search(raw_outcome) for pat in POSITIVE_RECOVERY_PATTERNS)
        has_negative_lesson = any(pat.search(lessons) for pat in NEGATIVE_DIRECTIVE_PATTERNS)
        has_incident_title = any(pat.search(title) for pat in INCIDENT_TITLE_PATTERNS)

        # If outcome was a successful recovery and had no remaining negative outcome text,
        # the action_taken was the remedy, not the cause of the failure.
        action_was_remedy = False
        if is_positive_recovery and not has_negative_outcome:
            action_was_remedy = True

        reasons: List[str] = []
        severity = salience

        if has_failure_tag:
            reasons.append("tag_incident")
        if has_negative_outcome:
            reasons.append("negative_outcome")
            severity = max(severity, 0.75)
        if has_negative_lesson:
            reasons.append("cautionary_lesson")
        if has_incident_title:
            reasons.append("incident_title")

        is_failure = bool(has_failure_tag or has_negative_outcome or has_negative_lesson or has_incident_title)
        reason_str = ", ".join(reasons) if reasons else "none"

        return is_failure, severity, reason_str, action_was_remedy

    def get_failure_episodes(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve historical episodes classified as failures, outages, or incidents."""
        raw_episodes = self.storage.list_episodes(limit=max(1, limit * 2))
        failure_episodes: List[Dict[str, Any]] = []

        for ep in raw_episodes:
            is_fail, _, _, _ = self.is_failure_episode(ep)
            if is_fail:
                failure_episodes.append(ep)
                if len(failure_episodes) >= limit:
                    break

        return failure_episodes

    def evaluate_action(
        self,
        action_proposal: str,
        context: str = "",
    ) -> GuardrailResult:
        """
        Evaluate proposed action against historical episodic failure records.

        Args:
            action_proposal: The action being proposed for execution.
            context: Optional situational context (e.g. environment, target service).

        Returns:
            GuardrailResult with risk level, warning message, and suggested mitigation.
        """
        if not action_proposal or not action_proposal.strip():
            return GuardrailResult(
                is_risky=False,
                risk_level="safe",
                matched_episode=None,
                warning_message=None,
                suggested_mitigation=None,
                similarity_score=0.0,
            )

        clean_proposal = action_proposal.strip()
        proposal_tokens = extract_tokens(clean_proposal)

        # 1. Retrieve candidate episodes from FTS and recent episodes
        candidates_map: Dict[str, Dict[str, Any]] = {}
        fts_episodes = self.storage.search_episodes(clean_proposal, limit=25)
        for ep in fts_episodes:
            candidates_map[ep["id"]] = ep

        if context and context.strip():
            context_fts = self.storage.search_episodes(context.strip(), limit=15)
            for ep in context_fts:
                candidates_map[ep["id"]] = ep

        # Also retrieve top salience episodes to ensure coverage of major historical incidents
        top_episodes = self.storage.list_episodes(limit=60)
        for ep in top_episodes:
            candidates_map[ep["id"]] = ep

        if not candidates_map:
            return GuardrailResult(
                is_risky=False,
                risk_level="safe",
                matched_episode=None,
                warning_message=None,
                suggested_mitigation=None,
                similarity_score=0.0,
            )

        # 2. Pre-vectorize the proposed action and optional context
        vec_proposal = self.vectorize(clean_proposal)
        vec_context = self.vectorize(context.strip()) if (context and context.strip()) else None

        best_score = 0.0
        best_episode: Optional[Dict[str, Any]] = None

        # 3. Score against all failure episodes
        for ep_id, ep in candidates_map.items():
            is_fail, severity_factor, reason, action_was_remedy = self.is_failure_episode(ep)
            if not is_fail:
                continue

            # Cache episode vectors to accelerate repeated evaluations
            act_text = str(ep.get("action_taken") or "").strip()
            lesson_text = str(ep.get("lessons_learned") or "").strip()
            situation_text = str(ep.get("situation") or "").strip()

            cache_key = f"{ep_id}:{hash(act_text)}:{hash(lesson_text)}:{hash(situation_text)}"
            if cache_key in self._vector_cache:
                vec_act, vec_lesson, vec_sit = self._vector_cache[cache_key]
            else:
                vec_act = self.vectorize(act_text) if act_text else [0.0] * self.vector_dim
                vec_lesson = self.vectorize(lesson_text) if lesson_text else [0.0] * self.vector_dim
                vec_sit = self.vectorize(situation_text) if situation_text else [0.0] * self.vector_dim
                self._vector_cache[cache_key] = (vec_act, vec_lesson, vec_sit)

            # A. Semantic Cosine Similarity
            sim_action = max(0.0, float(cosine_similarity(vec_proposal, vec_act))) if act_text else 0.0
            sim_lesson = max(0.0, float(cosine_similarity(vec_proposal, vec_lesson))) if lesson_text else 0.0

            # B. Lexical Token Overlap
            overlap_action = token_overlap_ratio(proposal_tokens, act_text)
            overlap_lesson = token_overlap_ratio(proposal_tokens, lesson_text)

            # C. Hybrid dense-sparse score per component (lexical boosts, never penalizes)
            action_score = max(sim_action, (0.60 * sim_action) + (0.40 * overlap_action)) if act_text else 0.0
            lesson_score = max(sim_lesson, (0.60 * sim_lesson) + (0.40 * overlap_lesson)) if lesson_text else 0.0

            # D. Remedy vs Failure Action handling:
            # If the episode documented a recovery remedy, only match against what lessons_learned warned against!
            if action_was_remedy:
                has_neg_lesson = any(pat.search(lesson_text) for pat in NEGATIVE_DIRECTIVE_PATTERNS)
                if has_neg_lesson:
                    candidate_score = lesson_score * 0.90
                else:
                    candidate_score = 0.0
            else:
                candidate_score = max(action_score, lesson_score * 0.95)

            # E. Situational Context Boost
            if vec_context is not None and situation_text and candidate_score > 0.0:
                sim_situation = max(0.0, float(cosine_similarity(vec_context, vec_sit)))
                overlap_situation = token_overlap_ratio(extract_tokens(context), situation_text)
                composite_sit = max(sim_situation, (0.7 * sim_situation) + (0.3 * overlap_situation))
                if composite_sit >= 0.35:
                    # Situational congruence elevates the risk
                    candidate_score = min(1.0, candidate_score * (1.0 + (0.30 * composite_sit)))

            # F. Salience Sensitivity
            ep_salience = float(ep.get("salience") or 0.5)
            if ep_salience >= 0.90 and candidate_score >= self.similarity_threshold:
                candidate_score = min(1.0, candidate_score * 1.05)

            if candidate_score > best_score:
                best_score = candidate_score
                best_episode = ep

        # 4. Evaluate Thresholds and Assign Risk Level
        if best_score < self.similarity_threshold or best_episode is None:
            return GuardrailResult(
                is_risky=False,
                risk_level="safe",
                matched_episode=None,
                warning_message=None,
                suggested_mitigation=None,
                similarity_score=round(best_score, 4),
            )

        # Determine risk level tier based on thresholds and salience
        ep_salience = float(best_episode.get("salience") or 0.5)

        if best_score >= self.critical_risk_threshold or (
            best_score >= self.high_risk_threshold and ep_salience >= 0.90
        ):
            risk_level = "critical"
        elif best_score >= self.high_risk_threshold or (
            best_score >= self.medium_risk_threshold and ep_salience >= 0.85
        ):
            risk_level = "high"
        elif best_score >= self.medium_risk_threshold or (
            best_score >= self.similarity_threshold and ep_salience >= 0.80
        ):
            risk_level = "medium"
        else:
            risk_level = "low"

        # 5. Extract Warning and Actionable Mitigation
        warning_msg = generate_warning_message(
            episode=best_episode,
            similarity_score=best_score,
            action_proposal=clean_proposal,
        )
        mitigation = extract_mitigation(str(best_episode.get("lessons_learned") or ""))

        return GuardrailResult(
            is_risky=True,
            risk_level=risk_level,
            matched_episode=best_episode,
            warning_message=warning_msg,
            suggested_mitigation=mitigation,
            similarity_score=round(best_score, 4),
        )

    def batch_evaluate(
        self,
        proposals: List[Union[str, Tuple[str, str]]],
    ) -> List[GuardrailResult]:
        """Evaluate a batch of action proposals (strings or (proposal, context) tuples)."""
        results: List[GuardrailResult] = []
        for item in proposals:
            if isinstance(item, tuple):
                action = item[0]
                ctx = item[1] if len(item) > 1 else ""
            else:
                action = str(item)
                ctx = ""
            results.append(self.evaluate_action(action, context=ctx))
        return results

    def explain_risk(self, result: GuardrailResult) -> str:
        """Format a human-readable explanation of the guardrail evaluation."""
        if not result.is_risky:
            return (
                f"[RISK LEVEL: SAFE] Action proposal passed counterfactual guardrails "
                f"(similarity score: {result.similarity_score:.4f})."
            )

        ep_title = (result.matched_episode or {}).get("title", "Historical Incident")
        lines = [
            f"[RISK LEVEL: {result.risk_level.upper()}] Action proposal flagged by Episodic Counterfactual Guardrail.",
            f"Similarity Score: {result.similarity_score:.4f}",
            f"Matched Incident: {ep_title}",
            f"Warning: {result.warning_message}",
            f"Suggested Mitigation: {result.suggested_mitigation}",
        ]
        return "\n".join(lines)
