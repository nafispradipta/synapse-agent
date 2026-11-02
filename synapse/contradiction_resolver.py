"""
Auto-Contradiction Resolver & Self-Correcting Engine for Cognitive Synapse Core.
Evaluates topic similarity and detects factual discrepancies (specs, status, directives),
automatically superseding outdated facts and reducing their strength to 0.05.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# Stopwords for keyword extraction
STOPWORDS: Set[str] = {
    # English
    "a", "an", "the", "in", "on", "at", "for", "to", "with", "by", "of", "and", "or",
    "is", "are", "was", "were", "be", "been", "this", "that", "these", "those", "it",
    "its", "as", "from", "into", "we", "our", "you", "your", "they", "their", "he", "she",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    # Framework and structural metadata stopwords
    "synapse", "architecture", "decision", "preference", "sop", "incident", "general",
    "task", "project", "item", "note", "rule", "status", "context",
    # Indonesian
    "di", "ke", "dari", "pada", "untuk", "dengan", "dan", "atau", "ini", "itu", "yang",
    "adalah", "sebagai", "oleh", "saat", "juga", "akan", "telah", "sudah", "menjadi",
    "menggunakan", "gunakan", "pakai", "sebesar", "sebanyak", "dalam", "bisa", "dapat",
    "kami", "kita", "mereka", "dia", "ia", "tersebut", "secara", "karena", "agar", "supaya",
}

# Key domain entities / concepts for subject matching
DOMAIN_ENTITIES: Set[str] = {
    "ram", "memory", "memori", "storage", "disk", "cpu", "vcpu", "core", "cores",
    "port", "timeout", "worker", "workers", "thread", "threads", "server", "vps",
    "redis", "postgres", "postgresql", "mysql", "sqlite", "database", "gateway",
    "auth", "token", "backup", "docker", "pm2", "service", "daemon", "api", "cluster",
    "proxy", "network", "ssl", "tls", "cert", "theme", "editor", "font",
    "cache", "queue", "kafka", "rabbitmq", "session", "model", "llm", "pipeline",
    "host", "ip", "dns", "domain", "endpoint", "url", "client", "connection"
}

# Specification attributes to look for in numeric claims
SPEC_ATTRIBUTES: Set[str] = {
    "ram", "memory", "memori", "storage", "disk", "cpu", "vcpu", "core", "cores",
    "port", "timeout", "worker", "workers", "thread", "threads", "limit", "batch_size",
    "capacity", "size", "bandwidth", "connections", "connection", "quota", "kuota",
    "retries", "delay", "ttl", "interval", "max_connections", "concurrency", "rate_limit",
    "keepalive", "port_api", "port_http", "port_grpc"
}

# Canonical normalization for attribute names (e.g. memori -> ram, kuota -> quota)
SPEC_CANONICAL_MAP: Dict[str, str] = {
    "memori": "ram",
    "memory": "ram",
    "kuota": "quota",
}

# Units recognized for specifications
KNOWN_UNITS: Set[str] = {
    "gb", "mb", "tb", "kb", "g", "m", "k", "b", "core", "cores", "cpu", "cpus",
    "vcpu", "vcpus", "hz", "khz", "mhz", "ghz", "s", "sec", "secs", "second", "seconds",
    "detik", "ms", "min", "minute", "minutes", "menit", "hour", "hours", "jam",
    "day", "days", "hari", "kbps", "mbps", "gbps", "%", "px"
}

# Unit normalizations
UNIT_NORMALIZATION: Dict[str, str] = {
    "gigabytes": "gb", "giga": "gb", "gb": "gb", "g": "gb",
    "megabytes": "mb", "mega": "mb", "mb": "mb", "m": "mb",
    "terabytes": "tb", "tera": "tb", "tb": "tb",
    "kilobytes": "kb", "kilo": "kb", "kb": "kb", "k": "kb",
    "second": "s", "seconds": "s", "sec": "s", "secs": "s", "s": "s", "detik": "s",
    "ms": "ms", "millisecond": "ms", "milliseconds": "ms",
    "min": "min", "minute": "min", "minutes": "min", "menit": "min",
    "hour": "hour", "hours": "hour", "jam": "hour",
    "day": "day", "days": "day", "hari": "day",
    "core": "core", "cores": "core",
    "cpu": "cpu", "cpus": "cpu", "vcpu": "cpu", "vcpus": "cpu",
}

# Operational status vocabulary
ACTIVE_STATUS_TERMS: Set[str] = {
    "active", "aktif", "running", "berjalan", "enabled", "enable", "online", "up", "hidup"
}
INACTIVE_STATUS_TERMS: Set[str] = {
    "inactive", "nonaktif", "mati", "stopped", "berhenti", "disabled", "disable", "off", "offline", "down"
}

# Normative directive vocabulary
MANDATORY_TERMS: Set[str] = {
    "mandatory", "wajib", "harus", "perlu", "must", "always", "required", "mandate", "adopt", "enable", "allow", "pakai", "gunakan"
}
FORBIDDEN_TERMS: Set[str] = {
    "never", "forbid", "prohibited", "avoid", "deny", "disable", "jangan", "dilarang", "tidak boleh", "hindari"
}

# Opposing patterns for technology or configuration options
OPPOSING_PATTERNS: List[Tuple[re.Pattern[str], re.Pattern[str]]] = [
    (
        re.compile(r"\b(prefer|always|enable|use|adopt|active|true|light|tabs|allow|permit|yes|wajib|harus|pakai|gunakan|boleh)\b", re.IGNORECASE),
        re.compile(r"\b(avoid|never|disable|dont|do not|deprecated|inactive|false|dark|spaces|deny|forbid|no|dilarang|larang|jangan|tolak|tidak boleh)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"\b(postgresql|postgres)\b", re.IGNORECASE),
        re.compile(r"\b(sqlite|mysql)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"\b(tabs)\b", re.IGNORECASE),
        re.compile(r"\b(spaces)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"\b(dark mode|theme:\s*dark)\b", re.IGNORECASE),
        re.compile(r"\b(light mode|theme:\s*light)\b", re.IGNORECASE),
    ),
]


def parse_timestamp(val: Any) -> float:
    """Parse unix timestamp or ISO string into a float epoch timestamp."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            pass
        try:
            return datetime.datetime.fromisoformat(val).timestamp()
        except Exception:
            return 0.0
    return 0.0


def extract_keywords(text: str) -> Set[str]:
    """Extract non-stopword tokens from text."""
    if not text:
        return set()
    words = re.findall(r"[a-zA-Z0-9_\-]+", text.lower())
    return {w for w in words if len(w) >= 2 and w not in STOPWORDS}


def extract_numeric_specs(text: str) -> Dict[str, Tuple[float, str]]:
    """
    Extract key numeric specifications from text, returning a map of:
    attribute -> (numeric_value, normalized_unit).
    Handles patterns like:
    - '2GB RAM', '8 GB RAM', '4 core CPU', '60s timeout'
    - 'RAM 2GB', 'RAM: 4GB', 'port 8080', 'timeout 30s'
    - 'port_api = 8080', 'worker_threads: 4'
    - 'PostgreSQL 14', 'Python 3.11'
    """
    specs: Dict[str, Tuple[float, str]] = {}
    if not text:
        return specs

    # 1. Number + unit followed by attribute (e.g. '8 GB RAM', '100 GB disk', '4 core CPU', '8 GB memori')
    for attr in SPEC_ATTRIBUTES:
        pat = (
            r"(?<![\w\-])(\d+(?:\.\d+)?)\s*([a-zA-Z%]+)\s+(?:of\s+)?\b"
            + re.escape(attr)
            + r"\b"
        )
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            raw_unit = m.group(2).lower()
            unit = UNIT_NORMALIZATION.get(raw_unit, raw_unit if raw_unit in KNOWN_UNITS else "")
            canonical_attr = SPEC_CANONICAL_MAP.get(attr, attr)
            specs[canonical_attr] = (val, unit)

    # 2. Spec attribute followed by optional relation and number + unit (without jumping over conjunctions)
    for attr in SPEC_ATTRIBUTES:
        canonical_attr = SPEC_CANONICAL_MAP.get(attr, attr)
        if canonical_attr in specs:
            continue
        pat2 = (
            r"\b"
            + re.escape(attr)
            + r"\b(?:\s+(?!and|dan|with|dengan|or|atau)[a-zA-Z0-9_\-]+){0,2}\s*(?::|=|adalah|sebesar|sebanyak|is)?\s*(?<![\w\-])(\d+(?:\.\d+)?)\s*([a-zA-Z%]+)?\b"
        )
        m2 = re.search(pat2, text, re.IGNORECASE)
        if m2:
            val = float(m2.group(1))
            raw_unit = (m2.group(2) or "").lower()
            unit = UNIT_NORMALIZATION.get(raw_unit, raw_unit if raw_unit in KNOWN_UNITS else "")
            specs[canonical_attr] = (val, unit)

    # 3. Explicit key:value syntax e.g. "worker_threads: 4", "batch_size = 100", "version: 16"
    p3 = re.finditer(r"\b([a-zA-Z_][a-zA-Z0-9_]{2,20})\s*(?::|=)\s*(?<![\w\-])(\d+(?:\.\d+)?)\s*([a-zA-Z%]+)?\b", text, re.IGNORECASE)
    for m in p3:
        raw_attr = m.group(1).lower()
        if raw_attr in STOPWORDS:
            continue
        canonical_attr = SPEC_CANONICAL_MAP.get(raw_attr, raw_attr)
        if canonical_attr not in specs:
            val = float(m.group(2))
            raw_unit = (m.group(3) or "").lower()
            unit = UNIT_NORMALIZATION.get(raw_unit, raw_unit if raw_unit in KNOWN_UNITS else "")
            specs[canonical_attr] = (val, unit)

    # 4. Software/engine version progression e.g. "PostgreSQL 14", "Postgres 16", "Python 3.11", "Node 20"
    p4 = re.finditer(r"\b(postgresql|postgres|sqlite|mysql|redis|python|node|ubuntu|go|linux|kafka)\s*(?:version|v)?\s*[:=]?\s*(\d+(?:\.\d+)?)\b", text, re.IGNORECASE)
    for m in p4:
        tech_name = m.group(1).lower()
        attr_key = "postgresql_version" if tech_name in ("postgresql", "postgres") else f"{tech_name}_version"
        if attr_key not in specs:
            specs[attr_key] = (float(m.group(2)), "")

    return specs


INACTIVE_PHRASE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(
        r"\b(not\s+active|is\s+not\s+active|tidak\s+aktif|tidak\s+berjalan|belum\s+jalan|tidak\s+hidup|tidak\s+menyala|non[\s\-_]+aktif)\b",
        re.IGNORECASE,
    )
]

PROGRESS_PHRASE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(
        r"\b(in[- ]progress|underway|ongoing|in\s+development|\bwip\b|sedang\s+berjalan|dalam\s+proses|sedang\s+dikerjakan|belum\s+selesai)\b",
        re.IGNORECASE,
    )
]

COMPLETED_PHRASE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(
        r"\b(completed|complete|finished|resolved|closed|shipped|sudah\s+selesai|selesai|tuntas|berhasil)\b",
        re.IGNORECASE,
    )
]


def extract_status_flags(
    text: str,
    custom_progress_terms: Optional[Set[str]] = None,
    custom_completed_terms: Optional[Set[str]] = None,
) -> Optional[str]:
    """Detect operational status (active vs inactive) or task lifecycle state from text."""
    if not text:
        return None
    lower = text.lower()

    # 1. Multi-word negation phrases for inactive status (with strict word boundaries)
    if any(p.search(lower) for p in INACTIVE_PHRASE_PATTERNS):
        return "inactive"

    # 2. Check task lifecycle progress vs completion (with strict word boundaries)
    if any(p.search(lower) for p in PROGRESS_PHRASE_PATTERNS) or (
        custom_progress_terms and any(re.search(r"\b" + re.escape(t) + r"\b", lower) for t in custom_progress_terms)
    ):
        return "in_progress"

    if any(p.search(lower) for p in COMPLETED_PHRASE_PATTERNS) or (
        custom_completed_terms and any(re.search(r"\b" + re.escape(t) + r"\b", lower) for t in custom_completed_terms)
    ):
        return "completed"

    # 3. Single-word token matching (strict regex tokenization)
    words = set(re.findall(r"[a-zA-Z0-9_\-]+", lower))

    is_active = bool(words & ACTIVE_STATUS_TERMS)
    is_inactive = bool(words & INACTIVE_STATUS_TERMS)

    if is_active and not is_inactive:
        return "active"
    if is_inactive and not is_active:
        return "inactive"
    return None


def extract_directive_flags(text: str) -> Optional[str]:
    """Detect normative directive polarity (mandatory vs forbidden) from text."""
    if not text:
        return None
    lower = text.lower()
    # Check multi-word forbidden indicators
    for phrase in ["do not", "must not", "shall not", "cannot", "tidak boleh", "dilarang", "di larang"]:
        if phrase in lower:
            return "forbidden"

    words = set(re.findall(r"[a-zA-Z0-9_\-]+", lower))
    if words & FORBIDDEN_TERMS:
        return "forbidden"
    if words & MANDATORY_TERMS:
        return "mandatory"
    return None


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm1 * norm2)))


class ContradictionResolver:
    """
    Cognitive Contradiction Resolver & Self-Correcting Engine.
    Detects factual clashes between memory nodes and eliminates outdated/hallucinated
    facts by marking them as 'superseded' with strength=0.05.
    """

    def __init__(
        self,
        storage: Optional[Any] = None,
        embedder: Optional[Any] = None,
        vector_threshold: float = 0.82,
    ) -> None:
        self.storage = storage
        self.embedder = embedder
        self.vector_threshold = vector_threshold

    def _get_node_text(self, node: Dict[str, Any]) -> str:
        """Compose all searchable descriptive text from a node dict."""
        parts = [
            node.get("l0_abstract", ""),
            node.get("l1_overview", ""),
            node.get("content", ""),
            node.get("title", ""),
        ]
        return " ".join(p for p in parts if p).strip()

    def _compute_vector_sim(
        self,
        node_a: Dict[str, Any],
        node_b: Dict[str, Any],
        text_a: str,
        text_b: str,
    ) -> Optional[float]:
        """Attempt to retrieve or compute vector similarity between two nodes."""
        vec_a: Optional[List[float]] = None
        vec_b: Optional[List[float]] = None

        if self.storage is not None:
            if "id" in node_a:
                vec_a = self.storage.get_vector(node_a["id"])
            if "id" in node_b:
                vec_b = self.storage.get_vector(node_b["id"])

        if vec_a is None and self.embedder is not None and text_a:
            try:
                vec_a = self.embedder.embed(text_a)
            except Exception:
                pass

        if vec_b is None and self.embedder is not None and text_b:
            try:
                vec_b = self.embedder.embed(text_b)
            except Exception:
                pass

        if vec_a is not None and vec_b is not None:
            return cosine_similarity(vec_a, vec_b)

        return None

    def is_same_topic(
        self,
        node_a: Dict[str, Any],
        node_b: Dict[str, Any],
        vector_sim: Optional[float] = None,
    ) -> Tuple[bool, float, str]:
        """
        Evaluate subject/topic similarity between two nodes.
        Uses keyword overlap or vector cosine similarity > 0.82 on the same category.
        Returns (is_same, score, method).
        """
        text_a = self._get_node_text(node_a)
        text_b = self._get_node_text(node_b)

        cat_a = str(node_a.get("category", "general")).lower()
        cat_b = str(node_b.get("category", "general")).lower()
        same_category = (cat_a == cat_b) or (cat_a == "general") or (cat_b == "general")

        # 1. Vector similarity check (> 0.82 on the same/compatible category)
        sim = vector_sim
        if sim is None:
            sim = self._compute_vector_sim(node_a, node_b, text_a, text_b)

        if sim is not None and sim >= self.vector_threshold:
            if same_category:
                return True, float(round(sim, 4)), "vector_similarity"

        # 2. Keyword overlap check
        words_a = extract_keywords(text_a)
        words_b = extract_keywords(text_b)
        shared = words_a & words_b
        union = words_a | words_b
        jaccard = len(shared) / max(1, len(union))

        # Check if they share key domain entities (e.g. ram, server, vps, port, postgresql, redis, etc.)
        shared_entities = shared & DOMAIN_ENTITIES
        if shared_entities:
            if len(shared) >= 2 and jaccard >= 0.15:
                return True, float(round(jaccard, 4)), "keyword_entity_overlap"

        # Check for significant general keyword overlap
        if len(shared) >= 2 and jaccard >= 0.20:
            return True, float(round(jaccard, 4)), "keyword_overlap"

        if jaccard >= 0.25:
            return True, float(round(jaccard, 4)), "keyword_jaccard"

        return False, float(round(jaccard, 4)), "none"

    def detect_claim_conflict(
        self,
        node_a: Dict[str, Any],
        node_b: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Detect differences in factual claims between two nodes.
        Checks:
        1. Numeric / specification differences (e.g. RAM 2GB vs 4GB, port 8080 vs 3000)
        2. Operational status conflicts (active vs inactive)
        3. Directive rule conflicts (mandatory vs forbidden)
        4. Opposing patterns (theme, spaces/tabs, opposing db engines)
        """
        text_a = self._get_node_text(node_a)
        text_b = self._get_node_text(node_b)

        # 1. Numeric / specification mismatch check
        specs_a = extract_numeric_specs(text_a)
        specs_b = extract_numeric_specs(text_b)
        for attr, (val_a, unit_a) in specs_a.items():
            if attr in specs_b:
                val_b, unit_b = specs_b[attr]
                # If units match or one is missing, but numbers differ
                if (unit_a == unit_b or not unit_a or not unit_b) and val_a != val_b:
                    if attr.endswith("_version"):
                        # Check if both nodes qualify the same component (e.g. both refer to "backend", "database", etc.)
                        words_a = extract_keywords(text_a)
                        words_b = extract_keywords(text_b)
                        tech_base = attr.replace("_version", "")
                        shared_qualifiers = (words_a & words_b) - {tech_base, "version"}
                        if not shared_qualifiers:
                            # Different components using the same technology (e.g. Backend API vs Migration script) -> NOT a contradiction!
                            continue

                    return {
                        "conflict_type": "numeric_specification",
                        "attribute": attr,
                        "val_a": val_a,
                        "val_b": val_b,
                        "unit_a": unit_a,
                        "unit_b": unit_b,
                        "reason": f"Factual specification conflict on '{attr}': {val_a}{unit_a} vs {val_b}{unit_b}",
                    }

        # 2. Operational status & Task lifecycle conflict (active vs inactive / in_progress vs completed)
        status_a = extract_status_flags(text_a)
        status_b = extract_status_flags(text_b)
        if status_a and status_b and status_a != status_b:
            words_a = extract_keywords(text_a)
            words_b = extract_keywords(text_b)
            status_stopwords = {
                "active", "inactive", "stopped", "running", "progress", "wip", "completed", "complete",
                "done", "finished", "closed"
            }
            shared_topics = (words_a & words_b) - status_stopwords
            is_task_transition = {status_a, status_b} == {"in_progress", "completed"}
            if is_task_transition and not shared_topics:
                # Different tasks (e.g. database migration vs frontend deployment) -> NOT a contradiction!
                pass
            else:
                topic_str = ", ".join(sorted(shared_topics)[:3]) if shared_topics else "service"
                return {
                    "conflict_type": "operational_status",
                    "status_a": status_a,
                    "status_b": status_b,
                    "reason": f"Operational status conflict: node A claims '{status_a}' while node B claims '{status_b}' for '{topic_str}'",
                }

        # 3. Normative directive rule conflict (mandatory vs forbidden)
        dir_a = extract_directive_flags(text_a)
        dir_b = extract_directive_flags(text_b)
        if dir_a and dir_b and dir_a != dir_b:
            words_a = extract_keywords(text_a)
            words_b = extract_keywords(text_b)
            shared_topics = words_a & words_b
            topic_str = ", ".join(sorted(shared_topics)[:3]) if shared_topics else "topic"
            return {
                "conflict_type": "directive_rule",
                "rule_a": dir_a,
                "rule_b": dir_b,
                "reason": f"Opposing directive rules: '{dir_a}' vs '{dir_b}' regarding '{topic_str}'",
            }

        # 4. Opposing patterns check (opposing architectural choices or preferences)
        lower_a = text_a.lower()
        lower_b = text_b.lower()
        for p1, p2 in OPPOSING_PATTERNS:
            if (p1.search(lower_a) and p2.search(lower_b)) or (p2.search(lower_a) and p1.search(lower_b)):
                return {
                    "conflict_type": "opposing_patterns",
                    "reason": "Opposing configurations or architectural directives detected",
                }

        return None

    def check_contradiction(
        self,
        node_a: Dict[str, Any],
        node_b: Dict[str, Any],
        vector_sim: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Check if node_a and node_b contradict each other on the same subject/topic.
        Returns conflict details dict if contradiction found, None otherwise.
        """
        is_same, score, method = self.is_same_topic(node_a, node_b, vector_sim=vector_sim)
        if not is_same:
            return None

        conflict = self.detect_claim_conflict(node_a, node_b)
        if not conflict:
            return None

        conflict["topic_method"] = method
        conflict["topic_score"] = score
        return conflict

    def determine_winner_and_loser(
        self,
        new_node: Dict[str, Any],
        old_node: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        """
        Determine winner and loser between new_node and old_node.
        Core Rule:
        If new_node has a newer timestamp OR higher salience, new_node wins and old_node is superseded.
        """
        ts_new = parse_timestamp(new_node.get("created_at") or new_node.get("last_accessed"))
        ts_old = parse_timestamp(old_node.get("created_at") or old_node.get("last_accessed"))
        sal_new = float(new_node.get("salience", 0.5))
        sal_old = float(old_node.get("salience", 0.5))

        # Check software version downgrade protection
        specs_new = extract_numeric_specs(self._get_node_text(new_node))
        specs_old = extract_numeric_specs(self._get_node_text(old_node))
        for attr, (val_new, _) in specs_new.items():
            if attr.endswith("_version") and attr in specs_old:
                val_old, _ = specs_old[attr]
                if val_new < val_old:
                    text_new_lower = self._get_node_text(new_node).lower()
                    downgrade_terms = ("downgrade", "rollback", "revert", "kembali ke", "turun versi")
                    if not any(k in text_new_lower for k in downgrade_terms):
                        # Lower version without explicit downgrade intent cannot supersede higher version
                        return old_node, new_node, f"old_node_wins (downgrade protection for {attr} {val_old} -> {val_new})"

        if ts_new > ts_old or sal_new > sal_old:
            return new_node, old_node, "new_node_wins (newer timestamp or higher salience)"
        elif ts_old > ts_new or sal_old > sal_new:
            return old_node, new_node, "old_node_wins (older node has higher salience and newer/equal timestamp)"
        else:
            # Tie breaker: default to incoming fact winning
            return new_node, old_node, "incoming_node_wins_tie_break"

    def find_conflicts_for_candidate(
        self,
        candidate: Dict[str, Any],
        candidate_nodes: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find any existing active nodes in storage that conflict with candidate before saving.
        """
        if self.storage is None:
            return []

        cat = candidate.get("category", "general")
        if candidate_nodes is None:
            active_nodes = self.storage.list_nodes(category=cat, status="active", limit=1000)
            if cat != "general":
                active_nodes.extend(self.storage.list_nodes(category="general", status="active", limit=500))
            candidate_nodes = active_nodes

        conflicts = []
        cand_id = candidate.get("id")
        for node in candidate_nodes:
            if cand_id and node.get("id") == cand_id:
                continue
            if node.get("status") != "active":
                continue

            conflict = self.check_contradiction(candidate, node)
            if conflict:
                conflicts.append({
                    "conflicting_node_id": node.get("id"),
                    "conflicting_node_uri": node.get("uri"),
                    "conflict_type": conflict.get("conflict_type"),
                    "reason": conflict.get("reason"),
                })
        return conflicts

    def resolve_node_contradictions(
        self,
        new_node: Dict[str, Any],
        candidate_nodes: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Resolve contradictions between a new node and existing active nodes.
        Supersedes the loser node in storage (status='superseded', strength=0.05)
        and adds graph links.
        """
        if self.storage is None:
            return []

        new_id = new_node.get("id")
        category = new_node.get("category")

        if candidate_nodes is None:
            active_nodes = self.storage.list_nodes(category=category, status="active", limit=2000)
            if category != "general":
                active_nodes.extend(self.storage.list_nodes(category="general", status="active", limit=1000))

            seen = set()
            unique_candidates = []
            for n in active_nodes:
                nid = n.get("id")
                if nid and nid != new_id and nid not in seen:
                    seen.add(nid)
                    unique_candidates.append(n)
            candidate_nodes = unique_candidates

        resolved: List[Dict[str, Any]] = []

        for cand in candidate_nodes:
            cand_id = cand.get("id")
            if not cand_id or cand_id == new_id:
                continue
            if cand.get("status") != "active":
                continue

            conflict = self.check_contradiction(new_node, cand)
            if conflict:
                winner, loser, reason = self.determine_winner_and_loser(new_node, cand)
                winner_id = winner.get("id")
                loser_id = loser.get("id")

                if loser_id:
                    self.storage.supersede_node(loser_id, superseded_by=winner_id)
                    loser["status"] = "superseded"
                    loser["strength"] = 0.05

                rec = {
                    "winner_id": winner_id,
                    "loser_id": loser_id,
                    "superseded_node_id": loser_id,
                    "win_reason": reason,
                    "conflict_type": conflict.get("conflict_type"),
                    "conflict_reason": conflict.get("reason"),
                }
                resolved.append(rec)
                logger.info(
                    "Contradiction resolved: %s (Winner: %s, Loser: %s)",
                    conflict.get("reason"),
                    winner_id,
                    loser_id,
                )

                if loser_id == new_id:
                    # The new node itself was superseded
                    break

        return resolved

    def audit_all_contradictions(self) -> List[Dict[str, Any]]:
        """
        Mass contradiction audit for nightly consolidation.
        Scans all active nodes, resolves any contradictions,
        and marks outdated loser nodes as superseded with strength=0.05.
        """
        if self.storage is None:
            return []

        active_nodes = self.storage.list_nodes(status="active", limit=5000)
        if len(active_nodes) < 2:
            return []

        resolved: List[Dict[str, Any]] = []
        superseded_ids: Set[str] = set()

        by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for node in active_nodes:
            cat = str(node.get("category", "general")).lower()
            by_category[cat].append(node)

        # 1. Audit within each category
        for cat, nodes_in_cat in by_category.items():
            n_count = len(nodes_in_cat)
            for i in range(n_count):
                node_a = nodes_in_cat[i]
                id_a = node_a.get("id")
                if id_a in superseded_ids:
                    continue

                for j in range(i + 1, n_count):
                    node_b = nodes_in_cat[j]
                    id_b = node_b.get("id")
                    if id_b in superseded_ids:
                        continue

                    conflict = self.check_contradiction(node_a, node_b)
                    if conflict:
                        winner, loser, reason = self.determine_winner_and_loser(node_b, node_a)
                        winner_id = winner.get("id")
                        loser_id = loser.get("id")

                        if loser_id:
                            superseded_ids.add(loser_id)
                            self.storage.supersede_node(loser_id, superseded_by=winner_id)
                            loser["status"] = "superseded"
                            loser["strength"] = 0.05

                        rec = {
                            "winner_id": winner_id,
                            "loser_id": loser_id,
                            "superseded_node_id": loser_id,
                            "conflict_type": conflict.get("conflict_type"),
                            "conflict_reason": conflict.get("reason"),
                            "win_reason": reason,
                        }
                        resolved.append(rec)
                        logger.info(
                            "Mass audit contradiction resolved: %s (Winner: %s, Loser: %s)",
                            conflict.get("reason"),
                            winner_id,
                            loser_id,
                        )

                        if id_a in superseded_ids:
                            break

        # 2. Audit general nodes against specific categories
        if "general" in by_category:
            general_nodes = by_category["general"]
            for cat, specific_nodes in by_category.items():
                if cat == "general":
                    continue
                for g_node in general_nodes:
                    id_g = g_node.get("id")
                    if id_g in superseded_ids:
                        continue
                    for s_node in specific_nodes:
                        id_s = s_node.get("id")
                        if id_s in superseded_ids:
                            continue

                        conflict = self.check_contradiction(g_node, s_node)
                        if conflict:
                            winner, loser, reason = self.determine_winner_and_loser(s_node, g_node)
                            winner_id = winner.get("id")
                            loser_id = loser.get("id")

                            if loser_id:
                                superseded_ids.add(loser_id)
                                self.storage.supersede_node(loser_id, superseded_by=winner_id)
                                loser["status"] = "superseded"
                                loser["strength"] = 0.05

                            resolved.append({
                                "winner_id": winner_id,
                                "loser_id": loser_id,
                                "superseded_node_id": loser_id,
                                "conflict_type": conflict.get("conflict_type"),
                                "conflict_reason": conflict.get("reason"),
                                "win_reason": reason,
                            })

                            if id_g in superseded_ids:
                                break

        return resolved
