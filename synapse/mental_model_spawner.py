"""
Auto-Spawn Mental Model Detection & Creation Engine for Synapse Memory.

Detects topic clusters in the synapse_nodes graph that have grown large enough
to deserve a consolidated mental model but do not yet have one, then asks the
LLM to synthesize a name / source_query / initial content and inserts it via
storage.set_mental_model().

Reusable entry point:
    from synapse.mental_model_spawner import run_auto_spawn_step
    run_auto_spawn_step(storage=SynapseStorage(DB_PATH), log_fn=log)

Design notes:
- Pure-Python semantic clustering on category + keyword (token) overlap — no
  numpy / sklearn dependency, runs inside the constrained nightly environment.
- Capped at MAX_NEW_MODELS_PER_RUN per invocation to prevent runaway growth.
- All actions are logged through an injectable log_fn (falls back to print).
- Idempotent: never re-creates a mental model whose theme overlaps an existing
  model's signature (name + source_query + tags + content headers).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from synapse.config import get_llm_api_key, get_llm_model, get_llm_base_url, call_llm_resilient

# LLM endpoint
LLM_API_URL = f"{get_llm_base_url()}/chat/completions"
MODEL_NAME = get_llm_model()
LLM_API_KEY = get_llm_api_key()

# A cluster must have at least this many active nodes to be considered for a
# new mental model. Keeps spawner conservative.
MIN_CLUSTER_SIZE = 15

# Hard cap on new mental models created per invocation (anti-runaway growth).
MAX_NEW_MODELS_PER_RUN = 2

# When scanning a category for theme keywords, only keywords appearing in at
# least this many nodes (unique-per-node) are treated as cluster anchors.
MIN_KEYWORD_FREQ = 15

# Stopwords for the lightweight tokenizer (English + Indonesian mix, since the
# corpus is bilingual). Extended with domain-generic terms that
# are too broad to anchor a meaningful mental model (years, months, and words
# that recur across nearly every node).
_STOPWORDS = set(
    (
        "a an the and or but if then else of to in on at by for with from into "
        "over under as is are was were be been being this that these those it its "
        "i you he she we they them us our your his her their my me no not nor so "
        "too very can will shall may might must do does did done has have had "
        "the a an of to in on at by for with from as is are was were be this that it its "
        # temporal / generic — never a useful theme anchor
        "2026 2025 2024 august january february march april may june july "
        "september october november december when use used using uses "
        "user users context conversation between involving migrated supermemory "
        # corpus-structural words
        "sections key points table tags title off between content note notes "
        "file files data via new today daily session snapshot"
    ).split()
)


def _tokenize(text: str) -> List[str]:
    """Lowercase, alnum tokenize, dropping stopwords and very short tokens."""
    if not text:
        return []
    return [
        t
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) >= 3 and t not in _STOPWORDS
    ]


def _slugify(text: str) -> str:
    """Convert a human model name into a kebab-case mental model id."""
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    # Truncate to a sane id length and ensure non-empty.
    return (s[:64] or "auto-spawned-model")


# Sentinel returned when no clusters qualify.
NO_CLUSTERS_FOUND: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Cluster detection
# ---------------------------------------------------------------------------

def _load_node_corpus(storage: "Any") -> List[Dict[str, Any]]:
    """
    Load the active synapse_nodes corpus as lightweight dicts.

    Uses SynapseStorage.list_nodes (its public API) with no category filter so
    the spawner stays decoupled from raw SQL and auto-adapts to any category
    that later gets added to the graph. A high limit fetches the whole active
    set in one call (ORDER BY last_accessed DESC).
    """
    return storage.list_nodes(status="active", limit=100000)


def _existing_model_signatures(storage: "Any") -> List[Dict[str, Any]]:
    """
    Build a normalized signature for every existing mental model so newly
    detected clusters can be checked for overlap.

    Each signature collects the model's keywords from its name + source_query
    + tags + markdown section headers, plus the original name/tags for display.
    """
    existing = storage.list_mental_models()
    signatures: List[Dict[str, Any]] = []
    for mm in existing:
        text = " ".join(
            [
                str(mm.get("name", "")),
                str(mm.get("source_query", "")),
                " ".join(mm.get("tags", []) or []),
            ]
        )
        # Also include section headers from content (## ... lines).
        content = mm.get("content") or ""
        headers = " ".join(
            line.lstrip("# ").strip()
            for line in content.splitlines()
            if line.startswith("#")
        )
        text = f"{text} {headers}"
        sig = {
            "id": mm.get("id"),
            "name": mm.get("name"),
            "tags": mm.get("tags", []) or [],
            "keywords": set(_tokenize(text)),
        }
        signatures.append(sig)
    return signatures


def _cluster_overlaps_existing(
    cluster_keywords: set, signatures: List[Dict[str, Any]]
) -> bool:
    """
    Decide whether a candidate cluster is already represented by an existing
    mental model. A duplicate requires a substantial fraction of the *existing*
    model's keywords to appear in the cluster (the existing model is the more
    specific anchor), with a floor so small keyword sets are still comparable.
    """
    if not cluster_keywords:
        return True  # nothing to model -> treat as covered, skip
    for sig in signatures:
        sig_kw = sig["keywords"]
        if not sig_kw:
            continue
        overlap = cluster_keywords & sig_kw
        if not overlap:
            continue
        # Require >=3 shared terms AND >=35% of the existing model's vocabulary.
        if len(overlap) >= 3 and len(overlap) >= max(3, int(0.35 * len(sig_kw))):
            return True
    return False


def detect_unmodeled_clusters(
    storage: "Any",
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    min_keyword_freq: int = MIN_KEYWORD_FREQ,
) -> List[Dict[str, Any]]:
    """
    Analyze the synapse_nodes graph and return topic clusters that lack a
    corresponding mental model.

    Algorithm (semantic clustering by category + keyword overlap):
      1. Load all active nodes, grouped by category.
      2. Within each category, count unique-per-node keyword frequency.
      3. Anchors = keywords appearing in >= min_keyword_freq nodes.
      4. For each anchor, the cluster = nodes in that category whose text
         contains the anchor keyword. Co-anchoring merges near-duplicate
         clusters whose anchors are themselves mutually present.
      5. Keep clusters with >= min_cluster_size nodes that do NOT overlap any
         existing mental model signature.
      6. Rank by size (descending) and de-duplicate overlapping clusters.

    Returns a list of dicts:
      { category, anchor_keyword, keywords:[...], size, sample_abstracts:[...] }
    """
    corpus = _load_node_corpus(storage)
    if len(corpus) < min_cluster_size:
        return NO_CLUSTERS_FOUND

    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for node in corpus:
        cat = node.get("category") or "general"
        # Build a per-node token set (unique tokens) so frequency reflects node
        # count, not term count.
        text = " ".join(
            [
                str(node.get("l0_abstract") or ""),
                str(node.get("l1_overview") or ""),
                str(node.get("id") or ""),
            ]
        )
        node["_tokens"] = set(_tokenize(text))
        node["_category"] = cat
        by_cat[cat].append(node)

    signatures = _existing_model_signatures(storage)

    candidate_clusters: List[Dict[str, Any]] = []

    for cat, nodes in by_cat.items():
        # Keyword frequency = number of distinct nodes containing the keyword.
        freq: "Counter[str]" = Counter()
        for n in nodes:
            freq.update(n["_tokens"])

        anchors = [kw for kw, f in freq.items() if f >= min_keyword_freq]
        # Most specific / frequent anchors first so merges keep the strongest.
        anchors.sort(key=lambda k: freq[k], reverse=True)

        used_anchors: set = set()
        for anchor in anchors:
            if anchor in used_anchors:
                continue
            members = [n for n in nodes if anchor in n["_tokens"]]
            if len(members) < min_cluster_size:
                continue

            # Co-anchors: anchors present in >40% of this cluster's members are
            # the same theme -> merge and mark used so we don't re-emit them.
            co_anchors: List[str] = []
            threshold = max(2, int(0.40 * len(members)))
            for other in anchors:
                if other == anchor or other in used_anchors:
                    continue
                cnt = sum(1 for n in members if other in n["_tokens"])
                if cnt >= threshold:
                    co_anchors.append(other)
                    used_anchors.add(other)
            used_anchors.add(anchor)

            cluster_kw = set([anchor] + co_anchors)
            # Themes must show genuine shared semantic overlap — a lone anchor
            # is too weak to justify a dedicated mental model.
            if len(cluster_kw) < 2:
                continue
            if _cluster_overlaps_existing(cluster_kw, signatures):
                continue

            sample = [
                (n.get("l0_abstract") or "")[:90]
                for n in members[:6]
            ]
            candidate_clusters.append(
                {
                    "category": cat,
                    "anchor_keyword": anchor,
                    "keywords": [anchor] + co_anchors,
                    "size": len(members),
                    "sample_abstracts": sample,
                }
            )

    # Rank largest first, then strip overlapping clusters (a cluster whose
    # keyword set is a near-subset of an already-accepted cluster is dropped).
    candidate_clusters.sort(key=lambda c: c["size"], reverse=True)
    accepted: List[Dict[str, Any]] = []
    for cand in candidate_clusters:
        cand_kw = set(cand["keywords"])
        dup = False
        for acc in accepted:
            acc_kw = set(acc["keywords"])
            inter = cand_kw & acc_kw
            if len(inter) >= 3 and (
                len(inter) >= 0.6 * len(cand_kw)
                or len(inter) >= 0.6 * len(acc_kw)
            ):
                dup = True
                break
        if not dup:
            accepted.append(cand)
    return accepted


# ---------------------------------------------------------------------------
# LLM model synthesis
# ---------------------------------------------------------------------------

def _call_llm(system_prompt: str, user_prompt: str, log_fn: Callable[[str], None]) -> Optional[str]:
    """
    Call the configured LLM endpoint with resilient loopback and retry mechanism.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return call_llm_resilient(
        messages=messages,
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=3000,
        timeout=120.0,
        max_retries=3,
        log_fn=log_fn,
    )


def _parse_llm_model_json(raw: str) -> Optional[Dict[str, str]]:
    """
    Robustly extract the model JSON object from an LLM response that may be
    wrapped in ```json fences or carry surrounding prose.
    """
    if not raw:
        return None
    cleaned = raw.strip()
    # Strip code fences.
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Try direct parse first.
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: extract the first {...} block.
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def _generate_mental_model_for_cluster(
    cluster: Dict[str, Any],
    log_fn: Callable[[str], None],
) -> Optional[Tuple[str, str, str, List[str]]]:
    """
    Ask the LLM to synthesize (model_id_slug, name, source_query, content,
    tags) for a single detected cluster. Returns None on failure.

    The LLM is instructed to return strict JSON matching the schema consumed by
    storage.set_mental_model().
    """
    kws = cluster.get("keywords", [])
    sample = cluster.get("sample_abstracts", [])

    system_prompt = (
        "You are the Auto-Spawn Mental Model Engine of Synapse Memory. "
        "Your task is to synthesize a single consolidated Mental Model for a cluster "
        "of memory nodes. Output MUST be pure JSON without extra prose.\n\n"
        "SAFETY RULES:\n"
        "1. DO NOT invent facts outside the provided node contexts.\n"
        "2. Name must be concise, descriptive, and professional.\n"
        "3. source_query must be a focused question this model answers.\n"
        "4. content must be clean markdown with headings and bullets from nodes.\n"
        "5. tags: 3-6 relevant keywords.\n\n"
        "REQUIRED JSON SCHEMA:\n"
        "{\n"
        '  "name": "Mental Model Title",\n'
        '  "source_query": "Focused question answered by this model?",\n'
        '  "content": "# Heading\\n- bullet\\n- bullet",\n'
        '  "tags": ["tag1", "tag2"]\n'
        "}"
    )

    user_prompt = (
        f"Detected Cluster:\n"
        f"- Category: {cluster.get('category')}\n"
        f"- Size: {cluster.get('size')} nodes\n"
        f"- Keyword anchors: {', '.join(kws)}\n"
        f"- Sample node abstracts:\n"
        + "\n".join(f"  - {s}" for s in sample)
        + "\n\nSynthesize one consolidated Mental Model as pure JSON."
    )

    raw = _call_llm(system_prompt, user_prompt, log_fn)
    if not raw:
        log_fn("[AutoSpawn] LLM returned empty response for cluster; skipping.")
        return None

    obj = _parse_llm_model_json(raw)
    if not obj:
        log_fn("[AutoSpawn] Failed to parse LLM JSON; skipping cluster.")
        return None

    name = (obj.get("name") or "").strip()
    source_query = (obj.get("source_query") or "").strip()
    content = (obj.get("content") or "").strip()
    tags_raw = obj.get("tags") or []
    tags = [t.strip() for t in tags_raw if isinstance(t, str) and t.strip()]

    if not name or not source_query or not content:
        log_fn(
            f"[AutoSpawn] LLM JSON missing required fields "
            f"(name/source_query/content); skipping cluster '{kws}'."
        )
        return None

    model_slug = _slugify(name)
    # Tags: ensure auto-spawn marker for traceability.
    if "auto-spawned" not in tags:
        tags.append("auto-spawned")

    return model_slug, name, source_query, content, tags


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_auto_spawn_step(
    storage: "Any",
    log_fn: Optional[Callable[[str], None]] = None,
    max_new: int = MAX_NEW_MODELS_PER_RUN,
) -> int:
    """
    Detect unmodeled topic clusters, LLM-synthesize a mental model for each (up
    to `max_new` per run), and persist via storage.set_mental_model().

    Returns the number of new mental models created this run.
    """
    log = log_fn or (lambda m: print(m))

    log(
        f"[AutoSpawn] Scanning synapse graph for unmodeled topic clusters "
        f"(min_cluster_size={MIN_CLUSTER_SIZE}, min_keyword_freq={MIN_KEYWORD_FREQ}, "
        f"cap={max_new})..."
    )

    try:
        clusters = detect_unmodeled_clusters(storage)
    except Exception as e:  # noqa: BLE001 - nightly must not crash
        log(f"[AutoSpawn] CRITICAL: cluster detection failed: {e}")
        return 0

    if not clusters:
        log("[AutoSpawn] No qualifying unmodeled clusters found. Skipping spawn.")
        return 0

    log(f"[AutoSpawn] Detected {len(clusters)} candidate cluster(s):")
    for c in clusters:
        log(
            f"[AutoSpawn]   - [{c['category']}] anchor='{c['anchor_keyword']}' "
            f"size={c['size']} kw={c['keywords']}"
        )

    created = 0
    existing_ids = {mm.get("id") for mm in storage.list_mental_models()}

    for cluster in clusters:
        if created >= max_new:
            log(f"[AutoSpawn] Reached per-run cap ({max_new}); stopping spawn.")
            break

        result = _generate_mental_model_for_cluster(cluster, log)
        if not result:
            continue
        model_slug, name, source_query, content, tags = result

        # Guard against id collision with an existing model.
        if model_slug in existing_ids:
            log(
                f"[AutoSpawn] Model id '{model_slug}' already exists; skipping "
                f"(would duplicate)."
            )
            continue

        try:
            storage.set_mental_model(
                model_id=model_slug,
                name=name,
                source_query=source_query,
                content=content,
                structured_content=None,
                trigger_mode="delta",
                tags=tags,
                max_tokens=2048,
            )
            existing_ids.add(model_slug)
            created += 1
            log(
                f"[AutoSpawn] CREATED new mental model '{model_slug}' "
                f"({name}) | source_query='{source_query[:60]}' "
                f"| tags={tags} from cluster anchor='{cluster['anchor_keyword']}'."
            )
        except Exception as e:  # noqa: BLE001
            log(f"[AutoSpawn] ERROR persisting model '{model_slug}': {e}")

    log(f"[AutoSpawn] Step 9.5 complete: {created} new mental model(s) created.")
    return created
