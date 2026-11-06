"""
SynapseRetriever - Hybrid Lexical, Semantic Vector & Hebbian Spreading Activation Retrieval Engine.
"""

from __future__ import annotations

import datetime
import math
import os
import re
import threading
import time
import zlib
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from synapse.storage import SynapseStorage
from synapse.embedder import SynapseEmbedder, DEFAULT_MODEL_NAME

# Native embedding dimension for the sentence-transformers backend
# (all-MiniLM-L6-v2 => 384). Used as the default vector_dim so that newly
# created vectors match the neural embedder's output dimension.
_ST_NATIVE_DIM = 384

def _default_vector_dim() -> int:
    """Return the unified default vector dimension (384)."""
    return _ST_NATIVE_DIM


def _as_float(value: Any, default: float) -> float:
    """Coerce a stored score to float, tolerating NULL and non-finite legacy rows."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


OPPOSING_PATTERNS: List[Tuple[re.Pattern[str], re.Pattern[str]]] = [
    (
        re.compile(r"\b(prefer|always|enable|use|adopt|active|true|light|tabs|allow|permit|yes|direct|wajib|harus|pakai|gunakan|boleh)\b", re.IGNORECASE),
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
        re.compile(r"\b(dark mode|theme: dark)\b", re.IGNORECASE),
        re.compile(r"\b(light mode|theme: light)\b", re.IGNORECASE),
    ),
]


def detect_contradictions(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Cognitive contradiction and metacognition detector between recalled memories.
    Identifies conflicting directives, opposing preferences, or contradicting decisions.
    Determines the winning memory based on Salience and Confidence weighting.
    """
    if not nodes or len(nodes) < 2:
        return []

    # Optimization: Pre-extract text and topic words once per node (O(N) instead of O(N^2))
    node_data = []
    stop_words = {"a", "an", "the", "in", "on", "for", "with", "to", "is", "of", "and", "or"}
    for n in nodes:
        t = f"{n.get('l0_abstract', '')} {n.get('l1_overview', '')} {n.get('title', '')} {n.get('lessons_learned', '')}".lower()
        words = set(re.findall(r"[a-z0-9_\-]+", t)) - stop_words
        node_data.append((n, t, words))

    contradictions: List[Dict[str, Any]] = []

    for i in range(len(node_data)):
        a, text_a, words_a = node_data[i]
        cat_a = a.get("category")
        for j in range(i + 1, len(node_data)):
            b, text_b, words_b = node_data[j]
            cat_b = b.get("category")

            # Fast topic filter: skip comparison if no common topic words and different category
            shared_words = words_a & words_b
            if not shared_words and cat_a != cat_b:
                continue

            is_conflict = False
            reason = ""

            # Check opposing patterns
            for p1, p2 in OPPOSING_PATTERNS:
                if (p1.search(text_a) and p2.search(text_b)) or (p2.search(text_a) and p1.search(text_b)):
                    is_conflict = True
                    reason = f"Opposing directives detected regarding topic '{', '.join(sorted(shared_words)[:3]) or cat_a or 'general'}'"
                    break

            if is_conflict:
                sal_a = _as_float(a.get("salience"), 0.5)
                sal_b = _as_float(b.get("salience"), 0.5)
                conf_a = _as_float(a.get("confidence"), 1.0)
                conf_b = _as_float(b.get("confidence"), 1.0)

                if sal_a > sal_b:
                    winner_id = a.get("id")
                    loser_id = b.get("id")
                elif sal_b > sal_a:
                    winner_id = b.get("id")
                    loser_id = a.get("id")
                elif conf_a >= conf_b:
                    winner_id = a.get("id")
                    loser_id = b.get("id")
                else:
                    winner_id = b.get("id")
                    loser_id = a.get("id")

                contradictions.append({
                    "node_a_id": a.get("id"),
                    "node_b_id": b.get("id"),
                    "node_a_uri": a.get("uri", a.get("title", "")),
                    "node_b_uri": b.get("uri", b.get("title", "")),
                    "conflict": reason or f"Contradiction between '{a.get('l0_abstract', a.get('title', ''))}' and '{b.get('l0_abstract', b.get('title', ''))}'",
                    "winner_id": winner_id,
                    "loser_id": loser_id,
                    "salience_diff": round(abs(sal_a - sal_b), 4),
                    "confidence_a": conf_a,
                    "confidence_b": conf_b,
                })

    return contradictions


def vectorize_text(text: str, dim: int = 384) -> List[float]:
    """
    Pure-Python deterministic zero-dependency subword n-gram / semantic hashing vectorizer.
    Produces unit-normalized L2 float embeddings of dimension `dim`.
    """
    if not text or not text.strip():
        return [0.0] * dim

    clean_text = text.lower().strip()
    words = re.findall(r"[a-z0-9_\-]+", clean_text)
    if not words:
        return [0.0] * dim

    vec = [0.0] * dim

    # 1. Whole word tokens
    for w in words:
        h = zlib.crc32(w.encode("utf-8"))
        bucket = h % dim
        sign = 1.0 if ((h >> 16) & 1) else -1.0
        vec[bucket] += sign * 1.0
        # Subword character n-grams (3-grams and 4-grams)
        if len(w) >= 3:
            for n in (3, 4):
                for i in range(len(w) - n + 1):
                    ngram = w[i : i + n]
                    hn = zlib.crc32(ngram.encode("utf-8"))
                    vec[hn % dim] += (1.0 if ((hn >> 16) & 1) else -1.0) * 0.5

    # 2. Word bigrams
    for i in range(len(words) - 1):
        bigram = f"{words[i]}_{words[i+1]}"
        hb = zlib.crc32(bigram.encode("utf-8"))
        vec[hb % dim] += (1.0 if ((hb >> 16) & 1) else -1.0) * 0.75

    # L2 normalize
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return [0.0] * dim

    return [round(x / norm, 6) for x in vec]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 * n2 == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (n1 * n2)))


WIKILINK_PATTERN = re.compile(r"\[\[(.*?)\]\]")


def extract_wikilinks_from_markdown(text: str) -> List[str]:
    """
    Extract clean wikilink target names/paths from Markdown text.
    Strips display aliases (|alias), heading anchors (#heading),
    and optional .md/.markdown file extensions.
    """
    if not text:
        return []
    targets: List[str] = []
    seen: Set[str] = set()
    for match in WIKILINK_PATTERN.finditer(text):
        raw = match.group(1).strip()
        if not raw:
            continue
        # Strip alias: [[target|label]]
        target = raw.split("|", 1)[0].strip()
        # Strip heading anchor: [[target#anchor]]
        target = target.split("#", 1)[0].strip()
        # Strip trailing extension if explicitly specified: [[target.md]]
        if target.lower().endswith(".md"):
            target = target[:-3].strip()
        elif target.lower().endswith(".markdown"):
            target = target[:-9].strip()
        if target and target.lower() not in seen:
            seen.add(target.lower())
            targets.append(target)
    return targets


class VaultWikilinkCache:
    """
    Lightweight, thread-safe in-memory cache for physical Vault markdown wikilinks.
    Tracks file modification timestamps (mtime) and file size, invalidating
    cached wikilinks only when physical disk contents change, avoiding redundant disk I/O.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._stats = {"hits": 0, "misses": 0}

    def get_wikilinks(self, file_path: Union[str, Path]) -> List[str]:
        """
        Get wikilinks for a physical markdown file.
        Returns cached wikilinks if file mtime and size are unchanged.
        """
        try:
            p = Path(file_path).expanduser().resolve()
            if not p.is_file():
                return []
            path_str = str(p)
            st = p.stat()
        except (OSError, ValueError):
            return []

        with self._lock:
            cached = self._cache.get(path_str)
            if cached and cached["mtime"] == st.st_mtime and cached["size"] == st.st_size:
                self._stats["hits"] += 1
                return list(cached["wikilinks"])

        try:
            content = p.read_text(encoding="utf-8", errors="replace")
            links = extract_wikilinks_from_markdown(content)
        except Exception:
            links = []

        with self._lock:
            self._stats["misses"] += 1
            self._cache[path_str] = {
                "mtime": st.st_mtime,
                "size": st.st_size,
                "cached_at": time.time(),
                "wikilinks": links,
            }
            return list(links)

    def get_stats(self) -> Dict[str, int]:
        """Return cache hit/miss/entries statistics."""
        with self._lock:
            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "entries": len(self._cache),
            }

    def clear(self) -> None:
        """Clear all cached entries and reset stats."""
        with self._lock:
            self._cache.clear()
            self._stats = {"hits": 0, "misses": 0}


class SynapseRetriever:
    """
    Hybrid retriever combining full-text lexical search (BM25), dense semantic
    vector similarity, Hebbian graph spreading activation, recency half-life decay,
    and importance/strength weighting across cognitive synapse nodes.
    """

    def __init__(
        self,
        storage: Optional[SynapseStorage] = None,
        bm25_weight: float = 0.35,
        vector_weight: float = 0.35,
        activation_weight: float = 0.15,
        recency_weight: float = 0.15,
        half_life_seconds: float = 604800.0,  # 7 days default
        vector_dim: Optional[int] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        disposition: Optional[Dict[str, int]] = None,
        vault_path: Optional[Union[str, Path]] = None,
        wikilink_weight: float = 0.8,
        wikilink_max_hops: int = 1,
        vault_cache: Optional[VaultWikilinkCache] = None,
    ) -> None:
        self.storage = storage if storage is not None else SynapseStorage()
        # Disposition traits: skepticism (1-5), literalism (1-5), empathy (1-5)
        self.disposition = disposition or {"skepticism": 5, "literalism": 4, "empathy": 2}

        # Adaptive weight tuning based on disposition:
        # High skepticism (4-5) boosts BM25 exact verification and reduces blind vector fuzziness
        skep_ratio = (self.disposition.get("skepticism", 5) - 3) * 0.05
        lit_ratio = (self.disposition.get("literalism", 4) - 3) * 0.05

        self.bm25_weight = max(0.2, min(0.6, bm25_weight + skep_ratio + lit_ratio))
        self.vector_weight = max(0.2, min(0.5, vector_weight - skep_ratio))
        self.activation_weight = activation_weight
        self.recency_weight = recency_weight
        self.half_life_seconds = half_life_seconds
        # When sentence-transformers is active, default to its native dim (384);
        # otherwise fall back to 64 for the CRC32 vectorizer. An explicit
        # vector_dim always wins so callers can force a specific dimension.
        self.vector_dim = vector_dim if vector_dim is not None else _default_vector_dim()
        self.embedder = SynapseEmbedder(dim=self.vector_dim, model_name=model_name)

        # Vault Wikilink Crawler configuration & caches
        raw_vault = vault_path or os.environ.get("SYNAPSE_VAULT_PATH")
        if not raw_vault and (Path.home() / "vault").is_dir():
            raw_vault = str(Path.home() / "vault")
        self.vault_path = Path(raw_vault).expanduser().resolve() if raw_vault else None
        self.vault_cache = vault_cache if vault_cache is not None else VaultWikilinkCache()
        self.wikilink_weight = float(wikilink_weight)
        self.wikilink_max_hops = int(wikilink_max_hops)

        self._vault_backlinks: Dict[str, Set[str]] = {}  # target_key -> set of source file paths
        self._wikilink_node_id_cache: Dict[str, Optional[str]] = {}
        self._vault_indexed_at: float = 0.0
        self._vault_index_lock = threading.Lock()

    def vectorize(self, text: str) -> List[float]:
        """Generate normalized semantic embedding vector from text."""
        return self.embedder.embed(text)

    def ensure_node_vector(
        self, node_id: str, node: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """Auto-generate and store embedding vector if missing for a node."""
        vec = self.storage.get_vector(node_id)
        if vec is not None:
            return vec
        if node is None:
            node = self.storage.get_node(node_id)
        if not node:
            return [0.0] * self.vector_dim
        text = f"{node.get('l0_abstract', '')} {node.get('l1_overview', '')}".strip()
        vec = self.vectorize(text)
        self.storage.upsert_vector(node_id, vec)
        return vec

    def auto_index_missing_vectors(self) -> int:
        """Generate and store vectors for any active nodes in storage that lack vectors."""
        missing = self.storage.list_nodes_missing_vectors(limit=100)

        indexed_count = 0
        for node in missing:
            text = f"{node.get('l0_abstract', '')} {node.get('l1_overview', '')}".strip()
            vec = self.vectorize(text)
            self.storage.upsert_vector(node["id"], vec)
            indexed_count += 1
        return indexed_count

    def _normalize_bm25_scores(
        self, seed_nodes: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Normalize BM25 rank scores from SQLite FTS5 to [0.0, 1.0].
        In FTS5 bm25(), rank is negative float where smaller (more negative)
        indicates higher relevance.
        """
        if not seed_nodes:
            return {}

        raw_scores: Dict[str, float] = {}
        for node in seed_nodes:
            r = node.get("rank", 0.0)
            # Convert negative rank to positive relevance
            relevance = -float(r) if float(r) < 0 else float(r)
            raw_scores[node["id"]] = max(0.0001, relevance)

        max_score = max(raw_scores.values())
        min_score = min(raw_scores.values())

        normalized: Dict[str, float] = {}
        for node_id, score in raw_scores.items():
            if max_score == min_score:
                normalized[node_id] = 1.0
            else:
                # Proportional score normalization relative to top result
                normalized[node_id] = score / max_score

        return normalized

    def detect_contradictions(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Instance helper to detect contradictions among retrieved items."""
        return detect_contradictions(nodes)

    def _normalize_wikilink_key(self, target: str) -> str:
        """Normalize target to stem key for dictionary lookups."""
        clean = str(target).strip()
        if clean.lower().endswith(".md"):
            clean = clean[:-3].strip()
        elif clean.lower().endswith(".markdown"):
            clean = clean[:-9].strip()
        return Path(clean).stem.lower()

    def resolve_node_markdown_path(self, node: Dict[str, Any]) -> Optional[Path]:
        """
        Locate the physical Markdown note in ~/vault/ or file system
        associated with this synapse node.
        """
        l2 = node.get("l2_target_path")
        if l2:
            try:
                p = Path(l2).expanduser()
                if not p.is_absolute() and self.vault_path:
                    p = (self.vault_path / p)
                if p.is_file():
                    return p.resolve()
            except Exception:
                pass

        uri = str(node.get("uri") or "")
        if uri.startswith("vault:"):
            raw = uri[6:].replace(":", "/")
            if self.vault_path and self.vault_path.is_dir():
                for cand in (
                    self.vault_path / f"{raw}.md",
                    self.vault_path / raw,
                    self.vault_path / f"{raw}.markdown",
                ):
                    if cand.is_file():
                        return cand.resolve()
                stem = Path(raw).stem
                for cand in self.vault_path.rglob(f"{stem}.md"):
                    if cand.is_file():
                        return cand.resolve()

        if uri.startswith("synapse://sop/"):
            slug = uri[len("synapse://sop/"):]
            if self.vault_path and self.vault_path.is_dir():
                for cand in (
                    self.vault_path / f"{slug}.md",
                    self.vault_path / slug,
                    self.vault_path / f"{slug}.markdown",
                ):
                    if cand.is_file():
                        return cand.resolve()
                stem = Path(slug).stem
                for cand in self.vault_path.rglob(f"{stem}.md"):
                    if cand.is_file():
                        return cand.resolve()

        return None

    def resolve_node_by_wikilink(self, target: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Resolve a wikilink target (or file path/stem) to an active SynapseStorage node.
        """
        raw_str = str(target).strip()
        if not raw_str:
            return None

        # Check in-memory resolution cache
        if raw_str in self._wikilink_node_id_cache:
            cached_id = self._wikilink_node_id_cache[raw_str]
            return self.storage.get_node(cached_id) if cached_id else None

        clean = raw_str
        if clean.lower().endswith(".md"):
            clean = clean[:-3].strip()
        elif clean.lower().endswith(".markdown"):
            clean = clean[:-9].strip()

        stem = Path(clean).stem.lower()
        clean_slug = re.sub(r"[^a-z0-9_\-/]+", "_", clean.lower()).strip("_/")
        clean_stem = re.sub(r"[^a-z0-9_\-]+", "_", stem).strip("_")

        # 1. Direct ID / URI lookup
        direct = self.storage.get_node(raw_str)
        if direct and direct.get("status") == "active":
            self._wikilink_node_id_cache[raw_str] = direct["id"]
            return direct

        vault_direct = self.storage.get_node(f"vault:{clean}")
        if vault_direct and vault_direct.get("status") == "active":
            self._wikilink_node_id_cache[raw_str] = vault_direct["id"]
            return vault_direct

        # 2. Database query matching uri, l2_target_path, clean slug, stem, etc.
        try:
            node = self.storage.resolve_wikilink_node(
                raw=raw_str,
                clean=clean,
                clean_slug=clean_slug,
                clean_stem=clean_stem,
                stem=stem,
            )
            if node:
                self._wikilink_node_id_cache[raw_str] = node["id"]
                return node
        except Exception:
            pass

        # 3. Physical file resolution under vault_path
        if self.vault_path and self.vault_path.is_dir():
            for cand in (
                self.vault_path / f"{clean}.md",
                self.vault_path / clean,
            ):
                if cand.is_file():
                    abs_cand = str(cand.resolve())
                    try:
                        node = self.storage.find_active_node_by_l2_paths(
                            [abs_cand, str(cand)]
                        )
                        if node:
                            self._wikilink_node_id_cache[raw_str] = node["id"]
                            return node
                    except Exception:
                        pass
                    break

        self._wikilink_node_id_cache[raw_str] = None
        return None

    def index_vault_wikilinks(self, force: bool = False, scan_vault_dir: bool = True) -> Dict[str, int]:
        """
        Build or refresh the in-memory bidirectional wikilink graph index.
        Scans physical Markdown files using VaultWikilinkCache (mtime-checked).
        Indexes forward links and reverse backlinks.
        """
        with self._vault_index_lock:
            now = time.time()
            if not force and self._vault_indexed_at > 0.0 and (now - self._vault_indexed_at < 60.0):
                return {
                    "indexed_files": len(self.vault_cache._cache),
                    "backlink_targets": len(self._vault_backlinks),
                }

            files_to_scan: Set[Path] = set()

            # 1. Scan storage nodes with l2_target_path
            if self.storage:
                try:
                    for row in self.storage.list_active_vault_targets():
                        l2 = row["l2_target_path"]
                        if l2:
                            p = Path(l2).expanduser()
                            if not p.is_absolute() and self.vault_path:
                                p = (self.vault_path / p)
                            if p.is_file():
                                files_to_scan.add(p.resolve())
                except Exception:
                    pass

            # 2. Scan vault_path directory
            if scan_vault_dir and self.vault_path and self.vault_path.is_dir():
                for ext in ("*.md", "*.markdown"):
                    for p in self.vault_path.rglob(ext):
                        path_str = str(p)
                        if "/." in path_str or "/_arsip/" in path_str:
                            continue
                        files_to_scan.add(p.resolve())

            backlinks: Dict[str, Set[str]] = {}
            for file_path in files_to_scan:
                path_str = str(file_path)
                links = self.vault_cache.get_wikilinks(file_path)
                for link in links:
                    key = self._normalize_wikilink_key(link)
                    if key not in backlinks:
                        backlinks[key] = set()
                    backlinks[key].add(path_str)

                    if self.vault_path and self.vault_path.is_dir():
                        try:
                            rel = file_path.relative_to(self.vault_path).with_suffix("").as_posix().lower()
                            if rel not in backlinks:
                                backlinks[rel] = set()
                            backlinks[rel].add(path_str)
                        except (ValueError, Exception):
                            pass

            self._vault_backlinks = backlinks
            self._vault_indexed_at = now
            return {
                "indexed_files": len(files_to_scan),
                "backlink_targets": len(backlinks),
            }

    def get_incoming_vault_backlinks(self, node: Dict[str, Any]) -> List[str]:
        """Find source file paths containing wikilinks that point to this node."""
        self.index_vault_wikilinks()

        keys_to_check: Set[str] = set()
        l2 = node.get("l2_target_path")
        if l2:
            keys_to_check.add(Path(l2).stem.lower())
            if self.vault_path:
                try:
                    rel = Path(l2).resolve().relative_to(self.vault_path).with_suffix("").as_posix().lower()
                    keys_to_check.add(rel)
                except Exception:
                    pass

        uri = str(node.get("uri") or "")
        if uri.startswith("vault:"):
            raw = uri[6:].replace(":", "/")
            keys_to_check.add(Path(raw).stem.lower())
            keys_to_check.add(raw.lower())
        elif uri.startswith("synapse://sop/"):
            slug = uri[len("synapse://sop/"):]
            keys_to_check.add(Path(slug).stem.lower())
            keys_to_check.add(slug.lower())

        if "id" in node:
            keys_to_check.add(node["id"])

        source_paths: Set[str] = set()
        for k in keys_to_check:
            if k in self._vault_backlinks:
                source_paths.update(self._vault_backlinks[k])

        return list(source_paths)

    def get_vault_wikilink_neighbors(
        self,
        node_id: str,
        node: Optional[Dict[str, Any]] = None,
        direction: str = "both",
    ) -> List[Dict[str, Any]]:
        """
        Bidirectional wikilink crawler for vault notes.
        - direction='out': returns nodes linked by this note (forward links)
        - direction='in': returns nodes linking to this note (backlinks)
        - direction='both': returns nodes from both outgoing wikilinks and incoming backlinks
        """
        if node is None:
            node = self.storage.get_node(node_id)
        if not node:
            return []

        neighbors: Dict[str, Dict[str, Any]] = {}

        # 1. Outgoing wikilinks (forward links)
        if direction in ("out", "both"):
            md_path = self.resolve_node_markdown_path(node)
            if md_path:
                targets = self.vault_cache.get_wikilinks(md_path)
                for target in targets:
                    target_node = self.resolve_node_by_wikilink(target)
                    if target_node and target_node.get("status") == "active":
                        if target_node["id"] != node_id and target_node["id"] not in neighbors:
                            neighbors[target_node["id"]] = target_node

        # 2. Incoming backlinks (reverse links)
        if direction in ("in", "both"):
            backlink_sources = self.get_incoming_vault_backlinks(node)
            for src_path in backlink_sources:
                src_node = self.resolve_node_by_wikilink(src_path)
                if src_node and src_node.get("status") == "active":
                    if src_node["id"] != node_id and src_node["id"] not in neighbors:
                        neighbors[src_node["id"]] = src_node

        return list(neighbors.values())

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        max_hops: int = 3,
        touch: bool = True,
        seed_limit: int = 15,
        hop_decay: float = 0.65,
        vector_threshold: float = 0.55,
        include_episodes: bool = True,
        task_context: Optional[str] = None,
        crawl_vault_wikilinks: bool = True,
        wikilink_max_hops: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant synapse nodes and episodic memories using Hybrid Dense-Sparse Fusion
        (BM25 + Semantic Vector Similarity + Spreading Activation + Recency/Importance/Salience/Confidence).

        Steps:
        1. Query storage.search_fts() for lexical seeds and normalize BM25 scores.
        2. Query storage.search_vectors() with query_vector for dense semantic seeds.
        3. Propagate activation energy through graph edges across max_hops.
        4. Query storage.search_episodes() for relevant episodic memories.
        5. Compute recency half-life decay, node strength weighting, salience & confidence scaling.
        6. Calculate composite score via hybrid fusion:
           composite_score = base_score * (salience / 0.5) * confidence (+ 0.5 if salience >= 0.9)
        7. Touch retrieved items if touch=True to reinforce strength and recency.
        """
        if not query or not query.strip():
            return []

        # Step 1: Query Vectorization
        query_vector = self.vectorize(query)

        # Step 2: Lexical Seed retrieval (Declarative/Semantic Nodes)
        raw_fts_results = self.storage.search_fts(query, limit=seed_limit)
        active_lexical_seeds: List[Dict[str, Any]] = [
            node for node in raw_fts_results if node.get("status") == "active"
        ]

        normalized_bm25 = self._normalize_bm25_scores(active_lexical_seeds)

        # Step 3: Dense Vector Seed retrieval (True Hybrid Retrieval)
        # Run dense vector search simultaneously with lexical search (no gating condition)
        self.auto_index_missing_vectors()
        vector_hits = self.storage.search_vectors(query_vector, top_k=seed_limit * 2)
        active_vector_seeds: List[Tuple[str, float, Dict[str, Any]]] = []
        for node_id, sim in vector_hits:
            if sim >= vector_threshold:
                node = self.storage.get_node(node_id)
                if node and node.get("status") == "active":
                    active_vector_seeds.append((node_id, float(sim), dict(node)))

        # Reciprocal Rank Fusion (RRF) combining Lexical & Dense Vector rankings
        rrf_k = 60
        rrf_scores: Dict[str, float] = {}
        for rank, node in enumerate(active_lexical_seeds, start=1):
            nid = node["id"]
            rrf_scores[nid] = rrf_scores.get(nid, 0.0) + (1.0 / (rrf_k + rank))
        for rank, (nid, sim, node) in enumerate(active_vector_seeds, start=1):
            rrf_scores[nid] = rrf_scores.get(nid, 0.0) + (1.0 / (rrf_k + rank))

        # Build unified candidate pool from both search modalities
        known_nodes: Dict[str, Dict[str, Any]] = {}
        for node in active_lexical_seeds:
            known_nodes[node["id"]] = dict(node)
        for nid, sim, node in active_vector_seeds:
            if nid not in known_nodes:
                known_nodes[nid] = node

        # Vector similarities for known nodes
        vector_similarities: Dict[str, float] = {}
        for node_id, node in known_nodes.items():
            node_vec = self.ensure_node_vector(node_id, node)
            sim = cosine_similarity(query_vector, node_vec)
            vector_similarities[node_id] = max(0.0, float(sim))

        # Step 4: Spreading Activation from seed nodes (incorporating RRF)
        max_possible_rrf = 2.0 / (rrf_k + 1.0)
        normalized_rrf: Dict[str, float] = {
            nid: score / max_possible_rrf for nid, score in rrf_scores.items()
        }

        activation_energies: Dict[str, float] = {}
        for node_id in known_nodes:
            bm_score = normalized_bm25.get(node_id, 0.0)
            v_score = vector_similarities.get(node_id, 0.0)
            r_score = normalized_rrf.get(node_id, 0.0)
            activation_energies[node_id] = max(bm_score, v_score, r_score)

        current_frontier: Set[str] = set(known_nodes.keys())

        # Pre-warm vault wikilink index once if crawling is enabled
        effective_wikilink_hops = (
            wikilink_max_hops if wikilink_max_hops is not None else self.wikilink_max_hops
        )
        if crawl_vault_wikilinks and effective_wikilink_hops > 0:
            self.index_vault_wikilinks()

        for hop in range(1, max_hops + 1):
            next_frontier: Set[str] = set()
            for curr_id in current_frontier:
                curr_energy = activation_energies.get(curr_id, 0.0)
                if curr_energy <= 0.0:
                    continue

                edges = self.storage.get_edges(curr_id, direction="both")
                for edge in edges:
                    neighbor_id = (
                        edge["target_id"]
                        if edge["source_id"] == curr_id
                        else edge["source_id"]
                    )

                    # Check or load neighbor node
                    if neighbor_id not in known_nodes:
                        neighbor_node = self.storage.get_node(neighbor_id)
                        if neighbor_node and neighbor_node.get("status") == "active":
                            known_nodes[neighbor_id] = neighbor_node
                            n_vec = self.ensure_node_vector(neighbor_id, neighbor_node)
                            vector_similarities[neighbor_id] = max(
                                0.0, float(cosine_similarity(query_vector, n_vec))
                            )
                        else:
                            continue
                    else:
                        if known_nodes[neighbor_id].get("status") != "active":
                            continue

                    edge_weight = _as_float(edge.get("weight"), 0.5)
                    # Temporal Spreading Activation: boost causal timeline transitions
                    rel_type = str(edge.get("relation", "")).lower()
                    temporal_multiplier = 1.0
                    if any(t in rel_type for t in ("before", "after", "preceded", "followed", "caused", "triggered", "then")):
                        temporal_multiplier = 1.35

                    propagated_energy = curr_energy * edge_weight * (hop_decay ** hop) * temporal_multiplier

                    prev_energy = activation_energies.get(neighbor_id, 0.0)
                    new_energy = min(1.0, prev_energy + propagated_energy)
                    activation_energies[neighbor_id] = new_energy

                    if neighbor_id not in current_frontier:
                        next_frontier.add(neighbor_id)

                # 2. Bidirectional Vault Wikilink Expansion
                if crawl_vault_wikilinks and hop <= effective_wikilink_hops:
                    curr_node = known_nodes.get(curr_id)
                    wikilink_neighbors = self.get_vault_wikilink_neighbors(
                        curr_id, node=curr_node, direction="both"
                    )
                    for neighbor_node in wikilink_neighbors:
                        neighbor_id = neighbor_node["id"]
                        if neighbor_id == curr_id:
                            continue

                        # Check or load neighbor node
                        if neighbor_id not in known_nodes:
                            if neighbor_node.get("status") != "active":
                                continue
                            known_nodes[neighbor_id] = neighbor_node
                            n_vec = self.ensure_node_vector(neighbor_id, neighbor_node)
                            vector_similarities[neighbor_id] = max(
                                0.0, float(cosine_similarity(query_vector, n_vec))
                            )
                        else:
                            if known_nodes[neighbor_id].get("status") != "active":
                                continue

                        propagated_energy = (
                            curr_energy * self.wikilink_weight * (hop_decay ** hop)
                        )
                        prev_energy = activation_energies.get(neighbor_id, 0.0)
                        new_energy = min(1.0, prev_energy + propagated_energy)
                        activation_energies[neighbor_id] = new_energy

                        if neighbor_id not in current_frontier:
                            next_frontier.add(neighbor_id)

            current_frontier = next_frontier
            if not current_frontier:
                break

        # Step 5: Recency, Strength, Importance, Salience, and Confidence Fusion Scoring
        now = int(time.time())
        candidates: List[Dict[str, Any]] = []

        for node_id, node in known_nodes.items():
            if node.get("status") != "active":
                continue

            # Recency calculation with robust timestamp parsing
            raw_acc = node.get("last_accessed") or node.get("created_at") or now
            if isinstance(raw_acc, str):
                try:
                    last_accessed = int(datetime.datetime.fromisoformat(raw_acc).timestamp())
                except Exception:
                    try:
                        last_accessed = int(raw_acc)
                    except Exception:
                        last_accessed = now
            else:
                last_accessed = int(raw_acc)

            delta_t = max(0, now - last_accessed)
            recency_decay = 0.5 ** (delta_t / self.half_life_seconds)

            # Strength weighting
            node_strength = _as_float(node.get("strength"), 1.0)
            recency_score = min(1.0, max(0.0, recency_decay * (node_strength / 1.0)))

            # Importance multiplier (0.5 default = 1.0x factor)
            importance = _as_float(node.get("importance"), 0.5)
            importance_scale = importance / 0.5

            norm_bm25 = normalized_bm25.get(node_id, 0.0)
            vec_sim = vector_similarities.get(node_id, 0.0)
            act_energy = activation_energies.get(node_id, 0.0)

            # Base Dense-Sparse Fusion formula
            base_score = (
                (norm_bm25 * self.bm25_weight)
                + (vec_sim * self.vector_weight)
                + (act_energy * self.activation_weight)
                + (recency_score * self.recency_weight) * importance_scale
            )

            # Executive Working Memory Gating (PFC Filter) - Dynamic Semantic & Category Modulation
            if task_context:
                task_vec = self.vectorize(task_context)
                node_raw_vec = self.ensure_node_vector(node_id, node)
                node_sim = max(0.0, float(cosine_similarity(task_vec, node_raw_vec)))

                # Dynamic semantic resonance scaling:
                # Topics aligned with task context are proportionally boosted,
                # orthogonal or unrelated topics are gently dampened without keyword hardcoding
                if node_sim >= 0.40:
                    base_score *= (1.0 + (node_sim * 0.75))
                elif node_sim < 0.15:
                    base_score *= 0.70

            salience = _as_float(node.get("salience"), 0.5)
            confidence = _as_float(node.get("confidence"), 1.0)

            salience_factor = salience / 0.5
            confidence_factor = max(0.1, confidence)

            composite_score = base_score * salience_factor * confidence_factor
            if salience >= 0.9:
                composite_score += 0.5  # Priority boost for High-Salience Invariant Directives

            item = dict(node)
            item["composite_score"] = round(composite_score, 4)
            item["bm25_score"] = round(norm_bm25, 4)
            item["vector_similarity"] = round(vec_sim, 4)
            item["vector_score"] = round(vec_sim, 4)
            item["activation_energy"] = round(act_energy, 4)
            item["recency_score"] = round(recency_score, 4)
            item["rrf_score"] = round(rrf_scores.get(node_id, 0.0), 6)
            item["salience"] = salience
            item["confidence"] = confidence
            item["verification_needed"] = confidence < 0.6
            item["verification_flag"] = confidence < 0.6
            candidates.append(item)

        # Step 6: Episodic Memory Retrieval (if enabled)
        if include_episodes:
            raw_episodes = self.storage.search_episodes(query, limit=seed_limit)
            if raw_episodes:
                norm_ep_bm25 = self._normalize_bm25_scores(raw_episodes)
                for ep in raw_episodes:
                    ep_id = ep["id"]
                    ep_text = f"{ep.get('title', '')} {ep.get('situation', '')} {ep.get('action_taken', '')} {ep.get('outcome', '')} {ep.get('lessons_learned', '')}"
                    ep_vec = self.vectorize(ep_text)
                    ep_vec_sim = max(0.0, float(cosine_similarity(query_vector, ep_vec)))
                    ep_norm_bm25 = norm_ep_bm25.get(ep_id, 0.5)

                    ep_last_accessed = ep.get("last_accessed") or ep.get("created_at") or now
                    ep_delta_t = max(0, now - ep_last_accessed)
                    ep_recency_score = min(1.0, max(0.0, 0.5 ** (ep_delta_t / self.half_life_seconds)))

                    ep_salience = _as_float(ep.get("salience"), 0.5)
                    ep_confidence = _as_float(ep.get("confidence"), 1.0)

                    ep_base_score = (
                        (ep_norm_bm25 * self.bm25_weight)
                        + (ep_vec_sim * self.vector_weight)
                        + (ep_recency_score * self.recency_weight)
                    )

                    ep_sal_factor = ep_salience / 0.5
                    ep_conf_factor = max(0.1, ep_confidence)
                    ep_composite = ep_base_score * ep_sal_factor * ep_conf_factor
                    if ep_salience >= 0.9:
                        ep_composite += 0.5

                    title = ep.get("title", "Episode")
                    situation = ep.get("situation", "")
                    action_taken = ep.get("action_taken", "")
                    outcome = ep.get("outcome", "")
                    lesson = ep.get("lessons_learned", "")

                    l0 = f"Episode: {title} - {lesson or outcome or situation}".strip(" -")
                    l1_parts = []
                    if situation:
                        l1_parts.append(f"Situation: {situation}")
                    if action_taken:
                        l1_parts.append(f"Action: {action_taken}")
                    if outcome:
                        l1_parts.append(f"Outcome: {outcome}")
                    if lesson:
                        l1_parts.append(f"Lesson: {lesson}")
                    l1 = " | ".join(l1_parts)

                    ep_item = dict(ep)
                    ep_item.update({
                        "type": "episodic",
                        "category": "episode",
                        "uri": f"synapse://episode/{ep_id}",
                        "l0_abstract": l0,
                        "l1_overview": l1,
                        "composite_score": round(ep_composite, 4),
                        "bm25_score": round(ep_norm_bm25, 4),
                        "vector_similarity": round(ep_vec_sim, 4),
                        "vector_score": round(ep_vec_sim, 4),
                        "activation_energy": 0.0,
                        "recency_score": round(ep_recency_score, 4),
                        "salience": ep_salience,
                        "confidence": ep_confidence,
                        "verification_needed": ep_confidence < 0.6,
                        "verification_flag": ep_confidence < 0.6,
                    })
                    candidates.append(ep_item)

        if not candidates:
            return []

        # Step 6.5: Post-Retrieval Belief Revision & Contradiction Resolution
        # Pre-sort candidates by composite score, then check contradictions only on top-K candidate pool
        # to prevent O(N^2) explosion (e.g. 479 nodes -> 114k pairs -> 12s latency).
        # Retrieval never mutates belief state: the lower-ranked contender is
        # hidden from this response only. Persistent supersede happens in the
        # retain-time resolver and the nightly consolidation audit.
        if len(candidates) >= 2:
            try:
                candidates.sort(
                    key=lambda x: (
                        _as_float(x.get("composite_score"), 0.0),
                        _as_float(x.get("salience"), 0.5),
                        _as_float(x.get("recency_score"), 0.0),
                        _as_float(x.get("strength"), 1.0),
                    ),
                    reverse=True,
                )
                top_pool_limit = min(len(candidates), max(limit * 4, 25))
                conflicts = self.detect_contradictions(candidates[:top_pool_limit])
                if conflicts:
                    loser_ids = {c["loser_id"] for c in conflicts if c.get("loser_id")}
                    if loser_ids:
                        candidates = [cand for cand in candidates if cand.get("id") not in loser_ids]
            except Exception:
                pass

        # Sort descending by composite score, then by salience, recency and strength
        candidates.sort(
            key=lambda x: (
                _as_float(x.get("composite_score"), 0.0),
                _as_float(x.get("salience"), 0.5),
                _as_float(x.get("recency_score"), 0.0),
                _as_float(x.get("strength"), 1.0),
            ),
            reverse=True,
        )

        top_results = candidates[: max(1, limit)]

        # Step 7: Touch and Hebbian reinforce retrieved items
        if touch:
            semantic_ids: List[str] = []
            for item in top_results:
                if item.get("type") == "episodic" or "situation" in item:
                    self.storage.update_episode(
                        item["id"],
                        last_accessed=now,
                    )
                    item["last_accessed"] = now
                else:
                    curr_str = _as_float(item.get("strength"), 1.0)
                    curr_sal = _as_float(item.get("salience"), 0.5)
                    reinforced_str = round(min(5.0, curr_str + 0.1), 3)
                    reinforced_sal = round(min(1.0, curr_sal + 0.02), 3)

                    if hasattr(self.storage, "record_node_access"):
                        self.storage.record_node_access(
                            item["id"],
                            delta_salience=0.02,
                            delta_strength=0.1,
                            bump_edges=True,
                        )
                    else:
                        self.storage.update_node(
                            item["id"],
                            strength=reinforced_str,
                            last_accessed=now,
                        )
                    item["strength"] = reinforced_str
                    item["salience"] = reinforced_sal
                    item["last_accessed"] = now
                    item["access_count"] = (item.get("access_count") or 0) + 1
                    semantic_ids.append(item["id"])

            # Hebbian co-activation: reinforce synaptic links between co-retrieved items
            if len(semantic_ids) >= 2:
                from itertools import combinations
                for s1, s2 in combinations(semantic_ids, 2):
                    try:
                        self.storage.add_edge(s1, s2, relation="co_activated", weight=0.6, increment_access=True)
                        self.storage.add_edge(s2, s1, relation="co_activated", weight=0.6, increment_access=True)
                    except Exception:
                        pass

        return top_results
