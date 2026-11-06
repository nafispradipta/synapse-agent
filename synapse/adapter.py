"""
SynapseAdapter - Pluggable BaseMemoryAdapter Implementation for Synapse Cognitive Memory Core.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import re
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from synapse.loader import SynapseContextLoader, estimate_tokens, split_sentences
from synapse.retriever import SynapseRetriever
from synapse.storage import VALID_CATEGORIES, SynapseStorage
from synapse.worker import SynapseWorker, resolve_synapse_llm_config
from synapse.mental_model_engine import MentalModelEngine
from synapse.observation_engine import ObservationEngine
from synapse.harvester import SynapseWebHarvester
from synapse.embedder import SynapseEmbedder
from synapse.contradiction_resolver import ContradictionResolver
from synapse.metacognitive_scorer import (
    MetacognitiveScorer,
    MetacognitiveConfig,
    UncertaintyReport,
)
from synapse.counterfactual_guardrail import (
    CounterfactualGuardrail,
    GuardrailResult,
)

logger = logging.getLogger(__name__)


class BaseMemoryAdapter(ABC):
    """
    Abstract Base Class for memory backends.
    """

    name: str = "base"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    @abstractmethod
    def recall(self, query: str, limit: int = 3, **kwargs: Any) -> List[str]:
        """Retrieve relevant memories for a query string."""
        raise NotImplementedError

    @abstractmethod
    def retain(self, content: str, context: str = "", metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> bool:
        """Persist a memory item into the backend."""
        raise NotImplementedError

    def reflect(self, query: str, **kwargs: Any) -> str:
        """Synthesize memories on a topic (optional capability)."""
        return "Reflection not supported by this memory backend."

    async def recall_async(self, query: str, limit: int = 3, **kwargs: Any) -> List[str]:
        """Asynchronous recall helper."""
        return await asyncio.to_thread(self.recall, query, limit, **kwargs)

    async def retain_async(self, content: str, context: str = "", metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> bool:
        """Asynchronous retain helper."""
        return await asyncio.to_thread(self.retain, content, context, metadata, **kwargs)


class SynapseAdapter(BaseMemoryAdapter):
    """
    Pluggable Cognitive Synapse Memory Adapter for Autonomous Agents.
    Combines persistent SQLite FTS5 graph storage, hybrid spreading activation,
    tiered context packing, background decay, and obsidian vault cortex sync.
    Supports seamless fallback to another BaseMemoryAdapter (e.g. LocalFile or NullAdapter).
    """

    name: str = "synapse"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        raw_db = (
            self.config.get("db_path")
            or os.environ.get("SYNAPSE_DB_PATH")
            or (Path.home() / ".synapse" / "synapse.db")
        )
        self.db_path = ":memory:" if str(raw_db) == ":memory:" else str(Path(raw_db).expanduser().resolve())
        self.token_budget = int(self.config.get("token_budget", 250))
        self.fallback_adapter: Optional[BaseMemoryAdapter] = self.config.get("fallback_adapter")
        self.llm_config = resolve_synapse_llm_config(self.config)
        self.embedder: Optional[SynapseEmbedder] = None
        raw_vault = (
            self.config.get("vault_path")
            or os.environ.get("SYNAPSE_VAULT_PATH")
            or os.environ.get("OBSIDIAN_VAULT_DIR")
        )
        self.vault_path = Path(raw_vault).expanduser().resolve() if raw_vault else None

        # Configurable cognitive taxonomy
        categories_config = self.config.get("categories")
        self.categories: List[str] = categories_config or [
            "architecture", "sop", "decision", "preference", "incident", "general"
        ]
        self.category_patterns: Dict[str, str] = self.config.get("category_patterns") or {
            "preference": r"\b(prefer|preference|preferensi|pribadi|theme|font|editor|config|profile)\b",
            "incident": r"\b(incident|insiden|bug|error|failure|gagal|kegagalan|masalah|kerusakan|gangguan|crash|exception|outage|fix|issue|regression)\b",
            "sop": r"\b(sop|procedure|prosedur|guide|panduan|runbook|playbook|workflow|step|langkah|instruction|instruksi|rule|aturan|standard|standar)\b",
            "decision": r"\b(decid|decision|keputusan|memutuskan|pilih|chose|agreed|selected|adopt|settle)\b",
            "architecture": r"\b(architect|architecture|arsitektur|structure|struktur|pattern|schema|skema|database|model|fastapi|sqlite|api|design|desain|component|komponen)\b",
        }
        self.tag_rules: Dict[str, str] = self.config.get("tag_rules") or {
            "server": r"\b(server|vps|ssh|caddy|systemd|pm2|nginx|docker)\b",
            "ai": r"\b(ai|llm|synapse|agent|gpt|model|neural)\b",
            "sop": r"\b(sop|runbook|playbook|procedure|prosedur|guide|panduan|step|langkah)\b",
            "finance": r"\b(finance|stock|money|budget|invest|trading|treasury|banking)\b",
            "marketing": r"\b(marketing|seo|ads|iklan|advertising|copywriting)\b",
            "meeting": r"\b(meeting|discussion|diskusi|standup|call|rapat)\b",
            "todo": r"\b(todo|task|action_item|tugas)\b",
            "idea": r"\b(idea|concept|proposal|ide|gagasan|konsep)\b",
            "personal": r"\b(preference|profile|personal|custom|preferensi|pribadi)\b",
        }

        try:
            self.storage = SynapseStorage(
                self.db_path,
                allowed_categories=set(self.categories) if categories_config else None,
                allow_custom_categories=bool(categories_config),
            )
            self.disposition = self.config.get("disposition") or {"skepticism": 5, "literalism": 4, "empathy": 2}
            self.retriever = SynapseRetriever(storage=self.storage, disposition=self.disposition)
            self.loader = SynapseContextLoader(storage=self.storage)
            self.worker = SynapseWorker(storage=self.storage, config=self.config)
            self.mental_model_engine = MentalModelEngine(storage=self.storage, llm_config=self.llm_config)
            self.observation_engine = ObservationEngine(storage=self.storage)
            research_dir = (
                self.config.get("research_dir")
                or (str(self.vault_path / "research" / "auto_harvest") if self.vault_path else os.environ.get("SYNAPSE_RESEARCH_DIR") or str(Path.home() / ".synapse" / "research" / "auto_harvest"))
            )
            self.harvester = SynapseWebHarvester(
                storage=self.storage,
                llm_config=self.llm_config,
                vault_research_dir=research_dir,
            )
            # Initialize or reuse embedder for in-flight vectorization
            if "embedder" in self.config:
                self.embedder = self.config["embedder"]
            elif self.retriever is not None and getattr(self.retriever, "embedder", None) is not None:
                self.embedder = self.retriever.embedder
            else:
                try:
                    self.embedder = SynapseEmbedder()
                except Exception as emb_err:
                    logger.warning("Failed to initialize SynapseEmbedder for SynapseAdapter: %s", emb_err)
                    self.embedder = None

            self.contradiction_resolver = ContradictionResolver(
                storage=self.storage,
                embedder=self.embedder,
            )
            self.metacognitive_scorer = MetacognitiveScorer(
                storage=self.storage,
                config=self.config.get("metacognitive"),
            )
            self.counterfactual_guardrail = CounterfactualGuardrail(
                storage=self.storage,
                embedder=self.embedder,
                config=self.config.get("guardrail"),
            )
            self._last_retrieved_node_ids: List[str] = []

            if not self.storage.is_memory and not str(self.db_path).endswith(".db"):
                try:
                    self.mental_model_engine.ensure_default_models()
                except Exception:
                    pass
        except Exception:
            self.storage = None  # type: ignore[assignment]
            self.retriever = None  # type: ignore[assignment]
            self.loader = None  # type: ignore[assignment]
            self.worker = None  # type: ignore[assignment]
            self.mental_model_engine = None  # type: ignore[assignment]
            self.observation_engine = None  # type: ignore[assignment]
            self.harvester = None  # type: ignore[assignment]
            self.contradiction_resolver = None  # type: ignore[assignment]
            self.metacognitive_scorer = None  # type: ignore[assignment]
            self.counterfactual_guardrail = None  # type: ignore[assignment]
            self.embedder = None

    def _infer_category(self, content: str, context: str = "") -> str:
        """Infer cognitive category from content and context heuristics using configurable taxonomy."""
        combined = f"{content} {context}".lower()

        patterns = getattr(self, "category_patterns", None) or {}
        for cat, pattern in patterns.items():
            if re.search(pattern, combined):
                return cat

        categories = getattr(self, "categories", None)
        return categories[-1] if categories else "general"

    def _slugify(self, text: str, max_words: int = 5) -> str:
        """Generate clean slug identifier from text snippet."""
        clean = re.sub(r"[^a-zA-Z0-9\s_-]", "", text).strip()
        words = clean.split()[:max_words]
        slug = "-".join(w.lower() for w in words)
        return slug or uuid.uuid4().hex[:8]

    def _get_embedder(self) -> Optional[SynapseEmbedder]:
        """Lazy access or initialization for SynapseEmbedder with safe fallback."""
        if self.embedder is not None:
            return self.embedder
        if self.retriever is not None and getattr(self.retriever, "embedder", None) is not None:
            self.embedder = self.retriever.embedder
            return self.embedder
        try:
            self.embedder = SynapseEmbedder()
            return self.embedder
        except Exception as e:
            logger.warning("SynapseEmbedder not available for in-flight vectorization: %s", e)
            return None

    def retain_episode(
        self,
        title: str,
        situation: str,
        action_taken: str,
        outcome: str,
        lessons_learned: str,
        salience: float = 0.5,
        confidence: float = 1.0,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Persist a structured cognitive episode (Situation -> Action -> Outcome -> Lesson)
        into Synapse storage.
        """
        if self.storage is None:
            return ""

        stored_id = self.storage.add_episode(
            title=title,
            situation=situation,
            action_taken=action_taken,
            outcome=outcome,
            lessons_learned=lessons_learned,
            salience=salience,
            confidence=confidence,
            tags=tags,
        )

        if stored_id:
            try:
                embedder = self._get_embedder()
                if embedder is not None:
                    ep_text = f"{title} {situation} {action_taken} {outcome} {lessons_learned}".strip()
                    vector = embedder.embed(ep_text)
                    if vector:
                        self.storage.upsert_vector(stored_id, vector)
            except Exception as e:
                logger.warning("In-flight vectorization failed for episode %s: %s", stored_id, e)

        return stored_id

    # Alias for episodic recording
    record_episode = retain_episode

    def recall_episodes(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search and retrieve relevant cognitive episodic experiences.
        """
        if self.storage is None:
            return []
        return self.storage.search_episodes(query, limit=limit)

    def get_episode(self, episode_id: str, touch: bool = True) -> Optional[Dict[str, Any]]:
        """Retrieve a specific cognitive episode by ID."""
        if self.storage is None:
            return None
        return self.storage.get_episode(episode_id, touch=touch)

    def delete_episode(self, episode_id: str) -> bool:
        """Delete a cognitive episode by ID."""
        if self.storage is None:
            return False
        return self.storage.delete_episode(episode_id)

    def recall(self, query: str, limit: int = 3, **kwargs: Any) -> List[str]:
        """
        Recall relevant memories (semantic nodes and cognitive episodes) adhering to token budget.
        If no results or storage error occurs, falls back to fallback_adapter.
        """
        budget = int(kwargs.get("token_budget") or kwargs.get("budget") or self.token_budget)
        if "limit" in kwargs and kwargs["limit"] is not None:
            limit = int(kwargs["limit"])
        elif limit == 3 and budget > 250:
            limit = max(3, min(12, budget // 50))

        nodes: List[Dict[str, Any]] = []

        include_mental_models = kwargs.get("include_mental_models", True)
        # Tier 1 Priority: Check relevant Mental Models first (Cognitive Memory hierarchy)
        mental_model_snippets: List[str] = []
        if self.storage is not None and include_mental_models:
            try:
                models = self.storage.list_mental_models()
                q_words = set(re.findall(r"[a-z0-9]+", query.lower()))
                for mm in models:
                    name_words = set(re.findall(r"[a-z0-9]+", mm["name"].lower()))
                    tag_words = set(t.lower() for t in mm.get("tags", []))
                    content_words = set(re.findall(r"[a-z0-9]+", mm["content"].lower()))
                    # Check if query words intersect with name, tags, or content
                    header_intersect = (q_words & name_words) | (q_words & tag_words)
                    content_intersect = (q_words & content_words) - {"and", "the", "with", "for", "from", "into", "that", "this", "these", "those"}

                    if header_intersect or len(content_intersect) >= 2:
                        # Extract relevant bullet points or top sections from mental model
                        mm_lines = [l.strip() for l in mm["content"].split("\n") if l.strip().startswith("- ") or l.strip().startswith("* ")]
                        sample_bullets = mm_lines[:3]
                        if sample_bullets:
                            snippet = f"• [MENTAL_MODEL] {mm['name']}:\n  " + "\n  ".join(sample_bullets)
                            mental_model_snippets.append(snippet)
            except Exception as e:
                logger.debug("Mental model recall exception: %s", e, exc_info=True)

        # Tier 2 Priority: Check Evidence-Grounded Observations (Proof Counts)
        observation_snippets: List[str] = []
        if self.observation_engine is not None:
            try:
                obs_limit = max(2, min(6, budget // 100))
                matched_obs = self.observation_engine.search_observations(query, limit=obs_limit)
                for obs in matched_obs:
                    proof_cnt = obs.get("proof_count", 1)
                    conf = obs.get("confidence", 1.0)
                    quote_sample = obs.get("supporting_facts", [])
                    quote_str = f" (Quote: \"{quote_sample[0][:100]}...\")" if quote_sample else ""
                    obs_snippet = f"• [OBSERVATION [Proof: {proof_cnt}, Conf: {conf:.2f}]] {obs['pattern']}{quote_str}"
                    observation_snippets.append(obs_snippet)
            except Exception as e:
                logger.debug("Observation recall exception: %s", e, exc_info=True)

        if self.retriever is not None:
            try:
                    nodes = self.retriever.retrieve(
                        query,
                        limit=max(limit, 6),
                        max_hops=kwargs.get("max_hops", 3),
                        touch=kwargs.get("touch", True),
                        include_episodes=kwargs.get("include_episodes", True),
                    )
            except Exception as e:
                logger.warning("Retriever query failed for %r: %s", query, e, exc_info=True)
                nodes = []

        if not nodes and not mental_model_snippets and not observation_snippets:
            if self.fallback_adapter is not None:
                return self.fallback_adapter.recall(query, limit=limit, **kwargs)
            return []

        # Hebbian reinforce co-activated nodes
        if nodes:
            try:
                node_ids = [n["id"] for n in nodes if "id" in n and n.get("type") != "episodic"]
                self._last_retrieved_node_ids = list(node_ids)
                if self.worker is not None and len(node_ids) >= 2:
                    self.worker.hebbian_reinforce(node_ids, delta=0.05)
            except Exception as e:
                logger.debug("Hebbian reinforcement exception: %s", e, exc_info=True)

        results: List[str] = []
        current_tokens = 0

        # Proportional budget allocation across tiers:
        # Tier 1 (Mental Models): ~30% max
        # Tier 2 (Observations): ~30% max + rollover from Tier 1
        # Tier 3 (Nodes): ~40% max + rollover from Tier 1 & 2
        t1_max = max(40, int(budget * 0.30))
        t2_max = max(40, int(budget * 0.30))

        # Tier 1 Priority: Inject Mental Models within proportional quota
        t1_used = 0
        remaining_mms: List[str] = []
        for mms in mental_model_snippets:
            if len(results) >= limit:
                break
            overhead = 1 if results else 0
            mms_tokens = estimate_tokens(mms)
            if t1_used + mms_tokens <= t1_max and current_tokens + mms_tokens + overhead <= budget:
                results.append(mms)
                t1_used += mms_tokens
                current_tokens += mms_tokens + overhead
            else:
                remaining_mms.append(mms)

        # Rollover unused Tier 1 quota to Tier 2
        t1_unused = max(0, t1_max - t1_used)
        t2_avail = t2_max + t1_unused

        # Tier 2 Priority: Inject Evidence-Grounded Observations within proportional quota
        t2_used = 0
        remaining_obs: List[str] = []
        for obs in observation_snippets:
            if len(results) >= limit:
                break
            overhead = 1 if results else 0
            obs_tokens = estimate_tokens(obs)
            if t2_used + obs_tokens <= t2_avail and current_tokens + obs_tokens + overhead <= budget:
                results.append(obs)
                t2_used += obs_tokens
                current_tokens += obs_tokens + overhead
            else:
                remaining_obs.append(obs)

        # Tier 3 Priority: Semantic Nodes & Episodes (receives all remaining budget)
        if self.metacognitive_scorer is not None and nodes:
            try:
                nodes = self.metacognitive_scorer.annotate_context(nodes)
            except Exception as e:
                logger.debug("Metacognitive annotation exception: %s", e, exc_info=True)

        for node in nodes:
            if len(results) >= limit:
                break
            if self.loader:
                l0 = self.loader.format_l0_abstract(node)
                l1 = self.loader.format_l1_overview(node)
            else:
                ref = node.get("l2_target_path") or node.get("uri") or ""
                ref_suffix = f" [ref: {ref}]" if ref else ""
                l0 = f"• [{node.get('category', 'general').upper()}] {node.get('uri', '')}: {node.get('l0_abstract', '')}{ref_suffix}"
                l1 = node.get("l1_overview", "").strip()
                if l1:
                    l1 = f"  - Overview: {l1}"

            candidate_text = f"{l0}\n{l1}" if l1 else l0

            overhead = 1 if results else 0
            tokens = estimate_tokens(candidate_text)
            if current_tokens + tokens + overhead <= budget:
                results.append(candidate_text)
                current_tokens += tokens + overhead
            else:
                # If combined doesn't fit, try fitting just L0
                l0_tokens = estimate_tokens(l0)
                if current_tokens + l0_tokens + overhead <= budget:
                    results.append(l0)
                    current_tokens += l0_tokens + overhead
                else:
                    break

        # Overflow catch: Backfill any remaining Tier 1 or Tier 2 items if budget and limit allow
        if current_tokens < budget - 20 and len(results) < limit:
            for mms in remaining_mms:
                if len(results) >= limit:
                    break
                overhead = 1 if results else 0
                tok = estimate_tokens(mms)
                if current_tokens + tok + overhead <= budget:
                    results.append(mms)
                    current_tokens += tok + overhead

            for obs in remaining_obs:
                if len(results) >= limit:
                    break
                overhead = 1 if results else 0
                tok = estimate_tokens(obs)
                if current_tokens + tok + overhead <= budget:
                    results.append(obs)
                    current_tokens += tok + overhead

        if not results and self.fallback_adapter is not None:
            return self.fallback_adapter.recall(query, limit=limit, **kwargs)

        return results

    def _load_vault_tags(self, vault_base: Path) -> List[str]:
        """Read allowed tags from tags.md in vault root, with strict fallback."""
        default_tags = [
            "idea", "meeting", "todo", "devops", "ai",
            "marketing", "finance", "personal", "server", "sop"
        ]
        tags_file = vault_base / "tags.md"
        if not tags_file.exists():
            return default_tags
        try:
            content = tags_file.read_text(encoding="utf-8")
            parsed: List[str] = []
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("- #"):
                    parsed.append(line[3:].strip().lower())
                elif line.startswith("-"):
                    t = line.lstrip("- ").strip()
                    if t.startswith("#"):
                        t = t[1:]
                    if t:
                        parsed.append(t.lower())
            return parsed if parsed else default_tags
        except Exception:
            return default_tags

    @staticmethod
    def _contained_dir(vault_base: Path, candidate: Path) -> Path:
        """Resolve a vault subdirectory, falling back when it escapes the vault."""
        base = Path(vault_base).resolve()
        resolved = Path(candidate).resolve()
        try:
            resolved.relative_to(base)
            return resolved
        except ValueError:
            logger.warning("Vault target %s escapes the vault; using the default folder", resolved)
            return base / "projects" / "synapse-knowledge"

    def _resolve_vault_target(
        self,
        vault_base: Path,
        category: str,
        content: str,
        context: str,
        uri: str,
        metadata: Dict[str, Any],
    ) -> Tuple[Path, str, str, List[str]]:
        """Determine target directory, index file stem, domain title, and baseline tags."""
        combined = f"{content} {context} {uri}".lower()
        proj_folder = "projects" if (vault_base / "projects").exists() else ("projek" if (vault_base / "projek").exists() else "projects")
        proj_index = "00-INDEX-PROJEK" if (vault_base / "projek" / "00-INDEX-PROJEK.md").exists() else "00-INDEX-PROJECTS"

        # 0. Custom vault routing rules from config
        custom_routes = (getattr(self, "config", None) or {}).get("vault_routes") or {}
        if category in custom_routes:
            rule = custom_routes[category]
            folder_name = rule.get("folder", "notes")
            idx_name = rule.get("index", f"00-INDEX-{folder_name.upper()}")
            title = rule.get("title", folder_name.title())
            base_tags = rule.get("tags", [category])
            return self._contained_dir(vault_base, vault_base / folder_name), idx_name, title, base_tags

        # 1. Explicit vault directory in metadata
        custom_dir = metadata.get("vault_dir") or metadata.get("folder")
        if custom_dir:
            target_dir = self._contained_dir(vault_base, vault_base / custom_dir)
            if "server" in str(target_dir):
                return target_dir, "00-INDEX-SERVER", "Server & DevOps", ["server", "devops"]
            if "cybersecurity" in str(target_dir):
                return target_dir, "00-INDEX-SOP-CYBERSECURITY", "Cybersecurity", ["sop"]
            return target_dir, proj_index, "Projects & Engineering", ["ai"]

        # 2. Server / DevOps / Infrastructure routing
        server_pattern = r"\b(server|devops|router|caddy|vps|ssh|systemd|pm2|nginx|docker|port|firewall|wireguard|kernel|swap|network)\b"
        if re.search(server_pattern, combined):
            return vault_base / "server", "00-INDEX-SERVER", "Server & DevOps", ["server", "devops"]

        # 3. Architecture routing
        if category == "architecture":
            target_dir = vault_base / proj_folder
            return target_dir, proj_index, "Projects & Engineering", ["ai", "devops"]

        # 4. SOP routing
        if category == "sop":
            if re.search(r"\b(security|cve|vuln|exploit|penetration|pentest|audit|threat)\b", combined):
                return vault_base / "cybersecurity", "00-INDEX-SOP-CYBERSECURITY", "Cybersecurity SOP", ["sop"]
            target_dir = vault_base / proj_folder / "synapse-knowledge"
            return target_dir, proj_index, "Synapse SOP Knowledge", ["sop", "ai"]

        # 5. Decision / Preference routing
        if category in ("decision", "preference"):
            target_dir = vault_base / proj_folder / "synapse-knowledge"
            initial_tags = ["personal", "ai"] if category == "preference" else ["idea", "ai"]
            return target_dir, proj_index, "Synapse Cognitive Knowledge", initial_tags

        return vault_base / proj_folder / "synapse-knowledge", proj_index, "Synapse Knowledge", ["ai"]

    def _select_vault_tags(
        self,
        vault_base: Path,
        category: str,
        combined_text: str,
        initial_tags: List[str],
        metadata: Dict[str, Any],
    ) -> List[str]:
        """Filter and select tags strictly conforming to tags.md."""
        allowed_tags = self._load_vault_tags(vault_base)
        allowed_set = set(allowed_tags)
        candidate_tags = set(initial_tags)

        if category == "sop":
            candidate_tags.add("sop")
        elif category == "preference":
            candidate_tags.add("personal")
        elif category == "decision":
            candidate_tags.add("idea")
        elif category == "architecture":
            candidate_tags.add("ai")

        lower_text = combined_text.lower()
        tag_rules = getattr(self, "tag_rules", None) or {}
        for tag, pattern in tag_rules.items():
            if re.search(pattern, lower_text):
                candidate_tags.add(tag)

        meta_tags = metadata.get("tags") or []
        if isinstance(meta_tags, str):
            meta_tags = [t.strip() for t in meta_tags.split(",")]
        for t in meta_tags:
            clean_t = re.sub(r"^#", "", str(t)).strip().lower()
            if clean_t in allowed_set:
                candidate_tags.add(clean_t)

        final_tags = [t for t in allowed_tags if t in candidate_tags]
        return final_tags if final_tags else ["ai"]

    def _update_domain_index(
        self,
        vault_base: Path,
        target_dir: Path,
        domain_index: str,
        slug: str,
        title: str,
    ) -> None:
        """Safely record bidirectional wikilink in the corresponding domain index file."""
        try:
            if domain_index == "00-INDEX-SERVER":
                index_dir = vault_base / "server"
            elif domain_index == "00-INDEX-PROJEK":
                index_dir = vault_base / "projek"
            elif domain_index == "00-INDEX-SOP-CYBERSECURITY":
                index_dir = vault_base / "cybersecurity"
            else:
                index_dir = target_dir

            index_file = self._contained_dir(vault_base, index_dir) / f"{domain_index}.md"
            wikilink = f"[[{slug}]]"

            if index_file.exists():
                text = index_file.read_text(encoding="utf-8")
                if wikilink not in text:
                    entry = f"- {wikilink} — {title}\n" if domain_index == "00-INDEX-PROJEK" else f"- {wikilink}\n"
                    if "## Archive" in text or "## Arsip" in text:
                        pos = text.find("## Archive") if "## Archive" in text else text.find("## Arsip")
                        new_text = text[:pos].rstrip() + "\n" + entry + "\n" + text[pos:]
                    elif "\n- [[" in text:
                        last_pos = text.rfind("\n- [[")
                        eol = text.find("\n", last_pos + 1)
                        if eol != -1:
                            new_text = text[:eol + 1] + entry + text[eol + 1:]
                        else:
                            new_text = text + "\n" + entry
                    else:
                        new_text = text.rstrip() + "\n\n" + entry
                    index_file.write_text(new_text, encoding="utf-8")
                    logger.info("Updated index %s with link %s", index_file, wikilink)
            else:
                index_dir.mkdir(parents=True, exist_ok=True)
                initial_index = (
                    f"# {domain_index}\n\n"
                    "List of connected documents:\n\n"
                    f"- {wikilink}\n"
                )
                index_file.write_text(initial_index, encoding="utf-8")

            # Local folder index for synapse-knowledge
            if target_dir.name == "synapse-knowledge":
                sk_index = target_dir / "00-INDEX-SYNAPSE-KNOWLEDGE.md"
                if not sk_index.exists():
                    sk_index.write_text(
                        "# Synapse Knowledge Base\n\n"
                        "Cognitive document index for Synapse Memory auto-writeback.\n\n"
                        f"- **Parent Index:** [[{domain_index}]]\n\n"
                        "## Documents\n\n",
                        encoding="utf-8",
                    )
                sk_text = sk_index.read_text(encoding="utf-8")
                if wikilink not in sk_text:
                    with open(sk_index, "a", encoding="utf-8") as f:
                        f.write(f"- {wikilink} — {title}\n")

        except Exception as e:
            logger.warning("Failed to update domain index for %s: %s", slug, e)

    @staticmethod
    def _safe_vault_slug(raw: Any) -> str:
        """Normalize a user-supplied slug to a single safe file stem."""
        text = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(raw)).strip("-") or "note"
        return text.strip(".") or "note"

    @staticmethod
    def _yaml_flow_scalar(value: Any) -> str:
        """Quote a YAML flow scalar only when it is not a bare word."""
        text = str(value)
        if re.fullmatch(r"[A-Za-z0-9_.-]+", text):
            return text
        return json.dumps(text, ensure_ascii=False)

    @staticmethod
    def _yaml_quote(value: Any) -> str:
        """Render a YAML-safe double-quoted scalar for arbitrary content."""
        return json.dumps(str(value), ensure_ascii=False)

    def _auto_writeback_vault(
        self,
        stored_id: int,
        content: str,
        context: str,
        category: str,
        salience: float,
        uri: str,
        l0: str,
        l1: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Perform non-blocking writeback to Obsidian Vault Master for high-salience knowledge nodes."""
        try:
            meta = metadata or {}
            # If running in-memory without an explicit vault path, skip filesystem writeback
            if self.config.get("db_path") == ":memory:" and not (
                meta.get("vault_path")
                or getattr(self, "vault_path", None)
                or self.config.get("vault_path")
                or os.environ.get("SYNAPSE_VAULT_PATH")
            ):
                return None

            vault_base_str = (
                meta.get("vault_path")
                or getattr(self, "vault_path", None)
                or self.config.get("vault_path")
                or os.environ.get("SYNAPSE_VAULT_PATH")
            )
            if not vault_base_str:
                return None
            vault_base = Path(vault_base_str).expanduser().resolve()

            target_dir, domain_index, domain_name, initial_tags = self._resolve_vault_target(
                vault_base=vault_base,
                category=category,
                content=content,
                context=context,
                uri=uri,
                metadata=meta,
            )
            target_dir.mkdir(parents=True, exist_ok=True)

            combined_text = f"{content} {context} {uri}".lower()
            final_tags = self._select_vault_tags(
                vault_base=vault_base,
                category=category,
                combined_text=combined_text,
                initial_tags=initial_tags,
                metadata=meta,
            )

            raw_title = meta.get("title")
            if not raw_title:
                first_line = l0.strip().split("\n")[0]
                first_sent = first_line.split(". ")[0].strip()
                raw_title = re.sub(r"^[#\s*`_-]+", "", first_sent).strip()
                if not raw_title or len(raw_title) < 5:
                    raw_title = f"{category.title()} - {self._slugify(content, max_words=5)}"

            words = raw_title.split()
            title = " ".join(words[:12]) if len(words) > 12 else raw_title
            title = title.replace("\n", " ").strip()

            slug = meta.get("slug")
            if slug:
                slug = self._safe_vault_slug(slug)
            else:
                clean_slug = re.sub(r"[^a-zA-Z0-9\s_-]", "", title.lower()).strip()
                slug_words = clean_slug.split()[:8]
                slug = "-".join(slug_words).strip("-") or f"{category}-{uuid.uuid4().hex[:6]}"
            if not slug:
                slug = f"{category}-{uuid.uuid4().hex[:6]}"

            date_str = datetime.date.today().isoformat()
            escaped_title = self._yaml_quote(title)
            tags_str = ", ".join(self._yaml_flow_scalar(tag) for tag in final_tags)

            frontmatter = (
                "---\n"
                f"title: {escaped_title}\n"
                f"tags: [{tags_str}]\n"
                f"date: {date_str}\n"
                f"salience: {salience:.2f}\n"
                f"uri: {self._yaml_quote(uri)}\n"
                "status: active\n"
                "---\n\n"
            )

            body_lines = [
                f"# {title}\n",
                "> Auto-generated by Synapse Memory Cortex (Auto-Writeback)\n",
                f"> Date: {date_str} | Salience: {salience:.2f} | Category: {category}\n",
                "## Summary",
                f"{l0}\n",
                "## Details & Context",
                f"{content.strip()}",
            ]

            if context and context.strip() and context.strip() != content.strip():
                body_lines.append(f"\n**Operational Context:** {context.strip()}")

            body_lines.extend([
                "",
                "## Navigation & Index",
                f"- **Back to Index:** [[{domain_index}]]",
                f"- **Node URI:** `{uri}`",
                "",
            ])

            markdown_content = frontmatter + "\n".join(body_lines)
            target_file = (target_dir / f"{slug}.md").resolve()
            try:
                target_file.relative_to(vault_base)
            except ValueError:
                logger.warning("Vault slug %r escapes vault; skipping writeback", slug)
                return None

            with open(target_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            logger.info("Synapse auto-writeback saved note %s to %s", uri, target_file)

            if not meta.get("skip_moc_index", False) and meta.get("update_index", True):
                self._update_domain_index(
                    vault_base=vault_base,
                    target_dir=target_dir,
                    domain_index=domain_index,
                    slug=slug,
                    title=title,
                )

            return target_file

        except Exception as e:
            logger.warning("Vault writeback failed for URI %s: %s", uri, e)
            return None

    def retain(
        self,
        content: str,
        context: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Persist a memory into Synapse graph storage:
        - Extracts L0 abstract (<40 tokens) and L1 overview (<300 tokens)
        - Infers category
        - Generates URI (synapse://category/...)
        - Inserts into storage and creates graph edges if relations exist
        - Automatically writes back high-salience knowledge nodes to Obsidian Vault Master
        """
        meta = dict(metadata or {})
        for k, v in kwargs.items():
            if k not in meta:
                meta[k] = v

        if self.storage is None:
            if self.fallback_adapter is not None:
                return self.fallback_adapter.retain(content, context, metadata=metadata, **kwargs)
            return False

        try:
            # 1. Determine category
            valid_cats = set(self.categories) if getattr(self, "categories", None) else VALID_CATEGORIES
            category = meta.get("category")
            if not category or category not in valid_cats:
                category = self._infer_category(content, context)

            # 2. Extract L0 abstract (<40 tokens)
            l0 = meta.get("l0_abstract")
            if not l0:
                sentences = split_sentences(content)
                if sentences:
                    first_s = sentences[0]
                    if estimate_tokens(first_s) <= 40:
                        l0 = first_s
                    else:
                        words = first_s.split()
                        l0 = " ".join(words[:25]) + "..."
                else:
                    l0 = content.strip()[:120]

            # 3. Extract L1 overview (<300 tokens)
            l1 = meta.get("l1_overview")
            if not l1:
                overview_src = content.strip()
                if context:
                    overview_src = f"{overview_src} (Context: {context.strip()})"
                if estimate_tokens(overview_src) <= 300:
                    l1 = overview_src
                else:
                    sentences = split_sentences(overview_src)
                    acc: List[str] = []
                    for s in sentences:
                        candidate = " ".join(acc + [s])
                        if estimate_tokens(candidate) <= 250:
                            acc.append(s)
                        else:
                            break
                    l1 = " ".join(acc) if acc else " ".join(overview_src.split()[:200]) + "..."

            # 4. Generate URI
            uri = meta.get("uri")
            if not uri:
                slug = self._slugify(l0)
                uri = f"synapse://{category}/{slug}_{uuid.uuid4().hex[:6]}"

            importance = float(meta.get("importance", 0.5))
            salience = float(meta.get("salience", 0.5))
            confidence = float(meta.get("confidence", 1.0))
            created_at = int(meta.get("created_at")) if meta.get("created_at") is not None else None
            l2_path = meta.get("l2_target_path")
            node_id = meta.get("id") or meta.get("node_id")

            # 4.5. Pre-Save Contradiction Check
            try:
                if getattr(self, "contradiction_resolver", None) is not None:
                    candidate_preview = {
                        "l0_abstract": l0,
                        "l1_overview": l1,
                        "category": category,
                        "salience": salience,
                        "created_at": created_at or int(time.time()),
                    }
                    pre_conflicts = self.contradiction_resolver.find_conflicts_for_candidate(candidate_preview)
                    if pre_conflicts:
                        logger.info("Pre-save contradiction check detected %d conflict(s) for '%s'", len(pre_conflicts), l0)
            except Exception as pre_err:
                logger.debug("Pre-save contradiction check error: %s", pre_err)

            # Check if node already exists by URI
            existing = self.storage.get_node(uri)
            if existing is not None:
                stored_id = existing["id"]
                update_kwargs: Dict[str, Any] = {
                    "l0_abstract": l0,
                    "l1_overview": l1,
                    "category": category,
                    "importance": importance,
                    "salience": salience,
                    "confidence": confidence,
                    "l2_target_path": l2_path,
                }
                if created_at is not None:
                    update_kwargs["created_at"] = created_at
                self.storage.update_node(stored_id, **update_kwargs)
            else:
                stored_id = self.storage.add_node(
                    uri=uri,
                    l0_abstract=l0,
                    l1_overview=l1,
                    l2_target_path=l2_path,
                    category=category,
                    importance=importance,
                    salience=salience,
                    confidence=confidence,
                    node_id=node_id,
                    created_at=created_at,
                )

            # 5. Link edges if specified
            links = meta.get("links") or meta.get("related_nodes") or meta.get("edges") or meta.get("source_memory_ids") or []
            if isinstance(links, list):
                for target in links:
                    if isinstance(target, str):
                        target_node = self.storage.get_node(target)
                        if target_node is not None:
                            self.storage.add_edge(stored_id, target_node["id"], relation="relates_to", weight=0.6)
                    elif isinstance(target, dict) and "target_id" in target:
                        self.storage.add_edge(
                            stored_id,
                            target["target_id"],
                            relation=target.get("relation", "relates_to"),
                            weight=float(target.get("weight", 0.6)),
                        )

            # 6. In-flight vectorization (real-time dense embedding)
            if stored_id:
                try:
                    embedder = self._get_embedder()
                    if embedder is not None:
                        embed_text = f"{l0} {l1}".strip() or content.strip()
                        vector = embedder.embed(embed_text)
                        if vector:
                            self.storage.upsert_vector(stored_id, vector)
                except Exception as e:
                    logger.warning("In-flight vectorization failed for node %s: %s", stored_id, e)

            # 6.5. Auto-Contradiction Resolver & Self-Correction
            if stored_id:
                try:
                    if getattr(self, "contradiction_resolver", None) is None:
                        from synapse.contradiction_resolver import ContradictionResolver
                        self.contradiction_resolver = ContradictionResolver(
                            storage=self.storage,
                            embedder=self._get_embedder(),
                        )
                    new_node = self.storage.get_node(stored_id)
                    if new_node:
                        resolved = self.contradiction_resolver.resolve_node_contradictions(new_node)
                        if resolved:
                            logger.info(
                                "Auto-contradiction resolver: resolved %d contradiction(s) for node %s",
                                len(resolved),
                                stored_id,
                            )
                except Exception as cr_err:
                    logger.warning("Contradiction resolver execution failed for node %s: %s", stored_id, cr_err)

            # 7. Auto-Writeback to Obsidian Vault Master
            if salience >= 0.85 and str(category).lower() in ("architecture", "sop", "decision", "preference"):
                try:
                    written_vault_file = self._auto_writeback_vault(
                        stored_id=stored_id,
                        content=content,
                        context=context,
                        category=str(category).lower(),
                        salience=salience,
                        uri=uri,
                        l0=l0,
                        l1=l1,
                        metadata=meta,
                    )
                    if written_vault_file and not l2_path:
                        try:
                            self.storage.update_node(stored_id, l2_target_path=str(written_vault_file))
                        except Exception as err:
                            logger.warning("Failed to update l2_target_path for node %s: %s", stored_id, err)
                except Exception as writeback_err:
                    logger.warning("Non-blocking vault writeback error caught for URI %s: %s", uri, writeback_err)

            return True

        except Exception:
            if self.fallback_adapter is not None:
                return self.fallback_adapter.retain(content, context, metadata=metadata, **kwargs)
            return False

    def get_harvester(self, config: Optional[Dict[str, Any]] = None) -> SynapseWebHarvester:
        """Return configured Web Harvester instance with Vault integration."""
        research_dir = (
            self.config.get("research_dir")
            or (str(self.vault_path / "research" / "auto_harvest") if self.vault_path else os.environ.get("SYNAPSE_RESEARCH_DIR") or str(Path.home() / ".synapse" / "research" / "auto_harvest"))
        )
        harvester_config = {
            "storage": self.storage,
            "llm_config": self.llm_config,
            "vault_research_dir": research_dir,
        }
        return SynapseWebHarvester(**harvester_config)

    def harvest_research(self, topic: str, title: str, source_url: str, raw_content: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Trigger autonomous web research harvest for a topic."""
        if self.harvester is None:
            return None
        return self.harvester.harvest_item(topic=topic, title=title, source_url=source_url, raw_content=raw_content)

    def list_external_research(self, topic: Optional[str] = None, status: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """List sandboxed Tier 4 external research records."""
        if self.storage is None:
            return []
        return self.storage.list_external_research(topic=topic, status=status, limit=limit)

    def promote_research(
        self,
        research_id: str,
        target_tier: int = 1,
        mental_model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Promotion Gate execution: Elevates quarantined Tier 4 research to Tier 1 Mental Model
        or Tier 2 Observation, updating both the database and active memory engines.
        """
        if self.storage is None:
            return {"error": "Storage not available"}

        promo = self.storage.promote_external_research(
            research_id=research_id,
            target_tier=target_tier,
            mental_model_id=mental_model_id,
        )

        # Ingest into active engine
        if target_tier == 1 and self.mental_model_engine:
            mm_id = promo.get("mental_model_id", "master-user-preferences")
            self.mental_model_engine.refresh_model(
                mm_id,
                new_facts=[promo["fact_snippet"]],
                force_full=False,
            )
            logger.info("Promoted research %s to Tier 1 Mental Model '%s'", research_id, mm_id)
        elif target_tier == 2 and self.observation_engine:
            self.observation_engine.ingest_turn_observation(
                promo["observation_pattern"],
                category="architecture",
                proof_delta=3,
            )
            logger.info("Promoted research %s to Tier 2 Observation", research_id)

        return promo

    def resolve_contradictions(
        self, node_or_id: Optional[Union[str, Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Resolve contradictions for a node or run full storage contradiction audit."""
        if getattr(self, "contradiction_resolver", None) is None:
            from synapse.contradiction_resolver import ContradictionResolver
            self.contradiction_resolver = ContradictionResolver(
                storage=self.storage,
                embedder=self._get_embedder(),
            )
        if node_or_id is None:
            return self.contradiction_resolver.audit_all_contradictions()
        if isinstance(node_or_id, str):
            node = self.storage.get_node(node_or_id) if self.storage else None
            if not node:
                return []
            return self.contradiction_resolver.resolve_node_contradictions(node)
        return self.contradiction_resolver.resolve_node_contradictions(node_or_id)

    def evaluate_action(self, action_proposal: str, context: str = "") -> GuardrailResult:
        """Evaluate proposed action against episodic failure memories with counterfactual reasoning."""
        if getattr(self, "counterfactual_guardrail", None) is None:
            if self.storage is not None:
                self.counterfactual_guardrail = CounterfactualGuardrail(
                    storage=self.storage,
                    embedder=self._get_embedder(),
                    config=self.config.get("guardrail"),
                )
            else:
                return GuardrailResult(
                    is_risky=False,
                    risk_level="LOW",
                    matched_episode=None,
                    similarity_score=0.0,
                    warning_message="",
                    suggested_mitigation="",
                )
        return self.counterfactual_guardrail.evaluate_action(action_proposal, context)

    def score_uncertainty(
        self, node: Dict[str, Any], current_time: Optional[int] = None
    ) -> UncertaintyReport:
        """Evaluate epistemic uncertainty of a memory node."""
        if getattr(self, "metacognitive_scorer", None) is None:
            if self.storage is not None:
                self.metacognitive_scorer = MetacognitiveScorer(
                    storage=self.storage,
                    config=self.config.get("metacognitive"),
                )
            else:
                return UncertaintyReport(
                    uncertainty_score=0.0,
                    confidence_level="HIGH",
                    is_uncertain=False,
                    reasons=[],
                    tag=None,
                )
        return self.metacognitive_scorer.score_memory(node, current_time)

    def reflect(self, query: str, **kwargs: Any) -> str:
        """
        Synthesize associative graph memories for a topic query.
        """
        if self.retriever is None:
            if self.fallback_adapter is not None:
                return self.fallback_adapter.reflect(query, **kwargs)
            return f"Reflection error: Synapse retriever unavailable for '{query}'."

        try:
            nodes = self.retriever.retrieve(
                query,
                limit=kwargs.get("limit", 6),
                max_hops=kwargs.get("max_hops", 2),
                touch=False,
            )
        except Exception:
            nodes = []

        if not nodes:
            if self.fallback_adapter is not None:
                return self.fallback_adapter.reflect(query, **kwargs)
            return f"No cognitive associative memories found for query: '{query}'."

        summary_lines: List[str] = [
            f"Cognitive Synapse Reflection on: '{query}'",
            f"Discovered {len(nodes)} associative memory node(s):",
        ]

        seen_ids = set()
        for idx, node in enumerate(nodes, 1):
            seen_ids.add(node["id"])
            cat = node.get("category", "general").upper()
            uri = node.get("uri", "")
            l0 = node.get("l0_abstract", "")
            summary_lines.append(f"{idx}. [{cat}] {uri}")
            summary_lines.append(f"   • Abstract: {l0}")
            if node.get("l1_overview"):
                summary_lines.append(f"   • Overview: {node['l1_overview']}")

        # Discover inter-node graph relationships
        connections: List[str] = []
        if self.storage is not None:
            for node_id in seen_ids:
                edges = self.storage.get_edges(node_id, direction="out")
                for e in edges:
                    if e["target_id"] in seen_ids:
                        connections.append(f"({node_id[:8]}.. -> {e['target_id'][:8]}.. via {e['relation']} [w={e['weight']}])")

        if connections:
            summary_lines.append("\nActive Synaptic Connections:")
            for conn in connections:
                summary_lines.append(f"  ↔ {conn}")

        # Contradiction Detection & Metacognition
        if self.retriever is not None:
            contradictions = self.retriever.detect_contradictions(nodes)
            if contradictions:
                summary_lines.append(f"\nDetected Contradictions ({len(contradictions)}):")
                for c in contradictions:
                    summary_lines.append(f"  - Conflict: {c['conflict']} (Winner: {c['winner_id']})")

        # Check if LLM-assisted reflection synthesis is configured / requested
        use_llm = bool(kwargs.get("use_llm", self.config.get("use_llm", False)))
        if use_llm:
            llm_result = self._reflect_with_llm(query, "\n".join(summary_lines))
            if llm_result:
                return llm_result

        return "\n".join(summary_lines)

    def _reflect_with_llm(self, query: str, context_text: str) -> Optional[str]:
        """Attempt LLM synthesis of recalled context using configured OpenAI-compatible endpoint."""
        try:
            import json
            import os
            import urllib.request
            from synapse.config import get_llm_base_url, get_llm_api_key, get_llm_model
            base_url = (
                self.llm_config.get("base_url")
                or os.environ.get("SYNAPSE_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or get_llm_base_url()
            ).rstrip("/")
            if not base_url:
                return None
            api_key = (
                self.llm_config.get("api_key")
                or get_llm_api_key()
            )
            model = (
                self.llm_config.get("model")
                or get_llm_model()
            )

            prompt_system = (
                "You are Synapse Cognitive Memory synthesizer. "
                "Synthesize the associative graph memories into a concise, high-signal "
                "Disposition & Operational Stance reflection for the agent. "
                "Highlight active preferences, architectural constraints, and settled decisions."
            )
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": prompt_system,
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nContext:\n{context_text}",
                    },
                ],
                "temperature": 0.2,
                "max_tokens": 500,
            }
            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    choices = data.get("choices", [])
                    if choices and "message" in choices[0]:
                        return choices[0]["message"].get("content", "").strip()
        except Exception:
            return None
        return None

    def apply_outcome_feedback(
        self,
        node_ids: Optional[List[str]] = None,
        outcome: Union[str, bool] = True,
        note: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Apply Hebbian outcome feedback (Active Plasticity: LTP and LTD)
        across given or recently co-activated memory nodes.
        Delegates to the underlying SynapseWorker.
        """
        if self.worker is None:
            is_success = outcome if isinstance(outcome, bool) else (str(outcome).lower() in ("success", "true", "1", "pass"))
            return {
                "outcome": "success" if is_success else "failure",
                "is_success": is_success,
                "edges_updated": 0,
                "edges_reinforced": 0,
                "edges_depressed": 0,
                "nodes_updated": 0,
                "reinforced_nodes": [],
                "penalized_nodes": [],
                "flagged_nodes": [],
                "protected_nodes": [],
                "note": note,
            }

        target_ids = node_ids
        if target_ids is None and hasattr(self, "_last_retrieved_node_ids"):
            target_ids = self._last_retrieved_node_ids

        return self.worker.apply_outcome_feedback(
            node_ids=target_ids or [],
            outcome=outcome,
            note=note,
            **kwargs,
        )


class LocalFileAdapter(BaseMemoryAdapter):
    """Adapter for local Markdown / text memory files without external server dependency."""

    name = "local"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        raw_file = self.config.get("memory_file") or "MEMORY.md"
        path_obj = Path(raw_file).expanduser()
        if path_obj.is_absolute() or str(raw_file).startswith((".", "/", "~")):
            self.file_path = path_obj.resolve()
        else:
            self.file_path = (Path.home() / ".synapse" / raw_file).resolve()

    def recall(self, query: str, limit: int = 3, **kwargs: Any) -> List[str]:
        if not self.file_path.exists():
            return []
        try:
            lines = [l.strip() for l in self.file_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            q_terms = [w.lower() for w in query.split() if len(w) > 2]
            scored = []
            for line in lines:
                score = sum(1 for term in q_terms if term in line.lower())
                if score > 0:
                    scored.append((score, line))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [s[1] for s in scored[:limit]]
        except Exception:
            return []

    def retain(self, content: str, context: str = "", metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> bool:
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            line = f"- {content.strip()}"
            if context:
                line += f" [Context: {context.strip()}]"
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(f"\n{line}")
            return True
        except Exception:
            return False


class NullAdapter(BaseMemoryAdapter):
    """Disabled memory backend."""

    name = "none"

    def recall(self, query: str, limit: int = 3, **kwargs: Any) -> List[str]:
        return []

    def retain(self, content: str, context: str = "", metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> bool:
        return True


def resolve_memory_adapter(
    config: Optional[Dict[str, Any]] = None,
    global_env: Optional[Dict[str, str]] = None,
) -> BaseMemoryAdapter:
    """Factory resolving the active memory adapter dynamically.

    Priority:
    1. config['memory_backend'] (per-profile dictionary or string)
    2. Environment variable MEMORY_BACKEND or MEMORY_TYPE
    3. Default SynapseAdapter
    """
    cfg = config or {}
    backend_cfg = cfg.get("memory_backend")

    if isinstance(backend_cfg, dict):
        btype = str(backend_cfg.get("type", "")).lower()
        if btype in ("synapse", "cognitive", "synapse_core", "default"):
            return SynapseAdapter(backend_cfg)
        if btype in ("local", "file"):
            return LocalFileAdapter(backend_cfg)
        if btype in ("none", "disabled", "off"):
            return NullAdapter()

    btype_str = str(
        backend_cfg
        or os.environ.get("MEMORY_BACKEND")
        or os.environ.get("MEMORY_TYPE")
        or ""
    ).lower().strip()

    if btype_str in ("synapse", "cognitive", "synapse_core"):
        return SynapseAdapter(cfg.get("synapse") or {})
    if btype_str in ("none", "off", "disabled"):
        return NullAdapter()
    if btype_str in ("local", "file"):
        return LocalFileAdapter({"memory_file": cfg.get("memory_file", "MEMORY.md")})

    return SynapseAdapter(cfg.get("synapse") or cfg)
