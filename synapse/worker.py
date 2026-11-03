"""
SynapseWorker - Background Cognitive Maintenance, Vault Cortex Sync & Hebbian Reinforcement.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from synapse.loader import estimate_tokens, split_sentences
from synapse.storage import VALID_CATEGORIES, SynapseStorage

DEFAULT_LTP_DELTA: float = 0.1
DEFAULT_LTD_DELTA: float = 0.15
DEFAULT_CONFIDENCE_PENALTY: float = 0.1
FLAG_CONFIDENCE_THRESHOLD: float = 0.3
SALIENCE_PROTECTION_THRESHOLD: float = 0.95


def resolve_synapse_llm_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Resolve generic OpenAI-compatible LLM configuration for cognitive memory tasks.

    Precedence order:
    1. Explicit config dictionary passed as argument (e.g. base_url, synapse_base_url, llm_config, etc.)
    2. SYNAPSE_* environment variables (SYNAPSE_BASE_URL, SYNAPSE_API_KEY, SYNAPSE_MODEL)
    3. OPENAI_* environment variables (OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL)
    4. Sane default endpoints:
       - base_url: 'http://127.0.0.1:11434/v1'
       - api_key: 'default_key'
       - model: 'gpt-4o-mini'
    """
    cfg: Dict[str, Any] = dict(config or {})
    if isinstance(cfg.get("llm_config"), dict):
        llm_sub = cfg["llm_config"]
        cfg = {**cfg, **llm_sub}

    # 1. Base URL: SYNAPSE_BASE_URL -> OPENAI_BASE_URL -> BASE_URL -> ''
    base_url = (
        cfg.get("base_url")
        or cfg.get("synapse_base_url")
        or cfg.get("openai_base_url")
        or cfg.get("SYNAPSE_BASE_URL")
        or cfg.get("OPENAI_BASE_URL")
        or cfg.get("BASE_URL")
        or os.environ.get("SYNAPSE_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("BASE_URL")
        or ""
    )

    # 2. API Key: SYNAPSE_API_KEY -> OPENAI_API_KEY -> API_KEY -> ''
    api_key = (
        cfg.get("api_key")
        or cfg.get("synapse_api_key")
        or cfg.get("openai_api_key")
        or cfg.get("SYNAPSE_API_KEY")
        or cfg.get("OPENAI_API_KEY")
        or cfg.get("API_KEY")
        or os.environ.get("SYNAPSE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("API_KEY")
        or ""
    )

    # 3. Model: SYNAPSE_MODEL -> OPENAI_MODEL -> MODEL_NAME -> MODEL -> ''
    model = (
        cfg.get("model")
        or cfg.get("synapse_model")
        or cfg.get("openai_model")
        or cfg.get("model_name")
        or cfg.get("SYNAPSE_MODEL")
        or cfg.get("OPENAI_MODEL")
        or cfg.get("MODEL_NAME")
        or cfg.get("MODEL")
        or os.environ.get("SYNAPSE_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("MODEL_NAME")
        or os.environ.get("MODEL")
        or ""
    )

    return {
        "base_url": str(base_url).strip(),
        "api_key": str(api_key).strip(),
        "model": str(model).strip(),
    }


class _HybridMaintenanceMethod:
    """
    Descriptor enabling run_in_process_maintenance_if_needed to be called either
    as a classmethod (SynapseWorker.run_in_process_maintenance_if_needed)
    or as an instance method (worker.run_in_process_maintenance_if_needed).
    """

    def __init__(self, func: Callable[..., bool]) -> None:
        self.func = func

    def __get__(self, instance: Optional[SynapseWorker], owner: type) -> Callable[..., bool]:
        if instance is not None:
            def _instance_wrapper(*args: Any, **kwargs: Any) -> bool:
                if "storage" not in kwargs and len(args) < 5:
                    kwargs.setdefault("storage", instance.storage)
                return self.func(instance, *args, **kwargs)
            return _instance_wrapper
        else:
            def _class_wrapper(*args: Any, **kwargs: Any) -> bool:
                return self.func(owner, *args, **kwargs)
            return _class_wrapper


class SynapseWorker:
    """
    Worker engine handling background cognitive maintenance:
    1. Half-life decay cycles and node archiving.
    2. Vault Cortex synchronization with Obsidian / Markdown SOP files.
    3. Hebbian synaptic edge weight reinforcement for co-activated nodes.
    4. Cross-platform universal in-process timestamp guard lifecycle.
    """

    _latest_background_thread: Optional[threading.Thread] = None

    def __init__(
        self,
        storage: Optional[SynapseStorage] = None,
        config: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.storage = storage if storage is not None else SynapseStorage()
        self.config = dict(config or {})
        if llm_config:
            self.config["llm_config"] = llm_config
        self.llm_config = resolve_synapse_llm_config(self.config)

    def run_in_process_maintenance_if_needed(
        self_or_cls: Any,
        vault_path: Optional[str] = None,
        interval_hours: float = 12.0,
        non_blocking: bool = True,
        sync_state_path: Optional[Union[Path, str]] = None,
        storage: Optional[SynapseStorage] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        In-Process Universal Lifecycle Maintenance Guard (Bab 6 Blueprint).
        Checks if elapsed time since last sync exceeds interval_hours (default 12 hours).

        If needed:
          - Runs vault_cortex_sync(vault_path) and decay_cycle()
          - Updates/records timestamp in state file upon completion
          - If non_blocking=True, spawns background daemon thread and returns True immediately
          - If non_blocking=False, executes synchronously and returns True
        If up to date (elapsed < interval_hours):
          - Returns False (skipped)
        """
        # Resolve target sync state path using Path.home()
        if sync_state_path is not None:
            target_state_path = Path(sync_state_path).expanduser().resolve()
        else:
            target_state_path = (Path.home() / ".synapse" / "synapse_last_sync.json").resolve()

        now = time.time()
        interval_seconds = max(0.0, float(interval_hours)) * 3600.0

        should_run = False
        if not target_state_path.exists():
            should_run = True
        else:
            try:
                content = target_state_path.read_text(encoding="utf-8").strip()
                if not content:
                    should_run = True
                else:
                    data = json.loads(content)
                    if not isinstance(data, dict):
                        should_run = True
                    else:
                        last_sync = data.get("last_sync")
                        if last_sync is None:
                            last_sync = data.get("timestamp")

                        if last_sync is None:
                            should_run = True
                        else:
                            elapsed = now - float(last_sync)
                            if elapsed >= interval_seconds:
                                should_run = True
            except Exception:
                should_run = True

        if not should_run:
            return False

        # Prepare execution logic
        def _execute_maintenance() -> None:
            try:
                # Instantiate worker with provided or instance storage
                if isinstance(self_or_cls, SynapseWorker):
                    worker_instance = self_or_cls
                else:
                    worker_instance = SynapseWorker(storage=storage, config=config)

                # 1. Vault Cortex Sync
                worker_instance.vault_cortex_sync(vault_path=vault_path)

                # 2. Episodic-to-Semantic Consolidation
                try:
                    worker_instance.consolidate_episodes_to_semantic(limit=10)
                except Exception:
                    pass

                # 3. Decay Cycle
                worker_instance.decay_cycle()

                # 4. Storage Optimization & Compaction
                worker_instance.storage.optimize_and_compact()

                # 4. Update / record timestamp in state file upon completion
                target_state_path.parent.mkdir(parents=True, exist_ok=True)
                completed_ts = time.time()
                target_vault = str(
                    vault_path
                    or os.environ.get("SYNAPSE_VAULT_PATH")
                    or (Path.home() / "vault").resolve()
                )
                payload = {
                    "last_sync": completed_ts,
                    "timestamp": completed_ts,
                    "last_sync_iso": datetime.datetime.fromtimestamp(
                        completed_ts, tz=datetime.timezone.utc
                    ).isoformat(),
                    "vault_path": target_vault,
                }
                target_state_path.write_text(
                    json.dumps(payload, indent=2), encoding="utf-8"
                )
            except Exception:
                pass

        if non_blocking:
            thread = threading.Thread(
                target=_execute_maintenance,
                name="SynapseMaintenanceDaemon",
                daemon=True,
            )
            SynapseWorker._latest_background_thread = thread
            thread.start()
            return True
        else:
            _execute_maintenance()
            return True

    # Register hybrid descriptor
    run_in_process_maintenance_if_needed = _HybridMaintenanceMethod(  # type: ignore[assignment]
        run_in_process_maintenance_if_needed
    )

    def decay_cycle(
        self,
        decay_rate: float = 0.05,
        archive_threshold: float = 0.1,
        importance_threshold: float = 0.8,
    ) -> Dict[str, int]:
        """
        Iterate over active nodes, apply exponential decay to strength based on elapsed
        time since last_accessed, and archive nodes falling below archive_threshold.
        Nodes/episodes with salience >= 0.9 (Master Directives / Critical Lessons) NEVER decay (immune to pruning).

        Returns:
            Dict containing counts: {'processed': N, 'decayed': D, 'archived': A, 'protected': P, 'immune': I}
        """
        stats: Dict[str, int] = {
            "processed": 0,
            "decayed": 0,
            "archived": 0,
            "protected": 0,
            "immune": 0,
        }

        active_nodes = self.storage.list_nodes(status="active", limit=100000)
        now = int(time.time())

        for node in active_nodes:
            stats["processed"] += 1
            node_id = node["id"]
            last_accessed = node.get("last_accessed") or node.get("created_at") or now
            elapsed_seconds = max(0, now - last_accessed)
            elapsed_days = elapsed_seconds / 86400.0

            salience = float(node.get("salience", 0.5))
            importance = float(node.get("importance", 0.5))

            # Master Directives / High Salience items (salience >= 0.9) NEVER decay (immune to pruning)
            if salience >= 0.9:
                stats["protected"] += 1
                stats["immune"] += 1
                continue

            if elapsed_days <= 0:
                continue

            current_strength = float(node.get("strength", 1.0))
            # Exponential decay: S(t) = S_0 * exp(-decay_rate * days)
            new_strength = max(0.0, current_strength * math.exp(-decay_rate * elapsed_days))
            stats["decayed"] += 1

            if new_strength < archive_threshold:
                if importance >= importance_threshold:
                    # Protected high-importance node: keep active but update strength
                    stats["protected"] += 1
                    self.storage.update_node(
                        node_id,
                        strength=round(new_strength, 4),
                    )
                else:
                    # Archive decaying low/medium importance node
                    stats["archived"] += 1
                    self.storage.update_node(
                        node_id,
                        strength=round(new_strength, 4),
                        status="archived",
                    )
            else:
                self.storage.update_node(
                    node_id,
                    strength=round(new_strength, 4),
                )

        return stats

    def decay_edges_cycle(
        self,
        decay_rate: float = 0.05,
        prune_threshold: float = 0.15,
        idle_days_threshold: float = 14.0,
    ) -> Dict[str, int]:
        """
        Execute edge decay cycle on associative graph edges.
        Dormant edges (> 14 days idle) decay exponentially and are pruned if below prune_threshold (0.15),
        unless connected to high-salience nodes (>= 0.90, e.g. Master Directives).
        """
        if hasattr(self.storage, "prune_decayed_edges"):
            return self.storage.prune_decayed_edges(
                decay_rate=decay_rate,
                prune_threshold=prune_threshold,
                idle_days_threshold=idle_days_threshold,
            )
        return {"processed": 0, "decayed": 0, "pruned": 0}

    def rollup_superseded_nodes(
        self,
        min_count: int = 1,
        backup_snapshot: bool = True,
        snapshot_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Abstract Rollup: Compresses superseded nodes into 1 master archive summary entity,
        safely archiving raw historical records in synapse_superseded_archive and a compressed
        JSON snapshot, and pruning superseded records from synapse_nodes, synapse_fts,
        synapse_vectors, and synapse_edges to maintain <10ms retrieval speeds.
        """
        rows = self.storage.list_superseded_nodes()

        if len(rows) < min_count:
            return {
                "status": "noop",
                "rolled_up_count": len(rows),
                "message": f"Superseded count ({len(rows)}) < threshold ({min_count})",
            }

        # 1. Aggregate statistics and categories
        cat_counts: Dict[str, int] = {}
        for r in rows:
            cat = r.get("category") or "general"
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        now_ts = int(time.time())
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        date_str = now_dt.strftime("%Y-%m-%d %H:%M UTC")

        # 2. Extract representative historical highlights
        theme_samples: Dict[str, List[str]] = {}
        for r in rows:
            cat = r.get("category") or "general"
            if cat not in theme_samples:
                theme_samples[cat] = []
            abstract = (r.get("l0_abstract") or "").strip()
            if abstract and len(theme_samples[cat]) < 3:
                theme_samples[cat].append(abstract[:100])

        cat_summary_lines = []
        for cat, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
            samples = theme_samples.get(cat, [])
            sample_text = "; ".join(samples) if samples else "Various superseded entries"
            cat_summary_lines.append(f"- **{cat.capitalize()}** ({count} nodes): e.g. {sample_text}")
        cat_breakdown = "\n".join(cat_summary_lines)

        # Master archive rollup entity definition
        master_id = "arch-rollup-superseded-master"
        master_uri = "synapse://archive/superseded_rollup"
        master_l0 = (
            f"Consolidated Archive Rollup: {len(rows):,} Superseded Nodes across "
            f"{', '.join(cat_counts.keys())} domains."
        )
        master_l1 = (
            f"# Archive Superseded Knowledge Rollup ({date_str})\n"
            f"Total historical superseded entries archived: {len(rows):,}\n\n"
            f"### Domain Breakdown & Evolution:\n{cat_breakdown}\n\n"
            f"All {len(rows):,} granular records safely preserved in 'synapse_superseded_archive'. "
            f"Search indices optimized for sub-10ms active retrieval."
        )

        # 3. Snapshot backup to disk if requested
        if backup_snapshot:
            try:
                import gzip
                base_dir = (
                    snapshot_dir
                    or os.path.join(os.path.dirname(self.storage.db_path), "..", "data")
                )
                os.makedirs(base_dir, exist_ok=True)
                snap_path = os.path.join(base_dir, "superseded_archive_snapshot.json.gz")
                with gzip.open(snap_path, "wt", encoding="utf-8") as gz:
                    json.dump(
                        {
                            "rollup_timestamp": now_ts,
                            "rollup_date_wib": date_str,
                            "total_records": len(rows),
                            "categories": cat_counts,
                            "records": rows,
                        },
                        gz,
                        indent=2,
                    )
            except Exception:
                pass

        # 4-8. Archive rows, upsert the master entity, purge superseded data.
        result = self.storage.commit_superseded_rollup(
            master={
                "id": master_id,
                "uri": master_uri,
                "l0_abstract": master_l0,
                "l1_overview": master_l1,
            },
            rollup_id=master_id,
            archived_at=now_ts,
        )

        return {
            "status": "success",
            "rolled_up_count": result["deleted_nodes"],
            "deleted_edges": result["deleted_edges"],
            "master_entity_id": master_id,
            "categories": cat_counts,
            "archived_at_wib": date_str,
        }

    def _extract_vault_l0(self, text: str, title_fallback: str) -> str:
        """
        Extract L0 abstract (<40 tokens) from markdown text or frontmatter.
        """
        clean_text = text.strip()
        # 1. Check YAML frontmatter summary or title
        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", clean_text, re.DOTALL)
        if fm_match:
            fm_content = fm_match.group(1)
            clean_text = clean_text[fm_match.end():].strip()

            summary_match = re.search(r"^(?:summary|description|abstract):\s*([^\n]+)", fm_content, re.IGNORECASE | re.MULTILINE)
            if summary_match:
                candidate = summary_match.group(1).strip().strip('"\'')
                if candidate:
                    return candidate

            title_match = re.search(r"^(?:title|name):\s*([^\n]+)", fm_content, re.IGNORECASE | re.MULTILINE)
            if title_match:
                candidate = title_match.group(1).strip().strip('"\'')
                if candidate:
                    title_fallback = candidate

        # 2. Check first top-level or second-level heading
        heading_match = re.search(r"^#{1,3}\s+(.+)$", clean_text, re.MULTILINE)
        if heading_match:
            heading = heading_match.group(1).strip()
            # If body has text right after heading
            body_start = heading_match.end()
            body_snippet = clean_text[body_start:].strip()
            first_sentence = split_sentences(body_snippet)
            if first_sentence:
                combined = f"{heading}: {first_sentence[0]}"
                if estimate_tokens(combined) <= 40:
                    return combined
            if estimate_tokens(heading) <= 40:
                return heading

        # 3. First non-empty paragraph or sentence
        sentences = split_sentences(clean_text)
        if sentences:
            first_s = sentences[0]
            if estimate_tokens(first_s) <= 40:
                return first_s
            words = first_s.split()
            truncated = " ".join(words[:25]) + "..."
            return truncated

        return title_fallback

    def _parse_markdown_tables(self, text: str) -> List[Dict[str, Any]]:
        """
        Lightweight AST-style markdown table parser without external heavy libraries.
        Identifies markdown table blocks (| col1 | col2 |), extracts headers and structured row mappings.
        """
        tables: List[Dict[str, Any]] = []
        # Strip code blocks to avoid false positives inside code fences
        clean = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        lines = clean.splitlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Potential table header row containing '|'
            if "|" in line and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # Separator row e.g. |---|---| or :--- | ---: or ---|---
                sep_match = re.match(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$", next_line)
                if sep_match and "|" in next_line:
                    header_cells = [c.strip() for c in line.strip("|").split("|")]
                    headers = [h for h in header_cells if h]

                    rows: List[List[str]] = []
                    j = i + 2
                    while j < len(lines):
                        row_line = lines[j].strip()
                        if not row_line or "|" not in row_line:
                            break
                        row_cells = [c.strip() for c in row_line.strip("|").split("|")]
                        if any(row_cells):
                            rows.append(row_cells)
                        j += 1

                    if headers and rows:
                        tables.append({
                            "headers": headers,
                            "rows": rows,
                        })
                    i = j
                    continue
            i += 1
        return tables

    def _extract_vault_l1(self, text: str) -> str:
        """
        Extract L1 overview (<300 tokens) summarizing key sections, markdown tables, and structure.
        """
        clean_text = text.strip()
        # Strip frontmatter if present
        clean_text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", clean_text, flags=re.DOTALL).strip()

        if not clean_text:
            return ""

        # Extract headings, tables, and bullet points
        headings = re.findall(r"^(#{1,4}\s+.+)$", clean_text, re.MULTILINE)
        bullets = re.findall(r"^\s*[-*+]\s+(.+)$", clean_text, re.MULTILINE)
        parsed_tables = self._parse_markdown_tables(clean_text)

        overview_parts: List[str] = []
        if headings:
            clean_headings = [re.sub(r"^#{1,4}\s+", "", h).strip() for h in headings[:8]]
            overview_parts.append(f"Sections: {', '.join(clean_headings)}")

        if parsed_tables:
            table_summaries: List[str] = []
            for t in parsed_tables[:3]:
                hdrs = t["headers"]
                hdr_str = " | ".join(hdrs[:4])
                row_items: List[str] = []
                for row in t["rows"][:5]:
                    if not row:
                        continue
                    if len(row) == 1:
                        row_items.append(row[0])
                    elif len(row) == 2:
                        row_items.append(f"{row[0]} -> {row[1]}")
                    else:
                        details = ", ".join(
                            f"{hdrs[k]}: {row[k]}" for k in range(2, min(len(hdrs), len(row))) if k < len(hdrs) and row[k]
                        )
                        if details:
                            row_items.append(f"{row[0]} -> {row[1]} ({details})")
                        else:
                            row_items.append(f"{row[0]} -> {row[1]}")
                if row_items:
                    table_summaries.append(f"Table [{hdr_str}]: {'; '.join(row_items)}")
            if table_summaries:
                overview_parts.append(" | ".join(table_summaries))

        if bullets:
            clean_bullets = bullets[:5]
            overview_parts.append(f"Key Points: {'; '.join(clean_bullets)}")

        if not overview_parts:
            # Fallback to initial paragraphs / sentences
            sentences = split_sentences(clean_text)
            accepted: List[str] = []
            for s in sentences:
                candidate = " ".join(accepted + [s])
                if estimate_tokens(candidate) <= 250:
                    accepted.append(s)
                else:
                    break
            overview_parts.append(" ".join(accepted))

        overview = " | ".join(overview_parts)
        # Ensure under 300 tokens
        if estimate_tokens(overview) > 300:
            words = overview.split()
            overview = " ".join(words[:200]) + "..."

        return overview

    def vault_cortex_sync(self, vault_path: Optional[str] = None) -> Dict[str, int]:
        """
        Scan markdown files in ~/vault/ (or specified path), extract L0 abstract
        and L1 overview, and register/update synapse_nodes with category='sop'
        and l2_target_path pointing to the markdown file path.

        Returns:
            Dict containing counts: {'scanned': S, 'created': C, 'updated': U, 'errors': E}
        """
        stats: Dict[str, int] = {
            "scanned": 0,
            "created": 0,
            "updated": 0,
            "errors": 0,
        }

        raw_path = vault_path or os.environ.get("SYNAPSE_VAULT_PATH", "~/vault")
        root_dir = Path(raw_path).expanduser().resolve()

        if not root_dir.exists() or not root_dir.is_dir():
            return stats

        md_files: List[Path] = []
        for ext in ("*.md", "*.markdown"):
            for p in root_dir.rglob(ext):
                try:
                    rel_p = p.relative_to(root_dir)
                    if "_arsip" in rel_p.parts or "_archive" in rel_p.parts or "daily" in rel_p.parts or any(part.startswith(".") for part in rel_p.parts):
                        continue
                except Exception:
                    continue
                md_files.append(p)

        for file_path in md_files:
            stats["scanned"] += 1
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                rel_path = file_path.relative_to(root_dir)
                slug_stem = str(rel_path.with_suffix("")).replace(os.sep, "/").lower()
                clean_slug = re.sub(r"[^a-z0-9_\-/]+", "_", slug_stem).strip("_/")

                # Extract category from frontmatter, preserving original metadata
                category = "sop"
                fm_match = re.search(r"^---\s*\n(.*?)\n---\s*\n", content, flags=re.DOTALL)
                if fm_match:
                    fm_raw = fm_match.group(1)
                    try:
                        import yaml
                        fm_data = yaml.safe_load(fm_raw)
                        if isinstance(fm_data, dict):
                            raw_cat = fm_data.get("category") or fm_data.get("type") or fm_data.get("domain")
                            if raw_cat and str(raw_cat).lower().strip() in VALID_CATEGORIES:
                                category = str(raw_cat).lower().strip()
                    except Exception:
                        cat_match = re.search(
                            r"^\s*(?:category|type|domain)\s*:\s*['\"]?([a-zA-Z0-9_\-]+)['\"]?",
                            fm_raw,
                            re.MULTILINE,
                        )
                        if cat_match:
                            cand = cat_match.group(1).lower().strip()
                            if cand in VALID_CATEGORIES:
                                category = cand

                # If category remains default "sop", check if top directory matches a known category
                if category == "sop" and rel_path.parts:
                    first_folder = rel_path.parts[0].lower()
                    if first_folder in VALID_CATEGORIES:
                        category = first_folder

                # Use uniform URI scheme synapse://memory/{category}/{slug}
                uri = f"synapse://memory/{category}/{clean_slug}"
                title_fallback = file_path.stem.replace("_", " ").title()

                l0_abstract = self._extract_vault_l0(content, title_fallback)
                l1_overview = self._extract_vault_l1(content)
                abs_path = str(file_path.resolve())

                # Seamlessly check both new uniform URI and legacy URI
                existing_node = self.storage.get_node(uri)
                if existing_node is None:
                    legacy_uri = f"synapse://sop/{clean_slug}"
                    existing_node = self.storage.get_node(legacy_uri)

                if existing_node is not None:
                    self.storage.update_node(
                        existing_node["id"],
                        uri=uri,
                        l0_abstract=l0_abstract,
                        l1_overview=l1_overview,
                        l2_target_path=abs_path,
                        category=category,
                    )
                    stats["updated"] += 1
                else:
                    self.storage.add_node(
                        uri=uri,
                        l0_abstract=l0_abstract,
                        l1_overview=l1_overview,
                        l2_target_path=abs_path,
                        category=category,
                        importance=0.6,
                    )
                    stats["created"] += 1

            except Exception:
                stats["errors"] += 1

        return stats

    def hebbian_reinforce(
        self,
        node_ids: List[str],
        delta: float = 0.1,
        relation: str = "co_activated",
    ) -> int:
        """
        Increases synaptic edge weights between co-activated node IDs,
        strengthens node salience and strength via interaction history,
        or creates new edges if none exist.

        Returns:
            Total number of edges reinforced or created.
        """
        if not node_ids or len(node_ids) < 2:
            return 0

        unique_ids = list(dict.fromkeys(node_ids))
        if len(unique_ids) < 2:
            return 0

        # Reinforce individual nodes (salience, strength, access_count)
        for nid in unique_ids:
            if hasattr(self.storage, "record_node_access"):
                self.storage.record_node_access(nid, delta_salience=0.02, delta_strength=delta, bump_edges=False)

        reinforced_count = 0

        for src, tgt in combinations(unique_ids, 2):
            for source_id, target_id in ((src, tgt), (tgt, src)):
                existing_edges = self.storage.get_edges(source_id, direction="out")
                target_edge = next(
                    (e for e in existing_edges if e["target_id"] == target_id), None
                )

                if target_edge is not None:
                    curr_weight = float(target_edge.get("weight", 0.5))
                    new_weight = min(1.0, curr_weight + delta)
                    rel = target_edge.get("relation") or relation
                else:
                    new_weight = min(1.0, 0.5 + delta)
                    rel = relation

                success = self.storage.add_edge(
                    source_id,
                    target_id,
                    relation=rel,
                    weight=round(new_weight, 4),
                    increment_access=True,
                )
                if success:
                    reinforced_count += 1

        return reinforced_count

    def apply_outcome_feedback(
        self,
        node_ids: List[str],
        outcome: Union[str, bool],
        note: str = "",
        delta: float = DEFAULT_LTP_DELTA,
        delta_ltd: float = DEFAULT_LTD_DELTA,
        confidence_penalty: float = DEFAULT_CONFIDENCE_PENALTY,
        salience_protection_threshold: float = SALIENCE_PROTECTION_THRESHOLD,
        flag_confidence_threshold: float = FLAG_CONFIDENCE_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Apply Hebbian Outcome Feedback Loop (Active Plasticity: LTP and LTD).

        Features:
        1. Long-Term Potentiation (LTP / success):
           - Strengthen synaptic edge weights (+delta, default 0.1)
           - Reinforce node strength (+delta)
        2. Long-Term Depression (LTD / failure):
           - Depress synaptic edge weights (-delta_ltd, default 0.15)
           - Penalize node confidence (-confidence_penalty, default -0.1)
           - If confidence < flag_confidence_threshold (0.3), flag node (status='flagged')
        3. Salience Protection:
           - Nodes with salience >= salience_protection_threshold (0.95) are completely
             protected from confidence decay/penalties.

        Returns:
            Dict containing outcome feedback execution metrics.
        """
        if isinstance(outcome, bool):
            is_success = outcome
        elif isinstance(outcome, (int, float)):
            is_success = outcome > 0
        elif isinstance(outcome, str):
            clean = outcome.strip().lower()
            if clean in {
                "1", "true", "yes", "y", "ok", "success", "pass",
                "passed", "positive", "reward", "win", "good", "ltp"
            }:
                is_success = True
            elif clean in {
                "0", "false", "no", "n", "failure", "fail",
                "failed", "negative", "penalty", "loss", "bad", "ltd", "error"
            }:
                is_success = False
            elif any(sub in clean for sub in ("fail", "error", "penalty", "ltd", "negative", "bad")):
                is_success = False
            elif any(sub in clean for sub in ("succ", "pass", "reward", "ltp", "positive", "good", "win")):
                is_success = True
            else:
                is_success = False
        else:
            is_success = bool(outcome)

        if not node_ids:
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

        unique_ids = list(dict.fromkeys(node_ids))
        edges_updated = 0
        reinforced_nodes: List[str] = []
        penalized_nodes: List[str] = []
        flagged_nodes: List[str] = []
        protected_nodes: List[str] = []

        if is_success:
            # LTP: Strengthen synaptic edge weights between co-activated pairs
            if len(unique_ids) >= 2:
                for src, tgt in combinations(unique_ids, 2):
                    for source_id, target_id in ((src, tgt), (tgt, src)):
                        existing_edges = self.storage.get_edges(source_id, direction="out")
                        target_edges = [
                            e for e in existing_edges if e.get("target_id") == target_id
                        ]
                        if target_edges:
                            for target_edge in target_edges:
                                curr_weight = float(target_edge.get("weight", 0.5))
                                new_weight = min(1.0, curr_weight + delta)
                                rel = target_edge.get("relation") or "co_activated"
                                if self.storage.add_edge(
                                    source_id,
                                    target_id,
                                    relation=rel,
                                    weight=round(new_weight, 4),
                                ):
                                    edges_updated += 1
                        else:
                            new_weight = min(1.0, 0.5 + delta)
                            if self.storage.add_edge(
                                source_id,
                                target_id,
                                relation="co_activated",
                                weight=round(new_weight, 4),
                            ):
                                edges_updated += 1

            # LTP: Reinforce node strength
            for node_id in unique_ids:
                node = self.storage.get_node(node_id)
                if node is None:
                    continue
                curr_strength = float(
                    node.get("strength", 1.0) if node.get("strength") is not None else 1.0
                )
                new_strength = round(curr_strength + delta, 4)
                if self.storage.update_node(node_id, strength=new_strength):
                    reinforced_nodes.append(node_id)

        else:
            # LTD: Depress synaptic edge weights between co-activated pairs
            if len(unique_ids) >= 2:
                for src, tgt in combinations(unique_ids, 2):
                    for source_id, target_id in ((src, tgt), (tgt, src)):
                        existing_edges = self.storage.get_edges(source_id, direction="out")
                        target_edges = [
                            e for e in existing_edges if e.get("target_id") == target_id
                        ]
                        if target_edges:
                            for target_edge in target_edges:
                                curr_weight = float(target_edge.get("weight", 0.5))
                                new_weight = max(0.0, curr_weight - delta_ltd)
                                rel = target_edge.get("relation") or "co_activated"
                                if self.storage.add_edge(
                                    source_id,
                                    target_id,
                                    relation=rel,
                                    weight=round(new_weight, 4),
                                ):
                                    edges_updated += 1
                        else:
                            new_weight = max(0.0, 0.5 - delta_ltd)
                            if self.storage.add_edge(
                                source_id,
                                target_id,
                                relation="co_activated",
                                weight=round(new_weight, 4),
                            ):
                                edges_updated += 1

            # LTD: Penalize node confidence with salience protection
            for node_id in unique_ids:
                node = self.storage.get_node(node_id)
                if node is None:
                    continue
                salience = float(
                    node.get("salience", 0.5) if node.get("salience") is not None else 0.5
                )

                # Salience Protection: Master directives completely protected from decay/penalties
                if salience >= salience_protection_threshold:
                    protected_nodes.append(node_id)
                    continue

                curr_conf = float(
                    node.get("confidence", 1.0) if node.get("confidence") is not None else 1.0
                )
                new_conf = round(max(0.0, curr_conf - confidence_penalty), 4)

                if new_conf < flag_confidence_threshold:
                    self.storage.update_node(node_id, confidence=new_conf, status="flagged")
                    flagged_nodes.append(node_id)
                else:
                    self.storage.update_node(node_id, confidence=new_conf)

                penalized_nodes.append(node_id)

        return {
            "outcome": "success" if is_success else "failure",
            "is_success": is_success,
            "edges_updated": edges_updated,
            "edges_reinforced": edges_updated if is_success else 0,
            "edges_depressed": edges_updated if not is_success else 0,
            "nodes_updated": len(reinforced_nodes) if is_success else len(penalized_nodes),
            "reinforced_nodes": reinforced_nodes,
            "penalized_nodes": penalized_nodes,
            "flagged_nodes": flagged_nodes,
            "protected_nodes": protected_nodes,
            "note": note,
        }

    def maintenance_cycle(self, vault_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute full synchronous cognitive maintenance cycle:
        1. Vault cortex synchronization
        2. Episodic-to-semantic consolidation
        3. Half-life decay cycle
        4. Storage optimization and compaction (PRAGMA optimize, wal_checkpoint, VACUUM)
        """
        vault_stats = self.vault_cortex_sync(vault_path=vault_path)
        consolidation_stats = self.consolidate_episodes_to_semantic(limit=10)
        decay_stats = self.decay_cycle()
        edge_decay_stats = self.decay_edges_cycle()
        opt_stats = self.storage.optimize_and_compact()
        return {
            "vault_sync": vault_stats,
            "consolidation": consolidation_stats,
            "decay": decay_stats,
            "edge_decay": edge_decay_stats,
            "optimization": opt_stats,
        }

    def _categorize_episode_lesson(self, ep: Dict[str, Any]) -> str:
        """
        Categorize an episodic experience into 'sop' or 'architecture' based on its
        lessons_learned, title, situation, and action_taken.
        """
        title = str(ep.get("title") or "")
        situation = str(ep.get("situation") or "")
        action = str(ep.get("action_taken") or "")
        lesson = str(ep.get("lessons_learned") or "")
        tags = str(ep.get("tags") or "")

        sop_regex = re.compile(
            r"\b(sop|procedure|guideline|guide|runbook|playbook|workflow|"
            r"step|instruction|rule|protocol|standard|"
            r"make sure|always|never|"
            r"must|ensure|remember to|verify|do not|checklist|best practice)\b",
            re.IGNORECASE,
        )

        arch_regex = re.compile(
            r"\b(architect\w*|structure|pattern\w*|schema|"
            r"database\w*|design\w*|topology|layer|modular|standalone|"
            r"repository|fastapi|sqlite|wal|postgresql|postgres|redis|kernel|vps|swap|"
            r"ram|memory|kswapd|tuning|overcommit|cluster|failover|gateway|router|proxy|"
            r"microservice|engine|framework|component\w*|interface|subsystem|"
            r"stack|infrastructure|concurrency|multithread|hardware|server)\b",
            re.IGNORECASE,
        )

        sop_score = (
            len(sop_regex.findall(lesson)) * 3
            + len(sop_regex.findall(title)) * 2
            + len(sop_regex.findall(action))
            + len(sop_regex.findall(situation))
            + len(sop_regex.findall(tags))
        )

        arch_score = (
            len(arch_regex.findall(lesson)) * 3
            + len(arch_regex.findall(title)) * 2
            + len(arch_regex.findall(action))
            + len(arch_regex.findall(situation))
            + len(arch_regex.findall(tags))
        )

        if sop_score > arch_score:
            return "sop"
        elif arch_score > sop_score:
            return "architecture"
        else:
            # Tie breaker: check title and starting words of lesson
            lesson_lower = lesson.lower().strip()
            title_lower = title.lower().strip()
            if any(w in title_lower for w in ("sop", "protocol", "workflow", "guide", "rule", "runbook")):
                return "sop"
            if any(lesson_lower.startswith(w) for w in ("always", "never", "must", "ensure", "do not")):
                return "sop"
            if any(w in lesson_lower for w in ("must ensure", "always verify", "always use")):
                return "sop"
            return "architecture"

    def _format_consolidated_l0(self, category: str, title: str, lesson: str) -> str:
        """Format an atomic, concise (<40 tokens) L0 abstract for a consolidated node."""
        prefix = f"[{category.upper()}] {title}: "
        if estimate_tokens(prefix) > 20:
            words = title.split()[:4]
            prefix = f"[{category.upper()}] {' '.join(words)}: "

        sentences = split_sentences(lesson)
        clean_lesson = sentences[0].strip() if sentences else lesson.strip()
        candidate = f"{prefix}{clean_lesson}"

        if estimate_tokens(candidate) <= 40:
            return candidate

        words = clean_lesson.split()
        for i in range(len(words), 0, -1):
            truncated = f"{prefix}{' '.join(words[:i])}..."
            if estimate_tokens(truncated) <= 40:
                return truncated

        return candidate[:150]

    def _format_consolidated_l1(self, ep: Dict[str, Any], category: str) -> str:
        """Format structured L1 overview markdown for a consolidated node."""
        title = ep.get("title") or "Consolidated Experience"
        lesson = ep.get("lessons_learned") or ""
        situation = ep.get("situation") or "N/A"
        action = ep.get("action_taken") or "N/A"
        outcome = ep.get("outcome") or "N/A"
        ep_id = ep.get("id") or "unknown"
        salience = float(ep.get("salience", 0.5))
        confidence = float(ep.get("confidence", 1.0))
        tags = ep.get("tags") or []
        if isinstance(tags, list):
            tags_str = ", ".join(str(t) for t in tags)
        else:
            tags_str = str(tags)

        return (
            f"# {title}\n\n"
            f"## Permanent Directive ({category.upper()})\n"
            f"{lesson}\n\n"
            f"## Operational Context\n"
            f"- **Situation:** {situation}\n"
            f"- **Action Taken:** {action}\n"
            f"- **Observed Outcome:** {outcome}\n\n"
            f"## Cognitive Provenance\n"
            f"- Source: Consolidated from Episode `{ep_id}`\n"
            f"- Salience: {salience:.2f} | Confidence: {confidence:.2f}\n"
            f"- Tags: {tags_str}\n"
        )

    def consolidate_episodes_to_semantic(
        self,
        limit: int = 10,
        min_salience: float = 0.5,
        update_mental_models: bool = True,
        use_llm: bool = False,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Consolidate active episodic memories into permanent semantic nodes (synapse_nodes).

        Reads episodes from the episodes table, analyzes lessons_learned, categorizes each
        into 'sop' or 'architecture' with high salience (salience >= 0.92) to confer Amigdala
        decay-immunity. Optionally updates core Mental Models with refined directives.

        Args:
            limit: Maximum number of candidate episodes to consolidate per cycle (default 10).
            min_salience: Minimum salience threshold for eligible episodes (default 0.5).
            update_mental_models: Whether to refresh foundational Mental Models (default True).
            use_llm: Whether to attempt LLM refinement (falls back to deterministic heuristics).
            force: If True, re-consolidates episodes even if previously marked consolidated.

        Returns:
            Dictionary with status, counts, node IDs created/updated, and mental model updates.
        """
        limit = max(1, int(limit))
        all_episodes = self.storage.list_consolidatable_episodes(min_salience)

        if not all_episodes:
            return {
                "status": "noop",
                "episodes_analyzed": 0,
                "nodes_created": [],
                "nodes_updated": [],
                "consolidated_count": 0,
                "mental_models_updated": [],
                "edges_created": 0,
            }

        candidates: List[Dict[str, Any]] = []
        for ep in all_episodes:
            ep_id = str(ep["id"])
            if not force:
                if self.storage.find_node_by_episode_ref(ep_id) is not None:
                    continue

            candidates.append(ep)
            if len(candidates) >= limit:
                break

        if not candidates:
            return {
                "status": "noop",
                "episodes_analyzed": len(all_episodes),
                "nodes_created": [],
                "nodes_updated": [],
                "consolidated_count": 0,
                "mental_models_updated": [],
                "edges_created": 0,
            }

        created_nodes: List[str] = []
        updated_nodes: List[str] = []
        consolidated_items: List[Dict[str, Any]] = []
        edges_created = 0
        now = int(time.time())

        for ep in candidates:
            ep_id = str(ep["id"])
            title = str(ep.get("title") or "Consolidated Experience").strip()
            lesson = str(ep.get("lessons_learned") or "").strip()
            category = self._categorize_episode_lesson(ep)

            # High salience for decay immunity (Amigdala protection threshold is 0.90)
            salience = max(0.92, float(ep.get("salience", 0.92)))
            importance = max(0.85, float(ep.get("importance", 0.85)) if "importance" in ep else 0.85)
            confidence = float(ep.get("confidence", 1.0))
            strength = 1.0

            l0 = self._format_consolidated_l0(category, title, lesson)
            l1 = self._format_consolidated_l1(ep, category)

            slug_seed = re.sub(r"[^a-zA-Z0-9\s_-]", "", title).strip()
            words = slug_seed.split()[:6]
            slug = "-".join(w.lower() for w in words) or f"ep_{ep_id[:8]}"
            target_uri = f"synapse://{category}/{slug}"

            existing_by_path = self.storage.get_node_by_l2_target_path(f"episode:{ep_id}")

            if existing_by_path is not None:
                # Update existing consolidated node
                node_id = existing_by_path["id"]
                self.storage.update_node(
                    node_id,
                    l0_abstract=l0,
                    l1_overview=l1,
                    category=category,
                    importance=importance,
                    salience=salience,
                    strength=strength,
                    confidence=confidence,
                    status="active",
                )
                updated_nodes.append(node_id)
            else:
                existing_by_uri = self.storage.get_node(target_uri)
                if existing_by_uri is not None:
                    # Disambiguate URI collision
                    target_uri = f"synapse://{category}/{slug}_{ep_id[:8]}"

                try:
                    node_id = self.storage.add_node(
                        uri=target_uri,
                        l0_abstract=l0,
                        l1_overview=l1,
                        l2_target_path=f"episode:{ep_id}",
                        category=category,
                        importance=importance,
                        strength=strength,
                        salience=salience,
                        confidence=confidence,
                        status="active",
                    )
                    created_nodes.append(node_id)
                except Exception:
                    unique_uri = f"synapse://{category}/{slug}_{uuid.uuid4().hex[:6]}"
                    node_id = self.storage.add_node(
                        uri=unique_uri,
                        l0_abstract=l0,
                        l1_overview=l1,
                        l2_target_path=f"episode:{ep_id}",
                        category=category,
                        importance=importance,
                        strength=strength,
                        salience=salience,
                        confidence=confidence,
                        status="active",
                    )
                    created_nodes.append(node_id)

            consolidated_items.append({
                "node_id": node_id,
                "episode_id": ep_id,
                "title": title,
                "category": category,
                "lessons_learned": lesson,
            })

            # Update tags on episode to mark consolidated
            raw_tags = ep.get("tags")
            tags_list: List[str] = []
            if isinstance(raw_tags, list):
                tags_list = list(raw_tags)
            elif isinstance(raw_tags, str) and raw_tags.strip():
                try:
                    parsed = json.loads(raw_tags)
                    if isinstance(parsed, list):
                        tags_list = parsed
                    else:
                        tags_list = [raw_tags.strip()]
                except Exception:
                    tags_list = [t.strip() for t in raw_tags.split(",") if t.strip()]

            if "consolidated" not in tags_list:
                tags_list.append("consolidated")

            self.storage.update_episode(
                ep_id,
                tags=tags_list,
                last_accessed=now,
            )

            # Build synaptic edges with related semantic nodes via FTS
            try:
                search_matches = self.storage.search_fts(title, limit=3)
                for sm in search_matches:
                    match_id = sm.get("id")
                    if match_id and match_id != node_id:
                        if self.storage.add_edge(node_id, match_id, relation="consolidated_from_episode", weight=0.75):
                            edges_created += 1
                        if self.storage.add_edge(match_id, node_id, relation="consolidated_from_episode", weight=0.75):
                            edges_created += 1
            except Exception:
                pass

        # Link co-consolidated nodes together
        all_active_node_ids = created_nodes + updated_nodes
        if len(all_active_node_ids) >= 2:
            for src, tgt in combinations(all_active_node_ids, 2):
                if self.storage.add_edge(src, tgt, relation="co_consolidated", weight=0.70):
                    edges_created += 1
                if self.storage.add_edge(tgt, src, relation="co_consolidated", weight=0.70):
                    edges_created += 1

        # Optionally refresh Mental Models
        updated_mental_models: List[str] = []
        if update_mental_models and consolidated_items:
            try:
                from synapse.mental_model_engine import MentalModelEngine
                mm_engine = MentalModelEngine(self.storage)
                mm_engine.ensure_default_models()

                conventions_facts = [
                    f"[{item['title']}] {item['lessons_learned']}"
                    for item in consolidated_items
                    if item["category"] in ("architecture", "sop")
                ]

                if conventions_facts:
                    target_id = "core-conventions" if self.storage.get_mental_model("core-conventions") else "master-core-conventions"
                    if mm_engine.refresh_model(target_id, new_facts=conventions_facts):
                        updated_mental_models.append(target_id)
            except Exception:
                pass

        return {
            "status": "success",
            "episodes_analyzed": len(candidates),
            "nodes_created": created_nodes,
            "nodes_updated": updated_nodes,
            "consolidated_count": len(created_nodes) + len(updated_nodes),
            "mental_models_updated": updated_mental_models,
            "edges_created": edges_created,
        }

    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Extract entity keywords, technology names, and identifiers from text."""
        entities: Set[str] = set()
        # 1. Quoted / backtick names
        for m in re.findall(r"[`'\"]([a-zA-Z0-9_\-\.\/]{2,50})[`'\"]", text):
            entities.add(m.strip())
        # 2. Capitalized Multi-word or Single Capitalized Words
        for m in re.findall(r"\b[A-Z][a-zA-Z0-9_]+(?:\s+[A-Z][a-zA-Z0-9_]+)*\b", text):
            if m.lower() not in {
                "the", "a", "an", "this", "that", "we", "i", "you", "user", "assistant",
                "if", "when", "then", "so", "for", "with", "and", "or", "but", "fixed",
                "always", "never", "step", "first", "second", "all", "our"
            }:
                entities.add(m.strip())
        # 3. Known technical patterns
        tech_pattern = r"\b(?:sqlite3?|postgresql|postgres|mysql|redis|fastapi|docker|python|linux|wal|fts5|react|astro|svelte|tailwind|css|html|obsidian|gvisor)\b"
        for m in re.findall(tech_pattern, text, re.IGNORECASE):
            entities.add(m.strip())
        return sorted(entities)

    def _distill_with_llm(
        self, messages: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Call OpenAI-compatible chat completion API to extract structured cognitive items
        (both atomic semantic memories and structured cognitive episodes).
        """
        endpoint = f"{self.llm_config['base_url'].rstrip('/')}/chat/completions"
        convo_parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                convo_parts.append(f"{role.upper()}: {content.strip()}")

        if not convo_parts:
            return None

        prompt_text = "\n".join(convo_parts)

        system_prompt = (
            "You are a cognitive memory distiller. Analyze the conversation and extract:\n"
            "1. Atomic semantic memories (facts, preferences, decisions, architectural choices, standard procedures).\n"
            "2. Structured cognitive episodes (Situation -> Action -> Outcome -> Lesson Learned).\n\n"
            "Return JSON matching this format:\n"
            "{\n"
            '  "memories": [\n'
            "    {\n"
            '      "l0_abstract": "Concise atomic fact (<30 words)",\n'
            '      "l1_overview": "Context or reasoning (<100 words)",\n'
            '      "category": "preference" | "decision" | "architecture" | "incident" | "sop" | "general",\n'
            '      "importance": 0.5 to 1.0,\n'
            '      "salience": 0.5 to 1.0,\n'
            '      "confidence": 0.5 to 1.0,\n'
            '      "entities": ["entity1", "entity2"]\n'
            "    }\n"
            "  ],\n"
            '  "episodes": [\n'
            "    {\n"
            '      "title": "Short descriptive title of experience/incident",\n'
            '      "situation": "Initial problem or context",\n'
            '      "action_taken": "Action or steps taken",\n'
            '      "outcome": "Result or resolution",\n'
            '      "lessons_learned": "Takeaway or cognitive lesson",\n'
            '      "salience": 0.5 to 1.0,\n'
            '      "confidence": 0.5 to 1.0,\n'
            '      "tags": ["tag1", "tag2"]\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "If no durable knowledge is found, return {\"memories\": [], \"episodes\": []}."
        )

        req_body = {
            "model": self.llm_config["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ],
            "temperature": 0.1,
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(req_body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_config['api_key']}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=8.0) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            content = resp_data["choices"][0]["message"]["content"]

            # Extract JSON block if wrapped in markdown code fence
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                parsed = json.loads(content)

            if isinstance(parsed, dict):
                return {
                    "memories": parsed.get("memories", []),
                    "episodes": parsed.get("episodes", []),
                }
            elif isinstance(parsed, list):
                return {"memories": parsed, "episodes": []}

        return None

    def _distill_episodes_rule_based(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rule-based episodic memory extraction (Situation -> Action -> Outcome -> Lesson).
        """
        episodes: List[Dict[str, Any]] = []
        incident_pattern = re.compile(
            r"\b(incident|bug|error|failure|crash|exception|outage|issue|problem|broken|down|corrupt)\b",
            re.IGNORECASE,
        )
        action_pattern = re.compile(
            r"\b(fixed by|resolved by|solution|repaired|implemented|patched|refactored|applied|ran|executed|switched to|migrated)\b",
            re.IGNORECASE,
        )
        lesson_pattern = re.compile(
            r"\b(lesson|takeaway|in the future|make sure|always|never|remember to|rule|must|note that)\b",
            re.IGNORECASE,
        )

        all_text_blocks = [m.get("content", "") for m in messages if isinstance(m.get("content"), str)]
        full_text = " \n ".join(all_text_blocks)
        sentences = split_sentences(full_text)

        situations: List[str] = []
        actions: List[str] = []
        lessons: List[str] = []

        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 10:
                continue
            if incident_pattern.search(s_clean):
                situations.append(s_clean)
            elif action_pattern.search(s_clean):
                actions.append(s_clean)
            elif lesson_pattern.search(s_clean):
                lessons.append(s_clean)

        if situations and (actions or lessons):
            sit = situations[0]
            act = actions[0] if actions else "Investigated and resolved the incident."
            out = "Resolved successfully."
            les = lessons[0] if lessons else "Verify configurations before deploying changes."

            title_words = sit.split()[:6]
            title = " ".join(title_words).strip(".,:;!?")
            if not title:
                title = "System Incident Resolution"

            entities = self._extract_entities_from_text(f"{sit} {act} {les}")

            episodes.append({
                "title": title,
                "situation": sit,
                "action_taken": act,
                "outcome": out,
                "lessons_learned": les,
                "salience": 0.85,
                "confidence": 0.95,
                "tags": entities,
            })

        return episodes

    def _distill_rule_based(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rule-based heuristic fallback for cognitive memory extraction.
        """
        items: List[Dict[str, Any]] = []

        patterns = [
            (
                "preference",
                re.compile(
                    r"\b(prefer\w*|preference\w*|theme\w*|font\w*|editor\w*|profile\w*|always want|i like)\b",
                    re.IGNORECASE,
                ),
                0.7,
            ),
            (
                "decision",
                re.compile(
                    r"\b(decid\w*|decision\w*|keputusan|chose|agreed|selected|adopt\w*|settle\w*|we will use)\b",
                    re.IGNORECASE,
                ),
                0.8,
            ),
            (
                "architecture",
                re.compile(
                    r"\b(architect\w*|struktur|pattern\w*|schema\w*|database\w*|fastapi|sqlite|postgresql|postgres|api|design\w*|layer\w*|component\w*)\b",
                    re.IGNORECASE,
                ),
                0.75,
            ),
            (
                "incident",
                re.compile(
                    r"\b(incident\w*|bug\w*|error\w*|failure\w*|crash\w*|exception\w*|outage\w*|fix\w*|issue\w*|regression\w*|vulnerability|cve)\b",
                    re.IGNORECASE,
                ),
                0.85,
            ),
            (
                "sop",
                re.compile(
                    r"\b(sop\w*|procedure\w*|guide\w*|runbook\w*|playbook\w*|workflow\w*|step\w*|instruction\w*|rule\w*|standard\w*|must ensure|always run)\b",
                    re.IGNORECASE,
                ),
                0.7,
            ),
        ]

        seen_sentences: Set[str] = set()

        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue

            sentences = split_sentences(content)
            for sentence in sentences:
                clean_s = sentence.strip()
                if len(clean_s) < 10 or clean_s in seen_sentences:
                    continue

                for cat, regex, default_imp in patterns:
                    if regex.search(clean_s):
                        seen_sentences.add(clean_s)
                        entities = self._extract_entities_from_text(clean_s)
                        items.append(
                            {
                                "l0_abstract": clean_s,
                                "l1_overview": content.strip(),
                                "category": cat,
                                "importance": default_imp,
                                "entities": entities,
                            }
                        )
                        break

        return items

    def distill_session(
        self, messages: List[Dict[str, Any]], use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Extract atomic facts, categorize them (preference/decision/architecture/incident/sop),
        extract entities, and persist them with synaptic edges into SynapseStorage.
        Also extracts structured cognitive episodes (Situation -> Action -> Outcome -> Lesson).
        Uses OpenAI-compatible LLM via resolve_synapse_llm_config() with robust rule-based fallback.
        """
        if not messages:
            return {
                "status": "empty",
                "distilled_count": 0,
                "nodes_created": [],
                "episodes_created": [],
                "edges_created": 0,
                "entities": [],
                "method": "none",
            }

        extracted_memories: List[Dict[str, Any]] = []
        extracted_episodes: List[Dict[str, Any]] = []
        method_used = "none"

        if use_llm:
            try:
                llm_res = self._distill_with_llm(messages)
                if isinstance(llm_res, dict):
                    extracted_memories = llm_res.get("memories", [])
                    extracted_episodes = llm_res.get("episodes", [])
                    if extracted_memories or extracted_episodes:
                        method_used = "llm"
                elif isinstance(llm_res, list):
                    extracted_memories = llm_res
                    if extracted_memories:
                        method_used = "llm"
            except Exception:
                extracted_memories = []
                extracted_episodes = []

        if not extracted_memories and not extracted_episodes:
            extracted_memories = self._distill_rule_based(messages)
            extracted_episodes = self._distill_episodes_rule_based(messages)
            method_used = "rule_based"

        created_node_ids: List[str] = []
        created_episode_ids: List[str] = []
        all_entities: Set[str] = set()
        node_entity_map: Dict[str, Set[str]] = {}

        now = int(time.time())

        for idx, item in enumerate(extracted_memories):
            l0 = str(item.get("l0_abstract", "")).strip()
            if not l0:
                continue

            # Ensure L0 abstract is concise (<40 tokens)
            if estimate_tokens(l0) > 40:
                sentences = split_sentences(l0)
                if sentences and estimate_tokens(sentences[0]) <= 40:
                    l0 = sentences[0]
                else:
                    words = l0.split()
                    l0 = " ".join(words[:25]) + "..."

            l1 = str(item.get("l1_overview", "")).strip()
            category = str(item.get("category", "general")).lower().strip()
            if category not in VALID_CATEGORIES:
                category = "general"

            importance = float(item.get("importance", 0.6))
            importance = max(0.1, min(1.0, importance))

            salience = float(item.get("salience", 0.5))
            confidence = float(item.get("confidence", 1.0))

            # Entities
            item_entities: Set[str] = set()
            raw_ents = item.get("entities") or []
            if isinstance(raw_ents, list):
                for e in raw_ents:
                    if isinstance(e, str) and e.strip():
                        item_entities.add(e.strip())
            # Augment with rule-based entity extraction
            item_entities.update(self._extract_entities_from_text(f"{l0} {l1}"))

            all_entities.update(item_entities)

            # Generate URI
            slug_seed = re.sub(r"[^a-zA-Z0-9\s_-]", "", l0).strip()
            words = slug_seed.split()[:5]
            slug = "-".join(w.lower() for w in words) or f"fact_{now}_{idx}"
            uri = f"synapse://{category}/{slug}"

            existing = self.storage.get_node(uri)
            if existing is not None:
                node_id = existing["id"]
                self.storage.update_node(
                    node_id,
                    l0_abstract=l0,
                    l1_overview=l1,
                    importance=importance,
                    salience=salience,
                    confidence=confidence,
                )
            else:
                try:
                    node_id = self.storage.add_node(
                        uri=uri,
                        l0_abstract=l0,
                        l1_overview=l1,
                        category=category,
                        importance=importance,
                        salience=salience,
                        confidence=confidence,
                    )
                except Exception:
                    uri_unique = f"synapse://{category}/{slug}_{uuid.uuid4().hex[:6]}"
                    node_id = self.storage.add_node(
                        uri=uri_unique,
                        l0_abstract=l0,
                        l1_overview=l1,
                        category=category,
                        importance=importance,
                        salience=salience,
                        confidence=confidence,
                    )

            created_node_ids.append(node_id)
            node_entity_map[node_id] = item_entities

        # Persist structured episodic memories
        for ep in extracted_episodes:
            title = str(ep.get("title") or "Experience").strip()
            situation = str(ep.get("situation") or "").strip()
            action_taken = str(ep.get("action_taken") or "").strip()
            outcome = str(ep.get("outcome") or "").strip()
            lessons_learned = str(ep.get("lessons_learned") or "").strip()
            salience = float(ep.get("salience", 0.5))
            confidence = float(ep.get("confidence", 1.0))
            tags = ep.get("tags") or []
            if isinstance(tags, list):
                for t in tags:
                    if isinstance(t, str) and t.strip():
                        all_entities.add(t.strip())

            ep_id = self.storage.add_episode(
                title=title,
                situation=situation,
                action_taken=action_taken,
                outcome=outcome,
                lessons_learned=lessons_learned,
                salience=salience,
                confidence=confidence,
                tags=tags,
            )
            created_episode_ids.append(ep_id)

        # Create Synaptic Edges between co-distilled nodes in this session
        edges_created = 0
        if len(created_node_ids) >= 2:
            for src, tgt in combinations(created_node_ids, 2):
                ok1 = self.storage.add_edge(src, tgt, relation="session_distilled", weight=0.6)
                ok2 = self.storage.add_edge(tgt, src, relation="session_distilled", weight=0.6)
                if ok1:
                    edges_created += 1
                if ok2:
                    edges_created += 1

                # Shared entity relation
                shared = node_entity_map.get(src, set()) & node_entity_map.get(tgt, set())
                if shared:
                    ok3 = self.storage.add_edge(src, tgt, relation="shares_entity", weight=0.75)
                    ok4 = self.storage.add_edge(tgt, src, relation="shares_entity", weight=0.75)
                    if ok3:
                        edges_created += 1
                    if ok4:
                        edges_created += 1

        return {
            "status": "success" if (created_node_ids or created_episode_ids) else "empty",
            "distilled_count": len(created_node_ids) + len(created_episode_ids),
            "nodes_created": created_node_ids,
            "episodes_created": created_episode_ids,
            "edges_created": edges_created,
            "entities": sorted(list(all_entities)),
            "method": method_used,
        }
