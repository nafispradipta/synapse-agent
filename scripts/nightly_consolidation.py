#!/usr/bin/env python3
"""Autonomous Nightly Cognitive Consolidation Engine for Synapse Memory.
Executes scheduled maintenance to reinforce active memories, prune decayed entries,
sync markdown notes, optimize FTS5 indices, and verify DB integrity.
Built with atomic transactions, double integrity checks, and high-salience protection.
"""
import os
import sys
import glob
import json
import sqlite3
import datetime
from typing import Any, Dict, List, Optional

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from synapse.embedder import SynapseEmbedder, fallback_vectorize
    _embedder = SynapseEmbedder()
except Exception:
    _embedder = None
    import math
    import zlib
    def fallback_vectorize(text: str, dim: int = 384):
        clean = text.lower().strip()
        words = [w for w in clean.split() if w]
        if not words:
            return [0.0] * dim
        vec = [0.0] * dim
        for w in words:
            h = zlib.crc32(w.encode("utf-8"))
            bucket = h % dim
            sign = 1.0 if ((h >> 16) & 1) else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm > 0.0 else vec

BASE_DIR = str(PROJECT_ROOT)
DATA_DIR = os.environ.get("SYNAPSE_DATA_DIR", os.path.join(BASE_DIR, "data"))
LOGS_DIR = os.environ.get("SYNAPSE_LOGS_DIR", os.path.join(BASE_DIR, "logs"))
DB_PATH = os.environ.get("SYNAPSE_DB_PATH", os.path.join(DATA_DIR, "synapse.db"))
VAULT_PROJECTS_DIR = os.environ.get(
    "SYNAPSE_VAULT_PATH",
    os.path.expanduser("~/vault/projects") if os.path.exists(os.path.expanduser("~/vault/projects")) else os.path.expanduser("~/vault/projek")
)

os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "consolidation.log")

def get_timestamp_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def get_utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def log(msg):
    now_str = get_timestamp_now()
    entry = f"[{now_str}] {msg}"
    print(entry)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")

def run_consolidation(
    db_path: Optional[str] = None,
    skip_external: bool = False,
    skip_vault_sync: bool = False,
):
    log("=== [START] Autonomous Nightly Synapse Consolidation ===")
    target_db = db_path or DB_PATH
    if not os.path.exists(target_db):
        log(f"CRITICAL ERROR: Database not found at {target_db}")
        sys.exit(1)

    conn = sqlite3.connect(target_db)
    cur = conn.cursor()

    try:
        # Step 1: Pre-Execution Integrity Check
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA busy_timeout=10000;")
        cur.execute("PRAGMA integrity_check")
        integrity_pre = cur.fetchone()[0]
        if integrity_pre != "ok":
            raise RuntimeError(f"Pre-check database integrity failed: {integrity_pre}")
        log("Step 1: Pre-Execution DB Integrity -> OK (WAL Mode Verified)")

        # Step 1.5: Trigger Micro-Episodic Watermark Extractor (Sync chat turns)
        if not skip_external:
            try:
                extractor_script = os.path.join(BASE_DIR, "scripts", "episodic_watermark_extractor.py")
                if os.path.exists(extractor_script):
                    log("Step 1.5: Syncing latest conversation turns via episodic extractor...")
                    os.system(f"/usr/bin/python3 {extractor_script}")
            except Exception as e:
                log(f"Warning on Step 1.5 episodic extraction: {e}")

        now_iso = get_utc_now()

        with conn:
            # Step 2: Amigdala Protection & Hard Reinforcement
            # Hard-lock all critical directives, preferences, and decisions to strength 1.0
            cur.execute("""
                UPDATE synapse_nodes
                SET strength = 1.0, confidence = 1.0
                WHERE salience >= 0.90 OR category IN ('preference', 'decision')
            """)
            protected_count = cur.rowcount
            log(f"Step 2: Hard-locked Amigdala Salience nodes (Salience >= 0.90): {protected_count} nodes")

        # Step 2.5: Episodic-to-Semantic Knowledge Consolidation
        try:
            sys.path.insert(0, BASE_DIR)
            from synapse.worker import SynapseWorker
            from synapse.storage import SynapseStorage

            log("Step 2.5: Consolidating active episodic experiences into permanent semantic nodes...")
            worker_storage = SynapseStorage(DB_PATH)
            try:
                worker = SynapseWorker(storage=worker_storage)
                consolidation_stats = worker.consolidate_episodes_to_semantic(limit=10)
                c_count = consolidation_stats.get("consolidated_count", 0)
                mm_count = len(consolidation_stats.get("mental_models_updated", []))
                status_str = consolidation_stats.get("status", "noop")
                log(f"Step 2.5: Episodic Consolidation complete ({status_str}) -> {c_count} node(s) consolidated, {mm_count} mental model(s) updated")
            finally:
                worker_storage.close()
        except Exception as e:
            log(f"Step 2.5 Warning: Episodic-to-semantic consolidation error: {e}")

        with conn:
            # Step 3: Hebbian Decay Calculation (Only for non-critical, unaccessed dormant nodes)
            # Handle mixed last_accessed types: integer unix timestamps AND ISO text strings
            import time as _time
            _cutoff_ts = int(_time.time()) - (14 * 86400)  # 14 days ago as unix timestamp
            _cutoff_iso = datetime.datetime.fromtimestamp(_cutoff_ts, tz=datetime.timezone.utc).isoformat()
            cur.execute("""
                UPDATE synapse_nodes
                SET strength = MAX(0.05, strength - 0.05)
                WHERE salience < 0.80
                  AND category NOT IN ('preference', 'decision', 'architecture', 'sop')
                  AND status = 'active'
                  AND (
                    (typeof(last_accessed) = 'integer' AND last_accessed < ?)
                    OR (typeof(last_accessed) = 'text' AND last_accessed < ?)
                  )
            """, (_cutoff_ts, _cutoff_iso))
            decayed_count = cur.rowcount
            log(f"Step 3: Applied Hebbian Decay to dormant nodes (>14 days idle): {decayed_count} nodes")

            # Step 4: Strict Safe Pruning (Only exhausted nodes with zero significance)
            cur.execute("""
                DELETE FROM synapse_nodes
                WHERE salience < 0.25
                  AND strength <= 0.05
                  AND category NOT IN ('preference', 'decision', 'architecture', 'sop')
            """)
            pruned_count = cur.rowcount
            log(f"Step 4: Safely pruned dead/exhausted nodes: {pruned_count} nodes")

            # Step 4.2: Automated Edge Decay & Pruning (Karpathy Graph Hygiene)
            # Prune weak edges (<0.15) whose connected nodes are not high-salience (>=0.90)
            cur.execute("""
                DELETE FROM synapse_edges
                WHERE weight < 0.15
                  AND source_id NOT IN (SELECT id FROM synapse_nodes WHERE salience >= 0.90)
                  AND target_id NOT IN (SELECT id FROM synapse_nodes WHERE salience >= 0.90)
            """)
            pruned_edges = cur.rowcount
            log(f"Step 4.2: Safely pruned weak associative edges (< 0.15 weight): {pruned_edges} edges")

        # Step 4.3: Abstract Rollup of Superseded Nodes (Maintain sub-10ms indices)
        try:
            sys.path.insert(0, BASE_DIR)
            from synapse.worker import SynapseWorker
            from synapse.storage import SynapseStorage
            rollup_storage = SynapseStorage(target_db)
            try:
                worker = SynapseWorker(storage=rollup_storage)
                rollup_res = worker.rollup_superseded_nodes(min_count=50, backup_snapshot=True)
                if rollup_res.get("status") == "success":
                    log(f"Step 4.3: Abstract Rollup compressed {rollup_res['rolled_up_count']} superseded nodes into master entity '{rollup_res['master_entity_id']}'")
                else:
                    log(f"Step 4.3: Abstract Rollup -> {rollup_res.get('message', 'noop')}")
            finally:
                rollup_storage.close()
        except Exception as e:
            log(f"Step 4.3 Warning: Abstract Rollup error: {e}")

        # Step 4.5: Automated MOC Index Bookkeeper (Karpathy LLM Wiki - Zero Orphan Notes)
        vault_dir_path = os.environ.get("SYNAPSE_VAULT_PATH") or (os.path.expanduser("~/vault") if os.path.isdir(os.path.expanduser("~/vault")) else None)
        if vault_dir_path and os.path.isdir(vault_dir_path):
            try:
                sys.path.insert(0, BASE_DIR)
                from synapse.moc_bookkeeper import run_moc_bookkeeper_step

                log(f"Step 4.5: Executing Automated MOC Index Bookkeeper across Vault ({vault_dir_path})...")
                moc_stats = run_moc_bookkeeper_step(vault_dir=vault_dir_path, log_fn=log)
                total_added = sum(len(notes) for notes in moc_stats.values())
                for domain_name, notes_added in moc_stats.items():
                    if notes_added:
                        log(f"Step 4.5: Domain '{domain_name}' -> added {len(notes_added)} note(s) to MOC: {notes_added}")
                log(f"Step 4.5: MOC Bookkeeper complete -> {total_added} orphan note(s) indexed")
            except Exception as e:
                log(f"Step 4.5 Warning: Automated MOC Bookkeeper error: {e}")
        else:
            log("Step 4.5: Vault directory not specified or not found. Skipping MOC Bookkeeper.")

        with conn:
            # Step 5: Auto-Ingest & Synchronize Vault Notes with FTS (Full Vault Scan)
            if not skip_vault_sync and vault_dir_path and os.path.isdir(vault_dir_path):
                vault_notes = glob.glob(os.path.join(vault_dir_path, "**", "*.md"), recursive=True)
                synced_vault = 0
                import time as _time
                _now_ts = int(_time.time())
                for note_path in vault_notes:
                    if "/_arsip/" in note_path or "/_archive/" in note_path or "/." in note_path or "/daily/" in note_path:
                        continue
                    rel_path = os.path.relpath(note_path, vault_dir_path)
                    node_id = f"vault:{rel_path.replace('/', ':').replace('.md', '')}"
                    
                    try:
                        with open(note_path, "r", encoding="utf-8") as nf:
                            content = nf.read()
                        
                        lines = [l.strip() for l in content.split("\n") if l.strip()]
                        title = rel_path
                        for l in lines:
                            if l.startswith("# "):
                                title = l.replace("# ", "").strip()
                                break
                        
                        l0 = title
                        l1 = "\n".join(lines[:6])[:300]
                        
                        category = "general"
                        rel_lower = rel_path.lower()
                        if "agent" in rel_lower or "engineering" in rel_lower:
                            category = "architecture"
                        elif "sop" in rel_lower:
                            category = "sop"
                        elif "server" in rel_lower or "cybersecurity" in rel_lower:
                            category = "architecture"
                        elif "finance" in rel_lower or "business" in rel_lower:
                            category = "decision"
                        elif "marketing" in rel_lower or "seo" in rel_lower:
                            category = "general"

                        # ON CONFLICT: update content but preserve last_accessed
                        # so decay cycle can properly age nodes not actively recalled
                        cur.execute("""
                            INSERT INTO synapse_nodes (id, uri, l0_abstract, l1_overview, l2_target_path, category, importance, strength, created_at, last_accessed, status, salience, confidence)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(uri) DO UPDATE SET
                                l0_abstract = excluded.l0_abstract,
                                l1_overview = excluded.l1_overview,
                                l2_target_path = excluded.l2_target_path,
                                status = 'active'
                        """, (node_id, f"vault://{rel_path}", l0, l1, note_path, category, 0.90, 1.0, _now_ts, _now_ts, "active", 0.95, 1.0))

                        # Auto-vectorization into synapse_vectors (384-dim)
                        try:
                            cur.execute("SELECT id FROM synapse_nodes WHERE uri = ?", (f"vault://{rel_path}",))
                            row_id = cur.fetchone()
                            target_id = row_id[0] if row_id else node_id

                            text_to_embed = f"{l0} {l1}".strip()
                            if _embedder is not None and text_to_embed:
                                vec = _embedder.embed(text_to_embed)
                            else:
                                vec = fallback_vectorize(text_to_embed, dim=384)

                            if vec:
                                cur.execute("""
                                    INSERT INTO synapse_vectors (id, vector_json, dim, updated_at)
                                    VALUES (?, ?, ?, ?)
                                    ON CONFLICT(id) DO UPDATE SET
                                        vector_json = excluded.vector_json,
                                        dim = excluded.dim,
                                        updated_at = excluded.updated_at
                                """, (target_id, json.dumps(vec), len(vec), _now_ts))
                        except Exception as ve:
                            log(f"Warning: Vector generation skipped for {node_id}: {ve}")
                        
                        synced_vault += 1
                    except Exception as e:
                        log(f"Warning: Sync skipped for {note_path}: {e}")

                log(f"Step 5: Vault Notes Ingestion -> {synced_vault} markdown notes synced")
            conn.commit()

        # Step 5.5: Semantic Graph Association (LLM Reflection) - Outside transaction block
        if not skip_external:
            try:
                graph_script = os.path.join(BASE_DIR, "scripts", "enrich_semantic_graph.py")
                if os.path.exists(graph_script):
                    log("Step 5.5: Synthesizing new associative knowledge graph edges...")
                    os.system(f"{sys.executable} {graph_script}")
            except Exception as e:
                log(f"Warning on Step 5.5 semantic graph: {e}")

        # Step 5.6: Auto-Contradiction Resolver & Mass Audit (Self-Correcting Engine)
        try:
            log("Step 5.6: Executing Auto-Contradiction Resolver & Mass Audit...")
            run_contradiction_audit_step(target_db, log_fn=log)
        except Exception as e:
            log(f"Step 5.6 Warning: Auto-contradiction resolver error: {e}")

        with conn:
            # Step 6: Optimize FTS5 Search Indices
            try:
                cur.execute("INSERT INTO synapse_fts(synapse_fts) VALUES('optimize')")
            except Exception as e:
                log(f"Note on FTS optimize: {e}")
            
            try:
                cur.execute("INSERT INTO episodes_fts(episodes_fts) VALUES('optimize')")
            except Exception:
                pass

        # Step 7: VACUUM & Post-Execution Verification
        cur.execute("VACUUM")
        cur.execute("PRAGMA integrity_check")
        integrity_post = cur.fetchone()[0]
        if integrity_post != "ok":
            raise RuntimeError(f"Post-execution integrity check failed: {integrity_post}")
        
        db_size_mb = os.path.getsize(target_db) / (1024 * 1024)
        log(f"Step 6 & 7: VACUUM & Post-Integrity Verified (OK). Final DB Size: {db_size_mb:.2f} MB")

    except Exception as e:
        log(f"CRITICAL ROLLBACK: An error occurred during consolidation: {e}")
        conn.rollback()
        conn.close()
        sys.exit(1)
    finally:
        conn.close()

    # Step 8: Trigger LLM Metacognitive Reflection Layer
    if not skip_external:
        log("Step 8: Invoking Layer 2 (LLM Metacognitive Reflection Engine)...")
        try:
            reflection_script = os.path.join(BASE_DIR, "scripts", "llm_reflection_engine.py")
            if os.path.exists(reflection_script):
                ret = os.system(f"/usr/bin/python3 {reflection_script}")
                if ret == 0:
                    log("Step 8: Layer 2 LLM Reflection completed successfully.")
                else:
                    log(f"Step 8 Warning: Layer 2 returned non-zero code ({ret}).")
            else:
                log("Step 8: Reflection script not found, skipping.")
        except Exception as e:
            log(f"Step 8 Warning: Layer 2 invocation error: {e}")

        # Step 9: Autonomous Tech Stack Harvester (Tier 4 External Research)
        log("Step 9: Invoking Tier 4 Autonomous Web Harvester for Core Tech Stacks...")
        try:
            harvester_script = os.path.join(BASE_DIR, "scripts", "nightly_harvester.py")
            if os.path.exists(harvester_script):
                ret = os.system(f"/usr/bin/python3 {harvester_script}")
                if ret == 0:
                    log("Step 9: Tier 4 Web Harvester completed successfully.")
                else:
                    log(f"Step 9 Warning: Harvester returned non-zero code ({ret}).")
            else:
                log("Step 9: Harvester script not found, skipping.")
        except Exception as e:
            log(f"Step 9 Warning: Harvester invocation error: {e}")

        # Step 9.5: Auto-Spawn Mental Model Detection & Creation
        log("Step 9.5: Auto-Spawn Mental Model Detection & Creation...")
        try:
            sys.path.insert(0, BASE_DIR)
            from synapse.storage import SynapseStorage
            from synapse.mental_model_spawner import run_auto_spawn_step

            spawner_storage = SynapseStorage(target_db)
            try:
                created_count = run_auto_spawn_step(storage=spawner_storage, log_fn=log)
            finally:
                spawner_storage.close()
            log(f"Step 9.5: Auto-Spawn complete -> {created_count} new mental model(s) created.")
        except Exception as e:
            log(f"Step 9.5 Warning: Auto-spawn mental model detection error: {e}")

    # Step 10: Obsidian Vault MOC Index Bookkeeper (Zero Orphan Policy)
    log("Step 10: Running Obsidian Vault MOC Index Bookkeeper...")
    try:
        run_moc_update(log_fn=log)
    except Exception as e:
        log(f"Step 10 Warning: MOC Bookkeeper execution error: {e}")

    log("=== [SUCCESS] Nightly Synapse Consolidation Cycle Finished ===")

def run_contradiction_audit_step(db_path: Optional[str] = None, log_fn=None) -> List[Dict[str, Any]]:
    """Execute Auto-Contradiction Resolver mass audit on target database."""
    _logger = log_fn or log
    target_db = db_path or DB_PATH
    _logger(f"Executing Auto-Contradiction Resolver mass audit on {target_db}...")
    sys.path.insert(0, BASE_DIR)
    from synapse.contradiction_resolver import ContradictionResolver
    from synapse.storage import SynapseStorage

    audit_storage = SynapseStorage(target_db)
    try:
        resolver = ContradictionResolver(storage=audit_storage)
        contradictions = resolver.audit_all_contradictions()
        _logger(f"Contradiction audit complete -> {len(contradictions)} contradiction(s) resolved (outdated facts superseded)")
        return contradictions
    finally:
        audit_storage.close()

def run_moc_update(vault_dir: Optional[str] = None, log_fn=None) -> Dict[str, List[str]]:
    """Synchronize Map of Content (MOC) index files across Obsidian Vault domains."""
    _logger = log_fn or log
    _logger("Executing MOC Index Bookkeeper across Obsidian Vault domains...")
    try:
        sys.path.insert(0, BASE_DIR)
        from synapse.moc_bookkeeper import run_moc_bookkeeper_step

        results = run_moc_bookkeeper_step(vault_dir=vault_dir, log_fn=_logger)
        total = sum(len(items) for items in results.values())
        _logger(f"MOC Bookkeeper completed -> {total} new note(s) indexed: {results}")
        return results
    except Exception as e:
        _logger(f"Warning: MOC Bookkeeper execution error: {e}")
        return {}

if __name__ == "__main__":
    run_consolidation()
