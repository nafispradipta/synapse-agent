"""Consolidated knowledge: mental models, observations, and sandboxed research."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional


class KnowledgeStore:
    """Derived knowledge artefacts rather than raw memories."""

    def set_mental_model(
        self,
        model_id: str,
        name: str,
        source_query: str,
        content: str,
        structured_content: Optional[str] = None,
        trigger_mode: str = "delta",
        tags: Optional[List[str]] = None,
        max_tokens: int = 2048,
    ) -> bool:
        """Create or update a consolidated mental model."""
        now = int(time.time())
        tags_json = json.dumps(tags or [])
        with self._write():
            cursor = self._conn.execute(
                "SELECT content, history FROM synapse_mental_models WHERE id = ?",
                (model_id,),
            )
            row = cursor.fetchone()
            if row:
                old_content = row["content"]
                try:
                    history = json.loads(row["history"] or "[]")
                except Exception:
                    history = []
                if old_content and old_content != content:
                    history.append({"content": old_content, "archived_at": now})
                    history = history[-10:]  # Cap last 10 versions

                self._conn.execute(
                    """
                    UPDATE synapse_mental_models
                    SET name = ?, source_query = ?, content = ?, structured_content = ?,
                        trigger_mode = ?, tags = ?, max_tokens = ?, last_refreshed_at = ?,
                        last_refreshed_source_query = ?, history = ?
                    WHERE id = ?
                    """,
                    (
                        name,
                        source_query,
                        content,
                        structured_content,
                        trigger_mode,
                        tags_json,
                        max_tokens,
                        now,
                        source_query,
                        json.dumps(history),
                        model_id,
                    ),
                )
            else:
                self._conn.execute(
                    """
                    INSERT INTO synapse_mental_models (
                        id, name, source_query, content, structured_content,
                        trigger_mode, tags, max_tokens, last_refreshed_at,
                        last_refreshed_source_query, history, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        model_id,
                        name,
                        source_query,
                        content,
                        structured_content,
                        trigger_mode,
                        tags_json,
                        max_tokens,
                        now,
                        source_query,
                        "[]",
                        now,
                    ),
                )
            self._conn.commit()
            return True

    def get_mental_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a mental model by ID."""
        with self._write():
            cursor = self._conn.execute(
                "SELECT * FROM synapse_mental_models WHERE id = ?",
                (model_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            try:
                res["tags"] = json.loads(res.get("tags") or "[]")
            except Exception:
                res["tags"] = []
            return res

    def list_mental_models(self) -> List[Dict[str, Any]]:
        """List all mental models."""
        with self._write():
            cursor = self._conn.execute(
                "SELECT id, name, source_query, content, trigger_mode, tags, last_refreshed_at, created_at FROM synapse_mental_models ORDER BY name ASC"
            )
            rows = cursor.fetchall()
            results: List[Dict[str, Any]] = []
            for r in rows:
                item = dict(r)
                try:
                    item["tags"] = json.loads(item.get("tags") or "[]")
                except Exception:
                    item["tags"] = []
                results.append(item)
            return results

    def add_or_update_observation(
        self,
        pattern: str,
        supporting_quote: str,
        category: str = "general",
        confidence: float = 1.0,
        observation_id: Optional[str] = None,
        proof_delta: int = 1,
    ) -> Dict[str, Any]:
        """
        Record or reinforce an evidence-grounded observation.
        If a matching pattern already exists, increment proof_count and append quote.
        """
        now = int(time.time())
        with self._write():
            if observation_id:
                cursor = self._conn.execute(
                    "SELECT * FROM synapse_observations WHERE id = ?",
                    (observation_id,),
                )
            else:
                cursor = self._conn.execute(
                    "SELECT * FROM synapse_observations WHERE pattern = ? AND status = 'active'",
                    (pattern,),
                )
            existing = cursor.fetchone()

            if existing:
                obs_id = existing["id"]
                current_proof = int(existing["proof_count"]) + max(1, int(proof_delta))
                try:
                    facts = json.loads(existing["supporting_facts"] or "[]")
                except Exception:
                    facts = []
                if supporting_quote and supporting_quote not in facts:
                    facts.append(supporting_quote)
                    facts = facts[-15:]  # Keep latest 15 exact quotes

                new_confidence = min(1.0, float(existing["confidence"]) + 0.05)
                self._conn.execute(
                    """
                    UPDATE synapse_observations
                    SET proof_count = ?, supporting_facts = ?, confidence = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (current_proof, json.dumps(facts), new_confidence, now, obs_id),
                )
                self._conn.commit()
                return {
                    "id": obs_id,
                    "pattern": pattern,
                    "proof_count": current_proof,
                    "confidence": new_confidence,
                    "status": existing["status"],
                    "action": "reinforced",
                }
            else:
                obs_id = observation_id or f"obs_{uuid.uuid4().hex[:10]}"
                facts = [supporting_quote] if supporting_quote else []
                self._conn.execute(
                    """
                    INSERT INTO synapse_observations (
                        id, pattern, supporting_facts, proof_count, confidence,
                        status, category, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        obs_id,
                        pattern,
                        json.dumps(facts),
                        1,
                        confidence,
                        "active",
                        category,
                        now,
                        now,
                    ),
                )
                self._conn.commit()
                return {
                    "id": obs_id,
                    "pattern": pattern,
                    "proof_count": 1,
                    "confidence": confidence,
                    "status": "active",
                    "action": "created",
                }

    def list_observations(
        self,
        category: Optional[str] = None,
        status: str = "active",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Retrieve active observations ordered by proof_count descending."""
        with self._write():
            if category:
                cursor = self._conn.execute(
                    """
                    SELECT * FROM synapse_observations
                    WHERE status = ? AND category = ?
                    ORDER BY proof_count DESC, updated_at DESC LIMIT ?
                    """,
                    (status, category, limit),
                )
            else:
                cursor = self._conn.execute(
                    """
                    SELECT * FROM synapse_observations
                    WHERE status = ?
                    ORDER BY proof_count DESC, updated_at DESC LIMIT ?
                    """,
                    (status, limit),
                )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                try:
                    item["supporting_facts"] = json.loads(item.get("supporting_facts") or "[]")
                except Exception:
                    item["supporting_facts"] = []
                results.append(item)
            return results

    def add_external_research(
        self,
        topic: str,
        title: str,
        source_url: str,
        domain: str,
        summary: str,
        raw_snippet: str = "",
        vault_path: str = "",
        status: str = "quarantined",
    ) -> Dict[str, Any]:
        """Record sandboxed Tier 4 external research into synapse_external_research."""
        research_id = f"res:{uuid.uuid4().hex[:10]}"
        now = int(time.time())
        with self._write():
            self._conn.execute(
                """
                INSERT INTO synapse_external_research (
                    id, topic, title, source_url, domain, summary, raw_snippet, tier_level, vault_path, fetched_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 4, ?, ?, ?)
                """,
                (research_id, topic, title, source_url, domain, summary, raw_snippet, vault_path, now, status),
            )
            self._conn.commit()
            return {
                "id": research_id,
                "topic": topic,
                "title": title,
                "source_url": source_url,
                "domain": domain,
                "summary": summary,
                "tier_level": 4,
                "vault_path": vault_path,
                "status": status,
            }

    def list_external_research(
        self,
        topic: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List sandboxed external research records."""
        with self._write():
            query = "SELECT * FROM synapse_external_research WHERE 1=1"
            params: List[Any] = []
            if topic:
                query += " AND topic = ?"
                params.append(topic)
            if status:
                query += " AND status = ?"
                params.append(status)
            query += " ORDER BY fetched_at DESC LIMIT ?"
            params.append(limit)

            cursor = self._conn.execute(query, tuple(params))
            return [dict(r) for r in cursor.fetchall()]

    def promote_external_research(
        self,
        research_id: str,
        target_tier: int = 1,
        mental_model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Promotion Gate: Elevate a quarantined Tier 4 research item to Tier 1 Mental Model
        or Tier 2 Observation upon user confirmation or high usage count.
        """
        with self._write():
            cur = self._conn.execute(
                "SELECT * FROM synapse_external_research WHERE id = ?", (research_id,)
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Research record '{research_id}' not found.")
            res = dict(row)

            # Update status in research table
            self._conn.execute(
                "UPDATE synapse_external_research SET status = 'verified', tier_level = ? WHERE id = ?",
                (target_tier, research_id),
            )
            self._conn.commit()

        # If promoting to Tier 1 Mental Model
        if target_tier == 1:
            mm_target = mental_model_id or "master-user-preferences"
            # Return instructions for caller engine
            return {
                "action": "promoted",
                "target_tier": 1,
                "research": res,
                "mental_model_id": mm_target,
                "fact_snippet": f"Verified Best Practice ({res['title']}): {res['summary'][:200]}",
            }

        # If promoting to Tier 2 Observation
        obs_pattern = f"Validated Technical Standard: {res['title']} ({res['topic']})"
        return {
            "action": "promoted",
            "target_tier": 2,
            "research": res,
            "observation_pattern": obs_pattern,
            "proof_count": 3,
        }
