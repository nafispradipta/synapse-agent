"""
Metacognitive Uncertainty Scorer for Cognitive Synapse Core.
Evaluates memory epistemic confidence, staleness, proof counts,
factual contradictions, and salience protection for master directives.
Annotates prompt context with live verification tags when uncertainty is high.
"""

from __future__ import annotations

import copy
import datetime
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def parse_timestamp(val: Any) -> float:
    """
    Parse a timestamp (int, float, numeric string, or ISO formatted string)
    into a float epoch timestamp. Returns 0.0 if unparseable.
    """
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        clean_str = val.strip()
        if not clean_str:
            return 0.0
        try:
            return float(clean_str)
        except ValueError:
            pass
        try:
            # Handle ISO string with or without trailing Z / offset
            iso_str = clean_str.replace("Z", "+00:00")
            return datetime.datetime.fromisoformat(iso_str).timestamp()
        except Exception:
            pass
    return 0.0


@dataclass
class MetacognitiveConfig:
    """
    Dynamic configuration for Metacognitive Uncertainty Scorer.
    Zero hardcoded values: all weights, thresholds, and templates are parameterized.
    """
    # Factor weights in raw uncertainty calculation
    freshness_weight: float = 0.50
    proof_count_weight: float = 0.50
    contradiction_weight: float = 0.30

    # Freshness / staleness decay configuration
    freshness_half_life_days: float = 30.0
    unknown_age_uncertainty: float = 0.40

    # Evidence & proof count configuration
    target_proof_count: int = 5
    default_proof_count: int = 1

    # Contradiction configuration
    contradiction_penalty: float = 1.00
    contradiction_min_uncertainty: float = 0.85  # minimum uncertainty floor when contradiction exists

    # Salience dampening & Master Directive protection
    salience_master_threshold: float = 0.95
    master_salience_dampener: float = 0.02        # dampening multiplier for salience >= 0.95
    max_master_uncertainty: float = 0.05          # hard ceiling for master directive uncertainty (near 0)
    salience_dampening_weight: float = 0.40       # proportional dynamic dampening for salience < 0.95

    # Uncertainty & confidence classification thresholds
    uncertainty_threshold: float = 0.60           # threshold above which is_uncertain = True
    confidence_high_cutoff: float = 0.35          # uncertainty <= 0.35 -> confidence "HIGH"
    confidence_medium_cutoff: float = 0.60        # uncertainty <= 0.60 -> confidence "MEDIUM", else "LOW"

    # Tags & Context annotation directives
    action_advice: str = "MUST VERIFY LIVE"
    tag_template: str = "[UNCERTAINTY: {level} - {advice}]"
    high_uncertainty_tag: Optional[str] = None
    medium_uncertainty_tag: Optional[str] = None

    def get_tag(self, level: str) -> str:
        """Get or dynamically format tag for a given uncertainty level."""
        level_upper = level.upper()
        if level_upper == "HIGH":
            if self.high_uncertainty_tag is not None:
                return self.high_uncertainty_tag
            return self.tag_template.format(level="HIGH", advice=self.action_advice)
        elif level_upper == "MEDIUM":
            if self.medium_uncertainty_tag is not None:
                return self.medium_uncertainty_tag
            return self.tag_template.format(level="MEDIUM", advice=self.action_advice)
        else:
            return self.tag_template.format(level=level_upper, advice=self.action_advice)


@dataclass
class UncertaintyReport(dict):
    """
    Dataclass and Dictionary hybrid representing the uncertainty evaluation report.
    Supports both attribute access (.uncertainty_score) and mapping access (['uncertainty_score']).
    """
    uncertainty_score: float
    confidence_level: str
    is_uncertain: bool
    reasons: List[str] = field(default_factory=list)
    tag: Optional[str] = None
    uncertainty_level: Optional[str] = None
    breakdown: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(
            uncertainty_score=self.uncertainty_score,
            confidence_level=self.confidence_level,
            is_uncertain=self.is_uncertain,
            reasons=list(self.reasons),
            tag=self.tag,
            uncertainty_level=self.uncertainty_level,
            breakdown=dict(self.breakdown),
        )

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if hasattr(self, "__dataclass_fields__") and name in self.__dataclass_fields__:
            dict.__setitem__(self, name, value)

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        if hasattr(self, "__dataclass_fields__") and key in self.__dataclass_fields__:
            super().__setattr__(key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Export report as standard dictionary."""
        return {
            "uncertainty_score": self.uncertainty_score,
            "confidence_level": self.confidence_level,
            "is_uncertain": self.is_uncertain,
            "reasons": list(self.reasons),
            "tag": self.tag,
            "uncertainty_level": self.uncertainty_level,
            "breakdown": dict(self.breakdown),
        }


class MetacognitiveScorer:
    """
    Metacognitive Uncertainty Scorer for evaluating retrieved memory reliability.
    Scores memories dynamically across 4 core dimensions:
      1. Freshness / Staleness (age decay since created or last accessed)
      2. Proof Count (low proof or ungrounded facts increase uncertainty)
      3. Contradictions (conflicts with storage or context candidate nodes)
      4. Salience Protection (master directives >= 0.95 dampened to near 0)
    """

    def __init__(
        self,
        storage: Optional[Any] = None,
        config: Optional[Union[MetacognitiveConfig, Dict[str, Any]]] = None,
    ) -> None:
        self.storage = storage

        if config is None:
            self.config = MetacognitiveConfig()
        elif isinstance(config, MetacognitiveConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = MetacognitiveConfig(**config)
        else:
            raise TypeError(f"Config must be MetacognitiveConfig or dict, got: {type(config)}")

        # Initialize contradiction resolver if available
        self.contradiction_resolver: Optional[Any] = None
        try:
            from synapse.contradiction_resolver import ContradictionResolver
            self.contradiction_resolver = ContradictionResolver(storage=self.storage)
        except Exception as e:
            logger.debug("ContradictionResolver not available: %s", e)

    def _extract_timestamp(self, node: Dict[str, Any]) -> float:
        """Extract the most relevant timestamp from a node."""
        for key in ("updated_at", "last_verified_at", "created_at", "last_accessed", "timestamp"):
            if key in node and node[key] is not None:
                ts = parse_timestamp(node[key])
                if ts > 0.0:
                    return ts
        return 0.0

    def _extract_proof_count(self, node: Dict[str, Any]) -> int:
        """Extract proof or evidence count from a node."""
        for key in ("proof_count", "evidence_count", "verification_count"):
            if key in node and node[key] is not None:
                try:
                    return max(0, int(node[key]))
                except (ValueError, TypeError):
                    pass

        # Check if a proofs list exists
        if "proofs" in node and isinstance(node["proofs"], list):
            return len(node["proofs"])
        if "supporting_facts" in node and isinstance(node["supporting_facts"], list):
            return len(node["supporting_facts"])

        return self.config.default_proof_count

    def _evaluate_freshness(
        self,
        node: Dict[str, Any],
        current_time: int,
        reasons: List[str],
    ) -> float:
        """
        Evaluate freshness decay. Older memories without recent access decay in confidence,
        following an exponential half-life curve.
        """
        ref_ts = self._extract_timestamp(node)
        if ref_ts <= 0.0:
            reasons.append("Missing timestamp for age calculation; applied baseline age uncertainty")
            return self.config.unknown_age_uncertainty

        age_seconds = max(0.0, float(current_time) - ref_ts)
        half_life_seconds = max(1.0, self.config.freshness_half_life_days * 86400.0)

        # Half-life decay: 1.0 - (0.5 ^ (age / half_life))
        freshness_decay = 1.0 - math.pow(0.5, age_seconds / half_life_seconds)
        freshness_decay = max(0.0, min(1.0, freshness_decay))

        age_days = age_seconds / 86400.0
        if freshness_decay >= 0.25:
            reasons.append(
                f"Age decay: {age_days:.1f} days elapsed since creation/access "
                f"(freshness uncertainty: {freshness_decay:.2f})"
            )

        return freshness_decay

    def _evaluate_proof_count(
        self,
        node: Dict[str, Any],
        reasons: List[str],
    ) -> float:
        """
        Evaluate proof count. Few or zero proofs increase uncertainty.
        Reaching target_proof_count reduces proof uncertainty to 0.
        """
        proofs = self._extract_proof_count(node)
        target = max(1, self.config.target_proof_count)

        if proofs <= 0:
            reasons.append(f"Zero verification proofs recorded (proof_count=0/{target})")
            return 1.0

        if proofs >= target:
            return 0.0

        uncertainty = (target - proofs) / float(target)
        uncertainty = max(0.0, min(1.0, uncertainty))
        reasons.append(
            f"Low verification proof count (proof_count={proofs}/{target}, "
            f"proof uncertainty: {uncertainty:.2f})"
        )
        return uncertainty

    def _evaluate_contradictions(
        self,
        node: Dict[str, Any],
        candidate_nodes: Optional[List[Dict[str, Any]]],
        reasons: List[str],
    ) -> float:
        """
        Evaluate factual contradictions against explicit flags, candidate nodes, or storage.
        """
        # 1. Direct status and explicit contradiction indicators
        if node.get("status") == "superseded":
            reasons.append("Memory status is 'superseded' due to factual contradiction/update")
            return self.config.contradiction_penalty

        if node.get("has_contradiction") or node.get("conflict"):
            reasons.append("Explicit contradiction/conflict flag present on memory")
            return self.config.contradiction_penalty

        if node.get("contradictions") and isinstance(node.get("contradictions"), (list, set)):
            if len(node["contradictions"]) > 0:
                reasons.append(f"Active contradiction list contains {len(node['contradictions'])} conflicts")
                return self.config.contradiction_penalty

        # 2. Cross-checking candidate context nodes
        node_id = node.get("id")
        if candidate_nodes:
            for cand in candidate_nodes:
                cand_id = cand.get("id")
                if cand is node or (node_id and cand_id and node_id == cand_id):
                    continue

                if self.contradiction_resolver:
                    conflict = self.contradiction_resolver.check_contradiction(node, cand)
                    if conflict:
                        conflict_desc = conflict.get("reason") or conflict.get("conflict_type") or "Factual conflict"
                        reasons.append(
                            f"Contradiction detected with context memory '{cand_id or cand.get('uri', 'unknown')}': {conflict_desc}"
                        )
                        return self.config.contradiction_penalty
                else:
                    # Basic fallback check if categories match and opposing patterns trigger
                    try:
                        from synapse.retriever import detect_contradictions
                        conflicts = detect_contradictions([node, cand])
                        if conflicts:
                            reasons.append(f"Contradiction detected with context memory '{cand_id}': {conflicts[0].get('conflict')}")
                            return self.config.contradiction_penalty
                    except Exception:
                        pass

        # 3. Check storage if provided and no candidates were provided
        if candidate_nodes is None and self.contradiction_resolver and self.storage:
            try:
                storage_conflicts = self.contradiction_resolver.find_conflicts_for_candidate(node)
                if storage_conflicts:
                    for c in storage_conflicts[:2]:
                        reasons.append(
                            f"Contradiction detected with storage memory '{c.get('conflicting_node_id')}': {c.get('reason')}"
                        )
                    return self.config.contradiction_penalty
            except Exception as e:
                logger.debug("Error checking storage contradictions: %s", e)

        return 0.0

    def score_memory(
        self,
        node: Dict[str, Any],
        current_time: Optional[int] = None,
        candidate_nodes: Optional[List[Dict[str, Any]]] = None,
    ) -> UncertaintyReport:
        """
        Score a memory node for epistemic uncertainty.

        Evaluates:
          - Freshness decay based on age since creation/access.
          - Proof count (low proofs increase uncertainty).
          - Factual contradictions (with storage or provided context candidates).
          - Salience dampening (salience >= 0.95 dampens uncertainty to near 0).

        Returns an UncertaintyReport dataclass/dict with uncertainty_score,
        confidence_level, is_uncertain, reasons, and tag.
        """
        curr_time = int(time.time()) if current_time is None else int(current_time)
        reasons: List[str] = []

        # 1. Component evaluations
        freshness_u = self._evaluate_freshness(node, curr_time, reasons)
        proof_u = self._evaluate_proof_count(node, reasons)
        contradiction_u = self._evaluate_contradictions(node, candidate_nodes, reasons)

        # 2. Raw uncertainty calculation
        w_f = self.config.freshness_weight
        w_p = self.config.proof_count_weight
        base_w = w_f + w_p
        if base_w <= 0.0:
            base_w = 1.0

        baseline_raw = (w_f * freshness_u + w_p * proof_u) / base_w

        if contradiction_u > 0.0:
            raw_uncertainty = max(
                baseline_raw,
                contradiction_u * self.config.contradiction_min_uncertainty,
                (w_f * freshness_u + w_p * proof_u + self.config.contradiction_weight * contradiction_u)
                / (base_w + self.config.contradiction_weight),
            )
        else:
            raw_uncertainty = baseline_raw

        # 3. Salience evaluation & Master Directive protection
        raw_salience = node.get("salience")
        salience = float(0.5 if raw_salience is None else raw_salience)
        salience = max(0.0, min(1.0, salience))

        if salience >= self.config.salience_master_threshold:
            # MASTER DIRECTIVE MANDATE:
            # Salience >= 0.95 must be protected from high uncertainty and dampened to near 0.
            dampened_uncertainty = min(
                self.config.max_master_uncertainty,
                raw_uncertainty * self.config.master_salience_dampener,
            )
            reasons.append(
                f"Master directive protected (salience={salience:.2f} >= "
                f"{self.config.salience_master_threshold}): uncertainty dampened to near-zero "
                f"({dampened_uncertainty:.4f})"
            )
        else:
            # Dynamic proportional dampening for standard salience
            dampened_uncertainty = raw_uncertainty * (1.0 - (salience * self.config.salience_dampening_weight))

        # Clamp final uncertainty score to [0.0, 1.0]
        final_uncertainty = round(max(0.0, min(1.0, dampened_uncertainty)), 4)

        # 4. Determine classification levels
        is_uncertain = final_uncertainty >= self.config.uncertainty_threshold

        if final_uncertainty >= self.config.uncertainty_threshold:
            uncertainty_level = "HIGH"
            confidence_level = "LOW"
        elif final_uncertainty <= self.config.confidence_high_cutoff:
            uncertainty_level = "LOW"
            confidence_level = "HIGH"
        else:
            uncertainty_level = "MEDIUM"
            confidence_level = "MEDIUM"

        # 5. Determine tag
        if is_uncertain:
            tag = self.config.get_tag(uncertainty_level)
        else:
            tag = None

        breakdown = {
            "freshness_uncertainty": round(freshness_u, 4),
            "proof_uncertainty": round(proof_u, 4),
            "contradiction_uncertainty": round(contradiction_u, 4),
            "raw_uncertainty": round(raw_uncertainty, 4),
            "salience": round(salience, 4),
            "final_uncertainty": final_uncertainty,
        }

        return UncertaintyReport(
            uncertainty_score=final_uncertainty,
            confidence_level=confidence_level,
            is_uncertain=is_uncertain,
            reasons=reasons,
            tag=tag,
            uncertainty_level=uncertainty_level,
            breakdown=breakdown,
        )

    def annotate_context(
        self,
        nodes: List[Dict[str, Any]],
        threshold: float = 0.6,
        current_time: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Annotate memory nodes in prompt context with uncertainty warning tags.

        For memories whose uncertainty score meets or exceeds `threshold`,
        appends an uncertainty tag like '[UNCERTAINTY: HIGH - MUST VERIFY LIVE]'
        to the abstract or context fields.

        Returns a list of annotated node dictionaries.
        """
        if not nodes:
            return []

        annotated_nodes: List[Dict[str, Any]] = []

        for node in nodes:
            node_copy = copy.deepcopy(node) if isinstance(node, dict) else dict(node)
            report = self.score_memory(node_copy, current_time=current_time, candidate_nodes=nodes)

            is_uncertain_for_threshold = report.uncertainty_score >= threshold

            if is_uncertain_for_threshold:
                # Select appropriate tag based on score
                level = report.uncertainty_level or ("HIGH" if report.uncertainty_score >= 0.60 else "MEDIUM")
                tag_to_apply = self.config.get_tag(level)

                # Append tag to abstracts / context
                applied = False
                for target_field in ("l0_abstract", "abstract", "context"):
                    if target_field in node_copy:
                        orig_val = str(node_copy.get(target_field) or "").strip()
                        if tag_to_apply not in orig_val:
                            node_copy[target_field] = f"{orig_val} {tag_to_apply}".strip() if orig_val else tag_to_apply
                        applied = True

                if not applied:
                    if "l1_overview" in node_copy:
                        orig_val = str(node_copy.get("l1_overview") or "").strip()
                        if tag_to_apply not in orig_val:
                            node_copy["l1_overview"] = f"{orig_val} {tag_to_apply}".strip() if orig_val else tag_to_apply
                    else:
                        node_copy["l0_abstract"] = tag_to_apply

                node_copy["is_uncertain"] = True
                node_copy["uncertainty_tag"] = tag_to_apply
            else:
                node_copy["is_uncertain"] = False
                node_copy["uncertainty_tag"] = None

            node_copy["uncertainty_score"] = report.uncertainty_score
            node_copy["confidence_level"] = report.confidence_level
            node_copy["uncertainty_report"] = report.to_dict()

            annotated_nodes.append(node_copy)

        return annotated_nodes
