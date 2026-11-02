"""
Synapse - Biomimetic Cognitive Graph Memory & Spreading Activation for Autonomous Agents.

Core Modules:
- storage: Persistent SQLite WAL Graph & BM25 Full-Text Search.
- retriever: Hybrid Spreading Activation, Vector & Lexical Retriever.
- loader: Progressive Context Packer (L0/L1 budget enforcement).
- worker: Background Consolidation, Hebbian Reinforcement & Abstract Rollup.
- adapter: Pluggable BaseMemoryAdapter & Framework Bindings.
- contradiction_resolver: Belief Revision & Cascading Contradiction Invalidation.
- counterfactual_guardrail: Episodic Negative-Invariant Safety Guardrails.
- metacognitive_scorer: Uncertainty Calibration & Salience Dampening.
- mental_model_engine: Dynamic Markdown AST & Delta Refinement.
"""

from synapse.storage import (
    SynapseStorage,
    VALID_CATEGORIES,
    VALID_STATUSES,
)
from synapse.retriever import (
    SynapseRetriever,
    VaultWikilinkCache,
    extract_wikilinks_from_markdown,
    detect_contradictions,
    vectorize_text,
    cosine_similarity,
)
from synapse.loader import (
    SynapseContextLoader,
    estimate_tokens,
    split_sentences,
)
from synapse.worker import SynapseWorker, resolve_synapse_llm_config
from synapse.adapter import (
    BaseMemoryAdapter,
    LocalFileAdapter,
    NullAdapter,
    SynapseAdapter,
    resolve_memory_adapter,
)
from synapse.session import SessionManager
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

__all__ = [
    "BaseMemoryAdapter",
    "LocalFileAdapter",
    "NullAdapter",
    "SynapseAdapter",
    "resolve_memory_adapter",
    "SessionManager",
    "SynapseStorage",
    "SynapseRetriever",
    "VaultWikilinkCache",
    "extract_wikilinks_from_markdown",
    "SynapseContextLoader",
    "SynapseWorker",
    "ContradictionResolver",
    "MetacognitiveScorer",
    "MetacognitiveConfig",
    "UncertaintyReport",
    "CounterfactualGuardrail",
    "GuardrailResult",
    "detect_contradictions",
    "resolve_synapse_llm_config",
    "vectorize_text",
    "cosine_similarity",
    "estimate_tokens",
    "split_sentences",
    "VALID_CATEGORIES",
    "VALID_STATUSES",
]
