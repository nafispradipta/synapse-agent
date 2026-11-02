# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.3] - 2026-11-02

### Added
- **100% Pluggable Cognitive Taxonomy:** Categories and categorization heuristics are no longer hardcoded or constrained in SQLite schema. Users can configure any custom categories (e.g. medical, legal, gaming, e-commerce, customer support) via `SynapseAdapter(config={"categories": [...], "category_patterns": {...}})`.
- **Dynamic Tag Rules & Vault Routing:** Users can provide arbitrary domain tag rules and custom markdown vault target routes (`vault_routes`) in adapter configuration.
- **Unconstrained Storage Schema:** Removed restrictive SQLite `CHECK(category IN (...))` constraint, allowing open extensible taxonomies across all domain memory stores.
- **Expanded Test Suite:** 254 automated tests verifying pluggable categories and custom taxonomies.

## [1.0.2] - 2026-11-02

### Added
- **Multi-Word Status Negation:** Operational status parsing now detects negative phrases (`tidak aktif`, `not active`, `tidak berjalan`, `belum jalan`, `tidak menyala`, `mati`) to avoid classifying negated phrases as active.
- **Bilingual Categorization & Auto-Tags:** `SynapseAdapter` infers cognitive categories (`incident`, `sop`, `decision`, `preference`) and tags (`finance`, `marketing`, `meeting`, `todo`, `idea`, `personal`) across English and Indonesian.
- **Universal Vector Dimension Mismatch Guards:** Added length guards across all vector operations (`storage.py`, `counterfactual_guardrail.py`, `retriever.py`) to safely ignore mismatched embedding dimensions instead of silent truncation.
- **Software Downgrade Protection:** Version progression prevents newer lower versions from silently superseding higher versions (e.g., PostgreSQL 16 vs 14) unless explicit rollback directives (`rollback`, `downgrade`, `revert`, `kembali ke`) are specified.
- **Expanded Test Suite:** 253 automated tests covering negation status, bilingual categorization, and downgrade protection.

## [1.0.1] - 2026-11-02

### Added
- **Bilingual Cognitive Processing (English + Indonesian):** Native recognition for operational status (`active`/`aktif`, `stopped`/`mati`), task lifecycle progression (`in progress`/`sedang berjalan`, `completed`/`sudah selesai`), directive polarity (`mandatory`/`wajib`, `forbidden`/`dilarang`), and specifications (`memori` -> `ram`, `kuota` -> `quota`).
- **Subject-Isolated Belief Revision:** Task progressions strictly enforce subject matching, preventing unrelated tasks from superseding one another.
- **PyPI Metadata URLs:** Added `[project.urls]` linking directly to Homepage, Repository, Issues, and Changelog.

## [1.0.0] - 2026-11-02

### Added
- **Biomimetic Cognitive Memory Cortex:** Tripartite architecture combining Mental Models, Observations, Graph Storage, and Research Vault.
- **Hybrid Spreading Activation Retriever:** FTS5 BM25 lexical search, dense neural vector similarity (384-dim), and multi-hop graph spreading activation.
- **Progressive Context Packer:** Multi-tiered token budgeting (L0 Abstract < 40 tokens, L1 Overview < 300 tokens, L2 Target raw references).
- **Subconscious Belief Revision:** Automated contradiction resolution, factual dispute arbitration, and Hebbian reinforcement with decay cycles.
- **Counterfactual Guardrails:** Structured episodic failure tracking (*Situation -> Action -> Outcome -> Lesson*) with actionable mitigation recommendations.
- **Metacognitive Calibration Scorer:** Uncertainty estimation, salience decay protection, and context annotation directives.
- **Model Context Protocol (MCP) Server:** Native JSON-RPC 2.0 MCP server for cross-agent memory sharing (`synapse_recall`, `synapse_retain`, `synapse_record_episode`, `synapse_get_stats`).
- **Comprehensive Test Suite:** 248 automated tests covering all memory, retrieval, consolidation, and lifecycle subsystems.
- **Packaging & MCP Stdio Integration:** Direct `synapse-mcp` entry point packaged in `synapse.mcp`.
- **OpenAI-Compatible Remote Embeddings:** Lightweight `/v1/embeddings` integration without heavy local PyTorch dependencies.
- **Enhanced Belief Revision:** Native software version progression (`PostgreSQL 14 → 16`) and task lifecycle state transitions (`In Progress → Completed`).
