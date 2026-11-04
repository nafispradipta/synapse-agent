# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Storage Encapsulation:** `Retriever` and `Worker` no longer reach into `SynapseStorage._conn` / `_lock`. All cross-module database work now goes through named data-access methods (`bump_access_count`, `load_vector_index`, `iter_due_nodes`, `commit_superseded_rollup`, and friends), so locking and transaction boundaries stay inside one owner.
- **Schema Initialisation Decomposed:** The 337-line `_init_schema` is now a short orchestrator over seven per-table helpers plus module-level `_DDL_*` constants and index/trigger tuples. The legacy `synapse_nodes` rebuild lives in its own method.
- **Storage Split into Bounded Contexts:** `SynapseStorage` is now a 154-line facade over six mixins in `synapse/stores/` — `SchemaMixin` (DDL and migrations), `GraphStore` (nodes, edges, FTS), `VectorStore` (embeddings and the matrix cache), `EpisodeStore`, `KnowledgeStore` (mental models, observations, research) and `MaintenanceStore` (decay, rollups, compaction). The facade keeps the connection, the embedder hook, and cross-context aggregates such as `get_stats`. Method bodies moved verbatim; `from synapse.storage import SynapseStorage, VALID_CATEGORIES, VALID_STATUSES` keeps working.
- **Maintenance Operations Live in the Storage Layer:** The eight SQL operations `scripts/nightly_consolidation.py` used to issue against its own connection are now named methods on `MaintenanceStore` — `lock_critical_nodes`, `apply_hebbian_decay`, `prune_exhausted_nodes`, `prune_weak_edges`, `upsert_vault_node`, `optimize_fts_indices`, `verify_integrity`, `wal_checkpoint`. The vault upsert preserves `salience`, `strength`, `last_accessed` and `access_count` on re-ingest. `scripts/wal_checkpoint.py` and the consolidation pipeline now talk to storage through `SynapseStorage`; neither module opens a raw `sqlite3` connection any more.
- **One Connection Per Consolidation Run:** Steps 2.5, 4.3, 9.5 and the contradiction audit no longer open a separate `SynapseStorage` instance; they reuse the single instance bound to the `--db-path` argument. `PRAGMA busy_timeout=10000` is now applied by storage so the consolidation job and the app can share the WAL without starving each other.
- **Expanded Test Suite:** 318 automated tests, including 17 encapsulation guards, 17 schema-contract characterisation tests, 12 bounded-context boundary tests and 14 nightly-maintenance behaviour tests.

### Fixed
- **Legacy Node Migration Dropped `access_count`:** Rebuilding `synapse_nodes` for pre-1.0.3 databases recreated the table without `access_count`, so creating `idx_synapse_nodes_access_count` failed with `no such column: access_count`. The rebuild now carries the column over.
- **Legacy Migration Silently Disabled Foreign Keys:** `PRAGMA foreign_keys=ON` was issued after the row copy, but the copy opens an implicit transaction and the pragma is a no-op inside one — so every `ON DELETE CASCADE` silently stopped firing after a pre-1.0.3 migration. Enforcement is now restored before the connection is used.
- **Consolidation Targeted the Wrong Database with `skip_external`:** Step 2.5 ignored the `--db-path` argument and consolidated the global default database instead of the requested one; it now uses `target_db` and honours `skip_external`.

## [1.0.5] - 2026-11-04

### Fixed
- **Belief Revision No Longer Erases Distinct Facts:** `extract_numeric_specs` now binds numbers only to their own attribute (explicit `key: value`, a closed bridge vocabulary after the attribute, or a number with a recognized unit in front of it). `Project Alpha service 7 uses port 8007 with timeout 30s` yields `port=8007` and `timeout=30s` instead of `port=7` and `timeout=8007`. Subject identifiers such as `service 7` vs `service 8` now make two facts disjoint subjects, so 20 distinct numeric facts stay active instead of superseding each other.
- **Retrieval Is Read-Only for Belief State:** `retrieve()` no longer calls `supersede_low_salience_nodes`. Contradicting candidates are filtered in-memory for the response only; persistent supersede happens at retain time and in the nightly consolidation audit.
- **Failed Writes Roll Back:** New `_write()` helper rolls back the connection when a write raises. Duplicate URIs, FK violations and duplicate episode IDs no longer leave an open transaction that blocks every second connection until `busy_timeout` expires (`database is locked`). The connection is closed when initialisation fails instead of leaking.
- **Legacy Node Migration Is Atomic and Recoverable:** The `synapse_nodes` rebuild now runs as one transaction and startup recovers databases left mid-rename by older releases (rows restored from `_synapse_nodes_old`) instead of exposing an empty table.
- **Prune Keeps Superseded Nodes Until Archival:** `prune_exhausted_nodes` now only deletes active rows, so superseded records reach the superseded-archive rollup instead of being lost.
- **MCP Server Reports Real Failures:** The stdio server forces UTF-8, answers malformed input with `-32700`/`-32600`/`-32602`/`-32601`/`-32603`, no longer crashes on array payloads or non-ASCII output, ignores notifications as JSON-RPC requires, reports `isError` when a write fails or storage is unavailable, exposes the package version and the full category enum (`general` included), and creates no database or log directory at import time (adapter and logging are lazy). `tests/test_mcp_protocol.py` covers the protocol end to end.
- **Write Depth Tracking Is Thread-Local:** The shared `_write` rollback counter is now per thread, so a background maintenance thread cannot disable rollback for every later write on the connection; failed writes in any thread roll back at their own outermost level. Recovery of an interrupted legacy migration now logs the ids of superseded rows it must drop when they conflict with live rows.

## [1.0.4] - 2026-11-02

### Fixed
- **Regex Substring Boundary Trap in Operational Status:** Replaced raw substring checks with strict regex word boundaries (`\b`) for negative and lifecycle status phrases. Words containing root substrings such as `mati` (e.g., `otomatis`, `automatic`, `sistematis`, `systematic`, `informatika`) are no longer erroneously parsed as `inactive`.
- **Expanded Test Suite:** 255 automated tests verifying word-boundary immunity and eliminating false-positive task supersession.

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
