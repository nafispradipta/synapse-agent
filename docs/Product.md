# Product

## What it is

`synapse-cortex` is a memory library and MCP server for autonomous AI agents. It stores facts, decisions, procedures, preferences, incidents, and episodic lessons in a local SQLite graph, then returns compact, ranked context that fits a token budget.

The product exists because agents forget between sessions, flat retrieval returns stale and current facts together, and dumping raw history into prompts wastes context.

## Who it is for

- Developers building long-running agents that need memory across sessions.
- Operators of multiple AI assistants that should share one local memory pool.
- Teams that want memory to stay on their own machine or server instead of a hosted vector database.
- Indonesian and English users, since categorisation, tags, and status heuristics are bilingual.

## Product principles

- Local ownership. The database is a file the user controls.
- Useful without a model. Core memory operations work offline with no API key.
- Compact by default. Recall returns packed L0/L1 context, not raw transcripts.
- Correctable. Stale beliefs are superseded, not silently mixed with current ones.
- Honest limits. The product states where it is weaker instead of hiding it.

## Capabilities

### Semantic memory

- `retain(content, category, salience, title)`: store an atomic fact, rule, decision, preference, or incident.
- `recall(query, limit, token_budget)`: hybrid retrieval over lexical, vector, graph, and recency signals.
- `reflect(query)`: synthesize a topic from associated memories.
- Custom categories, category patterns, tag rules, and vault routes.

### Episodic memory

- `record_episode(...)`: store situation, action, outcome, and lesson.
- Episode search and L0/L1 packing.
- Counterfactual guardrail: warn when a proposed action previously caused failure, data loss, or an outage.
- Session distillation: convert episodes into semantic lessons.

### Consolidated knowledge

- Mental models: topic-level documents refined over time.
- Observations: deduplicated patterns with proof counts and evidence.
- Rollups: superseded facts compressed into abstracts instead of discarded.

### Research quarantine

- Whitelisted harvesting of web content with redirect and size limits.
- Harvested research records and vault notes are `quarantined`; the derived graph node is `flagged`. Neither appears in recall until explicitly promoted.

### Integrations

- Python API through `SynapseAdapter`, including async helpers.
- MCP stdio server with four tools: `synapse_recall`, `synapse_retain`, `synapse_record_episode`, `synapse_get_stats`.
- Markdown vault writeback and wikilink synchronization for Obsidian-style knowledge bases.
- Session helper for system prompt assembly.

### Operations

- In-process maintenance and a nightly consolidation script.
- WAL checkpoint helper.
- Integrity verification and FTS optimization through the storage layer.

## Functional requirements

- A retain call must persist the node, its vector, and its FTS entry, or fail without leaving a partial transaction.
- A recall call must return complete sentences within the token budget and must not mutate belief state.
- A superseded fact must stop appearing as current after consolidation.
- A write with non-finite scores must be rejected.
- A harvested URL outside the whitelist must be rejected before content is stored.
- A vault write must stay inside the configured vault directory.
- The MCP server must answer protocol errors with valid JSON-RPC codes and must not write non-protocol data to stdout.

## Non-goals

- A hosted, multi-tenant memory service. There is no built-in authentication or tenant isolation.
- A replacement for high-scale vector databases or search engines.
- A general document management system or note-taking UI.
- A guarantee that stored content is factually true. The system tracks confidence and contradictions, not ground truth.
- Autonomous browsing without an explicit harvest call and a configured whitelist.
- A web dashboard. Interfaces are the Python API, MCP, and the Markdown vault.

## Honest limits

- Target scale is roughly under 10,000 active nodes.
- The fallback embedder matches wording, not meaning. Paraphrase and cross-language recall need neural embeddings.
- Deterministic belief revision covers numbers, versions, and lifecycle status. Prose-level revision needs LLM-assisted consolidation.
- SQLite allows one writer at a time. Heavy concurrent writes serialize.
- Quality depends on what the agent chooses to retain. Memory cannot recover facts that were never stored.

## Version

Current repository version: `1.1.1`. The latest published PyPI release may lag the source version; check PyPI before relying on a release. See the root `CHANGELOG.md` for released changes and [Status.md](Status.md) for the working state.
