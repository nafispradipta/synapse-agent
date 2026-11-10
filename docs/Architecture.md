# Architecture

## Purpose

`synapse-cortex` gives long-running AI agents a persistent memory that behaves like an associative graph rather than a flat vector index. It combines lexical search, dense vectors, graph spreading activation, belief revision, and progressive context packing on top of a single SQLite database.

## Design principles

- Local-first. A single SQLite file in WAL mode is the system of record. No server is required.
- Standard-library core. Every third-party import is optional and guarded. The package installs with `dependencies = []`.
- Bounded degradation. A missing model, endpoint, or accelerator reduces quality, never availability.
- One owner per storage concern. The facade owns the connection; mixins own the SQL.
- Evidence over assertion. Contradictions, uncertainty, and guardrails are computed from stored records, not guessed.

## Component map

```text
Agent runtime
    |
    v
SynapseAdapter  (synapse/adapter.py)
    |-- retain()  --> categorizer, salience, L0/L1 text, embedder, graph writer, vault writeback
    |-- recall()  --> SynapseRetriever --> SynapseContextLoader --> prompt blocks
    |-- reflect() --> retriever + contradiction resolver + optional LLM
    |-- episodes  --> EpisodeStore + counterfactual guardrail
    |-- research  --> SynapseWebHarvester --> quarantine (research=quarantined, node=flagged)
    |
    v
SynapseStorage facade  (synapse/storage.py)
    |-- SchemaMixin      DDL, indexes, FTS5, triggers, migrations
    |-- GraphStore       nodes, edges, FTS search, belief state
    |-- VectorStore      dense vectors, matrix cache, backfill
    |-- EpisodeStore     episodic records and search
    |-- KnowledgeStore   mental models, observations, external research
    |-- MaintenanceStore decay, rollups, compaction, vault upsert
    |
    v
SQLite database (WAL, synchronous=NORMAL, foreign_keys=ON, busy_timeout=10000)
```

## Modules

| Module | Responsibility |
| --- | --- |
| `synapse/adapter.py` | Public `SynapseAdapter`, `BaseMemoryAdapter`, `LocalFileAdapter`, `NullAdapter`, `resolve_memory_adapter` |
| `synapse/config.py` | Environment-driven LLM base URL, credential, and model resolution with retry |
| `synapse/storage.py` | Storage facade: connection, lock, pragmas, `get_stats` |
| `synapse/stores/` | Bounded-context persistence mixins |
| `synapse/retriever.py` | Hybrid retrieval, spreading activation, contradiction detection, vault wikilinks |
| `synapse/loader.py` | Token estimation and L0/L1/L2 packing |
| `synapse/worker.py` | Maintenance cycle, Hebbian reinforcement, rollups, vault sync, episode distillation, session distillation |
| `synapse/embedder.py` | Neural embeddings with deterministic fallback |
| `synapse/mcp.py` | MCP stdio server and JSON-RPC protocol |
| `synapse/session.py` | Session lifecycle hooks and system prompt assembly |
| `synapse/contradiction_resolver.py` | Numeric, version, and status belief revision |
| `synapse/metacognitive_scorer.py` | Uncertainty, staleness, and proof-count scoring |
| `synapse/counterfactual_guardrail.py` | Warns about actions that previously caused failures |
| `synapse/mental_model_engine.py` | Mental model content refinement |
| `synapse/mental_model_spawner.py` | Cluster detection that proposes new mental models |
| `synapse/observation_engine.py` | Evidence-grounded pattern consolidation |
| `synapse/harvester.py` | Whitelisted web research with SSRF controls |
| `synapse/moc_bookkeeper.py` | Markdown vault index maintenance |

## Retain flow

1. The adapter accepts `content`, optional `context`, `metadata`, and overrides such as `category`, `salience`, and `title`.
2. It infers a category and tags from bilingual regex heuristics unless the caller supplies them. Custom taxonomies replace the defaults through adapter configuration.
3. It builds an L0 abstract and an L1 overview, then writes the node through `GraphStore`.
4. The storage embedder produces a vector during the node insert, then the adapter re-embeds the final text and `VectorStore` stores the vector and invalidates the matrix cache.
5. Graph edges are written from shared entities, tags, and references. Nodes carry salience, strength, confidence, and access metadata; edges carry relation, weight, last-fired time, and access count.
6. FTS5 triggers mirror the text into `synapse_fts`.
7. If a vault path is configured and the node is salient enough, a Markdown note is written inside the vault with path containment checks.

A write failure rolls back the transaction at the outermost write level. Non-finite scores are rejected at the storage boundary.

## Recall flow

1. The query is tokenized for FTS5 BM25 and embedded for cosine similarity.
2. Lexical and vector candidates are merged with reciprocal rank fusion.
3. Activation spreads from seed nodes across graph edges up to 3 hops with a decay of 0.65 per hop. Temporal edges receive a multiplier.
4. Vault wikilinks contribute a separate activation path.
5. The final score combines BM25 (`0.35`), vector similarity (`0.35`), spreading activation (`0.15`), and recency (`0.15`), scaled by node strength, salience, confidence, and importance.
6. Contradicting candidates are filtered in memory for the response. Retrieval does not supersede or rewrite facts, but with `touch=True` (the default) it reinforces access metadata, salience, strength, and co-activation edges. Pass `touch=False` for a read-only recall.
7. Episodic memories are scored separately and merged.
8. `SynapseContextLoader` packs complete sentences into the requested token budget. L0 is about 40 tokens, L1 about 300. If L1 does not fit, the loader keeps complete L0 lines instead of cutting mid-sentence.

Defaults are tunable through the retriever. Disposition settings (`skepticism`, `literalism`) adapt the lexical and vector weights within bounded ranges. `empathy` is accepted in the configuration but is not used by the weight calculation.

## Reflect and maintenance flow

`reflect(query)` synthesizes a topic from retrieved graph context. Background maintenance is where belief state changes:

- Hebbian reinforcement raises edge and node strength for co-activated records and applies outcome feedback.
- Decay lowers strength over time and prunes exhausted nodes and weak edges.
- Superseded nodes roll up into compact abstracts and move to `synapse_superseded_archive`.
- The contradiction resolver audits active facts, supersedes stale ones, and reduces their strength.
- Episode distillation converts session episodes into semantic lessons, with an LLM when configured and deterministic rules otherwise.
- Mental model and observation engines consolidate clusters and evidence.
- Vault sync imports Markdown notes and updates maps of content.

## Storage schema

`PRAGMA user_version` is stamped to `1`. Tables:

| Table | Role |
| --- | --- |
| `synapse_nodes` | Active semantic facts, L0/L1 text, category, status, salience, confidence, strength, access metadata |
| `synapse_edges` | Directed associations with relation, weight, and last-fired timestamp |
| `synapse_superseded_archive` | Superseded facts kept for rollup and audit |
| `episodes` | Situation, action, outcome, lessons, tags, salience |
| `synapse_vectors` | Embedding JSON, model name, updated timestamp |
| `_synapse_vector_meta` | Vector version/counter used by the cache |
| `synapse_mental_models` | Consolidated topic documents and history |
| `synapse_observations` | Deduplicated patterns with proof counts and status |
| `synapse_external_research` | Quarantined harvested research |

FTS5 virtual tables: `synapse_fts` over node L0/L1 text, `episodes_fts` over episode fields. Triggers keep both in sync and fire only on text-column changes. `tests/test_schema_contract.py` pins the exact columns, triggers, and indexes.

## Embedding paths

| Path | When | Notes |
| --- | --- | --- |
| `sentence-transformers` | Installed and model available | `all-MiniLM-L6-v2` by default, 384 dimensions, local inference |
| Remote OpenAI-compatible endpoint | `OPENAI_EMBEDDING_*` configured | Uses the provider's dimension, which may not be 384 |
| CRC32 n-gram vectorizer | Always available | Deterministic, zero dependencies, strong on lexical variants, weak on paraphrase and cross-language semantics |

The vector matrix cache can use NumPy when installed. Without it, cosine similarity runs in pure Python over candidate nodes.

## Extension points

- `SynapseAdapter(config={...})`: `db_path`, `token_budget`, `vault_path`, `research_dir`, `categories`, `category_patterns`, `tag_rules`, `vault_routes`, `disposition`, `metacognitive`, `guardrail`, `fallback_adapter`, `use_llm`.
- Custom storage categories require `allow_custom_categories`.
- `SessionManager` accepts a `config_dir` and a `soul_config` for prompt identity.
- MCP clients connect over stdio through `synapse-mcp`.

## Known limits

- The design targets personal and team memory graphs below roughly 10,000 active nodes. It is not a replacement for Milvus, Pinecone, or other multi-million-vector systems.
- Cosine similarity over the candidate set is computed in process. Very large graphs need an external index.
- SQLite WAL supports one writer at a time. Concurrent writers wait up to the configured busy timeout.
- The bundled vectorizer is lexical. True paraphrase and cross-language matching require neural embeddings.
- Broad semantic belief revision depends on LLM-assisted consolidation. Deterministic rules cover numeric, version, and lifecycle contradictions.
