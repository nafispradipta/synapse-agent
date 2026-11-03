# Synapse Cortex (`synapse-cortex`)

> **Biomimetic Cognitive Graph Memory, Spreading Activation, and Progressive Context Packing for Autonomous AI Agents.**

[![PyPI: synapse-cortex](https://img.shields.io/pypi/v/synapse-cortex.svg)](https://pypi.org/project/synapse-cortex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](pyproject.toml)
[![Tests: 301 Passed](https://img.shields.io/badge/Tests-301%20Passed-success.svg)](tests/)
[![Architecture: Local--First](https://img.shields.io/badge/Architecture-Local--First%20SQLite%20WAL-purple.svg)](synapse/storage.py)

---

## Why Synapse?

AI agents without memory forget everything between sessions. However, naive vector databases (flat RAG) introduce critical cognitive flaws:

1. **Belief Drift & Hallucination:** Flat vector search blindly retrieves outdated memories alongside new ones. If a task status changes from *In Progress* to *Completed*, traditional RAG frequently returns both, causing the model to hallucinate progress.
2. **Disconnected Knowledge:** Real-world knowledge forms an associative graph. If an agent knows *"Module X manages authentication"* and *"Module X experienced an outage under load Y"*, it must associate these facts when debugging auth bottlenecks without explicit keywords.
3. **Context Window Bleed:** Dumping raw conversation transcripts into system prompts consumes thousands of tokens and triggers attention dilution.
4. **Episodic Amnesia:** Agents repeatedly attempt failed operations because they lack structured *counterfactual guardrails* (*Situation → Action → Outcome → Lesson*).

**Synapse solves this** with a biomimetic cognitive memory architecture designed specifically for long-running autonomous agents.

---

## Core Tripartite Architecture

```
                  ┌────────────────────────────────────────┐
                  │       Autonomous Agent Runtime         │
                  └───────┬────────────────────────▲───────┘
                          │ retain()               │ recall()
                          ▼                        │
       ┌───────────────────────────────────────────────────────────┐
       │                 SYNAPSE COGNITIVE CORTEX                  │
       │                                                           │
       │  [Tier 1: Mental Models]     [Tier 2: Observations]       │
       │  Dynamic AST Markdown        Deduplicated Pattern Clusters│
       │                                                           │
       │  [Tier 3: Graph Storage]     [Tier 4: Research Vault]     │
       │  FTS5 (BM25) + Dense Vector  Quarantined Harvested Docs   │
       └───────────────────────────▲───────────────────────────────┘
                                   │ reflect()
                          ┌────────┴────────┐
                          │ Background Loop │
                          │ Belief Revision │
                          │ Hebbian Decay   │
                          └─────────────────┘
```

1. **`retain()` — Micro-Episodic Storage:** Extracts atomic facts, categorizes knowledge (`architecture`, `sop`, `decision`, `preference`, `incident`), generates 384-dim dense vectors, and establishes synaptic associative graph edges.
2. **`recall()` — Hybrid Spreading Activation:** Combines SQLite FTS5 (BM25 lexical search), dense vector cosine similarity, and multi-hop graph spreading activation. Context is packed progressively using multi-tiered token budgeting (L0 Abstract < 40 tokens, L1 Overview < 300 tokens).
3. **`reflect()` — Cascading Belief Revision:** Resolves factual contradictions, supersedes stale beliefs, applies Hebbian reinforcement based on retrieval co-activation, and rolls up obsolete nodes into compact abstract summaries.

---

## Quickstart

### Installation

The quickest way to install Synapse Cortex and the `synapse-mcp` server binary:

```bash
pip install synapse-cortex
```

*(Or for optional development and testing tools: `pip install "synapse-cortex[dev]"`).*

---

### Installing from Source (Developers)

#### Linux & macOS

```bash
git clone https://github.com/nafispradipta/synapse-agent.git
cd synapse-agent

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package in editable mode
pip install -e ".[dev]"
```

#### Windows (PowerShell)

```powershell
git clone https://github.com/nafispradipta/synapse-agent.git
cd synapse-agent

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install package in editable mode
pip install -e .

# (Optional) Install development and testing dependencies
pip install -e ".[dev]"
```

#### Windows (Command Prompt - cmd.exe)

```cmd
git clone https://github.com/nafispradipta/synapse-agent.git
cd synapse-agent

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate.bat

# Install package in editable mode
pip install -e .
```

### Basic Usage

```python
from synapse.adapter import SynapseAdapter

# Initialize in-memory or persistent SQLite storage
adapter = SynapseAdapter({"db_path": "./data/synapse.db", "token_budget": 250})

# 1. Retain an architectural rule or invariant
adapter.retain(
    content="Always enable WAL mode and set busy_timeout >= 5000ms for concurrent SQLite operations.",
    category="architecture",
    salience=0.95,
    title="SQLite Concurrency Rule"
)

# 2. Record an episodic learning experience
adapter.record_episode(
    title="Database Lock Incident Under High Load",
    situation="Agent workers experienced database lock contention during concurrent batch writes.",
    action_taken="Switched journal mode to WAL and increased busy timeout to 5000ms.",
    outcome="Lock contention eliminated completely across all concurrent workers.",
    lessons_learned="Always enable WAL mode and configure busy_timeout >= 5000ms.",
    salience=0.95
)

# 3. Recall packed context for your agent prompt
recalled_context = adapter.recall("How to handle SQLite database locks under high load?")

for block in recalled_context:
    print(block)
```

---

## Model Context Protocol (MCP) Server

Synapse includes a built-in JSON-RPC 2.0 MCP server, enabling AI coding assistants (**Claude Desktop**, **Cursor**, **Windsurf**, **OpenCode**, **Codex**, **Pi CLI**, **Hermes**, **Zed**, etc.) to tap directly into the shared cognitive memory cortex.

### MCP Configuration

Add the `synapse` entry to your assistant's MCP configuration (`mcpServers` block):

```json
{
  "mcpServers": {
    "synapse": {
      "command": "synapse-mcp"
    }
  }
}
```

> **Shared Memory Pool:** By default, `synapse-mcp` automatically stores memory in `~/.synapse/synapse.db`. This means all your AI assistants (Claude, Cursor, terminal agents) **automatically share the exact same persistent brain** across all projects on your machine.
>
> If you prefer an isolated database for a specific project, simply add the `SYNAPSE_DB_PATH` environment variable:
> ```json
> {
>   "mcpServers": {
>     "synapse": {
>       "command": "synapse-mcp",
>       "env": {
>         "SYNAPSE_DB_PATH": "/path/to/project/data/synapse.db"
>       }
>     }
>   }
> }
> ```

### Direct CLI Execution

For terminal-based coding agents or stdio inspection:

```bash
synapse-mcp
```

### Available MCP Tools

| Tool | Description |
| :--- | :--- |
| `synapse_recall` | Hybrid semantic vector + BM25 search across all past memories, SOPs, and architectural decisions. |
| `synapse_retain` | Store an atomic rule, decision, or SOP with salience weighting. |
| `synapse_record_episode` | Record a structured episodic experience (*Situation → Action → Outcome → Lesson*). |
| `synapse_get_stats` | Retrieve real-time cortex statistics (active nodes, edges, vectors, superseded counts). |

---

## Configuration & Dual-Engine Architecture

Synapse operates using two complementary cognitive engines: **The Embedding Engine** (for dense associative vector search) and **The LLM Engine** (for subconscious reflection and mental model synthesis).

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SYNAPSE COGNITIVE ENGINES                       │
├──────────────────────────────────┬─────────────────────────────────────┤
│      1. Dense Embedding Engine   │          2. LLM Engine              │
│  (Vector Search & Association)   │ (Belief Revision & Reflection)      │
├──────────────────────────────────┼─────────────────────────────────────┤
│ Primary: sentence-transformers   │ Any OpenAI-Compatible Provider:     │
│ (all-MiniLM-L6-v2, 384-dim)      │ OpenAI, Ollama, vLLM, Groq, etc.    │
│                                  │                                     │
│ Fallback: Built-in pure-Python   │ Fallback: Built-in rule-based       │
│ CRC32 n-gram vectorizer          │ deterministic heuristics            │
│ (Zero dependencies, 0 token cost)│ (Operates offline without API keys) │
└──────────────────────────────────┴─────────────────────────────────────┘
```

### 1. The Embedding Engine (Vector Search)

Synapse generates 384-dimensional dense vectors to power hybrid retrieval (BM25 lexical + vector cosine similarity):

- **Zero-Dependency Built-In Fallback:** If no external libraries are installed, Synapse automatically uses a built-in deterministic subword n-gram vectorizer. It requires **zero external packages, zero model downloads, and zero API costs**.
- **Neural Embeddings (Optional):** If you install `sentence-transformers`, Synapse seamlessly upgrades to neural embeddings using `all-MiniLM-L6-v2` (or your configured `SYNAPSE_EMBEDDING_MODEL`):
  ```bash
  pip install sentence-transformers
  ```

### 2. The LLM Engine (Cognitive Synthesis)

The LLM is invoked during background consolidation to synthesize high-level Mental Models, extract abstract takeaways from episodic experiences, and resolve complex factual contradictions:

- **Provider-Agnostic:** Any standard OpenAI-compatible API is supported (local runners like Ollama, vLLM, LM Studio, or hosted endpoints like OpenAI, Groq, DeepSeek).
- **Graceful Fallback:** If no LLM endpoint or API key is provided, Synapse automatically uses deterministic rule-based heuristics. Core memory operations (`retain()`, `recall()`, `record_episode()`) never crash or block.

### Environment Configuration

Copy the environment template:

```bash
cp .env.example .env
```

Edit `.env` according to your requirements:

```bash
# SQLite Database location (default: ./data/synapse.db)
SYNAPSE_DB_PATH=./data/synapse.db

# Context Packing Token Budget
SYNAPSE_TOKEN_BUDGET=250

# Standard OpenAI-Compatible LLM Settings
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini

# (Optional) Dedicated Synapse Endpoint Override
# SYNAPSE_BASE_URL=http://localhost:11434/v1
# SYNAPSE_API_KEY=
# SYNAPSE_MODEL=

# (Optional) Embedding Model Override (defaults to all-MiniLM-L6-v2)
# SYNAPSE_EMBEDDING_MODEL=all-MiniLM-L6-v2

# (Optional) Markdown Knowledge Base / Vault Path
# SYNAPSE_VAULT_PATH=./vault
```

> **Note for Local Runners (Ollama, vLLM):** If you are running locally with Ollama or vLLM, set `OPENAI_BASE_URL=http://localhost:11434/v1` and leave `OPENAI_API_KEY` and `OPENAI_MODEL` blank to use your active model.

---

## Testing

Synapse comes with an exhaustive test suite covering all cognitive subsystems:

```bash
pytest
```

```text
============================= 301 passed in 6.30s =============================
```

- **Belief Revision & Contradiction Resolution:** `test_contradiction_resolver.py`
- **Counterfactual Guardrails:** `test_counterfactual_guardrail.py`
- **Spreading Activation & Retainment:** `test_synapse_retriever.py`, `test_synapse_storage.py`
- **Hebbian Reinforcement & Abstract Rollups:** `test_hebbian_and_rollup.py`, `test_hebbian_outcome_feedback.py`
- **Metacognitive Calibration:** `test_metacognitive_scorer.py`

---

## Technical Design & Honest Trade-Offs

To maintain transparency and help developers choose the right tool for their architecture:

1. **Semantic Search vs. Subword Fallback:**
   - The default zero-dependency vectorizer uses subword CRC32 n-gram hashing. It excels at fast keyword matching, spelling variations, and offline local operations without downloading heavy packages.
   - For true conceptual paraphrasing and cross-lingual understanding (e.g. "How is login token handled?" matching "Autentikasi API memakai JWT"), configure an OpenAI-compatible embedding endpoint (`OPENAI_EMBEDDING_MODEL` via Ollama or OpenAI) or install `sentence-transformers`.

2. **Automated Belief Revision Scope:**
   - Real-time arbitration resolves numeric attributes (ports, memory, limits), technology version updates (e.g. `PostgreSQL 14 → 16`), and task lifecycle states (`In Progress → Completed`).
   - Broad semantic belief revisions that require reasoning over prose are synthesized during background consolidation cycles with LLM assistance.

3. **Performance & Target Scale:**
   - Vector cosine similarity is computed in-memory across candidate graph nodes in Python.
   - This design is lightweight and optimal for personal and team agent memory graphs (<10,000 active nodes). It is not designed to replace high-scale multi-million vector databases (Milvus, Pinecone).

4. **Multi-Project Isolation:**
   - By default, `synapse-mcp` defaults to `~/.synapse/synapse.db` so multiple assistants share context across repositories.
   - For strictly isolated project environments, always set `SYNAPSE_DB_PATH=./data/synapse.db`.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
