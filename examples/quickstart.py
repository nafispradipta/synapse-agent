#!/usr/bin/env python3
"""
Quickstart Example for Synapse Memory Engine.

Demonstrates:
1. Retaining architectural decisions and permanent rules.
2. Recording episodic experiences (Situation -> Action -> Outcome -> Lesson).
3. Recalling context under strict token budget (<250 tokens) with Spreading Activation.
"""

from synapse.adapter import SynapseAdapter

def main():
    # Initialize in-memory or file-backed Synapse Adapter
    adapter = SynapseAdapter({"db_path": ":memory:", "token_budget": 250})
    print("[INFO] Initialized Synapse Cognitive Memory...")

    # 1. Retain an architectural decision
    adapter.retain(
        content="Use SQLite WAL mode with synchronous=NORMAL for concurrent agent read/writes.",
        category="architecture",
        salience=0.95,
        title="SQLite Concurrency Rule"
    )
    print("[SUCCESS] Stored Architectural Decision.")

    # 2. Record an episodic learning event
    adapter.record_episode(
        title="Database Lock Incident Under High Load",
        situation="Agent workers were experiencing database locked errors during batch consolidation.",
        action_taken="Switched SQLite journal mode to WAL and increased busy timeout to 5000ms.",
        outcome="Lock contention eliminated completely across all concurrent workers.",
        lessons_learned="Always enable WAL mode and set busy_timeout >= 5000ms for multi-agent SQLite.",
        salience=0.95
    )
    print("[SUCCESS] Stored Episodic Learning Event.")

    # 3. Recall memory matching a query
    print("\n[QUERY] Recalling memory for query: 'How to prevent SQLite locking?'")
    recalled_blocks = adapter.recall("How to prevent SQLite locking?")
    
    print("\n[CONTEXT] Injected Context for Agent System Prompt:")
    for block in recalled_blocks:
        print("---")
        print(block)

if __name__ == "__main__":
    main()
