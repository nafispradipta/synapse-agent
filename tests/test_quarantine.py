"""Tier 4 harvest output must not enter active recall."""

from __future__ import annotations

from pathlib import Path

from synapse.adapter import SynapseAdapter


def test_harvested_node_is_flagged_and_not_recallable(tmp_path: Path) -> None:
    adapter = SynapseAdapter(
        {
            "db_path": ":memory:",
            "research_dir": str(tmp_path / "research"),
            "llm_config": {"base_url": "", "api_key": "", "model": ""},
        }
    )
    try:
        adapter.harvest_research(
            topic="kubernetes",
            title="Kubernetes architecture notes",
            source_url="https://python.org/kubernetes-architecture",
            raw_content="Kubernetes control planes are composed of an API server, " * 8,
        )
        flagged = adapter.storage.list_nodes(status="flagged")
        assert len(flagged) == 1
        node = flagged[0]
        assert node["category"] == "sop"
        assert adapter.storage.list_nodes(status="active") == []

        recalled = adapter.recall("kubernetes control plane", limit=5)
        assert all("Kubernetes architecture notes" not in str(item) for item in recalled)
    finally:
        adapter.storage.close()
