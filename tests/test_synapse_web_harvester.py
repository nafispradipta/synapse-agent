"""
Comprehensive test suite for Tier 4 Web Harvester & External Research module.
Tests strict whitelist enforcement, Vault integration, and DB ingestion.
"""

import os
import tempfile
from pathlib import Path

import pytest

from synapse.storage import SynapseStorage
from synapse.harvester import SynapseWebHarvester, DEFAULT_WHITELIST_DOMAINS


@pytest.fixture
def temp_storage():
    """Create in-memory SQLite storage for testing."""
    storage = SynapseStorage(":memory:")
    yield storage
    storage.close()


@pytest.fixture
def temp_vault(tmp_path):
    """Create temporary vault research directory."""
    vault_dir = tmp_path / "research" / "auto_harvest"
    vault_dir.mkdir(parents=True, exist_ok=True)
    return str(vault_dir)


@pytest.fixture
def harvester(temp_storage, temp_vault):
    """Create SynapseWebHarvester instance with test config."""
    return SynapseWebHarvester(
        storage=temp_storage,
        vault_research_dir=temp_vault,
        llm_config={"base_url": "http://localhost:9999/v1", "api_key": "test_key", "model": "test-model"},
    )


class TestWhitelistEnforcement:
    """Test strict domain whitelist guardrails."""

    def test_whitelist_includes_github(self, harvester):
        """GitHub should be in the default whitelist."""
        assert harvester.is_domain_allowed("https://github.com/user/repo")

    def test_whitelist_includes_arxiv(self, harvester):
        """ArXiv should be in the default whitelist."""
        assert harvester.is_domain_allowed("https://arxiv.org/abs/2512.12818")

    def test_whitelist_blocks_unknown_domain(self, harvester):
        """Unknown/random domains should be blocked."""
        assert not harvester.is_domain_allowed("https://random-blog-slop.example.com/article")

    def test_whitelist_blocks_marketing_seo(self, harvester):
        """Marketing/SEO content domains should be blocked."""
        assert not harvester.is_domain_allowed("https://medium.com/@user/some-article")

    def test_whitelist_subdomain_allowed(self, harvester):
        """Subdomains of whitelisted domains should be allowed."""
        assert harvester.is_domain_allowed("https://docs.python.org/3/library/")
        assert harvester.is_domain_allowed("https://raw.githubusercontent.com/user/repo/main/file.py")


class TestHarvestItem:
    """Test harvest_item end-to-end with mock content."""

    def test_harvest_blocked_domain_returns_none(self, harvester):
        """Harvesting from non-whitelisted domain should return None."""
        result = harvester.harvest_item(
            topic="test",
            title="Blocked Source",
            source_url="https://random-slop.example.com/article",
            raw_content="Some content here that is long enough to pass the minimum length check.",
        )
        assert result is None

    def test_harvest_with_raw_content_produces_vault_file(self, harvester, temp_vault):
        """Harvesting with raw_content should create a Vault Markdown file."""
        result = harvester.harvest_item(
            topic="Go Concurrency Patterns",
            title="Goroutine Best Practices",
            source_url="https://go.dev/blog/pipelines",
            raw_content="This is a comprehensive guide on goroutine pipelines and concurrency patterns in Go. It covers fan-in fan-out, cancellation, and timeout patterns.",
        )

        assert result is not None
        assert result["topic"] == "Go Concurrency Patterns"
        assert result["status"] == "quarantined"
        assert result["tier_level"] == 4
        assert result["domain"] == "go.dev"

        # Verify Vault file was created
        vault_path = Path(result["vault_path"])
        assert vault_path.exists()
        content = vault_path.read_text()
        assert "tier: 4" in content
        assert "status: quarantined" in content
        assert "Go Concurrency Patterns" in content or "go-concurrency-patterns" in content.lower()

    def test_harvest_too_short_content_returns_none(self, harvester):
        """Content too short should return None."""
        result = harvester.harvest_item(
            topic="test",
            title="Short",
            source_url="https://github.com/user/repo",
            raw_content="ab",
        )
        assert result is None

    def test_harvest_records_in_db(self, harvester, temp_storage):
        """Harvested item should be recorded in synapse_external_research table."""
        result = harvester.harvest_item(
            topic="Python Async",
            title="asyncio Best Practices",
            source_url="https://docs.python.org/3/library/asyncio.html",
            raw_content="asyncio is a library to write concurrent code using async/await syntax. It is used as a foundation for multiple asynchronous frameworks.",
        )

        assert result is not None

        # Verify DB record
        records = temp_storage.list_external_research(topic="Python Async")
        assert len(records) >= 1
        assert records[0]["title"] == "asyncio Best Practices"
        assert records[0]["domain"] == "docs.python.org"
        assert records[0]["status"] == "quarantined"


class TestStorageExternalResearch:
    """Test storage layer methods for external research."""

    def test_add_external_research(self, temp_storage):
        """Test adding external research record."""
        result = temp_storage.add_external_research(
            topic="Python Concurrency",
            title="AsyncIO Architecture in Python",
            source_url="https://python.org/doc/asyncio",
            domain="python.org",
            summary="Comprehensive guide on AsyncIO task scheduling.",
        )
        assert result["id"].startswith("res:")
        assert result["status"] == "quarantined"

    def test_list_external_research_filter_by_topic(self, temp_storage):
        """Test listing external research filtered by topic."""
        temp_storage.add_external_research(
            topic="Go",
            title="Go Article 1",
            source_url="https://go.dev/blog/1",
            domain="go.dev",
            summary="Summary 1",
        )
        temp_storage.add_external_research(
            topic="Python",
            title="Python Article 1",
            source_url="https://docs.python.org/3/",
            domain="docs.python.org",
            summary="Summary 2",
        )

        go_records = temp_storage.list_external_research(topic="Go")
        assert len(go_records) == 1
        assert go_records[0]["topic"] == "Go"

    def test_list_external_research_filter_by_status(self, temp_storage):
        """Test listing external research filtered by status."""
        temp_storage.add_external_research(
            topic="Test",
            title="Quarantined Item",
            source_url="https://github.com/test/repo",
            domain="github.com",
            summary="Summary",
            status="quarantined",
        )
        temp_storage.add_external_research(
            topic="Test",
            title="Verified Item",
            source_url="https://github.com/test/repo2",
            domain="github.com",
            summary="Summary 2",
            status="verified",
        )

        quarantined = temp_storage.list_external_research(status="quarantined")
        assert all(r["status"] == "quarantined" for r in quarantined)

        verified = temp_storage.list_external_research(status="verified")
        assert all(r["status"] == "verified" for r in verified)


class TestAdapterIntegration:
    """Test SynapseAdapter integration with harvester."""

    def test_adapter_get_harvester(self):
        """Test that SynapseAdapter can instantiate harvester."""
        from synapse.adapter import SynapseAdapter

        adapter = SynapseAdapter({"db_path": ":memory:"})
        harvester = adapter.get_harvester()
        assert harvester is not None
        assert harvester.storage is not None

    def test_adapter_harvest_research(self):
        """Test adapter.harvest_research method."""
        from synapse.adapter import SynapseAdapter

        adapter = SynapseAdapter({"db_path": ":memory:"})
        result = adapter.harvest_research(
            topic="Test Topic",
            title="Test Title",
            source_url="https://github.com/test/repo",
            raw_content="This is test content for the harvest research method. It should be long enough to pass the minimum length threshold.",
        )
        assert result is not None
        assert result["topic"] == "Test Topic"

    def test_adapter_list_external_research(self):
        """Test adapter.list_external_research method."""
        from synapse.adapter import SynapseAdapter

        adapter = SynapseAdapter({"db_path": ":memory:"})
        adapter.harvest_research(
            topic="Query Topic",
            title="Query Title",
            source_url="https://arxiv.org/abs/1234",
            raw_content="This is test content for listing external research records from the synapse adapter module.",
        )
        records = adapter.list_external_research(topic="Query Topic")
        assert len(records) >= 1
