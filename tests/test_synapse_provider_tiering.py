"""
Unit tests for Generic Universal Provider Tiering & Cross-Platform Resolving
in Synapse Cognitive Memory Core.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synapse import (
    SynapseAdapter,
    SynapseStorage,
    SynapseWorker,
    resolve_synapse_llm_config,
)


@pytest.fixture(autouse=True)
def clean_synapse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear all relevant environment variables before each test."""
    env_keys = [
        "SYNAPSE_BASE_URL",
        "SYNAPSE_API_KEY",
        "SYNAPSE_MODEL",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "BASE_URL",
        "API_KEY",
        "MODEL_NAME",
        "MODEL",
        "SYNAPSE_DB_PATH",
        "SYNAPSE_VAULT_PATH",
    ]
    for key in env_keys:
        monkeypatch.delenv(key, raising=False)


def test_resolve_llm_config_defaults() -> None:
    """Test LLM config resolution with no environment variables and empty config."""
    config = resolve_synapse_llm_config()
    assert config["base_url"] == ""
    assert config["api_key"] == ""
    assert config["model"] == ""

    # Also test with empty dictionary
    config_empty = resolve_synapse_llm_config({})
    assert config_empty["base_url"] == ""
    assert config_empty["api_key"] == ""
    assert config_empty["model"] == ""


def test_resolve_llm_config_openai_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test LLM config resolution with OPENAI_* and MODEL fallback env vars."""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai-proxy.internal:8000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test-key-12345")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    config = resolve_synapse_llm_config()
    assert config["base_url"] == "http://openai-proxy.internal:8000/v1"
    assert config["api_key"] == "sk-openai-test-key-12345"
    assert config["model"] == "gpt-4o-mini"

    # Test fallback from OPENAI_MODEL to generic MODEL if OPENAI_MODEL is not set
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setenv("MODEL", "custom-model-v1")
    config_model_fallback = resolve_synapse_llm_config()
    assert config_model_fallback["model"] == "custom-model-v1"


def test_resolve_llm_config_legacy_env_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test LLM config resolution with legacy keys commonly found in env (BASE_URL, API_KEY, MODEL_NAME)."""
    monkeypatch.setenv("BASE_URL", "http://legacy-env-host:11434/v1")
    monkeypatch.setenv("API_KEY", "sk-legacy-env-key-999")
    monkeypatch.setenv("MODEL_NAME", "legacy-model-v1")

    config = resolve_synapse_llm_config()
    assert config["base_url"] == "http://legacy-env-host:11434/v1"
    assert config["api_key"] == "sk-legacy-env-key-999"
    assert config["model"] == "legacy-model-v1"


def test_resolve_llm_config_full_precedence_tiering(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test complete fallback resolution tiering for each config dimension:
    - base_url: SYNAPSE_BASE_URL -> OPENAI_BASE_URL -> BASE_URL -> default
    - api_key: SYNAPSE_API_KEY -> OPENAI_API_KEY -> API_KEY -> default
    - model: SYNAPSE_MODEL -> OPENAI_MODEL -> MODEL_NAME -> MODEL -> default
    """
    # 1. Start with lowest-tier legacy / generic variables
    monkeypatch.setenv("BASE_URL", "http://tier-legacy:8000/v1")
    monkeypatch.setenv("API_KEY", "key-tier-legacy")
    monkeypatch.setenv("MODEL", "model-tier-generic-model")
    cfg = resolve_synapse_llm_config()
    assert cfg["base_url"] == "http://tier-legacy:8000/v1"
    assert cfg["api_key"] == "key-tier-legacy"
    assert cfg["model"] == "model-tier-generic-model"

    # MODEL_NAME should take precedence over MODEL
    monkeypatch.setenv("MODEL_NAME", "model-tier-legacy-name")
    cfg = resolve_synapse_llm_config()
    assert cfg["model"] == "model-tier-legacy-name"

    # 2. OPENAI_* tier should take precedence over legacy keys
    monkeypatch.setenv("OPENAI_BASE_URL", "http://tier-openai:8000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "key-tier-openai")
    monkeypatch.setenv("OPENAI_MODEL", "model-tier-openai")
    cfg = resolve_synapse_llm_config()
    assert cfg["base_url"] == "http://tier-openai:8000/v1"
    assert cfg["api_key"] == "key-tier-openai"
    assert cfg["model"] == "model-tier-openai"

    # 3. SYNAPSE_* tier should take precedence over OPENAI_* tier
    monkeypatch.setenv("SYNAPSE_BASE_URL", "http://tier-synapse:8000/v1")
    monkeypatch.setenv("SYNAPSE_API_KEY", "key-tier-synapse")
    monkeypatch.setenv("SYNAPSE_MODEL", "model-tier-synapse")
    cfg = resolve_synapse_llm_config()
    assert cfg["base_url"] == "http://tier-synapse:8000/v1"
    assert cfg["api_key"] == "key-tier-synapse"
    assert cfg["model"] == "model-tier-synapse"

    # 4. Explicit dictionary argument should take precedence over everything
    cfg = resolve_synapse_llm_config({
        "base_url": "http://tier-explicit:8000/v1",
        "api_key": "key-tier-explicit",
        "model": "model-tier-explicit",
    })
    assert cfg["base_url"] == "http://tier-explicit:8000/v1"
    assert cfg["api_key"] == "key-tier-explicit"
    assert cfg["model"] == "model-tier-explicit"


def test_resolve_llm_config_dict_override_highest_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that explicit dictionary config takes highest precedence over all env vars."""
    # Set env vars
    monkeypatch.setenv("SYNAPSE_BASE_URL", "http://env-synapse:9090/v1")
    monkeypatch.setenv("SYNAPSE_API_KEY", "sk-env-synapse")
    monkeypatch.setenv("SYNAPSE_MODEL", "env-model-synapse")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env-openai:8000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-openai")
    monkeypatch.setenv("OPENAI_MODEL", "env-model-openai")

    # 1. Direct dictionary keys
    explicit_config = {
        "base_url": "http://direct-cfg-host:4000/v1",
        "api_key": "sk-direct-cfg-key",
        "model": "qwen-2.5-72b-instruct",
    }
    resolved = resolve_synapse_llm_config(explicit_config)
    assert resolved["base_url"] == "http://direct-cfg-host:4000/v1"
    assert resolved["api_key"] == "sk-direct-cfg-key"
    assert resolved["model"] == "qwen-2.5-72b-instruct"

    # 2. Nested llm_config dictionary
    nested_config = {
        "llm_config": {
            "base_url": "http://nested-cfg-host:5000/v1",
            "api_key": "sk-nested-key",
            "model": "mistral-large-2407",
        }
    }
    resolved_nested = resolve_synapse_llm_config(nested_config)
    assert resolved_nested["base_url"] == "http://nested-cfg-host:5000/v1"
    assert resolved_nested["api_key"] == "sk-nested-key"
    assert resolved_nested["model"] == "mistral-large-2407"


def test_cross_platform_path_home_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Test 100% cross-platform path resolution using Path.home()
    across SynapseStorage, SynapseWorker, and SynapseAdapter.
    """
    mock_home = tmp_path / "mock_user_home"
    mock_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: mock_home)

    # 1. SynapseStorage default path resolution
    storage = SynapseStorage()
    expected_db_path = str((mock_home / ".synapse" / "synapse.db").resolve())
    assert storage.db_path == expected_db_path
    assert Path(storage.db_path).parent.exists()
    storage.close()

    # 2. SynapseAdapter default db_path resolution
    adapter = SynapseAdapter()
    assert adapter.db_path == expected_db_path
    assert adapter.storage is not None
    assert adapter.storage.db_path == expected_db_path
    if adapter.storage:
        adapter.storage.close()

    # 3. SynapseWorker sync state & vault path resolution
    worker = SynapseWorker()
    default_sync_state = (mock_home / ".synapse" / "synapse_last_sync.json").resolve()
    mock_vault = mock_home / "vault"
    mock_vault.mkdir(parents=True, exist_ok=True)

    # Put a sample markdown SOP in mock vault
    sop_file = mock_vault / "deploy_guide.md"
    sop_file.write_text("# Deploy Guide\nFollow step by step deployment procedures.", encoding="utf-8")

    # Run in-process maintenance with default paths
    res = worker.run_in_process_maintenance_if_needed(non_blocking=False)
    assert res is True
    assert default_sync_state.exists()


def test_adapter_and_worker_llm_config_integration(tmp_path: Path) -> None:
    """Test that SynapseAdapter and SynapseWorker properly hold and use resolved LLM configs."""
    custom_cfg = {
        "db_path": str(tmp_path / "synapse_llm_test.db"),
        "base_url": "http://custom-test-host:20128/v1",
        "api_key": "sk-custom-test-key",
        "model": "custom-model-synapse",
    }

    adapter = SynapseAdapter(custom_cfg)
    assert adapter.llm_config["base_url"] == "http://custom-test-host:20128/v1"
    assert adapter.llm_config["api_key"] == "sk-custom-test-key"
    assert adapter.llm_config["model"] == "custom-model-synapse"

    assert adapter.worker is not None
    assert adapter.worker.llm_config["base_url"] == "http://custom-test-host:20128/v1"
    assert adapter.worker.llm_config["api_key"] == "sk-custom-test-key"
    assert adapter.worker.llm_config["model"] == "custom-model-synapse"

    # Retain a node and test reflect behavior
    adapter.retain("PostgreSQL supports JSONB for semi-structured document storage.", context="Database")
    reflection = adapter.reflect("PostgreSQL")
    assert "PostgreSQL" in reflection
    assert "Cognitive Synapse Reflection" in reflection
