"""Global test isolation: no test may touch the shared user environment."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_ENV_VARS = (
    "SYNAPSE_DB_PATH",
    "SYNAPSE_VAULT_PATH",
    "SYNAPSE_RESEARCH_DIR",
    "SYNAPSE_LOGS_DIR",
    "SYNAPSE_DATA_DIR",
    "SYNAPSE_SYNC_STATE_PATH",
    "SYNAPSE_TOKEN_BUDGET",
    "SYNAPSE_L0_BUDGET",
    "SYNAPSE_L1_BUDGET",
    "SYNAPSE_BASE_URL",
    "SYNAPSE_API_KEY",
    "SYNAPSE_MODEL",
    "SYNAPSE_EMBEDDING_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
)


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch, tmp_path: Path) -> None:
    """Redirect HOME-derived paths and strip Synapse env vars per test."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    if os.name == "nt":
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
