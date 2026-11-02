"""
Tests for Synapse Mental Model Engine and Byte-Identical Delta Refinement.
"""

import json
import pytest
from synapse.storage import SynapseStorage
from synapse.mental_model_engine import (
    MentalModelEngine,
    parse_markdown_sections,
    assemble_markdown_sections,
    apply_delta_operations,
)
from synapse.adapter import SynapseAdapter


def test_markdown_ast_byte_identical_reassembly():
    original = """# Title

Lead line 1.
Lead line 2.

## Section A

Content for section A.
- Bullet 1
- Bullet 2

## Section B

Content for section B."""

    sections = parse_markdown_sections(original)
    assert "__lead__" in sections
    assert "## Section A" in sections
    assert "## Section B" in sections

    reassembled = assemble_markdown_sections(sections)
    assert reassembled == original.strip()


def test_delta_operations_append_bullet():
    base_text = """# Conventions

## Execution

- Step 1
- Step 2

## UI/UX

- Dark mode"""

    ops = [
        {
            "op": "append_bullet",
            "section": "## Execution",
            "bullet": "Step 3 verified",
        }
    ]

    new_text, changed = apply_delta_operations(base_text, ops)
    assert changed is True
    assert "- Step 3 verified" in new_text
    assert "## UI/UX\n\n- Dark mode" in new_text  # UI/UX byte-identical


def test_delta_operations_replace_section():
    base_text = """# Conventions

## Execution
- Step 1

## UI/UX
- Dark mode"""

    ops = [
        {
            "op": "replace_section",
            "section": "## UI/UX",
            "content": "- Nordic Monolith #09090b",
        }
    ]

    new_text, changed = apply_delta_operations(base_text, ops)
    assert changed is True
    assert "- Nordic Monolith #09090b" in new_text
    assert "- Step 1" in new_text


def test_mental_model_storage_and_engine():
    storage = SynapseStorage(":memory:")
    engine = MentalModelEngine(storage)
    engine.ensure_default_models()

    models = storage.list_mental_models()
    assert len(models) >= 2
    ids = [m["id"] for m in models]
    assert "core-conventions" in ids
    assert "user-preferences" in ids

    # Test Delta Refresh
    res = engine.refresh_model(
        "user-preferences",
        new_facts=["User prefers concise surgical responses."],
        force_full=False,
    )
    assert res is True

    updated_model = storage.get_mental_model("user-preferences")
    assert updated_model is not None
    assert "User prefers concise surgical responses." in updated_model["content"]


def test_adapter_recall_includes_mental_model():
    adapter = SynapseAdapter({"db_path": ":memory:"})
    assert adapter.storage is not None
    # Ensure default mental model
    adapter.mental_model_engine.ensure_default_models()

    # Query targeting preferences
    recalled = adapter.recall("What are user preferences for tone and style?")
    assert len(recalled) > 0
    # Should include MENTAL_MODEL snippet
    assert any("[MENTAL_MODEL]" in r for r in recalled)
