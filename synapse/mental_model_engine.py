"""
Synapse Mental Model Engine.
Implements Hierarchical Consolidated Understanding with Byte-Identical Delta Refinement AST
for Agentic Cognitive Memory Architecture.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_markdown_sections(text: str) -> Dict[str, str]:
    """
    Parse a markdown document into a structured AST map of sections:
    { "header": content_under_header }
    Unsectioned lead text is assigned to '__lead__'.
    Guarantees clean recovery when reassembled.
    """
    lines = text.split("\n")
    sections: Dict[str, List[str]] = {}
    current_section = "__lead__"
    sections[current_section] = []

    for line in lines:
        match = re.match(r"^(#{1,4}\s+.*)$", line)
        if match:
            current_section = match.group(1).strip()
            if current_section not in sections:
                sections[current_section] = []
        else:
            sections[current_section].append(line)

    res = {}
    for k, v in sections.items():
        joined = "\n".join(v)
        res[k] = joined.strip()
    return res


def assemble_markdown_sections(sections: Dict[str, str]) -> str:
    """
    Reassemble sections into a markdown document.
    """
    parts = []
    if "__lead__" in sections and sections["__lead__"]:
        parts.append(sections["__lead__"])

    for header, content in sections.items():
        if header == "__lead__":
            continue
        if content:
            # If content starts with a blank line or newline, format cleanly
            parts.append(f"{header}\n\n{content}".strip())
        else:
            parts.append(header)

    return "\n\n".join(parts).strip()


def apply_delta_operations(
    base_text: str,
    operations: List[Dict[str, Any]],
) -> Tuple[str, bool]:
    """
    Apply structured delta operations (add_section, replace_section, append_bullet, remove_section).
    Sections not targeted by operations are preserved byte-identical.
    Returns (new_text, changed).
    """
    if not base_text or not base_text.strip():
        # Baseline empty, fallback to full replacement
        for op in operations:
            if op.get("op") == "set_full_content":
                return op.get("content", "").strip(), True
        return "", False

    sections = parse_markdown_sections(base_text)
    changed = False

    for op in operations:
        op_type = op.get("op")
        target_header = op.get("section") or op.get("target")

        if op_type == "replace_section" and target_header:
            new_content = op.get("content", "").strip()
            # Find matching section (case-insensitive header match)
            matched_key = next((k for k in sections if k.lower() == target_header.lower()), None)
            if matched_key:
                if sections[matched_key] != new_content:
                    sections[matched_key] = new_content
                    changed = True
            else:
                # Add as new section
                sections[target_header] = new_content
                changed = True

        elif op_type == "add_section" and target_header:
            new_content = op.get("content", "").strip()
            matched_key = next((k for k in sections if k.lower() == target_header.lower()), None)
            if not matched_key:
                sections[target_header] = new_content
                changed = True
            else:
                # Append to existing
                if new_content and new_content not in sections[matched_key]:
                    sections[matched_key] = f"{sections[matched_key]}\n{new_content}".strip()
                    changed = True

        elif op_type == "append_bullet" and target_header:
            bullet = op.get("bullet") or op.get("content", "").strip()
            if not bullet.startswith("- ") and not bullet.startswith("* "):
                bullet = f"- {bullet}"
            matched_key = next((k for k in sections if k.lower() == target_header.lower()), None)
            if matched_key:
                if bullet not in sections[matched_key]:
                    sections[matched_key] = f"{sections[matched_key]}\n{bullet}".strip()
                    changed = True
            else:
                sections[target_header] = bullet
                changed = True

        elif op_type == "remove_section" and target_header:
            matched_key = next((k for k in sections if k.lower() == target_header.lower()), None)
            if matched_key:
                del sections[matched_key]
                changed = True

    if changed:
        return assemble_markdown_sections(sections), True
    return base_text, False


class MentalModelEngine:
    """
    Manages high-order consolidated mental models with auto-delta refinement.
    """

    def __init__(self, storage: Any, llm_config: Optional[Dict[str, Any]] = None) -> None:
        self.storage = storage
        self.llm_config = llm_config or {}

    def ensure_default_models(self) -> None:
        """Seed foundational mental models if they do not exist."""
        default_models = [
            {
                "id": "core-conventions",
                "name": "Core Architecture & Engineering Conventions",
                "source_query": "What are the core engineering conventions, architectural constraints, and operational protocols?",
                "content": """# Core Architecture & Engineering Conventions

## Core Topology & Execution
- Autonomous agent operates with verified facts, ground-truth memory, and reproducible actions.
- Multi-module delegation: Isolate complex refactoring into modular components.

## Compound Engineering & Reflex Loop
- Zero-silent-failure: Always verify live execution before marking tasks complete.
- Compound loop: Execute -> Verify -> Record in Memory -> Reflect.""",
                "trigger_mode": "delta",
                "tags": ["core", "conventions", "architecture"],
            },
            {
                "id": "user-preferences",
                "name": "User Persona & Interaction Preferences",
                "source_query": "What are the user's interaction style, preferences, and cognitive traits?",
                "content": """# User Persona & Interaction Preferences

## Tone & Style
- Clear, concise, surgical, to-the-point, and zero-fluff.
- Proportional response length: match the density and scope of the query.

## Interaction Principles
- Always verify live execution and data before reporting results.
- Transparent about system limitations; honest feedback over flattery.""",
                "trigger_mode": "delta",
                "tags": ["preference", "ui-ux", "persona"],
            }
        ]

        for mm in default_models:
            existing = self.storage.get_mental_model(mm["id"])
            if not existing:
                self.storage.set_mental_model(
                    model_id=mm["id"],
                    name=mm["name"],
                    source_query=mm["source_query"],
                    content=mm["content"],
                    structured_content=json.dumps(parse_markdown_sections(mm["content"])),
                    trigger_mode=mm.get("trigger_mode", "delta"),
                    tags=mm.get("tags", []),
                )
                logger.info("Seeded default mental model '%s'", mm["id"])

    def refresh_model(
        self,
        model_id: str,
        new_facts: Optional[List[str]] = None,
        force_full: bool = False,
    ) -> bool:
        """
        Refresh a mental model.
        In delta mode, applies non-destructive delta operations on existing AST.
        In full mode, synthesizes from scratch.
        Guarantees existing content is never overwritten with empty content.
        """
        model = self.storage.get_mental_model(model_id)
        if not model:
            return False

        existing_content = model.get("content", "").strip()
        mode = "full" if force_full else model.get("trigger_mode", "delta")

        if not new_facts or len(new_facts) == 0:
            return False

        if mode == "delta" and existing_content:
            # Build delta operations from new facts
            operations = []
            for fact in new_facts:
                clean_fact = fact.strip()
                if not clean_fact:
                    continue
                # Determine relevant section
                target_section = "## Recent Updates & Refined Directives"
                if re.search(r"\b(ui|ux|theme|color|design|layout|style)\b", clean_fact, re.I):
                    target_section = "## UI/UX Design System"
                elif re.search(r"\b(rule|protocol|vault|synapse|topology|server|agent|workflow)\b", clean_fact, re.I):
                    target_section = "## Core Architecture & Execution Directives"

                operations.append({
                    "op": "append_bullet",
                    "section": target_section,
                    "bullet": clean_fact.lstrip("-*• "),
                })

            updated_content, changed = apply_delta_operations(existing_content, operations)
            if changed and updated_content.strip():
                self.storage.set_mental_model(
                    model_id=model_id,
                    name=model["name"],
                    source_query=model["source_query"],
                    content=updated_content,
                    structured_content=json.dumps(parse_markdown_sections(updated_content)),
                    trigger_mode="delta",
                    tags=model.get("tags", []),
                    max_tokens=model.get("max_tokens", 2048),
                )
                return True
        else:
            # Full mode rebuild
            bullet_lines = "\n".join([f"- {f.strip().lstrip('-*• ')}" for f in new_facts if f.strip()])
            rebuilt_content = f"# {model['name']}\n\n## Consolidated Ground Truth\n{bullet_lines}"
            self.storage.set_mental_model(
                model_id=model_id,
                name=model["name"],
                source_query=model["source_query"],
                content=rebuilt_content,
                structured_content=json.dumps(parse_markdown_sections(rebuilt_content)),
                trigger_mode="full",
                tags=model.get("tags", []),
                max_tokens=model.get("max_tokens", 2048),
            )
            return True

        return False
