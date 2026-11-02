"""
Synapse Session Helper & Lifecycle Integration.
Provides reference SessionManager for managing agent sessions, memory hooks, and prompt generation.
"""

from __future__ import annotations

import datetime
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from synapse.loader import SynapseContextLoader, estimate_tokens
from synapse.worker import SynapseWorker


class SessionManager:
    """
    Session Manager handling session lifecycle hooks, memory injection,
    and background Synapse maintenance.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        config_dir: Optional[Path] = None,
        soul_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:8]
        raw_dir = config_dir or os.environ.get("SYNAPSE_CONFIG_DIR") or (Path.home() / ".synapse")
        self.config_dir = Path(raw_dir).expanduser().resolve()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.soul_config = soul_config or {
            "identity": {"name": "Agent", "archetype": "Autonomous Agent"}
        }
        self.recalled_memories: List[str] = []
        self.recalled_episodes: List[str] = []
        self._loader = SynapseContextLoader()
        self.on_session_start()

    def on_session_start(self) -> bool:
        """Lifecycle hook invoked when initializing an agent session."""
        return self._trigger_synapse_lifecycle_maintenance()

    def _trigger_synapse_lifecycle_maintenance(self) -> bool:
        """Safely trigger Synapse background maintenance (vault sync & decay cycle) if needed."""
        try:
            sync_state = self.config_dir / "synapse_last_sync.json"
            return SynapseWorker.run_in_process_maintenance_if_needed(
                non_blocking=True,
                sync_state_path=sync_state,
            )
        except Exception:
            return False

    def set_recalled_memories(self, memories: List[str]) -> None:
        """Set recalled long-term memories retrieved from Synapse."""
        self.recalled_memories = list(memories)

    def set_recalled_episodes(
        self,
        episodes: List[Union[str, Dict[str, Any]]],
        token_budget: int = 250,
        include_l1: bool = True,
    ) -> None:
        """
        Format recalled episodic memories into concise L0/L1 format under strict token budget (<250 tokens).
        """
        self.recalled_episodes.clear()
        if not episodes:
            return

        current_tokens = 0
        for ep in episodes:
            if isinstance(ep, str):
                ep_text = ep.strip()
                t_count = estimate_tokens(ep_text)
                if current_tokens + t_count <= token_budget:
                    self.recalled_episodes.append(ep_text)
                    current_tokens += t_count
                else:
                    break
            elif isinstance(ep, dict):
                l0 = self._loader.format_episode_l0(ep)
                l1 = self._loader.format_episode_l1(ep) if include_l1 else ""
                combined = f"{l0}\n{l1}" if l1 else l0
                t_count = estimate_tokens(combined)
                if current_tokens + t_count <= token_budget:
                    self.recalled_episodes.append(combined)
                    current_tokens += t_count
                else:
                    # Fallback to only L0
                    l0_tokens = estimate_tokens(l0)
                    if current_tokens + l0_tokens <= token_budget:
                        self.recalled_episodes.append(l0)
                        current_tokens += l0_tokens
                    else:
                        break

    def clear_recalled_memories(self) -> None:
        """Clear recalled memory cache."""
        self.recalled_memories.clear()
        self.recalled_episodes.clear()

    def build_system_prompt(self, tool_names: Optional[List[str]] = None) -> str:
        """Generate dynamic system prompt combining soul persona, runtime environment, and long-term memory."""
        identity = self.soul_config.get("identity", {})
        name = identity.get("name", "Agent")
        archetype = identity.get("archetype", "Autonomous Agent")
        directives = self.soul_config.get("directives", [])

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"You are {name}, an {archetype}.",
            "",
            "## Identity & Persona:",
            f"- Name: {name}",
        ]

        if directives:
            lines.extend([
                "",
                "## Core Directives:",
            ])
            for d in directives:
                lines.append(f"- {d}")

        if self.recalled_memories:
            lines.extend([
                "",
                "## Relevant Long-term Context (Synapse Long-term Memory):",
                "- When answering using facts recalled from Synapse / Vault memory with [ref: <source>], provide a clean 1-line source attribution at the bottom of the answer (e.g. Reference: <source>).",
            ])
            for mem in self.recalled_memories:
                lines.append(f"- {mem}")

        if self.recalled_episodes:
            lines.extend([
                "",
                "## Relevant Episodic Experiences (Cognitive Episodes):",
            ])
            for ep in self.recalled_episodes:
                lines.append(f"- {ep}")

        if tool_names:
            lines.extend([
                "",
                "## Available Tools:",
                f"- You have access to tools: {', '.join(tool_names)}.",
            ])

        return "\n".join(lines)
