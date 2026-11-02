"""Automated MOC Index Bookkeeper for Synapse Memory & Markdown Knowledge Base.

Maintains Map of Content (MOC) index files across core vault domains:
- projects: ~/vault/projects/00-INDEX-PROJECTS.md (or projek: 00-INDEX-PROJEK.md)
- server: ~/vault/server/00-INDEX-SERVER.md
- cybersecurity: ~/vault/cybersecurity/00-INDEX-SOP-CYBERSECURITY.md
- daily: ~/vault/daily/00-INDEX-DAILY.md

Adopts the Karpathy LLM Wiki pattern to ensure Zero Orphan Notes.
Extracts 1-line deterministic summaries and inserts wikilinks [[note-name]]
into the appropriate domain index sections cleanly and idempotently.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set


class MOCBookkeeper:
    """Deterministic Map of Content (MOC) Bookkeeper for Obsidian Vault."""

    def __init__(
        self,
        vault_dir: Optional[str] = None,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        raw_vault = vault_dir or os.environ.get("SYNAPSE_VAULT_PATH") or os.environ.get("OBSIDIAN_VAULT_DIR") or "~/vault"
        self.vault_dir = Path(os.path.expanduser(raw_vault)).resolve()
        self.log = log_fn or print

    @staticmethod
    def extract_summary(file_path: Path) -> str:
        """Extract a clean, concise 1-line summary from frontmatter or content."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return file_path.stem.replace("-", " ").title()

        # 1. Frontmatter extraction
        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        fm_title: Optional[str] = None
        fm_desc: Optional[str] = None
        body = content

        if fm_match:
            fm_text = fm_match.group(1)
            body = content[fm_match.end() :]
            t_m = re.search(r'^title:\s*["\']?(.*?)["\']?\s*$', fm_text, re.MULTILINE)
            if t_m:
                fm_title = t_m.group(1).strip()
            d_m = re.search(
                r'^(?:description|summary):\s*["\']?(.*?)["\']?\s*$',
                fm_text,
                re.MULTILINE,
            )
            if d_m:
                fm_desc = d_m.group(1).strip()

        # 2. Heading & first non-empty paragraph
        h1_title: Optional[str] = None
        first_para: Optional[str] = None

        for line in body.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith("# ") and not h1_title:
                h1_title = re.sub(r"^#\s+", "", line_str).strip()
                continue
            if (
                not line_str.startswith("#")
                and not line_str.startswith(">")
                and not line_str.startswith("```")
                and not line_str.startswith("---")
                and not line_str.startswith("|")
                and not line_str.startswith("- ")
                and not line_str.startswith("* ")
            ):
                if not first_para:
                    first_para = line_str
                    if len(first_para) > 130:
                        first_para = first_para[:127] + "..."
                    break

        summary = (
            fm_desc
            or fm_title
            or h1_title
            or first_para
            or file_path.stem.replace("-", " ").title()
        )

        # 3. Clean wikilinks, formatting, markdown markers
        summary = re.sub(r"\[\[(?:[^|\]]*\|)?(.*?)\]\]", r"\1", summary)
        summary = re.sub(r"[*_`#]", "", summary).strip()
        summary = re.sub(r"\s+", " ", summary)
        if len(summary) > 140:
            summary = summary[:137] + "..."
        return summary

    @staticmethod
    def get_indexed_stems(index_path: Path) -> Set[str]:
        """Parse all linked target stems from an index file."""
        if not index_path.exists():
            return set()
        content = index_path.read_text(encoding="utf-8", errors="replace")
        stems: Set[str] = set()
        for match in re.finditer(r"\[\[(.*?)\]\]", content):
            raw = match.group(1).split("|")[0].strip()
            stems.add(raw)
            stems.add(Path(raw).name)
            stems.add(Path(raw).stem)
        return stems

    def _is_index_or_ignored(self, file_path: Path) -> bool:
        """Check if file should be ignored during indexing."""
        name = file_path.name.lower()
        if (
            name.startswith("00-index")
            or name.startswith("00 - index")
            or name == "readme.md"
            or name.startswith(".")
        ):
            return True
        try:
            rel_parts = file_path.relative_to(self.vault_dir).parts
            for part in rel_parts:
                if part.startswith(".") and part != ".":
                    return True
        except ValueError:
            name_check = file_path.name
            if name_check.startswith(".") and name_check != ".":
                return True
        return False

    @staticmethod
    def _atomic_write(file_path: Path, content: str) -> None:
        """Atomically write content to file via temporary file replace."""
        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, file_path)

    @staticmethod
    def _insert_into_section(
        content: str,
        section_regex: str,
        new_entry: str,
        fallback_header: Optional[str] = None,
        prepend: bool = False,
    ) -> str:
        """Insert a markdown bullet into a specific section cleanly."""
        lines = content.splitlines(keepends=True)
        header_idx = -1

        for i, line in enumerate(lines):
            if re.search(section_regex, line, re.IGNORECASE):
                header_idx = i
                break

        if header_idx == -1:
            if fallback_header:
                if not content.endswith("\n\n"):
                    content += "\n\n" if not content.endswith("\n") else "\n"
                content += f"{fallback_header}\n{new_entry}\n"
                return content
            else:
                content += f"\n{new_entry}\n"
                return content

        if prepend:
            insert_idx = header_idx + 1
            while insert_idx < len(lines) and not lines[insert_idx].strip():
                insert_idx += 1
            lines.insert(insert_idx, new_entry + ("\n" if not new_entry.endswith("\n") else ""))
            return "".join(lines)

        header_line = lines[header_idx]
        header_level = len(header_line) - len(header_line.lstrip("#"))
        last_bullet_idx = -1

        for j in range(header_idx + 1, len(lines)):
            curr = lines[j]
            curr_stripped = curr.strip()
            if curr_stripped.startswith("#"):
                curr_level = len(curr_stripped) - len(curr_stripped.lstrip("#"))
                if curr_level <= header_level:
                    break
            if curr_stripped == "---":
                break
            if curr_stripped.startswith("- ") or curr_stripped.startswith("* "):
                last_bullet_idx = j

        if last_bullet_idx != -1:
            insert_idx = last_bullet_idx + 1
        else:
            insert_idx = header_idx + 1
            while insert_idx < len(lines) and not lines[insert_idx].strip():
                insert_idx += 1

        lines.insert(insert_idx, new_entry + ("\n" if not new_entry.endswith("\n") else ""))
        return "".join(lines)

    def sync_daily(self) -> List[str]:
        """Scan ~/vault/daily/ and maintain 00-INDEX-DAILY.md."""
        domain_dir = self.vault_dir / "daily"
        index_file = domain_dir / "00-INDEX-DAILY.md"
        if not index_file.exists():
            return []

        indexed = self.get_indexed_stems(index_file)
        added: List[str] = []
        md_files = sorted(
            [f for f in domain_dir.glob("*.md") if not self._is_index_or_ignored(f)],
            key=lambda x: x.name,
            reverse=True,
        )

        content = index_file.read_text(encoding="utf-8")
        for f in md_files:
            if f.stem in indexed:
                continue
            summary = self.extract_summary(f)
            entry = f"- [[{f.stem}]] — {summary}"
            content = self._insert_into_section(
                content,
                r"##\s*.*(?:Daily Reflection Log Chronology|Kronologi Log Refleksi Harian)",
                entry,
                fallback_header="## Daily Reflection Log Chronology",
                prepend=True,
            )
            indexed.add(f.stem)
            added.append(f.name)

        if added:
            self._atomic_write(index_file, content)
            self.log(f"MOC Bookkeeper: Added {len(added)} note(s) to {index_file.name}")
        return added

    def sync_server(self) -> List[str]:
        """Scan ~/vault/server/ and maintain 00-INDEX-SERVER.md."""
        domain_dir = self.vault_dir / "server"
        index_file = domain_dir / "00-INDEX-SERVER.md"
        if not index_file.exists():
            return []

        indexed = self.get_indexed_stems(index_file)
        added: List[str] = []
        md_files = sorted(
            [f for f in domain_dir.rglob("*.md") if not self._is_index_or_ignored(f)],
            key=lambda x: x.stem,
        )

        content = index_file.read_text(encoding="utf-8")
        for f in md_files:
            if f.stem in indexed:
                continue
            summary = self.extract_summary(f)
            entry = (
                f"- [[{f.stem}]] — {summary}"
                if summary != f.stem
                else f"- [[{f.stem}]]"
            )

            if "_arsip" in f.parts or "_archive" in f.parts:
                content = self._insert_into_section(
                    content,
                    r"##\s*.*(?:Archive & Historical Documents|Arsip & Dokumen Historis)",
                    entry,
                    fallback_header="## Archive & Historical Documents",
                )
            else:
                lines = content.splitlines(keepends=True)
                arsip_idx = -1
                for i, line in enumerate(lines):
                    if re.search(r"##\s*.*(?:Archive|Arsip)", line, re.IGNORECASE):
                        arsip_idx = i
                        break
                if arsip_idx != -1:
                    last_b = -1
                    for j in range(arsip_idx - 1, -1, -1):
                        if lines[j].strip().startswith("- "):
                            last_b = j
                            break
                    ins_at = (last_b + 1) if last_b != -1 else arsip_idx
                    lines.insert(ins_at, entry + "\n")
                    content = "".join(lines)
                else:
                    content = self._insert_into_section(
                        content,
                        r"(?:List of connected documents|Berikut adalah daftar dokumen)",
                        entry,
                        fallback_header="## Server Documents",
                    )
            indexed.add(f.stem)
            added.append(f.name)

        if added:
            self._atomic_write(index_file, content)
            self.log(f"MOC Bookkeeper: Added {len(added)} note(s) to {index_file.name}")
        return added

    def sync_cybersecurity(self) -> List[str]:
        """Scan ~/vault/cybersecurity/ and maintain 00-INDEX-SOP-CYBERSECURITY.md."""
        domain_dir = self.vault_dir / "cybersecurity"
        index_file = domain_dir / "00-INDEX-SOP-CYBERSECURITY.md"
        if not index_file.exists():
            return []

        indexed = self.get_indexed_stems(index_file)
        added: List[str] = []
        # Exclude offensive/ since it has its own 00-INDEX-OFFENSIVE-SECURITY-SUITE.md
        md_files = sorted(
            [
                f
                for f in domain_dir.rglob("*.md")
                if not self._is_index_or_ignored(f) and "offensive" not in f.parts
            ],
            key=lambda x: x.stem,
        )

        content = index_file.read_text(encoding="utf-8")
        for f in md_files:
            if f.stem in indexed:
                continue
            summary = self.extract_summary(f)
            entry = f"- [[{f.stem}]] — {summary}"
            name_lower = f.name.lower()
            rel_lower = str(f.relative_to(domain_dir)).lower()

            if f.name.startswith("Wiki_"):
                content = self._insert_into_section(
                    content,
                    r"##\s*.*Knowledge Base & Research Wikis",
                    entry,
                    fallback_header="## Knowledge Base & Research Wikis",
                )
            elif any(
                k in name_lower or k in rel_lower
                for k in [
                    "audit",
                    "report",
                    "incident",
                    "insiden",
                    "engine",
                    "scanner",
                    "monitor",
                ]
            ):
                content = self._insert_into_section(
                    content,
                    r"##\s*.*Core Security Engines & Audits",
                    entry,
                    fallback_header="## Core Security Engines & Audits",
                )
            else:
                content = self._insert_into_section(
                    content,
                    r"##\s*.*(?:SOP & Operational Playbooks|SOP & Playbook)",
                    entry,
                    fallback_header="## SOP & Operational Playbooks",
                )
            indexed.add(f.stem)
            added.append(f.name)

        if added:
            self._atomic_write(index_file, content)
            self.log(f"MOC Bookkeeper: Added {len(added)} note(s) to {index_file.name}")
        return added

    def sync_projects(self) -> List[str]:
        """Scan ~/vault/projects/ (or ~/vault/projek/) and maintain index file."""
        domain_dir = self.vault_dir / "projects"
        index_file = domain_dir / "00-INDEX-PROJECTS.md"
        if not domain_dir.exists():
            domain_dir = self.vault_dir / "projek"
            index_file = domain_dir / "00-INDEX-PROJEK.md"
        if not index_file.exists():
            return []

        indexed = self.get_indexed_stems(index_file)
        added: List[str] = []
        # Detect sub-domains that maintain their own indices dynamically
        sub_indices = {p.parent for p in domain_dir.glob("*/00-INDEX-*.md")}
        exclude_dirs = {p.strip().lower() for p in os.environ.get("SYNAPSE_MOC_EXCLUDE_DIRS", "").split(",") if p.strip()}

        md_files = sorted(
            [
                f
                for f in domain_dir.rglob("*.md")
                if not self._is_index_or_ignored(f)
                and not any(parent in sub_indices for parent in f.parents)
                and not any(p.lower() in exclude_dirs for p in f.parts)
            ],
            key=lambda x: x.stem,
        )

        content = index_file.read_text(encoding="utf-8")
        for f in md_files:
            if f.stem in indexed:
                continue
            summary = self.extract_summary(f)
            entry = f"- [[{f.stem}]] — {summary}"
            rel_str = str(f.relative_to(domain_dir)).lower()

            if "agent" in rel_str:
                if any(
                    k in rel_str
                    for k in [
                        "router",
                        "gateway",
                        "proxy",
                    ]
                ):
                    content = self._insert_into_section(
                        content,
                        r"### 5\..*AI Gateway & Router Infrastructure",
                        entry,
                        fallback_header="### 5. AI Gateway & Router Infrastructure",
                    )
                else:
                    content = self._insert_into_section(
                        content,
                        r"### 1\..*Agent Core OS & Architecture",
                        entry,
                        fallback_header="### 1. Agent Core OS & Architecture",
                    )
            elif "dashboard" in rel_str or "proxy" in rel_str:
                content = self._insert_into_section(
                    content,
                    r"### 4\..*Gateway & Infra: Dashboard|### 4\..*",
                    entry,
                    fallback_header="### 4. Gateway & Infra: Dashboard",
                )
            else:
                content = self._insert_into_section(
                    content,
                    r"### 6\..*(?:Additional Projects & Modules|Proyek & Modul Tambahan)|### 6\..*",
                    entry,
                    fallback_header="### 6. Additional Projects & Modules",
                )
            indexed.add(f.stem)
            added.append(f.name)

        if added:
            self._atomic_write(index_file, content)
            self.log(f"MOC Bookkeeper: Added {len(added)} note(s) to {index_file.name}")
        return added

    # Backward compatibility alias
    sync_projek = sync_projects

    def sync_all(self) -> Dict[str, List[str]]:
        """Run synchronization across all 4 primary MOC indices."""
        return {
            "daily": self.sync_daily(),
            "server": self.sync_server(),
            "cybersecurity": self.sync_cybersecurity(),
            "projek": self.sync_projek(),
        }


def run_moc_bookkeeper_step(
    vault_dir: Optional[str] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, List[str]]:
    """Entry point for nightly consolidation workflow."""
    bookkeeper = MOCBookkeeper(vault_dir=vault_dir, log_fn=log_fn)
    return bookkeeper.sync_all()
