"""
SynapseWebHarvester - External Research Harvester Module.
Implements Tier 4 sandboxed autonomous web research with strict whitelisting
and integration with Synapse Worker + Adapter.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from synapse.storage import SynapseStorage
from synapse.worker import resolve_synapse_llm_config

logger = logging.getLogger(__name__)

# Strict Domain Whitelist for High-Signal Technical Engineering Sources
DEFAULT_WHITELIST_DOMAINS: Set[str] = {
    "github.com", "raw.githubusercontent.com", "arxiv.org", "docs.astro.build",
    "golang.org", "go.dev", "python.org", "pypi.org", "developer.mozilla.org",
    "blog.cloudflare.com", "vercel.com",
    "huggingface.co", "news.ycombinator.com", "cisa.gov",
}


MAX_FETCH_BYTES = 2 * 1024 * 1024  # 2 MiB hard cap on response bodies
_MAX_REDIRECTS = 5


class _WhitelistRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow redirects only while each hop stays on the whitelist."""

    def __init__(self, check_url: Any) -> None:
        super().__init__()
        self._check_url = check_url
        self._redirects = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if self._redirects >= _MAX_REDIRECTS:
            raise urllib.error.HTTPError(newurl, code, "too many redirects", headers, fp)
        self._redirects += 1
        if not self._check_url(newurl):
            raise urllib.error.HTTPError(
                newurl, code, f"redirect blocked by whitelist: {newurl}", headers, fp
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def sanitize_slug(text: str) -> str:
    """Convert string to safe filename slug."""
    clean = re.sub(r"[^\w\s\-]", "", text.lower()).strip()
    return re.sub(r"[\s\-]+", "-", clean)[:60]


class SynapseWebHarvester:
    """
    Autonomous Tier 4 External Research Harvester.
    """

    def __init__(
        self,
        storage: SynapseStorage,
        vault_research_dir: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        whitelisted_domains: Optional[Set[str]] = None,
    ) -> None:
        self.storage = storage
        self.llm_config = resolve_synapse_llm_config(llm_config)
        self.domains = whitelisted_domains or DEFAULT_WHITELIST_DOMAINS

        vault_base = (
            vault_research_dir
            or os.environ.get("SYNAPSE_RESEARCH_DIR")
            or str(Path.home() / ".synapse" / "research" / "auto_harvest")
        )
        self.vault_dir = Path(vault_base)
        self.vault_dir.mkdir(parents=True, exist_ok=True)

    def is_domain_allowed(self, url: str) -> bool:
        """Verify URL domain against strict whitelist."""
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
            if ":" in domain:
                domain = domain.split(":")[0]
            for allowed in self.domains:
                if domain == allowed or domain.endswith("." + allowed):
                    return True
            return False
        except Exception:
            return False

    def fetch_url_text(self, url: str, timeout: float = 10.0) -> Optional[str]:
        """Fetch raw HTML/text from URL safely."""
        if not self.is_domain_allowed(url):
            logger.warning("URL domain blocked by whitelist: %s", url)
            return None

        try:
            opener = urllib.request.build_opener(
                _WhitelistRedirectHandler(self.is_domain_allowed)
            )
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; SynapseHarvester/1.0; +https://github.com/synapse-agent/synapse-agent)",
                    "Accept": "text/html,application/xhtml+xml,application/xml,application/atom+xml;q=0.9,*/*;q=0.8",
                },
            )
            with opener.open(req, timeout=timeout) as resp:
                length = resp.headers.get("Content-Length")
                if length and length.isdigit() and int(length) > MAX_FETCH_BYTES:
                    logger.warning("URL body too large (%s bytes): %s", length, url)
                    return None
                raw = resp.read(MAX_FETCH_BYTES + 1)
                if len(raw) > MAX_FETCH_BYTES:
                    logger.warning("URL body exceeded %d bytes: %s", MAX_FETCH_BYTES, url)
                    return None
                content = raw.decode("utf-8", errors="ignore")
                clean_text = re.sub(r"<script.*?>.*?</script>", "", content, flags=re.DOTALL)
                clean_text = re.sub(r"<style.*?>.*?</style>", "", clean_text, flags=re.DOTALL)
                clean_text = re.sub(r"<[^>]+>", " ", clean_text)
                clean_text = re.sub(r"\s+", " ", clean_text).strip()
                return clean_text[:15000]
        except Exception as e:
            logger.debug("Failed to fetch URL %s: %s", url, e)
            return None

    def summarize_with_llm(self, topic: str, source_url: str, text_content: str) -> Dict[str, str]:
        """Summarize research material using Synapse LLM with strict technical focus."""
        prompt = (
            f"You are the Synapse Technical Research Agent.\n"
            f"Topic: {topic}\n"
            f"Source URL: {source_url}\n\n"
            f"Read the technical text below and extract:\n"
            f"1. Executive Summary (2-3 sentences)\n"
            f"2. Core Technical Methods / Best Practices\n"
            f"3. Code Examples or Architectural Patterns (if any)\n\n"
            f"Output in clean Markdown. Raw text:\n{text_content[:4000]}"
        )

        messages = [
            {"role": "system", "content": "You are a surgical technical research summarizer. Be concise, factual, and code-focused."},
            {"role": "user", "content": prompt},
        ]

        try:
            from synapse.config import call_llm_resilient
            content = call_llm_resilient(
                messages=messages,
                model=self.llm_config.get("model"),
                temperature=0.2,
                max_tokens=1000,
                timeout=30.0,
                max_retries=3,
            )
            if content:
                return {"summary": content.strip(), "status": "ok"}
        except Exception as e:
            logger.debug("LLM summarization failed: %s, falling back to heuristic summary", e)

        lines = [l.strip() for l in text_content.split(".") if len(l.strip()) > 20][:5]
        heuristic = "### Research Summary\n" + "\n".join(f"- {l}." for l in lines)
        return {"summary": heuristic, "status": "fallback"}

    def harvest_item(
        self,
        topic: str,
        title: str,
        source_url: str,
        raw_content: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single research item, write Vault Markdown file, and insert to synapse_external_research.
        """
        if not self.is_domain_allowed(source_url):
            logger.info("Skipping non-whitelisted source URL: %s", source_url)
            return None

        domain = urllib.parse.urlparse(source_url).netloc

        content = raw_content
        if not content:
            content = self.fetch_url_text(source_url)
        if not content or len(content) < 50:
            return None

        summary_res = self.summarize_with_llm(topic, source_url, content)
        summary_md = summary_res["summary"]

        slug = sanitize_slug(title)
        date_str = datetime.date.today().isoformat()
        topic_dir = self.vault_dir / sanitize_slug(topic)
        topic_dir.mkdir(parents=True, exist_ok=True)
        vault_file = topic_dir / f"{date_str}-{slug}.md"

        frontmatter = (
            f"---\n"
            f"title: \"{title}\"\n"
            f"topic: \"{topic}\"\n"
            f"source_url: \"{source_url}\"\n"
            f"domain: \"{domain}\"\n"
            f"tier: 4\n"
            f"status: quarantined\n"
            f"date: {date_str}\n"
            f"tags: [research, auto-harvest, synapse-tier4, {sanitize_slug(topic)}]\n"
            f"---\n\n"
            f"# {title}\n\n"
            f"**Source:** [{domain}]({source_url})\n"
            f"**Tier:** Tier 4 (External Research Sandbox — Does not override Tier 1/2 Ground Truth)\n\n"
            f"{summary_md}\n"
        )

        with open(vault_file, "w", encoding="utf-8") as f:
            f.write(frontmatter)

        db_res = self.storage.add_external_research(
            topic=topic,
            title=title,
            source_url=source_url,
            domain=domain,
            summary=summary_md,
            raw_snippet=content[:1000],
            vault_path=str(vault_file),
            status="quarantined",
        )

        # Ingest into synapse_nodes and synapse_vectors with exact research ID
        try:
            rel_uri = f"vault://research/auto_harvest/{sanitize_slug(topic)}/{date_str}-{slug}.md"
            self.storage.add_node(
                uri=rel_uri,
                l0_abstract=title,
                l1_overview=f"Sections: {title}, Research Summary | Key Points: {summary_md[:400]} | Source: {source_url}",
                l2_target_path=str(vault_file),
                category="sop",
                importance=0.6,
                strength=0.1,
                node_id=db_res["id"],
                # Tier 4 research enters memory as flagged so unvetted web
                # content never appears in recall until promoted.
                status="flagged",
            )
        except Exception as e:
            logger.debug("Auto-retaining harvested research into nodes/vectors notice: %s", e)

        return db_res
