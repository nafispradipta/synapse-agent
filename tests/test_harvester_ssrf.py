"""Harvester redirects and body sizes must be bounded and whitelisted."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from synapse.storage import SynapseStorage
from synapse.harvester import SynapseWebHarvester


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 (http.server contract)
        if self.path == "/redirect-evil":
            port = self.server.server_address[1]
            self.send_response(301)
            # Same machine, but a hostname that is NOT in the whitelist.
            self.send_header("Location", f"http://127.0.0.1:{port}/secret")
            self.end_headers()
            return
        if self.path == "/redirect-ok":
            self.send_response(301)
            self.send_header("Location", "/ok")
            self.end_headers()
            return
        self.send_response(200)
        if self.path == "/big":
            content = b"a" * (3 * 1024 * 1024)
        elif self.path == "/secret":
            content = b"secret content"
        else:
            content = b"public content"
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):  # silence test server noise
        return


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    thread.join(timeout=5)


@pytest.fixture
def harvester(tmp_path):
    storage = SynapseStorage(":memory:")
    harvester = SynapseWebHarvester(
        storage=storage,
        vault_research_dir=str(tmp_path / "research"),
        whitelisted_domains={"localhost"},
        llm_config={"base_url": "http://127.0.0.1:1/v1", "api_key": "k", "model": "m"},
    )
    yield harvester
    storage.close()


def test_redirect_to_non_whitelisted_host_is_blocked(server, harvester):
    url = f"http://localhost:{server.server_address[1]}/redirect-evil"
    assert harvester.fetch_url_text(url, timeout=5) is None


def test_redirect_to_whitelisted_host_is_allowed(server, harvester):
    url = f"http://localhost:{server.server_address[1]}/redirect-ok"
    text = harvester.fetch_url_text(url, timeout=5)
    assert text is not None
    assert "public content" in text


def test_oversized_body_is_rejected(server, harvester):
    url = f"http://localhost:{server.server_address[1]}/big"
    assert harvester.fetch_url_text(url, timeout=10) is None
