import synapse.mcp as mcp


class _BoomAdapter:
    def recall(self, *args, **kwargs):
        raise RuntimeError("/home/user/secret/path.db is locked")


def test_internal_error_text_is_not_returned(monkeypatch):
    monkeypatch.setattr(mcp, "_adapter", _BoomAdapter())
    result = mcp.handle_call_tool("synapse_recall", {"query": "x"})
    assert result["isError"] is True
    assert "secret" not in result["content"][0]["text"]
    assert "internal error" in result["content"][0]["text"].lower()


def test_handle_call_tool_result_has_structured_content(monkeypatch):
    class _OkAdapter:
        def recall(self, *a, **k):
            return []

    monkeypatch.setattr(mcp, "_adapter", _OkAdapter())
    result = mcp.handle_call_tool("synapse_recall", {"query": "x"})
    assert "structuredContent" in result
    assert result["structuredContent"].get("results") == []

