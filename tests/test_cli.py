from types import SimpleNamespace

from product_memory.cli import tool_result_payload


def test_tool_result_payload_prefers_structured_content() -> None:
    result = SimpleNamespace(structured_content={"ok": True}, content=[])

    assert tool_result_payload(result) == {"ok": True}


def test_tool_result_payload_parses_json_text_content() -> None:
    result = SimpleNamespace(
        structured_content=None,
        content=[SimpleNamespace(text='{"query": "demo", "chunks": []}')],
    )

    assert tool_result_payload(result) == {"query": "demo", "chunks": []}


def test_tool_result_payload_wraps_plain_text_content() -> None:
    result = SimpleNamespace(structured_content=None, content=[SimpleNamespace(text="plain")])

    assert tool_result_payload(result) == {"text": "plain"}
