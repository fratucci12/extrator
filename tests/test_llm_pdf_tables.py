from __future__ import annotations

import json

import pytest

from disdem_etl.llm_pdf_tables import (
    DEFAULT_SYSTEM_PROMPT,
    TableCandidate,
    build_messages,
    parse_json_response,
    process_tables,
)


def test_build_messages_includes_instruction_and_table():
    candidate = TableCandidate(
        page=2,
        source="camelot",
        text="Item | Quantidade\n001 | 10",
        lines=["Item | Quantidade", "001 | 10"],
    )

    messages = build_messages(
        candidate,
        instruction="Extraia para JSON.",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    assert messages[0]["role"] == "system"
    assert "JSON" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "Tabela detectada na página 2" in messages[1]["content"]
    assert "Item | Quantidade" in messages[1]["content"]


@pytest.mark.parametrize(
    "payload",
    [
        '{"page":1,"rows":[]}',
        "```json\n{\"page\": 1, \"rows\": []}\n```",
    ],
)
def test_parse_json_response_handles_simple_payloads(payload: str):
    parsed = parse_json_response(payload)
    assert parsed == {"page": 1, "rows": []}


def test_process_tables_uses_client_and_returns_data():
    candidate = TableCandidate(
        page=3,
        source="table_transformer",
        text="Item 002 Mesa 5 unidades",
        lines=["Item 002 Mesa 5 unidades"],
    )
    output_payload = {"page": 3, "rows": [{"item": "002", "quantidade": "5"}]}

    class DummyClient:
        def __init__(self) -> None:
            self.messages = []

        def complete(self, messages):
            self.messages.append(messages)
            return json.dumps(output_payload)

    client = DummyClient()
    result = process_tables(
        tables=[candidate],
        llm_client=client,
        instruction="Responda em JSON.",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    assert result[0]["page"] == 3
    assert result[0]["data"] == output_payload
    assert client.messages, "complete() não foi chamado"
