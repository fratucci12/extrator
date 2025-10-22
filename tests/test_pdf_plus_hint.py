from __future__ import annotations

from disdem_etl import pdf_plus_hint_to_gt as pdf_gt


def test_extract_quantity_ampla_from_text():
    text = "1 Lote Mesa Reunião 20 Sim R$ 1.234,56"
    quantidade, ampla = pdf_gt._extract_quantity_ampla_from_text(text)  # type: ignore[attr-defined]  # noqa: SLF001
    assert quantidade == "20"
    assert ampla == "Sim"


def test_extract_valor_unitario():
    text = "Algum item Não R$ 9.876,54"
    valor = pdf_gt._extract_valor_unitario(text)  # type: ignore[attr-defined]  # noqa: SLF001
    assert valor == "R$ 9.876,54"


def test_filter_header_with_pdf():
    header = {"orgao_codigo": "Orgão XYZ", "processo": "123", "valor_teto": ""}
    pdf_text = [(1, ["Orgão XYZ - Processo 123"])]
    filtered = pdf_gt.filter_header_with_pdf(header, pdf_text)
    assert filtered["orgao_codigo"] == "Orgão XYZ"
    assert filtered["processo"] == "123"
    assert filtered["valor_teto"] == ""
