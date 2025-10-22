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


def test_generate_gt_dataframe_without_hints(monkeypatch, tmp_path):
    sample_pdf = tmp_path / "sample.pdf"
    sample_pdf.write_bytes(b"placeholder content")

    class DummyReader:
        def __init__(self, *_args, **_kwargs):
            self.pages = [object(), object()]

    records = [
        pdf_gt.ItemRecord(
            page=1,
            lote="01",
            item="001",
            nome="CADEIRA GIRATÓRIA",
            quantidade=10,
            valor_unitario=123.45,
            fonte="tabela",
            priority=1,
        )
    ]

    monkeypatch.setattr(pdf_gt, "PdfReader", DummyReader)
    monkeypatch.setattr(
        pdf_gt,
        "extract_pdf_text_lines",
        lambda *_args, **_kwargs: [(1, ["Item 001 - Cadeira giratória com braço R$ 123,45"])],
    )
    monkeypatch.setattr(pdf_gt, "extract_tables", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(pdf_gt, "compute_sha256", lambda *_args, **_kwargs: "hash123")
    monkeypatch.setattr(pdf_gt, "extract_regex_records", lambda *_args, **_kwargs: records)

    df = pdf_gt.generate_gt_dataframe(sample_pdf, None, min_score=60, max_pages=0)
    assert not df.empty
    assert list(df["item"]) == ["001"]
    assert df.iloc[0]["filename_pdf"] == "sample.pdf"
    assert df.iloc[0]["fonte"] == "pdf_tabela"
    assert df.iloc[0]["valor_unitario"] == "R$ 123,45"
    assert df.iloc[0]["hint_produto"] == ""
