from __future__ import annotations

from disdem_etl import html_to_unified as h2u
from disdem_etl import utils


def test_to_int_br():
    assert utils.to_int_br("1.234") == 1234
    assert utils.to_int_br("abc") is None


def test_to_decimal_br():
    value = utils.to_decimal_br("R$ 1.234,56")
    assert str(value) == "1234.56"


def test_strip_tags_removes_script():
    html = "<html><head><style>body{}</style></head><body><script>alert()</script><span>Texto</span></body></html>"
    text = utils.strip_tags(html)
    assert "Texto" in text
    assert "alert" not in text


def test_parse_items_detects_sim_nao():
    lines = [
        "Cabeçalho",
        "Items do DISDEM",
        "1 1",
        "Cadeira Teste Executiva 10 Sim",
        "R$ 1.234,56",
        "Total de itens",
    ]
    df = h2u.parse_items(lines)
    assert not df.empty
    row = df.iloc[0]
    assert row["item"] == "1"
    assert row["lote"] == "1"
    assert row["ampla_concorrencia"] == "Sim"
    assert row["quantidade"] == "10"
    assert row["valor_unitario"] == "R$ 1.234,56"


def test_parse_header_handles_nbsp():
    lines = [
        "Introdução",
        "Cabeçalho",
        "Órgão",
        "Orgão XYZ",
        "Data da Publicação",
        "01/02/2024",
        "Valor Teto",
        "R$ 9.999,99",
        "Items do DISDEM",
    ]
    header = h2u.parse_header(lines)
    assert header["orgao_codigo"] == "Orgão XYZ"
    assert header["data_publicacao"] == "01/02/2024"
    assert header["valor_teto"] == "R$ 9.999,99"

