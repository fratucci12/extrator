from edital_extractor.extractors import heuristic_markdown_tables


def test_markdown_table_extraction_basic():
    markdown = """
| Item | Descrição | Quantidade | Valor Unitário |
| --- | --- | --- | --- |
| 1 | Cadeira giratória com braço regulável | 12 | R$ 500,00 |
| 2 | Mesa em L | 3,5 | R$ 1.250,40 |
""".strip()

    recs = heuristic_markdown_tables(2, markdown, "4")
    items = {(rec.item, rec.quantidade, rec.valor_unitario) for rec in recs}

    assert ("1", 12, 500.0) in items
    assert ("2", 3.5, 1250.40) in items
