from edital_extractor.preprocessing import build_page_slices


def test_build_page_slices_marks_informative_when_keywords_present():
    pages_text = [
        "Introdução geral sem dados relevantes.",
        "Item 01 - Descrição de cadeira giratória Lote 3 Quantidade 10 Valor Unitário R$ 500,00",
        "ANEXO I - Termo de Referência",
    ]
    markdown = ["", "", ""]
    active_lote = {1: None, 2: "3", 3: None}

    slices = build_page_slices(pages_text, markdown, active_lote)
    informative_pages = [ps.page_number for ps in slices if ps.is_informative]

    assert informative_pages == [2]
    assert slices[1].lote_hint == "3"
    assert "has_lote" in slices[1].tags


def test_build_page_slices_fallback_when_score_low():
    pages_text = ["Página sem sinal claro.", "Outra página descritiva."]
    markdown = ["", ""]
    active_lote = {1: None, 2: None}

    slices = build_page_slices(pages_text, markdown, active_lote)
    assert any(ps.is_informative for ps in slices)
