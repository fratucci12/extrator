# edital_extractor

Extrator em **lote** de itens de **editais em PDF** com **regex robusta** e heurísticas de fallback.
Gera **XLSX** (abas `docs` e `labels_campos`) e **CSV** (`labels_campos`).

## Instalação
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## Uso
```bash
python -m edital_extractor.cli --input ./meus_pdfs --out-xlsx golden_set.xlsx --out-csv golden_set.csv
python -m edital_extractor.cli --input ./meus_pdfs --pages 25-27 --out-xlsx p25a27.xlsx
```

## Saída
- `docs`: metadados por PDF
- `labels_campos`: formato longo com `pag` e `fonte`

MIT License.
