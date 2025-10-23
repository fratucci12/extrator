# edital_extractor

Extrator em **lote** de itens de **editais** com duas frentes principais:

- `html2csv`: converte telas do **DISDEM** salvas como HTML para CSVs normalizados (cabeçalho + itens) mantendo o nome-base do arquivo.
- `pdf2gt`: usa os CSVs como **dica** e confirma os dados diretamente nos **PDFs** gerando o Ground Truth (`*_GT.csv`).

O pacote legado (`edital_extractor`) permanece disponível para fluxos existentes.

## Instalação
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## Uso
### CLI `html2csv`

```bash
html2csv --input ./html --out-dir ./csv_out --merge --add-source-col
```

- `--input`: pasta com `.html/.htm`.
- `--out-dir`: onde salvar os CSVs (um por HTML, nome original preservado).
- `--merge`: gera `merged.csv` somando todas as linhas.
- `--merge-out`: caminho alternativo para o merged.
- `--add-source-col`: acrescenta `source_html` apenas no merged.
- Colunas do CSV: `amostra`, `catalogo`, `item`, `lote`, `produto`, `quantidade`, `ampla_concorrencia`, `valor_unitario`.

### CLI `pdf2gt`

```bash
pdf2gt --pdf-dir ./meus_pdfs --csv-dir ./csv_out --out-dir ./gt --merge --min-score 70 --jobs 4
```

- `--pdf-dir`: PDFs originais.
- `--csv-dir`: CSVs produzidos pelo `html2csv` (mesmo nome-base dos PDFs). Opcional; se omitido ou se o arquivo estiver vazio, o extrator usa apenas heurísticas no PDF.
- `--out-dir`: um `_GT.csv` por PDF + `GT_merged.csv` (se `--merge`).
- `--min-score`: limiar RapidFuzz (default 60).
- `--max-pages`: limita páginas processadas (0 = todas).
- `--jobs`: paralelismo por processos (0 = serial).
- `--dedoc`: ativa extração de texto/tabelas via [Dedoc](https://github.com/ispras/dedoc). O pipeline também aproveita o Table Transformer para detectar tabelas antes de recorrer ao Camelot.
- `--dedoc-language`: define idiomas do Dedoc (ex.: `por`, `por+eng`).
- `--dedoc`: ativa extração de texto/tabelas via [Dedoc](https://github.com/ispras/dedoc) quando o ambiente estiver preparado.
- `--dedoc-language`: define idiomas do Dedoc (ex.: `por`, `por+eng`).

O pipeline usa Camelot (tabelas) e pdfplumber (texto), priorizando tabelas; se nenhum match for encontrado a linha sai com `fonte=nao_encontrado` e `evidencia=NAO_ENCONTRADO`.

### Quick check

Há um utilitário rápido:

```bash
python scripts/quick_check.py --type html csv_out/edital1.csv
python scripts/quick_check.py --type gt gt/edital1_GT.csv
```

Ele valida colunas obrigatórias e duplicidades `(document_id, lote, item)` no GT.

## Dependências extras

- `camelot-py[cv]` (exige **Ghostscript 64-bit** no PATH do Windows).
- `pdfplumber`
- `rapidfuzz`
- `pypdf`

No Windows certifique-se de instalar Ghostscript e reiniciar o terminal para que `gswin64c.exe` fique acessível.

Os leitores de HTML usam `utf-8`, `latin-1` e `cp1252` com `errors="ignore"`, garantindo compatibilidade com arquivos heterogêneos.

## Pipeline legado

O fluxo antigo continua disponível, agora com pré-processamento inspirado no blueprint Docling → seções → extração determinística:

```bash
python -m edital_extractor.cli --input ./meus_pdfs --out-xlsx golden_set.xlsx --out-csv golden_set.csv
python -m edital_extractor.cli --input ./meus_pdfs --pages 25-27 --out-xlsx p25a27.xlsx
```

- `docs`: metadados por PDF
- `labels_campos`: formato longo com `pag` e `fonte`
- Pipeline: (1) normaliza o PDF para Markdown via Docling (quando instalado) e marca páginas informativas; (2) roda Camelot para tabelas preservando coordenadas (`bbox`); (3) completa lacunas com heurísticas por texto (regex/anchors); (4) faz merge determinístico com prioridade para a fonte mais estruturada.
- As colunas `bbox (x1,y1,x2,y2)` passam a vir preenchidas quando o Camelot identifica a linha correspondente.
- O Docling é opcional; instale com `pip install "docling[pdf]"` para habilitar a normalização e o tagging automático. Sem ele, o pipeline usa somente PyPDF + heurísticas e mantém o comportamento anterior.
- As heurísticas agora entendem tabelas em Markdown (quando Docling converte) e aceitam quantidades decimais/unidades variadas. Sempre que uma linha não gera match, você pode ligar o modo debug para inspecionar rapidamente o trecho que falhou.

### Debug das heurísticas de regex

```bash
EXTRATOR_DEBUG_REGEX=1 python -m edital_extractor.cli --input ./meus_pdfs --out-xlsx golden_set.xlsx --out-csv golden_set.csv
```

- Arquivos de log são gravados em `build/regex_debug/` (use `EXTRATOR_DEBUG_DIR=/outro/caminho` para customizar).
- Cada log lista as páginas sem matches e um snippet do texto/Markdown analisado, facilitando a criação de novas heurísticas ou o ajuste das existentes.

## Backlog

- [x] `pdf2gt` sem hints em produção
  - Objetivo: permitir que o CLI gere `_GT.csv` apenas a partir dos PDFs quando o CSV de hints não existir ou estiver vazio, mantendo o pipeline atual para quem fornece as dicas.
  - Formato de saída: preservar `GT_COLUMNS`; preencher `document_id`, `filename_pdf`, `pdf_sha256`, `num_pages`, `lote`, `item`, `nome_real`, `quantidade`, `valor_unitario`, `ampla_concorrencia`, `pagina`, `fonte`, `evidencia` e `match_score` com os dados das heurísticas (`extract_regex_records`, `match_from_tables`, `match_from_text`). Definir `hint_produto` e as colunas de cabeçalho importadas do HTML como string vazia quando não houver hints.
  - Abordagem técnica: tornar `--csv-dir` opcional no CLI, detectar ausência/conteúdo vazio no `generate_gt_dataframe` e seguir com um fluxo “no-hints” que de-duplica registros por `(lote,item,página)`, gera evidência a partir dos PDFs e sinaliza a origem em `fonte`. Manter o log de aviso atual, mas acrescentar informação de fallback.
  - Testes: adicionar cenários em `tests/` garantindo que a saída sem hints preserva colunas, nomes de arquivo e produz pelo menos uma linha por item detectado por heurísticas.

MIT License.
