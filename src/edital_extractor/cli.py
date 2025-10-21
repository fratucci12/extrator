# -*- coding: utf-8 -*-
import argparse
import hashlib
import os
import re
from typing import Optional, Tuple, List

from tqdm import tqdm

from .pipeline import process_pdf
from .writers import write_outputs

def parse_pages_arg(pages: str) -> Optional[Tuple[int,int]]:
    if not pages:
        return None
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)?\s*$", pages)
    if not m:
        raise SystemExit("--pages deve ser algo como '25-27' ou '28-'")
    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) else 10**9
    return (start, end)

def main():
    ap = argparse.ArgumentParser(description="Extrair itens de editais (PDF) em lote com regex robusta.")
    ap.add_argument("--input", required=True, help="Pasta com PDFs")
    ap.add_argument("--out-xlsx", required=True, help="Arquivo XLSX de saída (golden set)")
    ap.add_argument("--out-csv", default="", help="Arquivo CSV de saída (labels_campos)")
    ap.add_argument("--pages", default="", help="Faixa de páginas (ex.: 25-27 ou 28-). Em branco = todas.")
    args = ap.parse_args()

    input_dir = args.input
    out_xlsx = args.out_xlsx
    out_csv = args.out_csv
    pages_range = parse_pages_arg(args.pages)

    pdfs = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    if not pdfs:
        raise SystemExit("Nenhum PDF encontrado em --input")

    docs_rows_all = []
    labels_rows_all = []

    for pdf_path in tqdm(pdfs, desc="Processando PDFs"):
        try:
            docs_row, labels_rows = process_pdf(pdf_path, pages_range=pages_range)
            docs_rows_all.append(docs_row)
            labels_rows_all.extend(labels_rows)
        except Exception as e:
            # registra erro no docs
            with open(pdf_path, "rb") as fh:
                doc_id = hashlib.sha256(fh.read()).hexdigest()
            filename = os.path.basename(pdf_path)
            docs_rows_all.append([doc_id, filename, doc_id, "pdf", "", "", "", "tabelado", f"ERRO: {e}"])

    write_outputs(out_xlsx, out_csv, docs_rows_all, labels_rows_all)

if __name__ == "__main__":
    main()

