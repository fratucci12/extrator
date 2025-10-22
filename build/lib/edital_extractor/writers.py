# -*- coding: utf-8 -*-
import re
from typing import List

import pandas as pd
from openpyxl import Workbook

DOCS_COLS = ["document_id","filename","sha256","tipo_pdf","paginas","idioma","qualidade_scan (1–5)","layout (narrativo/tabelado/misto)","observacoes"]
LABELS_COLS = ["document_id","filename","lote","item","campo","valor_verdade","pag","bbox (x1,y1,x2,y2)","fonte (tabela/parágrafo/cabeçalho)","comentario_annot"]

_ILLEGAL_XLS_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def _sanitize_cell(value):
    if isinstance(value, str):
        return _ILLEGAL_XLS_CHARS_RE.sub("", value)
    return value


def _sanitize_row(row: List):
    return [_sanitize_cell(value) for value in row]


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    text_cols = df.select_dtypes(include="object").columns
    if not len(text_cols):
        return df
    df[text_cols] = df[text_cols].applymap(_sanitize_cell)
    return df


def write_outputs(out_xlsx: str, out_csv: str, docs_rows: list, labels_rows: list):
    df_docs = pd.DataFrame(docs_rows, columns=DOCS_COLS)
    df_labels = pd.DataFrame(labels_rows, columns=LABELS_COLS)
    df_docs = _sanitize_df(df_docs)
    df_labels = _sanitize_df(df_labels)
    df_docs = df_docs.drop_duplicates().reset_index(drop=True)
    df_labels = df_labels.drop_duplicates().reset_index(drop=True)

    # CSV (labels)
    if out_csv:
        df_labels.to_csv(out_csv, index=False, encoding="utf-8")

    # XLSX (docs + labels_campos)
    wb = Workbook()
    ws_docs = wb.active; ws_docs.title = "docs"
    ws_docs.append(DOCS_COLS)
    for row in df_docs.itertuples(index=False):
        ws_docs.append(_sanitize_row(list(row)))

    ws_lab = wb.create_sheet("labels_campos")
    ws_lab.append(LABELS_COLS)
    for row in df_labels.itertuples(index=False):
        ws_lab.append(_sanitize_row(list(row)))

    wb.save(out_xlsx)
