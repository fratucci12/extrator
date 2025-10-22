# -*- coding: utf-8 -*-
import re
from typing import Dict, List, Optional

MONEY_BR = r"\d{1,3}(?:\.\d{3})*(?:,\d{2})?"
QTY = r"\d{1,3}(?:\.\d{3})*(?:,\d{1,3})?"
UNITS = r"(?:unidade|un|und|unid|unid\.?|und\.?|un\.?|pc|pcs|pç|pçs|peça|peca|peças|pecas|conjunto|kit|jogo|par)"
ITEM_ID = r"\d{1,4}(?:[\.\-]\d{1,4})*"

def normalize_pdf_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"(\d),\s*\n\s*(\d)", r"\1\2", t)
    t = re.sub(r"(\d)\s*\n\s*(\d{1,3})(?=[\.,]\d{2})", r"\1\2", t)
    t = re.sub(r"R\$\s*", "R$ ", t)
    t = re.sub(r"(\d)([A-ZÁÂÃÀÉÊÍÓÔÕÚÜÇ])", r"\1 \2", t)
    t = re.sub(r"([A-ZÁÂÃÀÉÊÍÓÔÕÚÜÇ])(\d)", r"\1 \2", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t

def _norm_num(s: str) -> str:
    s = s.replace(".", "").replace(",", ".")
    s = re.sub(r"[^\d\.]", "", s)
    return s

def to_int_br(s: str):
    s = _norm_num(s)
    try:
        return int(round(float(s))) if s else None
    except Exception:
        return None

def to_money_br(s: str):
    s = _norm_num(s)
    try:
        return float(s) if s else None
    except Exception:
        return None

PAT_LOTE = re.compile(r"\bLote\b\s*[:\-]?\s*([0-9A-Za-z\-\.]+)", re.IGNORECASE)

def active_lote_per_page(pages_text: List[str]) -> Dict[int, Optional[str]]:
    out = {}
    current = None
    for i, txt in enumerate(pages_text, start=1):
        for m in PAT_LOTE.finditer(txt):
            candidate = m.group(1)
            if not re.search(r"\d", candidate):
                continue
            current = candidate
        out[i] = current
    return out

def row_buffers(text: str) -> list:
    rows, buf = [], ""
    upper_chars = "A-ZÁÂÃÀÉÊÍÓÔÕÚÜÇ"
    start_pat = re.compile(
        rf"^\s*(?:-+\s*)?{ITEM_ID}(?=\s+(?:\d{3}\.\d{2}\.\d{5}|(?:\d+\s+)?[{upper_chars}]|(?:\d+\s*)+R\$))"
    )
    header_pat = re.compile(r"^Página\s+\d+\s+de\s+\d+|^ITENS\s+DESTINADOS", re.IGNORECASE)
    for ln in [l for l in text.splitlines() if l.strip()]:
        if header_pat.match(ln):
            continue
        if start_pat.match(ln):
            if buf:
                rows.append(buf.strip())
            buf = ln
        else:
            buf = (buf + " " + ln).strip() if buf else ln
    if buf:
        rows.append(buf.strip())

    item_block_re = re.compile(r"\b\d{1,4}\s+\d{3}\.\d{2}\.\d{5}\s*-")
    matches = list(item_block_re.finditer(text))
    if len(rows) <= 1 and len(matches) > 1:
        rows = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            rows.append(text[start:end].strip())

    final_rows: List[str] = []
    lote_splitter = re.compile(r"(LOTE\s*[-:]\s*\d+)", re.IGNORECASE)
    for row in rows:
        parts = lote_splitter.split(row)
        buffer = ""
        for part in parts:
            if not part:
                continue
            if lote_splitter.fullmatch(part.strip()):
                if buffer.strip():
                    final_rows.append(buffer.strip())
                    buffer = ""
                final_rows.append(part.strip())
            else:
                buffer = (buffer + " " + part).strip() if buffer else part.strip()
        if buffer.strip():
            final_rows.append(buffer.strip())

    return final_rows
