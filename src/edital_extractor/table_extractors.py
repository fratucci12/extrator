from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import camelot  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    camelot = None  # type: ignore

from .extractors import (
    ItemRecord,
    PAT_ITEM_ROW,
    PAT_MONEY,
    PAT_QTY_UNIT,
    PAT_UNIT_TRAIL,
    _clean_nome,
    _normalize_item_lote,
)
from .utils import QTY, to_int_br, to_money_br


def _row_bbox(table, row_idx: int) -> Optional[Tuple[float, float, float, float]]:
    try:
        row_cells = table.cells[row_idx]
    except Exception:
        return None
    coords = [
        (
            getattr(cell, "x1", math.nan),
            getattr(cell, "y1", math.nan),
            getattr(cell, "x2", math.nan),
            getattr(cell, "y2", math.nan),
        )
        for cell in row_cells
    ]
    coords = [c for c in coords if not any(math.isnan(v) for v in c)]
    if not coords:
        return None
    x1 = min(c[0] for c in coords)
    y1 = min(c[1] for c in coords)
    x2 = max(c[2] for c in coords)
    y2 = max(c[3] for c in coords)
    return (float(x1), float(y1), float(x2), float(y2))


def _parse_table_row(
    row_text: str,
    page_no: int,
    lote_hint: Optional[str],
) -> Optional[ItemRecord]:
    m_item = PAT_ITEM_ROW.match(row_text)
    if not m_item:
        return None
    raw_item = m_item.group("item").strip()
    lote_norm, item_norm = _normalize_item_lote(raw_item, lote_hint)
    remainder = m_item.group("rest").strip()
    if not remainder:
        return None

    price_value = None
    price_match = None
    for m_price in PAT_MONEY.finditer(remainder):
        raw_val = m_price.group(1)
        whole = m_price.group(0)
        if "r$" not in whole.lower():
            continue
        candidate = to_money_br(raw_val)
        if candidate is None:
            continue
        price_match = m_price
        price_value = candidate
        break

    qty_value = None
    qty_match = PAT_QTY_UNIT.search(remainder)
    if qty_match:
        qty_value = to_int_br(qty_match.group("qty"))
    else:
        simple_qty = None
        for candidate in PAT_UNIT_TRAIL.split(remainder):
            m_qty = re_search_qty(candidate)
            if m_qty is not None:
                simple_qty = m_qty
                break
        if simple_qty is not None:
            qty_value = simple_qty

    cutpoints: List[int] = []
    if qty_match:
        cutpoints.append(qty_match.start())
    if price_match:
        cutpoints.append(price_match.start())
    cut_at = min([p for p in cutpoints if p > 0], default=len(remainder))
    nome_raw = remainder[:cut_at].strip() if remainder else ""
    nome_raw = PAT_UNIT_TRAIL.sub("", nome_raw).strip(" -–—:,.;")

    nome = _clean_nome(nome_raw)
    return ItemRecord(
        page=page_no,
        lote=str(lote_norm or ""),
        item=item_norm,
        nome=nome,
        quantidade=qty_value,
        valor_unitario=price_value,
        fonte="tabela",
        priority=0,
    )


def re_search_qty(segment: str) -> Optional[int]:
    import re

    match = re.search(rf"\b({QTY})\b", segment)
    if not match:
        return None
    return to_int_br(match.group(1))


def extract_items_from_tables(
    pdf_path: str,
    page_numbers: Sequence[int],
    lote_per_page: Dict[int, Optional[str]],
) -> List[ItemRecord]:
    if not camelot:
        return []

    records: List[ItemRecord] = []
    for page_no in page_numbers:
        tables = _read_tables(pdf_path, page_no)
        if not tables:
            continue
        lote_hint = lote_per_page.get(page_no)
        for table in tables:
            for row_idx, row in enumerate(table.df.itertuples(index=False)):
                row_values: List[str] = []
                for cell in row:
                    text = str(cell).strip()
                    if not text or text.lower() == "nan":
                        continue
                    row_values.append(text)
                if not row_values:
                    continue
                row_text = " ".join(row_values)
                rec = _parse_table_row(row_text, page_no, lote_hint)
                if rec is None:
                    continue
                rec.bbox = _row_bbox(table, row_idx)
                records.append(rec)
    return records


def _read_tables(pdf_path: str, page_no: int):
    tables = []
    for flavor in ("lattice", "stream"):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="No tables found in table area*",
                    category=UserWarning,
                    module="camelot",
                )
                result = camelot.read_pdf(
                    pdf_path,
                    pages=str(page_no),
                    flavor=flavor,
                    strip_text="\n",
                )
        except Exception:
            continue
        tables.extend(result)
        if tables:
            break
    return tables
