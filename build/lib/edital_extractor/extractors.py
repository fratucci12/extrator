# -*- coding: utf-8 -*-
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .utils import MONEY_BR, QTY, UNITS, ITEM_ID, to_int_br, to_money_br, row_buffers

@dataclass
class ItemRecord:
    page: int
    lote: str
    item: str
    nome: Optional[str] = None
    quantidade: Optional[int] = None
    valor_unitario: Optional[float] = None
    fonte: str = "parágrafo"  # "tabela" | "parágrafo"
    priority: int = 999

def _dedup_best(recs: List[ItemRecord]) -> List[ItemRecord]:
    by_key: Dict[Tuple[int, str, str], ItemRecord] = {}
    for r in recs:
        key = (r.page, r.lote or "", r.item)
        if key not in by_key:
            by_key[key] = r
            continue
        cur = by_key[key]
        cur_info = (
            int(bool(cur.nome)),
            int(bool(cur.quantidade)),
            int(cur.valor_unitario is not None and cur.valor_unitario > 0),
        )
        new_info = (
            int(bool(r.nome)),
            int(bool(r.quantidade)),
            int(r.valor_unitario is not None and r.valor_unitario > 0),
        )
        score_cur = (cur.priority, -sum(cur_info), -cur_info[1])
        score_new = (r.priority, -sum(new_info), -new_info[1])
        if score_new < score_cur:
            by_key[key] = r
    return list(by_key.values())

def _normalize_item_lote(raw_item: str, lote_hint: Optional[str]) -> Tuple[str, str]:
    item = raw_item.strip()
    lote = (lote_hint or "").strip()
    if item.isdigit() and len(item) > 2:
        lote_part = item[:-2]
        item_part = item[-2:]
        if not lote:
            lote = lote_part.lstrip("0") or lote_part
        item = item_part.zfill(len(item_part))
    return lote, item

def _clean_nome(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = re.sub(r"\s+", " ", text).strip(" -–—:,.;")
    for sep in (" - ", " – ", " — "):
        if sep in text:
            candidate = text.split(sep, 1)[0].strip(" -–—:,.;")
            if len(candidate) >= 3:
                text = candidate
                break
    if len(text) > 120:
        cropped = text[:120]
        if " " in cropped:
            cropped = cropped.rsplit(" ", 1)[0]
        text = cropped.strip(" -–—:,.;")
    text = re.sub(r"\b\d{1,2}\b\s*$", "", text).strip(" -–—:,.;")
    return text if text else None

PAT_H1 = re.compile(
    rf"""^\s*(?P<item>{ITEM_ID})\s+(?P<name>.+?)\s+{UNITS}\s+(?P<qty>{QTY}).*?R\$\s*(?P<price>{MONEY_BR})\b""",
    re.IGNORECASE | re.DOTALL | re.MULTILINE | re.VERBOSE
)

def heuristic_H1_table_lines(page_no: int, text: str, lote_here: str) -> List[ItemRecord]:
    recs: List[ItemRecord] = []
    for m in PAT_H1.finditer(text):
        raw_item = m.group("item").strip()
        lote_norm, item_norm = _normalize_item_lote(raw_item, lote_here)
        nome_raw = m.group("name").strip()
        nome = re.sub(r"\b\d{5,}\b", "", nome_raw).strip(" -–—:,.;")
        nome = _clean_nome(nome)
        qtd = to_int_br(m.group("qty").strip())
        vu = to_money_br(m.group("price").strip())
        recs.append(ItemRecord(page=page_no, lote=str(lote_norm or ""), item=item_norm,
                               nome=nome, quantidade=qtd, valor_unitario=vu,
                               fonte="tabela", priority=1))
    return _dedup_best(recs)

PAT_ITEM_ROW = re.compile(rf"^\s*-?\s*(?P<item>{ITEM_ID})(?P<rest>.*)$", re.IGNORECASE)
PAT_QTY_UNIT = re.compile(rf"(?P<qty>{QTY})\s*(?P<unit>{UNITS})", re.IGNORECASE)
PAT_MONEY = re.compile(rf"(?:R?\$?\s*)({MONEY_BR})", re.IGNORECASE)
PAT_UNIT_TRAIL = re.compile(r"(?:\b(unidade|un|und|unid|peça|peca|peças|pecas|pc|pcs|pç|pçs|conjunto|kit|jogo|par)\b\s*)+$", re.IGNORECASE)

def heuristic_H2_row_buffer(page_no: int, text: str, lote_here: str) -> List[ItemRecord]:
    recs: List[ItemRecord] = []
    current_lote = lote_here
    last_item_value: Optional[int] = None
    for row in row_buffers(text):
        lote_match = re.search(r"LOTE\s*[-:]\s*(\d+)", row, flags=re.IGNORECASE)
        if lote_match and len(row.split()) < 6:
            current_lote = lote_match.group(1).strip()
            continue
        m_item = PAT_ITEM_ROW.match(row)
        if not m_item:
            continue
        raw_item = m_item.group("item").strip()
        digits_only = re.sub(r"\D", "", raw_item)
        if digits_only and len(digits_only) > 6:
            continue
        lote_norm, item_norm = _normalize_item_lote(raw_item, current_lote)
        try:
            item_number = int(item_norm)
        except Exception:
            item_number = None
        else:
            if (
                last_item_value is not None
                and item_number <= last_item_value
                and len(raw_item.strip()) <= 1
            ):
                item_number = last_item_value + 1
                item_norm = str(item_number)
            last_item_value = item_number
        remainder = m_item.group("rest").strip()
        if not remainder:
            continue

        qty_val = None
        name_prefix = ""
        tail = remainder

        qty_matches = list(PAT_QTY_UNIT.finditer(remainder))
        if qty_matches:
            qty_m = qty_matches[-1]
            unit = qty_m.group("unit").lower()
            valid_units = {"un", "un.", "unidade", "und"}
            if unit in valid_units:
                qty_val = to_int_br(qty_m.group("qty"))
                name_prefix = remainder[:qty_m.start()].strip()
                tail = remainder[qty_m.end():].strip()
            else:
                qty_matches = []
                qty_val = None
                tail = remainder
        else:
            simple_qty = re.match(rf"^(?P<qty>{QTY})\b\s*(?P<rest>.*)$", remainder)
            if simple_qty:
                qty_val = to_int_br(simple_qty.group("qty"))
                tail = simple_qty.group("rest").strip()
                name_prefix = ""
            else:
                tail = remainder
                name_prefix = ""

        price = None
        price_match = None
        price_start = price_end = None
        best_price_score = None
        for m_price in PAT_MONEY.finditer(tail):
            raw_val = m_price.group(1)
            whole = m_price.group(0)
            context_start = max(0, m_price.start() - 30)
            context_end = min(len(tail), m_price.end() + 30)
            context = tail[context_start:context_end].lower()
            value = to_money_br(raw_val)
            if value is None:
                continue

            has_currency = "r$" in whole.lower()
            if not has_currency:
                continue

            is_total = "total" in context
            if is_total:
                continue

            is_unit = re.search(r"unit[aá]r", context) is not None
            is_average = "méd" in context

            score = (
                0 if is_unit else 1,
                0 if is_average else 1,
                value,
            )

            if best_price_score is None or score < best_price_score:
                best_price_score = score
                price = value
                price_match = m_price
                price_start = m_price.start()
                price_end = m_price.end()

        name_parts: List[str] = []
        if name_prefix:
            name_parts.append(name_prefix)
        if price_match:
            desc_segment = tail[:price_start].strip()
        else:
            desc_segment = tail.strip()
        if desc_segment:
            name_parts.append(desc_segment)

        nome_raw = " ".join(name_parts).strip()
        if not nome_raw:
            continue

        # Remove trailing unit words or dangling numbers (ex.: quantidade mínima)
        nome_raw = PAT_UNIT_TRAIL.sub("", nome_raw).strip(" -–—:,.;")
        nome_raw = re.sub(r"^\d{6,}\s+", "", nome_raw)
        if qty_val is not None:
            nome_raw = re.sub(rf"\b{qty_val}\b\s*$", "", nome_raw).strip(" -–—:,.;")
        nome_raw = re.sub(r"\b\d{1,3}\b\s*$", "", nome_raw).strip(" -–—:,.;")
        nome_raw = re.sub(r"^\d{3}\.\d{2}\.\d{5}\s*-\s*", "", nome_raw)
        nome_raw = re.sub(r"\bVALOR TOTAL.*$", "", nome_raw, flags=re.IGNORECASE).strip(" -–—:,.;")

        if qty_val is None:
            segment = desc_segment if desc_segment else tail
            if price_match:
                segment = tail[max(0, price_start - 40):price_start]
            else:
                segment = segment[-120:]
            qty_candidates = re.findall(r"\d{1,3}(?:\.\d{3})*(?:,\d{1,3})?", segment)
            for candidate in reversed(qty_candidates):
                if ',' in candidate:
                    continue
                qty_val = to_int_br(candidate)
                if qty_val:
                    break

        if qty_val is None:
            continue

        nome = _clean_nome(nome_raw)

        recs.append(ItemRecord(page=page_no, lote=str(lote_norm or ""), item=item_norm,
                               nome=nome, quantidade=qty_val, valor_unitario=price,
                               fonte="tabela", priority=2))
    return _dedup_best(recs)

PAT_ITEM_ANCHOR = re.compile(rf"\bItem\b\s*[:\-]?\s*({ITEM_ID})\b", re.IGNORECASE)
PAT_QTD = re.compile(r"Quantidad[ea]\s*[:\-]?\s*([\d\.\,]+)", re.IGNORECASE)
PAT_VU  = re.compile(r"(?:Valor|Pre[cç]o)\s+Unit[aá]rio\s*[:\-]?\s*R?\$?\s*(" + MONEY_BR + ")", re.IGNORECASE)

def heuristic_H3_item_anchor(page_no: int, text: str, lote_here: str) -> List[ItemRecord]:
    recs: List[ItemRecord] = []
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        mi = PAT_ITEM_ANCHOR.search(ln)
        if not mi: 
            continue
        raw_item = mi.group(1)
        lote_norm, item_norm = _normalize_item_lote(raw_item, lote_here)
        remainder = ln[mi.end():].strip()
        for sep in (":","–","-","—"):
            if sep in remainder:
                remainder = remainder.split(sep,1)[-1].strip()
        name_line = remainder if remainder and not re.search(r"\b(Item|Lote|Quantidade|Valor|Unit[aá]rio)\b", remainder, flags=re.IGNORECASE) else ""
        if not name_line:
            j = i+1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                name_line = lines[j].strip()

        window = "\n".join(lines[i:i+12])
        mq = PAT_QTD.search(window)
        mv = PAT_VU.search(window)
        nome = _clean_nome(name_line)

        recs.append(ItemRecord(page=page_no, lote=str(lote_norm or ""), item=item_norm,
                               nome=nome,
                               quantidade=to_int_br(mq.group(1)) if mq else None,
                               valor_unitario=to_money_br(mv.group(1)) if mv else None,
                               fonte="parágrafo", priority=3))
    return _dedup_best(recs)
