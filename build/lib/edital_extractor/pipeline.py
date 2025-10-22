# -*- coding: utf-8 -*-
import hashlib
import os
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Tuple

import PyPDF2

from .utils import normalize_pdf_text, active_lote_per_page
from .extractors import (
    heuristic_H1_table_lines,
    heuristic_H2_row_buffer,
    heuristic_H3_item_anchor,
    ItemRecord,
)

import re

FURNITURE_KEYWORDS = [
    "cadeir",
    "armari",
    "armario",
    "armar",
    "movel",
    "moveis",
    "mobiliario",
    "mesa",
    "estacao",
    "estante",
    "gaveteiro",
    "balcao",
    "bancada",
    "poltrona",
    "sofa",
    "arquivo",
    "rack",
    "longarina",
    "modulo",
    "modular",
]


def _normalize_match_text(value: str) -> str:
    if not isinstance(value, str):
        value = str(value or "")
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    return value.lower()


def _record_rank(rec: ItemRecord) -> Tuple[int, int]:
    has_alpha = bool(rec.nome and re.search(r"[A-Za-zÁÂÃÀÉÊÍÓÔÕÚÜÇ]", rec.nome))
    info = int(rec.nome is not None) + int(rec.quantidade is not None) + int(rec.valor_unitario is not None)
    name_penalty = 0 if has_alpha else 1
    return (name_penalty, rec.priority, -info)


def _record_sort_key(rec: ItemRecord) -> Tuple[str, str, str, int]:
    lote_key = rec.lote or ""
    item_field = rec.item or ""
    digits = re.sub(r"\D+", "", item_field)
    if digits:
        try:
            num_val = int(digits)
            return (lote_key, "0", f"{num_val:06d}", rec.page)
        except Exception:
            pass
    return (lote_key, "1", item_field.lower(), rec.page)


def _maybe_store(found: Dict[Tuple[str, str], ItemRecord], rec: ItemRecord):
    key = ((rec.lote or "").strip(), rec.page, rec.item)
    existing = found.get(key)
    if existing is None or _record_rank(rec) < _record_rank(existing):
        found[key] = rec


def _is_furniture_record(rec: ItemRecord) -> bool:
    candidates: List[str] = []
    if rec.nome:
        candidates.append(rec.nome)
    if not candidates:
        return False
    for text in candidates:
        norm = _normalize_match_text(text)
        if any(keyword in norm for keyword in FURNITURE_KEYWORDS):
            return True
        norm_compact = norm.replace(" ", "")
        if any(keyword in norm_compact for keyword in FURNITURE_KEYWORDS):
            return True
    return False


def _strip_common_headers(pages_text: List[str]) -> List[str]:
    """Remove header lines that repeat across the majority of pages."""
    if len(pages_text) <= 1:
        return pages_text

    header_freq: Counter[str] = Counter()
    raw_lines_per_page: List[List[str]] = []

    for txt in pages_text:
        lines = txt.splitlines()
        raw_lines_per_page.append(lines)
        for idx in range(min(5, len(lines))):
            trimmed = lines[idx].strip()
            if not trimmed or len(trimmed) < 4:
                continue
            norm = _normalize_match_text(trimmed)
            if not norm:
                continue
            header_freq[norm] += 1

    threshold = max(2, len(pages_text) // 2)
    common_headers = {norm for norm, count in header_freq.items() if count >= threshold}

    if not common_headers:
        return pages_text

    cleaned_pages: List[str] = []
    for lines in raw_lines_per_page:
        filtered: List[str] = []
        for idx, original_line in enumerate(lines):
            trimmed = original_line.strip()
            norm = _normalize_match_text(trimmed)
            if idx < 5 and trimmed and norm in common_headers:
                continue
            filtered.append(original_line)
        cleaned_pages.append("\n".join(filtered))
    return cleaned_pages

def search_pdf_for_bool(pages_text: List[str], pattern_union: str):
    pat = re.compile(rf"({pattern_union})\s*[:\-]?\s*(Sim|N[aã]o)", re.IGNORECASE)
    for idx, txt in enumerate(pages_text, start=1):
        m = pat.search(txt)
        if m:
            return (idx, True if m.group(2).lower().startswith("s") else False)
    return (None, None)

def search_pdf_for_garantia(pages_text: List[str]):
    pat = re.compile(r"(Garantia|Meses de garantia)\s*[:\-]?\s*(\d+)", re.IGNORECASE)
    for idx, txt in enumerate(pages_text, start=1):
        m = pat.search(txt)
        if m:
            try:
                return (idx, int(m.group(2)))
            except Exception:
                return (idx, None)
    return (None, None)

def search_pdf_for_normas(pages_text: List[str]):
    pat = re.compile(r"\b(?:ABNT\s*)?NBR\s*\d+|\bNR\s*\d+\b|\bISO\s*\d+\b", re.IGNORECASE)
    for idx, txt in enumerate(pages_text, start=1):
        matches = pat.findall(txt)
        if matches:
            norm_list = sorted(set([re.sub(r"\s+", " ", m).upper() for m in matches]))
            return (idx, norm_list)
    return (None, [])

def load_pdf_text(pdf_path: str) -> List[str]:
    reader = PyPDF2.PdfReader(pdf_path)
    pages_text: List[str] = []
    for page in reader.pages:
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        pages_text.append(normalize_pdf_text(raw))
    return pages_text

def process_pdf(pdf_path: str, pages_range: Optional[Tuple[int, int]] = None):
    with open(pdf_path, "rb") as fh:
        pdf_bytes = fh.read()
    doc_id = hashlib.sha256(pdf_bytes).hexdigest()
    filename = os.path.basename(pdf_path)

    pages_text_raw = load_pdf_text(pdf_path)
    n_pages = len(pages_text_raw)
    active_lote = active_lote_per_page(pages_text_raw)
    pages_text = _strip_common_headers(pages_text_raw)

    # Cabeçalho
    pag_amostra, amostra = search_pdf_for_bool(pages_text, "Amostra")
    pag_catalogo, catalogo = search_pdf_for_bool(pages_text, "Cat[aá]logo|Catalogo")
    pag_garantia, garantia = search_pdf_for_garantia(pages_text)
    pag_normas, normas = search_pdf_for_normas(pages_text)

    start = pages_range[0] if pages_range else 1
    end = min(pages_range[1], n_pages) if pages_range else n_pages

    found_by_key: Dict[Tuple[str, str], ItemRecord] = {}
    for pno in range(start, end + 1):
        txt = pages_text[pno - 1]
        lote_here = active_lote.get(pno)

        # Em ordem de prioridade
        for rec in heuristic_H1_table_lines(pno, txt, lote_here):
            _maybe_store(found_by_key, rec)
        for rec in heuristic_H2_row_buffer(pno, txt, lote_here):
            _maybe_store(found_by_key, rec)
        for rec in heuristic_H3_item_anchor(pno, txt, lote_here):
            _maybe_store(found_by_key, rec)

    # docs row
    all_records = list(found_by_key.values())
    furniture_records = [rec for rec in all_records if _is_furniture_record(rec)]
    furniture_records.sort(key=_record_sort_key)

    selected_records: List[ItemRecord] = []
    selected_pages: List[int] = []

    def _clusters_from_pages(pages: List[int], max_gap: int = 1) -> List[List[int]]:
        clusters: List[List[int]] = []
        current: List[int] = []
        for page in pages:
            if not current or page - current[-1] <= max_gap + 1:
                current.append(page)
            else:
                clusters.append(current)
                current = [page]
        if current:
            clusters.append(current)
        return clusters

    if furniture_records:
        by_page: Dict[int, List[ItemRecord]] = {}
        for rec in furniture_records:
            by_page.setdefault(rec.page, []).append(rec)

        table_pages = sorted({rec.page for rec in furniture_records if rec.fonte == "tabela"})

        if table_pages:
            clusters = _clusters_from_pages(table_pages)

            cluster_infos = []
            for cluster in clusters:
                cluster_recs = [
                    rec for page in cluster for rec in by_page.get(page, []) if rec.fonte == "tabela"
                ]
                if not cluster_recs:
                    cluster_recs = [
                        rec for page in cluster for rec in by_page.get(page, [])  # fallback
                    ]
                if not cluster_recs:
                    continue
                digit_values = []
                for rec in cluster_recs:
                    digits = re.sub(r"\D", "", rec.item or "")
                    try:
                        if digits:
                            digit_values.append(int(digits))
                    except Exception:
                        pass
                min_item = min(digit_values) if digit_values else None
                max_item = max(digit_values) if digit_values else None
                has_price = any(
                    rec.valor_unitario is not None for rec in cluster_recs
                )
                cluster_infos.append(
                    {
                        "pages": cluster,
                        "recs": cluster_recs,
                        "min_item": min_item,
                        "max_item": max_item,
                        "has_price": has_price,
                    }
                )

            cluster_infos.sort(key=lambda info: (info["pages"][0]))

            def _select_clusters(infos):
                if not infos:
                    return []
                # pick starting cluster prioritising lowest numbered item
                start_idx = 0
                best_start_score = (1, float("inf"), infos[0]["pages"][0])
                for idx, info in enumerate(infos):
                    price_flag = 0 if info.get("has_price") else 1
                    min_item = info["min_item"] if info["min_item"] is not None else float("inf")
                    score = (price_flag, min_item, info["pages"][0])
                    if score < best_start_score:
                        best_start_score = score
                        start_idx = idx
                selected = [infos[start_idx]]
                last = infos[start_idx]
                for info in infos[start_idx + 1 :]:
                    page_gap = info["pages"][0] - last["pages"][-1]
                    if page_gap > 4:
                        break
                    if selected[0].get("has_price") and not info.get("has_price"):
                        break
                    last_max = last["max_item"]
                    curr_min = info["min_item"]
                    if last_max is not None and curr_min is not None and curr_min < last_max:
                        break
                    selected.append(info)
                    last = info
                return selected

            selected_infos = _select_clusters(cluster_infos)

            selected_pages = sorted({page for info in selected_infos for page in info["pages"]})
            selected_page_set = set(selected_pages)
            selected_records = [
                rec
                for rec in furniture_records
                if rec.page in selected_page_set and rec.fonte == "tabela"
            ]
            if any(info.get("has_price") for info in selected_infos):
                priced_records = [rec for rec in selected_records if rec.valor_unitario is not None]
                if priced_records:
                    selected_records = priced_records
                    selected_pages = sorted({rec.page for rec in selected_records})
                    selected_page_set = set(selected_pages)
            if not selected_records:
                selected_records = [
                    rec for rec in furniture_records if rec.page in selected_page_set
                ]
        else:
            pages_all = sorted({rec.page for rec in furniture_records})
            clusters = _clusters_from_pages(pages_all)

            def cluster_score_all(cluster: List[int]) -> Tuple[int, int, int]:
                total = sum(len(by_page.get(page, [])) for page in cluster)
                return (total, len(cluster), -cluster[0])

            best_cluster = max(clusters, key=cluster_score_all)
            selected_pages = best_cluster
            selected_records = [
                rec for rec in furniture_records if rec.page in selected_pages
            ]

    furniture_records = selected_records
    if furniture_records and selected_pages:
        if len(selected_pages) == 1:
            observacao = f"extraído (cadeiras/móveis, pág. {selected_pages[0]})"
        else:
            observacao = (
                f"extraído (cadeiras/móveis, págs. {selected_pages[0]}-{selected_pages[-1]})"
            )
    else:
        observacao = "sem cadeiras/móveis identificados"
    docs_row = [doc_id, filename, doc_id, "pdf", n_pages, "", "", "tabelado", observacao]

    # labels long
    labels_rows = []
    for rec in furniture_records:
        fonte = rec.fonte
        labels_rows.append([doc_id, filename, rec.lote, rec.item, "lote", rec.lote, rec.page, "", fonte, "batch"])
        labels_rows.append([doc_id, filename, rec.lote, rec.item, "item", rec.item, rec.page, "", fonte, "batch"])
        if rec.nome:
            labels_rows.append([doc_id, filename, rec.lote, rec.item, "nome do produto", rec.nome, rec.page, "", fonte, "batch"])
        if rec.quantidade is not None:
            labels_rows.append([doc_id, filename, rec.lote, rec.item, "quantidade", rec.quantidade, rec.page, "", fonte, "batch"])
        if rec.valor_unitario is not None:
            labels_rows.append([doc_id, filename, rec.lote, rec.item, "valor_unitario", rec.valor_unitario, rec.page, "", fonte, "batch"])

    # Propaga cabeçalho
    if furniture_records:
        for rec in furniture_records:
            if amostra is not None:
                labels_rows.append([doc_id, filename, rec.lote, rec.item, "amostra", amostra, pag_amostra, "", "cabeçalho", "batch"])
            if catalogo is not None:
                labels_rows.append([doc_id, filename, rec.lote, rec.item, "catálogo", catalogo, pag_catalogo, "", "cabeçalho", "batch"])
            if garantia is not None:
                labels_rows.append([doc_id, filename, rec.lote, rec.item, "garantia", garantia, pag_garantia, "", "cabeçalho", "batch"])
            if normas:
                labels_rows.append([doc_id, filename, rec.lote, rec.item, "normas_técnicas", "; ".join(normas), pag_normas, "", "cabeçalho", "batch"])

    return docs_row, labels_rows
