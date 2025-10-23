"""Generate Ground Truth CSVs from PDFs using HTML hints."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:  # pragma: no cover - optional heavy deps
    import torch
except ImportError:  # pragma: no cover - runtime guard
    torch = None  # type: ignore[assignment]

import pandas as pd

from rapidfuzz import fuzz

from edital_extractor.extractors import (
    ItemRecord,
    heuristic_H1_table_lines,
    heuristic_H2_row_buffer,
    heuristic_H3_item_anchor,
)
from edital_extractor.utils import normalize_pdf_text, active_lote_per_page
from edital_extractor.pipeline import _strip_common_headers

try:  # pragma: no cover - optional heavy deps
    import camelot
except ImportError as exc:  # pragma: no cover - runtime guard
    camelot = None  # type: ignore[assignment]

try:  # pragma: no cover - optional heavy deps
    import pdfplumber
except ImportError as exc:  # pragma: no cover - runtime guard
    pdfplumber = None  # type: ignore[assignment]

try:
    from pypdf import PdfReader
except ImportError as exc:  # pragma: no cover - runtime guard
    PdfReader = None  # type: ignore[assignment]

try:  # pragma: no cover - optional heavy deps
    from dedoc import DedocManager
except ImportError:  # pragma: no cover - runtime guard
    DedocManager = None  # type: ignore[assignment]

try:  # pragma: no cover - optional heavy deps
    from pdf2image import convert_from_path
except ImportError:  # pragma: no cover - runtime guard
    convert_from_path = None  # type: ignore[assignment]

try:  # pragma: no cover - optional heavy deps
    from transformers import DetrImageProcessor, TableTransformerForObjectDetection
except ImportError:  # pragma: no cover - runtime guard
    DetrImageProcessor = None  # type: ignore[assignment]
    TableTransformerForObjectDetection = None  # type: ignore[assignment]

from .html_to_unified import HEADER_COLUMNS
from .utils import (
    HintRow,
    compute_sha256,
    configure_logging,
    ensure_directory,
    list_pdf_files,
    normalize_label,
    normalize_whitespace,
)

logger = logging.getLogger("disdem_etl.pdf")

_dedoc_manager_cache: Any | None = None
_table_transformer_processor: Any | None = None
_table_transformer_model: Any | None = None


GT_COLUMNS: List[str] = [
    "document_id",
    "filename_pdf",
    "pdf_sha256",
    "num_pages",
    "orgao_codigo",
    "orgao_nome",
    "processo",
    "pregão",
    "conlicitacao",
    "etapa",
    "oc_bb_uasg",
    "uf",
    "quadrante",
    "data_publicacao",
    "data_hora_disputa",
    "data_solicitacao",
    "data_agendamento",
    "responsavel",
    "realizado_analise",
    "amostra",
    "catalogo",
    "valor_teto",
    "ticket_medio",
    "meses_garantia",
    "lote",
    "item",
    "nome_real",
    "quantidade",
    "valor_unitario",
    "ampla_concorrencia",
    "produto_cadastrado",
    "garantia",
    "amostra_item",
    "catalogo_item",
    "normas_tecnicas",
    "pagina",
    "fonte",
    "evidencia",
    "match_score",
    "hint_produto",
]


@dataclass(slots=True)
class TableRow:
    page: int
    text: str
    cells: List[str]

    def contains_item(self, item: str) -> bool:
        return bool(re.search(rf"\b{re.escape(item)}\b", self.text))


def _get_dedoc_manager() -> Any | None:
    global _dedoc_manager_cache
    if DedocManager is None:
        return None
    if _dedoc_manager_cache is None:
        try:
            _dedoc_manager_cache = DedocManager()
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("Falha ao inicializar DedocManager: %s", exc)
            _dedoc_manager_cache = None
    return _dedoc_manager_cache


def _collect_dedoc_lines(node: Dict[str, Any], lines_by_page: Dict[int, List[str]]) -> None:
    text = node.get("text") or ""
    meta = node.get("metadata") or {}
    page_id = meta.get("page_id")
    if text and isinstance(page_id, int):
        for fragment in text.splitlines():
            cleaned = normalize_whitespace(fragment)
            if cleaned:
                lines_by_page[page_id + 1].append(cleaned)
    for child in node.get("subparagraphs") or []:
        if isinstance(child, dict):
            _collect_dedoc_lines(child, lines_by_page)


def _parse_dedoc_tables(raw_tables: Any) -> List[TableRow]:
    if not isinstance(raw_tables, list):
        return []
    rows: List[TableRow] = []
    for table in raw_tables:
        if not isinstance(table, dict):
            continue
        cells = table.get("cells") or []
        if not isinstance(cells, list):
            continue
        page_id = None
        grouped: Dict[int, Dict[int, str]] = defaultdict(dict)
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            text = normalize_whitespace(str(cell.get("text", "")).strip())
            if not text:
                continue
            row_idx = cell.get("row") if cell.get("row") is not None else cell.get("row_id")
            col_idx = cell.get("column") if cell.get("column") is not None else cell.get("column_id")
            if row_idx is None or col_idx is None:
                continue
            try:
                r = int(row_idx)
                c = int(col_idx)
            except (TypeError, ValueError):
                continue
            grouped[r][c] = text
            if page_id is None:
                meta = cell.get("metadata")
                if isinstance(meta, dict) and isinstance(meta.get("page_id"), int):
                    page_id = meta["page_id"]
        if not grouped:
            continue
        if page_id is None:
            meta = table.get("metadata")
            if isinstance(meta, dict) and isinstance(meta.get("page_id"), int):
                page_id = meta["page_id"]
        page = int(page_id) + 1 if isinstance(page_id, int) else 1
        for r_idx in sorted(grouped):
            cols = grouped[r_idx]
            ordered_cells = [cols[c] for c in sorted(cols)]
            row_text = normalize_whitespace(" ".join(ordered_cells))
            if row_text:
                rows.append(TableRow(page=page, text=row_text, cells=ordered_cells))
    return rows


def _extract_with_dedoc(
    pdf_path: Path,
    *,
    language: str,
    max_pages: int,
) -> Tuple[List[Tuple[int, List[str]]], List[TableRow]]:
    manager = _get_dedoc_manager()
    if manager is None:
        logger.warning("dedoc não está disponível; ignorando integração opcional.")
        return [], []

    params: Dict[str, str] = {
        "language": language,
        "need_pdf_table_analysis": "true",
        "need_content_analysis": "false",
        "document_orientation": "auto",
        "is_one_column_document": "auto",
    }
    if max_pages > 0:
        params["pages"] = f"1:{max_pages}"

    try:
        parsed = manager.parse(str(pdf_path), parameters=params)
        payload = parsed.to_api_schema().model_dump()
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.warning("Dedoc falhou em %s: %s", pdf_path.name, exc)
        return [], []

    structure = payload.get("content", {}).get("structure")
    lines_by_page: Dict[int, List[str]] = defaultdict(list)
    if isinstance(structure, dict):
        _collect_dedoc_lines(structure, lines_by_page)
    dedoc_text = sorted(
        ((page, lines) for page, lines in lines_by_page.items()),
        key=lambda item: item[0],
    )
    dedoc_tables = _parse_dedoc_tables(payload.get("content", {}).get("tables"))
    logger.debug(
        "Dedoc retornou %d página(s) de texto e %d tabela(s) para %s",
        len(dedoc_text),
        len(dedoc_tables),
        pdf_path.name,
    )
    return dedoc_text, dedoc_tables


def _get_table_transformer_components() -> Tuple[Any | None, Any | None]:
    global _table_transformer_processor, _table_transformer_model
    if DetrImageProcessor is None or TableTransformerForObjectDetection is None:
        return None, None
    if _table_transformer_processor is None or _table_transformer_model is None:
        try:
            _table_transformer_processor = DetrImageProcessor.from_pretrained(
                "microsoft/table-transformer-detection"
            )
            _table_transformer_model = TableTransformerForObjectDetection.from_pretrained(
                "microsoft/table-transformer-detection"
            )
            _table_transformer_model.eval()
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("Falha ao carregar Table Transformer: %s", exc)
            _table_transformer_processor = None
            _table_transformer_model = None
    return _table_transformer_processor, _table_transformer_model


def _extract_with_table_transformer(
    pdf_path: Path,
    *,
    max_pages: int,
    threshold: float = 0.9,
) -> List[TableRow]:
    processor, model = _get_table_transformer_components()
    if (
        processor is None
        or model is None
        or convert_from_path is None
        or pdfplumber is None
        or torch is None
    ):
        return []

    rows: List[TableRow] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            limit = total_pages if max_pages <= 0 else min(max_pages, total_pages)
            if limit <= 0:
                return []

            try:
                images = convert_from_path(
                    str(pdf_path),
                    first_page=1,
                    last_page=limit,
                )
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.warning(
                    "Table Transformer falhou em %s: %s", pdf_path.name, exc
                )
                return []

            page_count = min(len(images), limit)
            if page_count < limit:
                logger.debug(
                    "Table Transformer gerou %d imagem(ns) para %d página(s) em %s",
                    page_count,
                    limit,
                    pdf_path.name,
                )

            for page_index in range(page_count):
                image = images[page_index]
                page = pdf.pages[page_index]

                inputs = processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)

                target_sizes = torch.tensor([image.size[::-1]])
                try:
                    detections = processor.post_process_object_detection(
                        outputs,
                        target_sizes=target_sizes,
                        threshold=threshold,
                    )[0]
                except Exception:  # pragma: no cover - runtime guard
                    continue

                boxes_tensor = detections.get("boxes")
                scores_tensor = detections.get("scores")
                labels_tensor = detections.get("labels")

                if boxes_tensor is None or len(boxes_tensor) == 0:
                    continue

                boxes = boxes_tensor.detach().cpu().tolist()
                scores = (
                    scores_tensor.detach().cpu().tolist()
                    if scores_tensor is not None
                    else [1.0] * len(boxes)
                )
                labels = (
                    labels_tensor.detach().cpu().tolist()
                    if labels_tensor is not None
                    else [0] * len(boxes)
                )

                pdf_width, pdf_height = page.width, page.height
                img_w, img_h = image.size

                for score, label, box in zip(scores, labels, boxes):
                    label_name = model.config.id2label.get(int(label), str(int(label)))
                    if label_name.lower() not in {"table", "table rotated"}:
                        continue

                    left = max(0.0, float(box[0]) / img_w * pdf_width)
                    right = min(pdf_width, float(box[2]) / img_w * pdf_width)
                    top_img = float(box[1])
                    bottom_img = float(box[3])
                    top_pdf = pdf_height - (top_img / img_h * pdf_height)
                    bottom_pdf = pdf_height - (bottom_img / img_h * pdf_height)
                    top = max(top_pdf, bottom_pdf)
                    bottom = min(top_pdf, bottom_pdf)
                    bbox = (
                        max(0.0, left),
                        max(0.0, bottom),
                        min(pdf_width, right),
                        min(pdf_height, top),
                    )

                    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                        continue

                    try:
                        cropped = page.crop(bbox)
                        text = cropped.extract_text() or ""
                    except Exception:  # pragma: no cover - runtime guard
                        text = ""

                    cells = [
                        normalize_whitespace(line)
                        for line in (text.splitlines() if text else [])
                        if normalize_whitespace(line)
                    ]

                    if not cells and text:
                        cells = [normalize_whitespace(text)]

                    if not cells:
                        continue

                    row_text = normalize_whitespace(" ".join(cells))
                    rows.append(
                        TableRow(
                            page=page_index + 1,
                            text=row_text,
                            cells=cells,
                        )
                    )

                if boxes:
                    logger.debug(
                        "Table Transformer detectou %d tabela(s) na página %d de %s",
                        len(boxes),
                        page_index + 1,
                        pdf_path.name,
                    )
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.warning("Table Transformer falhou em %s: %s", pdf_path.name, exc)
        return []

    return rows


def _normalize_item_code(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"\D+", "", value)


def _items_match(hint_value: str, record_value: Optional[str]) -> bool:
    if not hint_value or not record_value:
        return False
    if hint_value.strip() == record_value.strip():
        return True
    return _normalize_item_code(hint_value) == _normalize_item_code(record_value)


def _regex_record_score(rec: ItemRecord) -> Tuple[int, int, int]:
    filled = (
        int(bool(rec.nome)),
        int(rec.quantidade is not None and rec.quantidade > 0),
        int(rec.valor_unitario is not None and rec.valor_unitario > 0),
    )
    return (rec.priority, -sum(filled), rec.page)


def _format_money_br(value: Optional[float]) -> str:
    if value is None or value <= 0:
        return ""
    formatted = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {formatted}"


def _build_regex_evidence(
    rec: ItemRecord,
    pdf_lines: Dict[int, List[str]],
) -> str:
    lines = pdf_lines.get(rec.page) or []
    for line in lines:
        if rec.item and re.search(rf"\b{re.escape(rec.item)}\b", line):
            return line
        if rec.nome and rec.nome.lower() in line.lower():
            return line
    parts = [f"Item {rec.item}".strip()]
    if rec.nome:
        parts.append(rec.nome)
    return " - ".join(part for part in parts if part)


def extract_regex_records(reader, limit: int) -> List[ItemRecord]:
    if reader is None:
        return []

    total_pages = len(reader.pages)
    if total_pages == 0:
        return []
    page_limit = total_pages if limit <= 0 else min(limit, total_pages)

    pages_text_raw: List[str] = []
    for idx in range(page_limit):
        try:
            raw = reader.pages[idx].extract_text() or ""
        except Exception:
            raw = ""
        pages_text_raw.append(normalize_pdf_text(raw))

    if not pages_text_raw:
        return []

    pages_text = _strip_common_headers(pages_text_raw)
    active_lotes = active_lote_per_page(pages_text_raw)

    records: List[ItemRecord] = []
    for page_no, txt in enumerate(pages_text[:page_limit], start=1):
        lote_here = active_lotes.get(page_no)
        for extractor in (
            heuristic_H1_table_lines,
            heuristic_H2_row_buffer,
            heuristic_H3_item_anchor,
        ):
            for rec in extractor(page_no, txt, lote_here):
                if not rec.lote and lote_here:
                    rec.lote = str(lote_here)
                records.append(rec)
    return records


def match_from_regex(
    hint: HintRow,
    regex_records: Sequence[ItemRecord],
    used_keys: Set[Tuple[str, str, int]],
    pdf_text_map: Dict[int, List[str]],
) -> Tuple[str, str, str, str, int]:
    if not hint.item:
        return "", "", "", "", 0

    lot_hint_norm = _normalize_item_code(hint.lote)
    candidates: List[ItemRecord] = []
    for rec in regex_records:
        key = (rec.lote or "", rec.item or "", rec.page)
        if key in used_keys:
            continue
        if not _items_match(hint.item, rec.item):
            continue
        if lot_hint_norm and _normalize_item_code(rec.lote) != lot_hint_norm:
            continue
        candidates.append(rec)

    if not candidates:
        return "", "", "", "", 0

    best = min(candidates, key=_regex_record_score)
    used_keys.add((best.lote or "", best.item or "", best.page))

    snippet = _build_regex_evidence(best, pdf_text_map)
    quantidade = str(best.quantidade) if best.quantidade is not None else ""
    valor_unitario = _format_money_br(best.valor_unitario)
    return (
        snippet,
        quantidade,
        valor_unitario,
        "",
        best.page,
    )


def load_hint_rows(csv_path: Path) -> List[HintRow]:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    hints: List[HintRow] = []
    for _, row in df.iterrows():
        header = {col: str(row.get(col, "") or "") for col in HEADER_COLUMNS}
        hints.append(
            HintRow(
                item=str(row.get("item", "") or ""),
                lote=str(row.get("lote", "") or ""),
                produto=str(row.get("produto", "") or ""),
                tipo_produto=str(row.get("tipo_produto", "") or ""),
                descricao_tipo=str(row.get("descricao_tipo", "") or ""),
                produto_referencia=str(row.get("produto_referencia", "") or ""),
                quantidade=str(row.get("quantidade", "") or ""),
                ampla_concorrencia=str(row.get("ampla_concorrencia", "") or ""),
                valor_unitario=str(row.get("valor_unitario", "") or ""),
                header=header,
            )
        )
    return hints


def extract_tables(
    pdf_path: Path,
    *,
    max_pages: int,
) -> List[TableRow]:
    if camelot is None:
        logger.warning("Camelot não está instalado; extração de tabelas indisponível.")
        return []

    page_spec = "1-end" if max_pages <= 0 else f"1-{max_pages}"
    table_rows: List[TableRow] = []
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(
                str(pdf_path),
                pages=page_spec,
                flavor=flavor,
                strip_text="\n",
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.debug("Camelot falhou (%s): %s", flavor, exc)
            continue

        for table in tables:
            try:
                page_no = int(table.page)
            except Exception:
                page_no = 1
            df = table.df.replace("\n", " ", regex=True)
            for row_idx in range(len(df)):
                cells = [normalize_whitespace(str(cell)) for cell in df.iloc[row_idx].tolist()]
                row_text = normalize_whitespace(" ".join(cells))
                if not row_text:
                    continue
                if normalize_label(row_text).startswith("item"):
                    continue
                table_rows.append(TableRow(page=page_no, text=row_text, cells=cells))
        if table_rows:
            break
    return table_rows


def extract_pdf_text_lines(
    pdf_path: Path,
    *,
    max_pages: int,
) -> List[Tuple[int, List[str]]]:
    if pdfplumber is None:
        logger.warning("pdfplumber não está instalado; fallback em texto indisponível.")
        return []

    lines_per_page: List[Tuple[int, List[str]]] = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        limit = total if max_pages <= 0 else min(max_pages, total)
        for page_idx in range(limit):
            page = pdf.pages[page_idx]
            text = page.extract_text() or ""
            lines = [
                normalize_whitespace(ln)
                for ln in text.splitlines()
                if normalize_whitespace(ln)
            ]
            lines_per_page.append((page_idx + 1, lines))
    return lines_per_page


def _extract_quantity_ampla_from_text(text: str) -> Tuple[str, str]:
    tokens = text.split()
    quantidade = ""
    ampla = ""
    for idx, tok in enumerate(tokens):
        low = tok.lower()
        if low in {"sim", "não", "nao"}:
            ampla = "Sim" if low == "sim" else "Não"
            for rev in range(idx - 1, -1, -1):
                candidate = re.sub(r"[^\d]", "", tokens[rev])
                if candidate.isdigit():
                    quantidade = candidate
                    break
            break
    return quantidade, ampla


def _extract_valor_unitario(text: str) -> str:
    match = re.search(r"R\$\s*([\d\.\,]+)", text)
    if not match:
        return ""
    valor = match.group(1)
    return f"R$ {valor}"


def _attach_header_fields(row_data: Dict[str, str], header: Dict[str, str]) -> None:
    for column in HEADER_COLUMNS:
        row_data[column] = header.get(column, "")


def _rows_from_records_without_hints(
    *,
    document_id: str,
    pdf_path: Path,
    pdf_hash: str,
    num_pages: int,
    records: Sequence[ItemRecord],
    pdf_text_map: Dict[int, List[str]],
) -> List[Dict[str, str]]:
    header_empty = {column: "" for column in HEADER_COLUMNS}
    rows: List[Dict[str, str]] = []
    seen_keys: Set[Tuple[str, str, int]] = set()

    for rec in records:
        item_value = (rec.item or "").strip()
        if not item_value:
            continue
        key = ((rec.lote or "").strip(), item_value, rec.page)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        snippet = _build_regex_evidence(rec, pdf_text_map)
        quantidade = str(rec.quantidade) if rec.quantidade is not None else ""
        valor_unitario = _format_money_br(rec.valor_unitario)
        fonte = "pdf_tabela" if rec.fonte == "tabela" else "pdf_text"
        evidencia_label = "TABELA" if fonte == "pdf_tabela" else "TEXTO"
        evidencia_payload = snippet or (rec.nome or "")
        evidencia = (
            f'{evidencia_label} p.{rec.page} | trecho="{evidencia_payload}"'
            if evidencia_payload
            else f"{evidencia_label} p.{rec.page}"
        )

        row = {
            "document_id": document_id,
            "filename_pdf": pdf_path.name,
            "pdf_sha256": pdf_hash,
            "num_pages": str(num_pages),
            "lote": rec.lote or "",
            "item": item_value,
            "nome_real": rec.nome or evidencia_payload,
            "quantidade": quantidade,
            "valor_unitario": valor_unitario,
            "ampla_concorrencia": "",
            "produto_cadastrado": "",
            "garantia": "",
            "amostra_item": "",
            "catalogo_item": "",
            "normas_tecnicas": "",
            "pagina": str(rec.page),
            "fonte": fonte,
            "evidencia": evidencia,
            "match_score": "0",
            "hint_produto": "",
        }
        _attach_header_fields(row, header_empty)
        rows.append(row)
    return rows


def match_from_tables(
    hint: HintRow,
    table_rows: Sequence[TableRow],
    *,
    min_score: int,
) -> Tuple[str, str, str, str, int]:
    best_score = -1
    best_row: Optional[TableRow] = None
    hint_text = hint.combined_hint or hint.produto
    for row in table_rows:
        if not row.contains_item(hint.item):
            continue
        if not hint_text:
            continue
        score = fuzz.token_set_ratio(row.text, hint_text)
        if score > best_score:
            best_score = score
            best_row = row
    if best_row and best_score >= min_score:
        quantidade, ampla = _extract_quantity_ampla_from_text(best_row.text)
        valor = _extract_valor_unitario(best_row.text)
        return (
            best_row.text,
            quantidade,
            valor,
            ampla,
            best_row.page,
        )
    return "", "", "", "", 0


def match_from_text(
    hint: HintRow,
    pdf_text: Sequence[Tuple[int, List[str]]],
    *,
    min_score: int,
) -> Tuple[str, str, str, str, int]:
    hint_text = hint.combined_hint or hint.produto
    best_score = -1
    best_snippet = ""
    best_page = 0
    if not hint_text:
        return "", "", "", "", 0
    for page_no, lines in pdf_text:
        for idx, line in enumerate(lines):
            if not re.search(rf"\b{re.escape(hint.item)}\b", line):
                continue
            window = " ".join(lines[idx : idx + 3])
            score = fuzz.token_set_ratio(window, hint_text)
            if score > best_score:
                best_score = score
                best_snippet = window
                best_page = page_no
    if best_snippet and best_score >= min_score:
        quantidade, ampla = _extract_quantity_ampla_from_text(best_snippet)
        valor = _extract_valor_unitario(best_snippet)
        return (
            best_snippet,
            quantidade,
            valor,
            ampla,
            best_page,
        )
    return "", "", "", "", 0


def filter_header_with_pdf(
    header: Dict[str, str],
    pdf_text: Sequence[Tuple[int, List[str]]],
) -> Dict[str, str]:
    """Keep header values only if they appear in the PDF text."""
    flat_text = " ".join(" ".join(lines) for _, lines in pdf_text)
    filtered: Dict[str, str] = {}
    for key, value in header.items():
        if not value:
            filtered[key] = ""
            continue
        if value in flat_text:
            filtered[key] = value
        else:
            filtered[key] = ""
    return filtered


def generate_gt_dataframe(
    pdf_path: Path,
    csv_path: Path | None,
    *,
    min_score: int,
    max_pages: int,
    use_dedoc: bool,
    dedoc_language: str,
) -> pd.DataFrame:
    hints: List[HintRow] = []
    use_hints = False

    if csv_path is None:
        logger.info(
            "CSV dica não fornecido para %s; usando fallback apenas com heurísticas.",
            pdf_path.name,
        )
    else:
        if not csv_path.exists():
            logger.warning("CSV dica não encontrado para %s", pdf_path.name)
            logger.info(
                "Prosseguindo com fallback heurístico para %s.",
                pdf_path.name,
            )
        else:
            hints = load_hint_rows(csv_path)
            if not hints:
                logger.warning("CSV %s não possui linhas.", csv_path.name)
                logger.info(
                    "Prosseguindo com fallback heurístico para %s.",
                    pdf_path.name,
                )
            else:
                use_hints = True

    if PdfReader is None:  # pragma: no cover - runtime guard
        raise RuntimeError("pypdf precisa estar instalado para o pdf2gt.")

    reader = PdfReader(str(pdf_path))
    num_pages = len(reader.pages)
    limited_pages = num_pages if max_pages <= 0 else min(max_pages, num_pages)

    table_transformer_tables = _extract_with_table_transformer(
        pdf_path,
        max_pages=limited_pages,
    )

    dedoc_text: List[Tuple[int, List[str]]] = []
    dedoc_tables: List[TableRow] = []
    if use_dedoc:
        dedoc_text, dedoc_tables = _extract_with_dedoc(
            pdf_path,
            language=dedoc_language,
            max_pages=limited_pages,
        )

    pdf_text = dedoc_text or extract_pdf_text_lines(pdf_path, max_pages=limited_pages)
    pdf_text_map: Dict[int, List[str]] = {page: lines for page, lines in pdf_text}
    table_rows = table_transformer_tables or dedoc_tables or extract_tables(
        pdf_path, max_pages=limited_pages
    )
    pdf_hash = compute_sha256(pdf_path)
    document_id = pdf_path.stem
    regex_records = extract_regex_records(reader, limited_pages)
    regex_used: Set[Tuple[str, str, int]] = set()

    rows: List[Dict[str, str]] = []

    if use_hints:
        for hint in hints:
            table_match = match_from_tables(hint, table_rows, min_score=min_score)
            regex_match = match_from_regex(
                hint,
                regex_records,
                regex_used,
                pdf_text_map,
            ) if not table_match[0] else ("", "", "", "", 0)
            text_match = (
                match_from_text(hint, pdf_text, min_score=min_score)
                if not table_match[0] and not regex_match[0]
                else ("", "", "", "", 0)
            )

            fonte = "nao_encontrado"
            evidencia = "NAO_ENCONTRADO"
            nome_real = ""
            quantidade = ""
            valor_unitario = ""
            ampla_concorrencia = ""
            pagina = ""
            match_score = 0

            if table_match[0]:
                fonte = "pdf_tabela"
                evidencia = f'TABELA p.{table_match[4]} | linha="{table_match[0]}"'
                nome_real = table_match[0]
                quantidade = table_match[1]
                valor_unitario = table_match[2]
                ampla_concorrencia = table_match[3]
                pagina = str(table_match[4]) if table_match[4] else ""
                match_score = fuzz.token_set_ratio(table_match[0], hint.combined_hint or hint.produto)
            elif regex_match[0]:
                fonte = "regex_pdf"
                evidencia = f'REGEX p.{regex_match[4]} | trecho="{regex_match[0]}"'
                nome_real = regex_match[0]
                quantidade = regex_match[1]
                valor_unitario = regex_match[2]
                ampla_concorrencia = regex_match[3]
                pagina = str(regex_match[4]) if regex_match[4] else ""
                match_score = fuzz.token_set_ratio(regex_match[0], hint.combined_hint or hint.produto)
            elif text_match[0]:
                fonte = "pdf_text"
                evidencia = f'TEXTO p.{text_match[4]} | trecho="{text_match[0]}"'
                nome_real = text_match[0]
                quantidade = text_match[1]
                valor_unitario = text_match[2]
                ampla_concorrencia = text_match[3]
                pagina = str(text_match[4]) if text_match[4] else ""
                match_score = fuzz.token_set_ratio(text_match[0], hint.combined_hint or hint.produto)

            header_values = filter_header_with_pdf(hint.header, pdf_text) if pdf_text else hint.header

            row = {
                "document_id": document_id,
                "filename_pdf": pdf_path.name,
                "pdf_sha256": pdf_hash,
                "num_pages": str(num_pages),
                "lote": hint.lote,
                "item": hint.item,
                "nome_real": nome_real,
                "quantidade": quantidade,
                "valor_unitario": valor_unitario,
                "ampla_concorrencia": ampla_concorrencia,
                "produto_cadastrado": "",
                "garantia": "",
                "amostra_item": "",
                "catalogo_item": "",
                "normas_tecnicas": "",
                "pagina": pagina,
                "fonte": fonte,
                "evidencia": evidencia,
                "match_score": str(match_score),
                "hint_produto": hint.combined_hint,
            }
            _attach_header_fields(row, header_values)
            rows.append(row)
    else:
        rows.extend(
            _rows_from_records_without_hints(
                document_id=document_id,
                pdf_path=pdf_path,
                pdf_hash=pdf_hash,
                num_pages=num_pages,
                records=regex_records,
                pdf_text_map=pdf_text_map,
            )
        )

    if not rows:
        return pd.DataFrame(columns=GT_COLUMNS)

    sanitized_rows: List[Dict[str, str]] = []
    for row in rows:
        clean: Dict[str, str] = {}
        for key, value in row.items():
            key_str = str(key)
            if value is None:
                clean[key_str] = ""
            elif isinstance(value, (list, tuple, set)):
                clean[key_str] = ", ".join(str(item) for item in value)
            elif hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
                converted = value.tolist()
                if isinstance(converted, (list, tuple, set)):
                    clean[key_str] = ", ".join(str(item) for item in converted)
                else:
                    clean[key_str] = str(converted)
            else:
                clean[key_str] = str(value)
        sanitized_rows.append(clean)

    df = pd.DataFrame(sanitized_rows)
    df = df.reindex(columns=GT_COLUMNS).fillna("")
    return df


def process_pdf_file(
    pdf_path: Path,
    csv_dir: Path | None,
    out_dir: Path,
    *,
    min_score: int,
    max_pages: int,
    use_dedoc: bool,
    dedoc_language: str,
) -> Path:
    csv_path = (csv_dir / f"{pdf_path.stem}.csv") if csv_dir is not None else None
    df = generate_gt_dataframe(
        pdf_path,
        csv_path,
        min_score=min_score,
        max_pages=max_pages,
        use_dedoc=use_dedoc,
        dedoc_language=dedoc_language,
    )
    out_path = out_dir / f"{pdf_path.stem}_GT.csv"
    df.to_csv(
        out_path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
    )
    logger.info("GT gerado: %s (%d linha(s))", out_path.name, len(df))
    return out_path


def _parallel_worker(args: Tuple[Path, Optional[Path], Path, int, int, bool, str]) -> Path:
    pdf_path, csv_dir, out_dir, min_score, max_pages, use_dedoc, dedoc_language = args
    return process_pdf_file(
        pdf_path,
        csv_dir,
        out_dir,
        min_score=min_score,
        max_pages=max_pages,
        use_dedoc=use_dedoc,
        dedoc_language=dedoc_language,
    )


def run_pipeline(
    pdf_dir: Path | str,
    csv_dir: Path | str | None,
    out_dir: Path | str,
    *,
    merge: bool,
    min_score: int,
    max_pages: int,
    jobs: Optional[int] = None,
    use_dedoc: bool = False,
    dedoc_language: str = "por",
) -> List[Path]:
    pdf_dir_path = Path(pdf_dir)
    csv_dir_path = Path(csv_dir) if csv_dir is not None else None
    out_dir_path = ensure_directory(out_dir)

    pdf_files = list_pdf_files(pdf_dir_path)
    if not pdf_files:
        raise FileNotFoundError(f"Nenhum PDF encontrado em {pdf_dir_path}")

    output_paths: List[Path] = []
    merged_frames: List[pd.DataFrame] = []

    if jobs and jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            future_to_pdf = {
                executor.submit(
                    _parallel_worker,
                    (
                        pdf_path,
                        csv_dir_path,
                        out_dir_path,
                        min_score,
                        max_pages,
                        use_dedoc,
                        dedoc_language,
                    ),
                ): pdf_path
                for pdf_path in pdf_files
            }
            for future in as_completed(future_to_pdf):
                path = future.result()
                output_paths.append(path)
        if merge:
            for path in output_paths:
                merged_frames.append(pd.read_csv(path, dtype=str).fillna(""))
    else:
        for pdf_path in pdf_files:
            out_path = process_pdf_file(
                pdf_path,
                csv_dir_path,
                out_dir_path,
                min_score=min_score,
                max_pages=max_pages,
                use_dedoc=use_dedoc,
                dedoc_language=dedoc_language,
            )
            output_paths.append(out_path)
            if merge:
                merged_frames.append(pd.read_csv(out_path, dtype=str).fillna(""))

    if merge and merged_frames:
        merged_df = pd.concat(merged_frames, ignore_index=True)
        merge_path = out_dir_path / "GT_merged.csv"
        merged_df.to_csv(merge_path, index=False, encoding="utf-8")
        logger.info("Arquivo GT consolidado: %s (%d linha(s))", merge_path.name, len(merged_df))
        output_paths.append(merge_path)

    return output_paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gera Ground Truth a partir de PDFs usando CSVs como dica."
    )
    parser.add_argument("--pdf-dir", required=True, help="Diretório com arquivos PDF")
    parser.add_argument(
        "--csv-dir",
        help="Diretório com CSVs gerados pelo html2csv (opcional para fallback sem hints)",
    )
    parser.add_argument("--out-dir", required=True, help="Diretório de saída para os GTs")
    parser.add_argument("--merge", action="store_true", help="Gera GT_merged.csv com todas as linhas")
    parser.add_argument(
        "--min-score",
        type=int,
        default=60,
        help="Score mínimo de similaridade (RapidFuzz) para aceitar um match",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Limita o número de páginas processadas (0 = todas)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Quantidade de processos em paralelo (0 = auto)",
    )
    parser.add_argument(
        "--dedoc",
        action="store_true",
        help="Ativa extração de texto/tabelas via Dedoc (opcional).",
    )
    parser.add_argument(
        "--dedoc-language",
        default="por",
        help="Idioma(s) utilizados pelo Dedoc (ex.: 'por', 'por+eng').",
    )
    parser.add_argument("--verbose", action="store_true", help="Ativa logs detalhados")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbose)

    jobs = None
    if args.jobs and args.jobs > 0:
        jobs = args.jobs

    try:
        run_pipeline(
            args.pdf_dir,
            args.csv_dir,
            args.out_dir,
            merge=args.merge,
            min_score=args.min_score,
            max_pages=args.max_pages,
            jobs=jobs,
            use_dedoc=args.dedoc,
            dedoc_language=args.dedoc_language,
        )
    except Exception as exc:  # pragma: no cover - CLI safety
        logger.exception("Falha ao gerar GT: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
