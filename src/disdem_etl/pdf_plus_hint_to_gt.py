"""Generate Ground Truth CSVs from PDFs using HTML hints."""

from __future__ import annotations

import argparse
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from rapidfuzz import fuzz

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
    csv_path: Path,
    *,
    min_score: int,
    max_pages: int,
) -> pd.DataFrame:
    if not csv_path.exists():
        logger.warning("CSV dica não encontrado para %s", pdf_path.name)
        return pd.DataFrame(columns=GT_COLUMNS)

    hints = load_hint_rows(csv_path)
    if not hints:
        logger.warning("CSV %s não possui linhas.", csv_path.name)
        return pd.DataFrame(columns=GT_COLUMNS)

    if PdfReader is None:  # pragma: no cover - runtime guard
        raise RuntimeError("pypdf precisa estar instalado para o pdf2gt.")

    reader = PdfReader(str(pdf_path))
    num_pages = len(reader.pages)
    limited_pages = num_pages if max_pages <= 0 else min(max_pages, num_pages)

    pdf_text = extract_pdf_text_lines(pdf_path, max_pages=limited_pages)
    table_rows = extract_tables(pdf_path, max_pages=limited_pages)
    pdf_hash = compute_sha256(pdf_path)
    document_id = pdf_path.stem

    rows: List[Dict[str, str]] = []
    for hint in hints:
        table_match = match_from_tables(hint, table_rows, min_score=min_score)
        text_match = match_from_text(hint, pdf_text, min_score=min_score) if not table_match[0] else ("", "", "", "", 0)

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

    df = pd.DataFrame(rows, columns=GT_COLUMNS).fillna("")
    return df


def process_pdf_file(
    pdf_path: Path,
    csv_dir: Path,
    out_dir: Path,
    *,
    min_score: int,
    max_pages: int,
) -> Path:
    csv_path = csv_dir / f"{pdf_path.stem}.csv"
    df = generate_gt_dataframe(pdf_path, csv_path, min_score=min_score, max_pages=max_pages)
    out_path = out_dir / f"{pdf_path.stem}_GT.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("GT gerado: %s (%d linha(s))", out_path.name, len(df))
    return out_path


def _parallel_worker(args: Tuple[Path, Path, Path, int, int]) -> Path:
    pdf_path, csv_dir, out_dir, min_score, max_pages = args
    return process_pdf_file(pdf_path, csv_dir, out_dir, min_score=min_score, max_pages=max_pages)


def run_pipeline(
    pdf_dir: Path | str,
    csv_dir: Path | str,
    out_dir: Path | str,
    *,
    merge: bool,
    min_score: int,
    max_pages: int,
    jobs: Optional[int] = None,
) -> List[Path]:
    pdf_dir_path = Path(pdf_dir)
    csv_dir_path = Path(csv_dir)
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
                    (pdf_path, csv_dir_path, out_dir_path, min_score, max_pages),
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
    parser.add_argument("--csv-dir", required=True, help="Diretório com CSVs gerados pelo html2csv")
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
        )
    except Exception as exc:  # pragma: no cover - CLI safety
        logger.exception("Falha ao gerar GT: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
