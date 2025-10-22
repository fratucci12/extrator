"""Conversion of DISDEM HTML pages to normalized CSV outputs."""

from __future__ import annotations

import argparse
import html as html_lib
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from .utils import (
    configure_logging,
    list_html_files,
    normalize_label,
    normalize_whitespace,
    read_text_file,
    sanitize_windows_filename,
    strip_tags,
)

logger = logging.getLogger("disdem_etl.html")


HEADER_COLUMNS: List[str] = [
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
]

ITEM_COLUMNS: List[str] = [
    "item",
    "lote",
    "produto",
    "tipo_produto",
    "descricao_tipo",
    "produto_referencia",
    "quantidade",
    "ampla_concorrencia",
    "valor_unitario",
]

OUTPUT_COLUMNS: List[str] = [
    "amostra",
    "catalogo",
    "item",
    "lote",
    "produto",
    "quantidade",
    "ampla_concorrencia",
    "valor_unitario",
]

HEADER_ALIASES: Dict[str, Sequence[str]] = {
    "orgao_codigo": ["orgao", "órgão"],
    "orgao_nome": ["nome do órgão", "orgao nome"],
    "processo": ["processo"],
    "pregão": ["pregao", "pregão"],
    "conlicitacao": ["conlicitacao", "conlicitação"],
    "etapa": ["etapa"],
    "oc_bb_uasg": ["oc/bb/uasg", "oc bb uasg", "oc/ bb/ uasg"],
    "uf": ["uf", "estado"],
    "quadrante": ["quadrante"],
    "data_publicacao": ["data da publicação", "data publicacao"],
    "data_hora_disputa": ["data/hora disputa", "data hora disputa"],
    "data_solicitacao": ["data da solicitação", "data solicitacao"],
    "data_agendamento": ["data do agendamento", "data agendamento"],
    "responsavel": ["responsavel", "responsável"],
    "realizado_analise": ["realizado analise", "realizado análise"],
    "amostra": ["amostra"],
    "catalogo": ["catalogo", "catálogo"],
    "valor_teto": ["valor teto"],
    "ticket_medio": ["ticket medio", "ticket médio"],
    "meses_garantia": ["meses de garantia", "garantia (meses)"],
}

HEADER_PATTERNS: Dict[str, str] = {
    "pregão": r"[A-Z]{1,3}/\d+/\d{4}",
    "conlicitacao": r"\d{4,}",
    "oc_bb_uasg": r"\d{3,}",
    "uf": r"[A-ZÁÉÍÓÚÂÊÔÃÕa-zà-úç ]+\([A-Z]{2}\)",
    "data_publicacao": r"\d{2}/\d{2}/\d{4}",
    "data_hora_disputa": r"\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}",
    "data_solicitacao": r"\d{2}/\d{2}/\d{4}",
    "data_agendamento": r"\d{2}/\d{2}/\d{4}",
    "realizado_analise": r"\b(?:Sim|Não|Nao)\b",
    "amostra": r"\b(?:Sim|Não|Nao)\b",
    "catalogo": r"\b(?:Sim|Não|Nao)\b",
    "valor_teto": r"R\$\s*[\d\.,]+",
    "ticket_medio": r"R\$\s*[\d\.,]+",
    "meses_garantia": r"\d+",
}

ITEM_HEADER_LABEL = "Items do DISDEM"

SKIP_LABELS = {
    normalize_label(lbl)
    for labels in HEADER_ALIASES.values()
    for lbl in labels
}


def read_html_text(path: Path | str) -> str:
    """Read HTML file and return cleaned plain text."""
    raw = read_text_file(Path(path))
    raw = re.sub(
        r"<(script|style)[^>]*?>.*?</\1>", " ", raw, flags=re.IGNORECASE | re.DOTALL
    )
    raw = re.sub(
        r"</(div|p|li|tr|td|th|label|span|h\d)>",
        r"</\1>\n",
        raw,
        flags=re.IGNORECASE,
    )
    text = strip_tags(raw)
    text = html_lib.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text


def lines_from_text(text: str) -> List[str]:
    """Split cleaned text into trimmed non-empty lines."""
    lines = []
    for ln in text.split("\n"):
        line = ln.strip()
        if line:
            lines.append(line)
    return lines


def _slice_window(lines: Sequence[str], start_label: str, end_labels: Sequence[str]) -> List[str]:
    """Extract lines within a window defined by label markers."""
    start_idx = next(
        (idx for idx, line in enumerate(lines) if normalize_label(line) == normalize_label(start_label)),
        -1,
    )
    if start_idx == -1:
        return list(lines)
    end_idx = len(lines)
    for end_label in end_labels:
        found = next(
            (
                idx
                for idx, line in enumerate(lines)
                if idx > start_idx and normalize_label(line) == normalize_label(end_label)
            ),
            -1,
        )
        if found != -1:
            end_idx = min(end_idx, found)
    return list(lines[start_idx:end_idx])


def _find_after(
    context_lines: Sequence[str],
    labels: Sequence[str],
    *,
    pattern: str | None = None,
    max_distance: int = 12,
) -> str:
    """Return the first value after any of the provided labels."""
    normalized_labels = [normalize_label(lbl) for lbl in labels]
    for idx, line in enumerate(context_lines):
        current_label = normalize_label(line)
        if current_label in normalized_labels:
            for offset in range(1, max_distance + 1):
                pos = idx + offset
                if pos >= len(context_lines):
                    break
                candidate = context_lines[pos].strip()
                if not candidate or normalize_label(candidate) in normalized_labels:
                    continue
                if candidate.endswith(":"):
                    continue
                if pattern:
                    if re.search(pattern, candidate, flags=re.IGNORECASE):
                        return candidate
                    # look for pattern within candidate even if the full label is not present
                    continue
                if normalize_label(candidate) in SKIP_LABELS:
                    continue
                return candidate
    if pattern:
        regex = re.compile(pattern, flags=re.IGNORECASE)
        for line in context_lines:
            match = regex.search(line)
            if match:
                return match.group(0)
    return ""


def parse_header(lines: Sequence[str]) -> Dict[str, str]:
    """Parse header window looking for configured labels."""
    header_window = _slice_window(lines, "Cabeçalho", ("Cotações", ITEM_HEADER_LABEL))
    header: Dict[str, str] = {column: "" for column in HEADER_COLUMNS}

    # First pass straightforward lookups
    for column in HEADER_COLUMNS:
        aliases = HEADER_ALIASES.get(column, (column,))
        header[column] = _find_after(
            header_window,
            aliases,
            pattern=HEADER_PATTERNS.get(column),
        )

    # Special case: órgão nome can follow órgão código when same alias is used twice
    if header["orgao_nome"] == "" and header["orgao_codigo"]:
        codigo_index = next(
            (
                idx
                for idx, line in enumerate(header_window)
                if header["orgao_codigo"] in line and normalize_label(line) not in SKIP_LABELS
            ),
            -1,
        )
        if codigo_index != -1:
            for offset in range(1, 8):
                pos = codigo_index + offset
                if pos >= len(header_window):
                    break
                candidate = header_window[pos].strip()
                if normalize_label(candidate) in SKIP_LABELS or candidate.endswith(":"):
                    continue
                header["orgao_nome"] = candidate
                break

    # Normalize boolean fields
    for col in ("realizado_analise", "amostra", "catalogo"):
        val = header[col].strip().lower()
        if val.startswith("sim"):
            header[col] = "Sim"
        elif val in {"nao", "não"}:
            header[col] = "Não"
        elif val:
            header[col] = header[col].strip()

    return header


def _extract_quantity_ampla(tokens: List[str]) -> tuple[str, str]:
    ampla = ""
    quantidade = ""
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


def _extract_produto_parts(tokens: List[str]) -> tuple[str, str, str, str]:
    if not tokens:
        return "", "", "", ""
    produto_referencia = ""
    body_tokens = tokens[:]

    # Look for explicit "Produto Referência" marker
    for idx, tok in enumerate(body_tokens):
        if normalize_label(tok) == "produto referencia":
            produto_referencia = normalize_whitespace(" ".join(body_tokens[idx + 1 :]))
            body_tokens = body_tokens[:idx]
            break

    if len(body_tokens) >= 3:
        produto = normalize_whitespace(" ".join(body_tokens[:-2]))
        tipo_produto = body_tokens[-2]
        descricao_tipo = body_tokens[-1]
    elif len(body_tokens) == 2:
        produto = body_tokens[0]
        tipo_produto = body_tokens[1]
        descricao_tipo = ""
    else:
        produto = body_tokens[0]
        tipo_produto = ""
        descricao_tipo = ""

    return produto, tipo_produto, descricao_tipo, produto_referencia


def parse_items(lines: Sequence[str]) -> pd.DataFrame:
    """Parse item grid from the textual lines."""
    try:
        start_idx = next(
            idx for idx, line in enumerate(lines) if normalize_label(line) == normalize_label(ITEM_HEADER_LABEL)
        )
    except StopIteration:
        return pd.DataFrame(columns=ITEM_COLUMNS)

    relevant_lines: List[str] = []
    skip_markers = {
        "item lote produto tipo de produto",
        "descricao tipo produto referencia quantidade ampla concorrencia valor unitario acoes",
        "item",
        "lote",
        "produto",
        "tipo de produto",
        "descricao tipo",
        "produto referencia",
        "quantidade",
        "ampla concorrencia",
        "valor unitario",
        "acoes",
    }
    for raw in lines[start_idx + 1 :]:
        normalized = normalize_label(raw)
        if raw.startswith("Total de itens"):
            break
        if normalized in skip_markers:
            continue
        relevant_lines.append(raw.strip())

    items: List[dict] = []
    current: dict | None = None

    for line in relevant_lines:
        if not line:
            continue

        if re.match(r"^\d+\b", line):
            parts = line.split()
            item = parts[0]
            lote = parts[1] if len(parts) > 1 else ""
            current = {
                "item": item,
                "lote": lote,
                "raw_lines": [],
            }
            continue

        if current is None:
            continue

        if line.startswith("R$"):
            valor = line.replace("R$ ", "").replace("R$", "").strip()
            valor_unitario = f"R$ {valor}"
            detail_tokens = []
            for raw_line in current["raw_lines"]:
                detail_tokens.extend(raw_line.split())
            quantidade, ampla = _extract_quantity_ampla(detail_tokens)
            produto_tokens = detail_tokens
            produto, tipo_produto, descricao_tipo, produto_referencia = _extract_produto_parts(produto_tokens[:-2] if quantidade else produto_tokens)

            item_row = {
                "item": current["item"],
                "lote": current["lote"],
                "produto": produto,
                "tipo_produto": tipo_produto,
                "descricao_tipo": descricao_tipo,
                "produto_referencia": produto_referencia,
                "quantidade": quantidade,
                "ampla_concorrencia": ampla,
                "valor_unitario": valor_unitario,
            }
            items.append(item_row)
            current = None
            continue

        current["raw_lines"].append(line)

    if not items:
        return pd.DataFrame(columns=ITEM_COLUMNS)

    return pd.DataFrame(items, columns=ITEM_COLUMNS)


def convert_single_html(path: Path | str) -> pd.DataFrame:
    """Convert a single HTML file into a normalized DataFrame."""
    text = read_html_text(path)
    lines = lines_from_text(text)
    header = parse_header(lines)
    df_items = parse_items(lines)
    if df_items.empty:
        df_items = pd.DataFrame([{
            "item": "",
            "lote": "",
            "produto": "",
            "tipo_produto": "",
            "descricao_tipo": "",
            "produto_referencia": "",
            "quantidade": "",
            "ampla_concorrencia": "",
            "valor_unitario": "",
        }])
    df_items["amostra"] = header.get("amostra", "")
    df_items["catalogo"] = header.get("catalogo", "")
    for column in OUTPUT_COLUMNS:
        if column not in df_items.columns:
            df_items[column] = ""
    df_items = df_items[OUTPUT_COLUMNS]
    return df_items


def convert_directory(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    merge: bool = False,
    merge_out: Path | None = None,
    add_source_col: bool = False,
) -> List[Path]:
    """Convert all HTML files from a directory and optionally merge the results."""
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    html_files = list_html_files(in_dir)
    if not html_files:
        raise FileNotFoundError(f"Nenhum arquivo HTML encontrado em {in_dir}")

    merged_frames: List[pd.DataFrame] = []
    output_paths: List[Path] = []

    for html_path in html_files:
        logger.info("Processando %s", html_path.name)
        df = convert_single_html(html_path)
        base_name = sanitize_windows_filename(html_path.stem)
        csv_path = out_dir / f"{base_name}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        merged_df = df.copy()
        if add_source_col:
            merged_df["source_html"] = html_path.name
        merged_frames.append(merged_df)
        output_paths.append(csv_path)
        logger.info("Gerado %s (%d linha(s))", csv_path.name, len(df))

    if merge and merged_frames:
        merged_df = pd.concat(merged_frames, ignore_index=True)
        merge_path = merge_out or (out_dir / "merged.csv")
        merged_df.to_csv(merge_path, index=False, encoding="utf-8")
        output_paths.append(Path(merge_path))
        logger.info("Merge salvo em %s (%d linha(s))", merge_path, len(merged_df))

    return output_paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Converte HTMLs do DISDEM para CSVs normalizados.")
    parser.add_argument("--input", required=True, help="Diretório com arquivos .html/.htm")
    parser.add_argument("--out-dir", required=True, help="Diretório de saída para os CSVs gerados")
    parser.add_argument("--merge", action="store_true", help="Gera um CSV único com todos os arquivos")
    parser.add_argument("--merge-out", help="Caminho do CSV mesclado (padrão: <out-dir>/merged.csv)")
    parser.add_argument(
        "--add-source-col",
        action="store_true",
        help="Ao mesclar, adiciona a coluna source_html com o nome do arquivo de origem",
    )
    parser.add_argument("--verbose", action="store_true", help="Ativa logs em nível DEBUG")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbose)
    try:
        convert_directory(
            args.input,
            args.out_dir,
            merge=args.merge,
            merge_out=Path(args.merge_out) if args.merge_out else None,
            add_source_col=args.add_source_col,
        )
    except Exception as exc:  # pragma: no cover - CLI safety
        logger.exception("Falha ao processar HTMLs: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
