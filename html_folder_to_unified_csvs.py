# html_folder_to_unified_csvs.py
# Lê TODOS os HTMLs de uma pasta e gera CSVs "únicos" (cabeçalho + itens)
# mantendo o NOME ORIGINAL do HTML (apenas trocando a extensão p/ .csv).
# Opcional: --merge cria um CSV mesclado com todas as linhas.
#
# Uso:
#   python html_folder_to_unified_csvs.py --input ./htmls --out-dir ./csv_out --merge
#
# Requer: pandas  (pip install pandas)

import argparse
import html as htmllib
import os
import re
import sys
from typing import List, Dict

import pandas as pd


# ----------------------------
# 1) Ler HTML e linearizar texto
# ----------------------------
def read_html_text(path: str) -> str:
    encodings = ("utf-8", "latin-1", "cp1252")
    raw = ""
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                raw = f.read()
            break
        except Exception as e:
            last_err = e
    if not raw and last_err:
        raise last_err

    raw = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", raw, flags=re.IGNORECASE)
    raw = re.sub(r"</(div|p|li|tr|td|th|label|span|h\d)>", r"</\1>\n", raw, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", raw)
    text = htmllib.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text


def lines_from_text(text: str) -> List[str]:
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


# ----------------------------
# 2) Janela do cabeçalho + busca após rótulos
# ----------------------------
def slice_window(lines: List[str], start_label="Cabeçalho",
                 end_labels=("Cotações", "Items do DISDEM")) -> List[str]:
    s = next((i for i, l in enumerate(lines) if l.lower().startswith(start_label.lower())), -1)
    e = len(lines)
    for el in end_labels:
        j = next((i for i, l in enumerate(lines) if i > s and l.lower().startswith(el.lower())), -1)
        if j != -1:
            e = min(e, j)
    return lines[s:e] if s != -1 else lines


def find_after(hlines: List[str], label: str, pattern: str = None, skip_labels=()) -> str:
    try:
        idx = hlines.index(label)
    except ValueError:
        idx = next((i for i, l in enumerate(hlines) if l.lower() == label.lower()), -1)
    if idx == -1:
        return ""

    if pattern:
        rx = re.compile(pattern, re.IGNORECASE)
        for j in range(idx + 1, min(idx + 30, len(hlines))):
            v = hlines[j]
            if not v or v in skip_labels or v.endswith(":"):
                continue
            m = rx.search(v)
            if m:
                return m.group(0)
        return ""
    else:
        for j in range(idx + 1, min(idx + 10, len(hlines))):
            v = hlines[j]
            if v and v not in skip_labels and not v.endswith(":"):
                return v
        return ""


# ----------------------------
# 3) Extração do cabeçalho e dos itens
# ----------------------------
def parse_header(lines: List[str]) -> Dict[str, str]:
    hlines = slice_window(lines, "Cabeçalho", ("Cotações", "Items do DISDEM"))

    skip = {
        "Processo", "Pregão", "Conlicitação", "Etapa", "OC/BB/UASG", "UF", "Quadrante",
        "Data da Publicação", "Data/Hora Disputa", "Data da Solicitação", "Data do Agendamento",
        "Responsável", "Realizado análise", "Amostra", "Catálogo", "Valor Teto", "Ticket Médio", "Meses de garantia"
    }

    orgao_codigo = find_after(hlines, "Órgão")
    orgao_nome = find_after(hlines, orgao_codigo) if orgao_codigo else ""

    processo = find_after(hlines, "Processo")
    if processo and re.fullmatch(r"(?i)(preg[aã]o|[A-Z]{1,3}/\d+/\d{4})", processo):
        processo = ""

    pregao = find_after(hlines, "Pregão", pattern=r"[A-Z]{1,3}/\d+/\d{4}", skip_labels=skip) or find_after(hlines, "Pregão")
    conlicitacao = find_after(hlines, "Conlicitação", pattern=r"\d{6,}", skip_labels=skip) or find_after(hlines, "Conlicitação")

    etapa = find_after(hlines, "Etapa")
    oc_bb_uasg = find_after(hlines, "OC/BB/UASG", pattern=r"\d{3,}", skip_labels=skip) or find_after(hlines, "OC/BB/UASG")

    uf = find_after(hlines, "UF", pattern=r"[A-ZÁÉÍÓÚÂÊÔÃÕa-zà-úç ]+\s\([A-Z]{2}\)", skip_labels=skip) or find_after(hlines, "UF")
    quadrante = find_after(hlines, "Quadrante", pattern=r"^[A-Z]\b", skip_labels=skip) or find_after(hlines, "Quadrante")

    date_pat = r"\d{2}/\d{2}/\d{4}"
    dt_pat = r"\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}"

    data_publicacao = find_after(hlines, "Data da Publicação", pattern=date_pat, skip_labels=skip) or find_after(hlines, "Data da Publicação")
    data_hora_disputa = find_after(hlines, "Data/Hora Disputa", pattern=dt_pat, skip_labels=skip) or find_after(hlines, "Data/Hora Disputa")
    data_solicitacao = find_after(hlines, "Data da Solicitação", pattern=date_pat, skip_labels=skip)
    data_agendamento = find_after(hlines, "Data do Agendamento", pattern=date_pat, skip_labels=skip)

    responsavel = find_after(hlines, "Responsável")
    simnao = r"Sim|Não"
    realizado_analise = find_after(hlines, "Realizado análise", pattern=simnao, skip_labels=skip) or find_after(hlines, "Realizado análise")
    amostra = find_after(hlines, "Amostra", pattern=simnao, skip_labels=skip) or find_after(hlines, "Amostra")
    catalogo = find_after(hlines, "Catálogo", pattern=simnao, skip_labels=skip) or find_after(hlines, "Catálogo")

    money_pat = r"R\$\s*[\d\.\,]+"
    valor_teto = find_after(hlines, "Valor Teto", pattern=money_pat, skip_labels=skip) or find_after(hlines, "Valor Teto")
    ticket_medio = find_after(hlines, "Ticket Médio", pattern=money_pat, skip_labels=skip) or find_after(hlines, "Ticket Médio")
    meses_garantia = find_after(hlines, "Meses de garantia", pattern=r"\d+", skip_labels=skip) or find_after(hlines, "Meses de garantia")

    return {
        "orgao_codigo": orgao_codigo or "",
        "orgao_nome": orgao_nome or "",
        "processo": processo or "",
        "pregão": pregao or "",
        "conlicitacao": conlicitacao or "",
        "etapa": etapa or "",
        "oc_bb_uasg": oc_bb_uasg or "",
        "uf": uf or "",
        "quadrante": quadrante or "",
        "data_publicacao": data_publicacao or "",
        "data_hora_disputa": data_hora_disputa or "",
        "data_solicitacao": data_solicitacao or "",
        "data_agendamento": data_agendamento or "",
        "responsavel": responsavel or "",
        "realizado_analise": realizado_analise or "",
        "amostra": amostra or "",
        "catalogo": catalogo or "",
        "valor_teto": valor_teto or "",
        "ticket_medio": ticket_medio or "",
        "meses_garantia": meses_garantia or "",
    }


def parse_items(lines: List[str]) -> pd.DataFrame:
    try:
        start_idx = lines.index("Items do DISDEM")
    except ValueError:
        start_idx = -1

    columns = [
        "item", "lote", "produto", "tipo_produto", "descricao_tipo",
        "produto_referencia", "quantidade", "ampla_concorrencia", "valor_unitario"
    ]

    if start_idx == -1:
        return pd.DataFrame(columns=columns)

    skip_exact = {
        "Item Lote Produto Tipo de Produto",
        "Descrição tipo Produto Referência Quantidade Ampla Concorrência Valor Unitário Ações",
        "Item", "Lote", "Produto", "Tipo de Produto", "Descrição tipo",
        "Produto Referência", "Quantidade", "Ampla Concorrência", "Valor Unitário", "Ações",
    }

    relevant_lines: List[str] = []
    for raw_ln in lines[start_idx + 1:]:
        if raw_ln.startswith("Total de itens"):
            break
        if raw_ln in skip_exact:
            continue
        relevant_lines.append(raw_ln.strip().replace("\xa0", " "))

    items: List[List[str]] = []
    current = None  # type: ignore[var-annotated]

    for ln in relevant_lines:
        if not ln:
            continue

        if re.match(r"^\d+", ln):
            # New item row starts with item/lote numbers.
            parts = ln.split()
            item = parts[0]
            lote = parts[1] if len(parts) > 1 else ""
            current = {
                "item": item,
                "lote": lote,
                "produto": "",
                "tipo_produto": "",
                "descricao_tipo": "",
                "produto_referencia": "",
                "quantidade": "",
                "ampla_concorrencia": "",
                "valor_unitario": "",
            }
            continue

        if ln.startswith("R$"):
            if current:
                valor = ln.replace("R$ ", "").replace("R$", "").strip()
                current["valor_unitario"] = f"R$ {valor}"
                items.append([
                    current["item"],
                    current["lote"],
                    current["produto"],
                    current["tipo_produto"],
                    current["descricao_tipo"],
                    current["produto_referencia"],
                    current["quantidade"],
                    current["ampla_concorrencia"],
                    current["valor_unitario"],
                ])
                current = None
            continue

        if current:
            toks = ln.split()
            if len(toks) < 3:
                continue
            ampla = toks[-1].capitalize()
            quantidade = toks[-2]
            body = toks[:-2]
            if len(body) >= 2:
                tipo = body[-2]
                desc = body[-1]
                produto = " ".join(body[:-2]).strip()
            elif body:
                tipo = body[-1]
                desc = ""
                produto = " ".join(body[:-1]).strip()
            else:
                tipo = ""
                desc = ""
                produto = ""

            current["produto"] = produto
            current["tipo_produto"] = tipo
            current["descricao_tipo"] = desc
            current["quantidade"] = quantidade
            current["ampla_concorrencia"] = ampla

    return pd.DataFrame(items, columns=columns)


def run_single_html(html_path: str) -> pd.DataFrame:
    text = read_html_text(html_path)
    lines = lines_from_text(text)
    header = parse_header(lines)
    df_items = parse_items(lines)

    if df_items.empty:
        # cria 1 linha vazia para preservar cabeçalho no merge
        df_items = pd.DataFrame([{
            "item": "", "lote": "", "produto": "", "tipo_produto": "", "descricao_tipo": "",
            "produto_referencia": "", "quantidade": "", "ampla_concorrencia": "", "valor_unitario": ""
        }])

    desired_columns = [
        "amostra", "catalogo", "item", "lote", "produto",
        "quantidade", "ampla_concorrencia", "valor_unitario"
    ]

    for k in ("amostra", "catalogo"):
        df_items[k] = header.get(k, "")

    for col in desired_columns:
        if col not in df_items.columns:
            df_items[col] = ""

    return df_items[desired_columns]


# ----------------------------
# 4) Batch + MERGE
# ----------------------------
def sanitize_for_windows(name: str) -> str:
    """Substitui apenas caracteres realmente proibidos em Windows. Mantém o restante."""
    return re.sub(r'[\\/:*?"<>|]', "_", name).rstrip(" .")

def main():
    ap = argparse.ArgumentParser(description="Processa TODOS os HTMLs de uma pasta e gera CSVs com o MESMO nome-base do HTML. Opcional: --merge")
    ap.add_argument("--input", required=True, help="Pasta com arquivos .html/.htm")
    ap.add_argument("--out-dir", required=True, help="Pasta de saída dos CSVs")
    ap.add_argument("--merge", action="store_true", help="Se definido, gera também um CSV mesclado com todas as linhas")
    ap.add_argument("--merge-out", default="", help="Caminho do CSV mesclado (default: <out-dir>/merged.csv)")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    html_files = [os.path.join(in_dir, f) for f in os.listdir(in_dir)
                  if f.lower().endswith((".html", ".htm"))]
    html_files.sort()

    if not html_files:
        print("Nenhum .html encontrado em --input", file=sys.stderr)
        sys.exit(2)

    merged_frames = []
    merged_columns = None

    for path in html_files:
        try:
            df = run_single_html(path)

            if merged_columns is None:
                merged_columns = list(df.columns)

            base = os.path.splitext(os.path.basename(path))[0]
            # mantém o nome original (só troca extensão); sanitiza caracteres proibidos no Windows
            base_sanit = sanitize_for_windows(base)
            out_path = os.path.join(out_dir, base_sanit + ".csv")

            df.to_csv(out_path, index=False, encoding="utf-8")
            merged_frames.append(df if list(df.columns) == merged_columns else df[merged_columns])

            print(f"[OK] {os.path.basename(path)} -> {os.path.basename(out_path)}  ({len(df)} linha(s))")
        except Exception as e:
            print(f"[ERRO] {os.path.basename(path)}: {e}", file=sys.stderr)

    if args.merge and merged_frames:
        merged_df = pd.concat(merged_frames, ignore_index=True)
        merge_out = args.merge_out or os.path.join(out_dir, "merged.csv")
        merged_df.to_csv(merge_out, index=False, encoding="utf-8")
        print(f"\n[OK] Merge gerado: {merge_out}  ({len(merged_df)} linha(s))")

    print(f"\nConcluído. Saída em: {out_dir}")


if __name__ == "__main__":
    main()
