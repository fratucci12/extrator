"""LLM-assisted extraction of structured data from PDF tables."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Protocol, Sequence

from . import pdf_plus_hint_to_gt as pdf_gt
from .utils import normalize_whitespace

logger = logging.getLogger("disdem_etl.llm_tables")


DEFAULT_SYSTEM_PROMPT = (
    "Você converte tabelas de editais em JSON estruturado. "
    "Responda apenas com JSON válido, sem markdown nem comentários."
)

DEFAULT_USER_INSTRUCTION = (
    "Retorne um objeto JSON com os campos 'page' (número da página) e 'rows' "
    "(lista de linhas com pares chave/valor). Use os rótulos da tabela como chaves. "
    "Se não houver dados confiáveis, devolva rows como lista vazia."
)


@dataclass(slots=True)
class TableCandidate:
    page: int
    source: str
    text: str
    lines: List[str]
    rows: Optional[List[List[str]]] = None

    def prompt_block(self) -> str:
        if self.rows:
            rendered = "\n".join(" | ".join(row) for row in self.rows if any(cell for cell in row))
            if rendered:
                return rendered
        if self.lines:
            return "\n".join(self.lines)
        return self.text


class ChatClient(Protocol):
    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        """Return the content of the assistant message."""


class OpenAIChatClient:
    """Minimal OpenAI compatible chat client."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[dict[str, Any]] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise RuntimeError(
                "Instale o pacote 'openai>=1.0' para usar a integração LLM."
            ) from exc

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY não definido; forneça --api-key ou variável de ambiente.")

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_output_tokens
        self._response_format = response_format

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=list(messages),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            response_format=self._response_format,
        )
        content = response.choices[0].message.content or ""
        return content.strip()


def detect_tables_with_table_transformer(pdf_path: Path, max_pages: int) -> List[TableCandidate]:
    rows = pdf_gt._extract_with_table_transformer(  # type: ignore[attr-defined]  # noqa: SLF001
        pdf_path,
        max_pages=max_pages,
    )
    candidates: List[TableCandidate] = []
    for row in rows:
        lines = [normalize_whitespace(cell) for cell in row.cells] if row.cells else []
        text = row.text or "\n".join(lines)
        candidates.append(
            TableCandidate(
                page=row.page,
                source="table_transformer",
                text=text,
                lines=[ln for ln in lines if ln],
            )
        )
    return candidates


def detect_tables_with_camelot(pdf_path: Path, max_pages: int) -> List[TableCandidate]:
    if pdf_gt.camelot is None:  # pragma: no cover - runtime guard
        logger.warning("Camelot não está instalado; skipping Camelot detection.")
        return []

    page_spec = "1-end" if max_pages <= 0 else f"1-{max_pages}"
    try:
        tables = pdf_gt.camelot.read_pdf(  # type: ignore[attr-defined]  # noqa: SLF001
            str(pdf_path),
            pages=page_spec,
            flavor="stream",
            strip_text="\n",
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.warning("Falha na leitura com Camelot: %s", exc)
        return []

    candidates: List[TableCandidate] = []
    for table in tables:
        try:
            page_no = int(getattr(table, "page", 1) or 1)
        except Exception:
            page_no = 1
        dataframe = table.df.replace("\n", " ", regex=True)
        raw_rows: List[List[str]] = []
        for idx in range(len(dataframe)):
            cells = [
                normalize_whitespace(str(cell))
                for cell in dataframe.iloc[idx].tolist()
            ]
            if not any(cells):
                continue
            raw_rows.append(cells)
        text = "\n".join(" | ".join(row) for row in raw_rows)
        candidates.append(
            TableCandidate(
                page=page_no,
                source="camelot",
                text=text,
                lines=[" | ".join(row) for row in raw_rows],
                rows=raw_rows,
            )
        )
    return candidates


def detect_tables(
    pdf_path: Path,
    detector: str,
    max_pages: int,
) -> List[TableCandidate]:
    detector = detector.lower()
    candidates: List[TableCandidate] = []

    if detector in {"table-transformer", "auto"}:
        candidates = detect_tables_with_table_transformer(pdf_path, max_pages)
        if candidates:
            return candidates
        if detector == "table-transformer":
            return []

    if detector in {"camelot", "auto"}:
        camelot_candidates = detect_tables_with_camelot(pdf_path, max_pages)
        if camelot_candidates:
            candidates.extend(camelot_candidates)

    return candidates


def build_messages(
    table: TableCandidate,
    *,
    instruction: str,
    system_prompt: str,
) -> List[dict[str, str]]:
    table_block = table.prompt_block()
    user_prompt = textwrap.dedent(
        f"""
        Tabela detectada na página {table.page} (origem: {table.source}).
        {instruction.strip()}
        Tabela bruta:
        \"\"\"
        {table_block}
        \"\"\"
        """
    ).strip()
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)```$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    match = CODE_FENCE_RE.match(text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_json_response(content: str) -> Any:
    cleaned = _strip_code_fence(content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            return json.loads(snippet)
        raise


def process_tables(
    tables: Iterable[TableCandidate],
    *,
    llm_client: ChatClient,
    instruction: str,
    system_prompt: str,
) -> List[dict[str, Any]]:
    results: List[dict[str, Any]] = []
    for table in tables:
        messages = build_messages(
            table,
            instruction=instruction,
            system_prompt=system_prompt,
        )
        response = llm_client.complete(messages)
        try:
            parsed = parse_json_response(response)
        except json.JSONDecodeError as exc:
            logger.error(
                "Falha ao interpretar resposta do LLM para página %s: %s",
                table.page,
                exc,
            )
            raise
        results.append(
            {
                "page": table.page,
                "source": table.source,
                "raw_text": table.text,
                "data": parsed,
            }
        )
    return results


def run_cli(args: argparse.Namespace) -> dict[str, Any]:
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    candidates = detect_tables(
        pdf_path,
        detector=args.detector,
        max_pages=args.max_pages,
    )
    if not candidates:
        logger.warning("Nenhuma tabela detectada em %s.", pdf_path.name)
        return {"pdf": pdf_path.name, "tables": []}

    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT
    instruction = args.instruction or DEFAULT_USER_INSTRUCTION

    if args.dry_run:
        logger.info("Modo dry-run ativado; pulando chamadas ao LLM.")
        tables_payload = [
            {
                "page": table.page,
                "source": table.source,
                "raw_text": table.text,
            }
            for table in candidates
        ]
        return {"pdf": pdf_path.name, "tables": tables_payload}

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = args.api_base or os.environ.get("OPENAI_BASE_URL")
    response_format = {"type": "json_object"} if args.force_json else None

    client = OpenAIChatClient(
        api_key=api_key,
        model=args.model,
        base_url=base_url,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        response_format=response_format,
    )

    tables_payload = process_tables(
        candidates[: args.limit or None],
        llm_client=client,
        instruction=instruction,
        system_prompt=system_prompt,
    )
    return {"pdf": pdf_path.name, "tables": tables_payload}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Detecta tabelas em PDFs e usa um LLM para convertê-las em JSON estruturado.",
    )
    parser.add_argument("pdf", type=str, help="Caminho do PDF a ser processado.")
    parser.add_argument(
        "--detector",
        choices=["auto", "table-transformer", "camelot"],
        default="auto",
        help="Mecanismo para localizar tabelas (default: auto).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Limita o número de páginas analisadas (0 = todas).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Processa apenas as N primeiras tabelas detectadas (0 = todas).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Modelo do provedor OpenAI (ex.: gpt-4o-mini).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Chave de API (override de OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Endpoint alternativo compatível com OpenAI.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperatura usada pelo LLM (default: 0.0).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Limite de tokens de saída.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=DEFAULT_USER_INSTRUCTION,
        help="Instrução adicional enviada ao LLM.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="Prompt de sistema enviado ao LLM.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Arquivo onde salvar o JSON gerado (stdout se omitido).",
    )
    parser.add_argument(
        "--force-json",
        action="store_true",
        help="Força o uso de response_format=json_object (quando suportado).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Executa apenas a detecção de tabelas, sem chamar o LLM.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Nível de log (default: INFO).",
    )

    parsed_args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, parsed_args.log_level.upper(), logging.INFO))

    result = run_cli(parsed_args)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if parsed_args.out:
        parsed_args.out.parent.mkdir(parents=True, exist_ok=True)
        parsed_args.out.write_text(payload, encoding="utf-8")
        print(f"[OK] Resultado salvo em {parsed_args.out}")
    else:
        print(payload)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
