"""Utility helpers shared across CLI tools."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

DEFAULT_ENCODINGS: Tuple[str, ...] = ("utf-8", "latin-1", "cp1252")
WINDOWS_FORBIDDEN = r'\\/:*?"<>|'

logger = logging.getLogger("disdem_etl")


def read_text_file(
    path: Path | str,
    encodings: Sequence[str] = DEFAULT_ENCODINGS,
    *,
    errors: str = "ignore",
) -> str:
    """Read a text file trying multiple encodings."""
    last_exc: Exception | None = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors=errors) as fh:
                return fh.read()
        except Exception as exc:  # pragma: no cover - defensive
            last_exc = exc
            continue
    if last_exc:  # pragma: no cover - defensive
        raise last_exc
    return ""


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace sequences to single spaces and trim."""
    return re.sub(r"\s+", " ", text).strip()


def sanitize_windows_filename(name: str) -> str:
    """Replace characters that Windows forbids in filenames."""
    sanitized = re.sub(f"[{WINDOWS_FORBIDDEN}]", "_", name)
    return sanitized.rstrip(" .")


def to_int_br(value: str | int | float | None) -> int | None:
    """Convert Brazilian formatted integer (with thousand separator) to int."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    digits = re.sub(r"[^\d-]", "", str(value))
    if digits == "":
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def to_decimal_br(
    value: str | int | float | None,
    *,
    places: int = 2,
) -> Decimal | None:
    """Convert Brazilian formatted decimal to Decimal."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        quant = Decimal(str(value))
    else:
        cleaned = str(value).strip()
        cleaned = cleaned.replace("R$", "").replace("r$", "")
        cleaned = cleaned.replace(".", "").replace(",", ".")
        cleaned = re.sub(r"[^\d\-.]", "", cleaned)
        if cleaned == "":
            return None
        quant = Decimal(cleaned)
    try:
        return quant.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return None


def money_str_from_decimal(value: Decimal | None) -> str:
    """Format decimal value to Brazilian currency string."""
    if value is None:
        return ""
    q = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    as_str = f"{q:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {as_str}"


def compute_sha256(path: Path | str, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash for a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            sha.update(chunk)
    return sha.hexdigest()


def ensure_directory(path: Path | str) -> Path:
    """Create directory if necessary and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_html_files(directory: Path | str) -> List[Path]:
    """List HTML/HTM files inside directory sorted by name."""
    dir_path = Path(directory)
    return sorted(
        [p for p in dir_path.iterdir() if p.suffix.lower() in {".html", ".htm"}]
    )


def list_pdf_files(directory: Path | str) -> List[Path]:
    """List PDF files in directory sorted by name."""
    dir_path = Path(directory)
    return sorted([p for p in dir_path.iterdir() if p.suffix.lower() == ".pdf"])


def normalize_label(label: str) -> str:
    """Normalize labels by removing punctuation and lowering case."""
    text = label.lower()
    text = text.replace("ã", "a").replace("á", "a").replace("à", "a").replace("â", "a")
    text = text.replace("é", "e").replace("ê", "e")
    text = text.replace("í", "i")
    text = text.replace("ó", "o").replace("ô", "o").replace("õ", "o")
    text = text.replace("ú", "u")
    text = text.replace("ç", "c")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return normalize_whitespace(text)


def chunked(iterable: Sequence, size: int) -> Iterator[Sequence]:
    """Yield fixed-size chunks from a sequence."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def strip_tags(html: str) -> str:
    """Remove HTML tags, scripts, and styles."""
    without_scripts = re.sub(
        r"<(script|style)[^>]*?>.*?</\1>", " ", html, flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(r"<[^>]+>", " ", without_scripts)
    return text


def configure_logging(verbose: bool = False) -> None:
    """Configure default logging for the package."""
    level = logging.DEBUG if verbose else logging.INFO
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    else:  # pragma: no cover - defensive
        logging.getLogger().setLevel(level)


@dataclass(slots=True)
class HintRow:
    """Represent a hint row read from the HTML CSV."""

    item: str
    lote: str
    produto: str
    tipo_produto: str
    descricao_tipo: str
    produto_referencia: str
    quantidade: str
    ampla_concorrencia: str
    valor_unitario: str
    header: dict

    @property
    def combined_hint(self) -> str:
        parts = [
            self.produto,
            self.tipo_produto,
            self.descricao_tipo,
            self.produto_referencia,
        ]
        return normalize_whitespace(" ".join(p for p in parts if p))

