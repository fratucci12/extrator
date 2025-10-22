import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class PageSlice:
    page_number: int
    text: str
    markdown: str
    score: float
    is_informative: bool
    lote_hint: Optional[str] = None
    tags: Sequence[str] = ()


@lru_cache(maxsize=1)
def _load_docling_converter():
    try:
        from docling.document_converter import DocumentConverter
        from docling.document_converter import DocumentInput
    except Exception:
        return None, None
    try:
        converter = DocumentConverter()
    except Exception:
        return None, None
    return converter, DocumentInput


def markdown_by_page(pdf_path: str) -> List[str]:
    converter, DocumentInput = _load_docling_converter()
    if not converter or not DocumentInput:
        return []
    try:
        input_factory = getattr(DocumentInput, "with_auto_file_opening", None)
        if callable(input_factory):
            doc_input = input_factory(path=pdf_path)
        else:
            doc_input = DocumentInput(path=pdf_path)  # type: ignore[arg-type]
        result = converter.convert(doc_input)
    except Exception:
        return []

    document = getattr(result, "document", None)
    if document is None:
        return []

    page_markdown: List[str] = []
    try:
        pages = getattr(document, "pages", None)
        if pages:
            for page in pages:
                exporter = getattr(page, "export_to_markdown", None)
                if callable(exporter):
                    try:
                        page_markdown.append(exporter())
                    except Exception:
                        page_markdown.append("")
                else:
                    page_markdown.append("")
            return page_markdown
    except Exception:
        pass

    exporter = getattr(document, "export_to_markdown", None)
    if callable(exporter):
        try:
            return [exporter()]
        except Exception:
            return []
    return []


KEYWORD_PATTERNS: Tuple[Tuple[str, float], ...] = (
    (r"\bitem\s*\d{1,3}\b", 1.8),
    (r"\blote\s*\d+\b", 1.5),
    (r"\bquantidad[ea]\b", 1.2),
    (r"\bvalor\s+unit[a\u00E1]rio\b", 1.5),
    (r"\bpre[c\u00E7]o\s+unit[a\u00E1]rio\b", 1.5),
    (r"\bampla\s+concorr[e\u00EA]ncia\b", 0.8),
    (r"\bregistro\s+de\s+pre[c\u00E7]o\b", 0.6),
    (r"\bn[ºo]\s*do\s+item\b", 0.6),
    (r"r\$\s*\d", 1.2),
    (r"\bsaldo\s*(?:de)?\s*item\b", 0.6),
    (r"\bdescri[c\u00E7][a\u00E3]o\s+do\s+item\b", 1.0),
)


def _page_score(text: str, markdown: str) -> Tuple[float, Sequence[str]]:
    content = " ".join(part for part in (text, markdown) if part).lower()
    tags: List[str] = []
    if not content:
        return 0.0, tags
    score = 0.0
    for pattern, weight in KEYWORD_PATTERNS:
        if re.search(pattern, content, flags=re.IGNORECASE):
            score += weight
            if pattern.startswith(r"\blote"):
                tags.append("has_lote")
            elif pattern.startswith(r"\bampla"):
                tags.append("has_ampla_concorrencia")
    # Penalize pages that look like annexes or indexes
    if re.search(r"\banexo\b", content):
        score *= 0.6
        tags.append("possible_attachment")
    if re.search(r"refer[êe]ncia", content) and "termo" in content:
        tags.append("possible_ref_term")
    return score, tags


def build_page_slices(
    pages_text: List[str],
    markdown_pages: List[str],
    active_lote: Dict[int, Optional[str]],
) -> List[PageSlice]:
    slices: List[PageSlice] = []
    for idx, text in enumerate(pages_text):
        page_no = idx + 1
        md = markdown_pages[idx] if idx < len(markdown_pages) else ""
        score, tags = _page_score(text, md)
        is_informative = score >= 2.0
        lote_hint = active_lote.get(page_no)
        if lote_hint and "has_lote" not in tags:
            tags = list(tags) + ["has_lote"]
        slices.append(
            PageSlice(
                page_number=page_no,
                text=text,
                markdown=md,
                score=score,
                is_informative=is_informative,
                lote_hint=lote_hint,
                tags=tuple(tags),
            )
        )

    if not any(s.is_informative for s in slices):
        # When nothing hits the threshold pick the top scoring quartile
        sorted_by_score = sorted(slices, key=lambda s: s.score, reverse=True)
        top_k = max(1, len(sorted_by_score) // 4)
        for candidate in sorted_by_score[:top_k]:
            candidate.is_informative = True  # type: ignore[attr-defined]
    return slices
