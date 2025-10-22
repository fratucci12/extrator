"""
Quick validation helpers for generated CSV files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from disdem_etl.html_to_unified import OUTPUT_COLUMNS
from disdem_etl.pdf_plus_hint_to_gt import GT_COLUMNS


def check_html_csv(path: Path) -> list[str]:
    df = pd.read_csv(path, dtype=str)
    required = set(OUTPUT_COLUMNS)
    missing = sorted(required - set(df.columns))
    return missing


def check_gt_csv(path: Path) -> list[str]:
    df = pd.read_csv(path, dtype=str)
    missing = sorted(set(GT_COLUMNS) - set(df.columns))
    dupes = df.duplicated(subset=["document_id", "lote", "item"]).any()
    report = missing[:]
    if dupes:
        report.append("duplicated document_id+lote+item")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Valida CSVs do pipeline DISDEM.")
    parser.add_argument("path", help="Arquivo CSV a validar")
    parser.add_argument(
        "--type",
        choices=("html", "gt"),
        required=True,
        help="Tipo de CSV esperado",
    )
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        parser.error(f"Arquivo não encontrado: {path}")

    if args.type == "html":
        issues = check_html_csv(path)
    else:
        issues = check_gt_csv(path)

    if issues:
        print("Problemas detectados:")
        for issue in issues:
            print(f" - {issue}")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
