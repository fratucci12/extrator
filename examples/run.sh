#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python -m edital_extractor.cli --input ./meus_pdfs --out-xlsx golden_set.xlsx --out-csv golden_set.csv
