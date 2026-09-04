# HM Land Registry – Data Science Technical Test
[![Tests](https://github.com/oyetzmedic/hmlr-challenge/actions/workflows/tests.yml/badge.svg)](https://github.com/oyetzmedic/hmlr-challenge/actions/workflows/tests.yml)

A reproducible pipeline for classifying historical planning-decision PDF pages and extracting key entities.

## What the solution does

1. **Classifies each PDF page** into one of four document-type categories (or `other`).
2. **Extracts application numbers** (e.g. `P/00/0759`, `02/80/1609`).
3. **Extracts applicant names** (personal and company names).
4. **Exports** per-page results to `CSV` and `JSON`.

## Document categories

| Category key | Description |
|---|---|
| `planning_charges_register` | Part 3 / Planning Charges register page |
| `application_for_planning_permission` | Application form with Part I / Part II sections and "Notice of Approval" heading |
| `grant_of_conditional_planning_permission` | Council grant letter, older Town & Country Planning Act format |
| `notice_of_approval_of_details` | Newer "Notice of Approval of Details" form variant |
| `other` | Fallback for pages that do not match known templates |

## Approach

The pipeline uses a single NLP-first method as required by the brief:

```
PDF → render pages (PyMuPDF) → OCR (Tesseract) → zero-shot NLP classification
     (facebook/bart-large-mnli) → regex entity extraction → CSV + JSON output
```

A deterministic keyword fallback is available via `--no-transformer` for offline use.

See [`docs/analysis_report.md`](docs/analysis_report.md) for EDA observations, limitations, and alternative methods.

## Repository structure

```
.
├── src/
│   └── pipeline.py          # End-to-end pipeline + CLI
├── tests/
│   └── test_pipeline.py     # Pytest unit tests (34 tests)
├── docs/
│   └── analysis_report.md   # One-page analysis report
├── outputs/                  # Generated prediction files (gitignored)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Prerequisites

**Python**: 3.10+

**Tesseract OCR binary** must be installed on the host OS:

```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract
```

## Setup

```bash
git clone <your-repo-url>
cd hmlr-challenge

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Run on the anonymised PDF

```bash
python -m src.pipeline path/to/anonymised.pdf
```

This writes `outputs/page_predictions.csv` and `outputs/page_predictions.json`, then prints the JSON to stdout.

### Options

```
positional arguments:
  pdf_path              Path to the input PDF.

optional arguments:
  --output-dir DIR      Directory for output files (default: outputs/)
  --no-transformer      Use keyword-only classification (no internet/model required)
  --model NAME          HuggingFace zero-shot model (default: facebook/bart-large-mnli)
  --workers N           Number of parallel OCR workers (default: 1)
  --use-ner             Enable spaCy NER for additional name extraction (requires spacy + en_core_web_sm)
  --dpi INT             Rendering DPI for OCR quality (default: 250)
```

**Offline / no-model mode** (keyword classifier only):

```bash
python -m src.pipeline path/to/anonymised.pdf --no-transformer
```

**With spaCy NER** (install the optional dependency first):

```bash
pip install spacy
python -m spacy download en_core_web_sm
python -m src.pipeline path/to/anonymised.pdf --use-ner
```

## Output schema

### CSV (`outputs/page_predictions.csv`)

| Column | Type | Description |
|---|---|---|
| `page_number` | int | 1-indexed page number |
| `page_type` | str | Predicted document category |
| `confidence` | float | Classification confidence (0–1) |
| `application_numbers` | str | Semicolon-separated application references |
| `applicant_names` | str | Semicolon-separated applicant names |

### JSON (`outputs/page_predictions.json`)

Array of objects with the same fields; `application_numbers` and `applicant_names` are JSON arrays.

## Run tests

```bash
pytest -v
```

All 34 tests run without any external dependencies (no transformer model, Tesseract, or PDF files needed) by testing classification and extraction logic directly on text fixtures.

## Example output

```json
[
  {
    "page_number": 1,
    "page_type": "planning_charges_register",
    "confidence": 0.6,
    "application_numbers": ["02/80/1609", "02/81/1237"],
    "applicant_names": ["Mr. & Mrs. J M Doe", "My First Company Ltd"]
  },
  {
    "page_number": 2,
    "page_type": "application_for_planning_permission",
    "confidence": 0.6,
    "application_numbers": ["P/00/0759"],
    "applicant_names": ["Mr M Dale"]
  },
  {
    "page_number": 3,
    "page_type": "grant_of_conditional_planning_permission",
    "confidence": 0.7,
    "application_numbers": [],
    "applicant_names": []
  },
  {
    "page_number": 4,
    "page_type": "notice_of_approval_of_details",
    "confidence": 0.6,
    "application_numbers": ["P/98/0964"],
    "applicant_names": ["Mrs AM Stephens"]
  }
]
```
