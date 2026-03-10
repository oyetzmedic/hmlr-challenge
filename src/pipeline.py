"""
HM Land Registry – Data Science Technical Test Pipeline
========================================================
Classifies each page of a planning-decision PDF into a document category and
extracts application numbers and applicant names.

Approach
--------
1. Render each PDF page to a high-resolution image (PyMuPDF).
2. OCR the image (Tesseract via pytesseract).
3. Classify the resulting text with a zero-shot NLP transformer
   (facebook/bart-large-mnli). A deterministic keyword fallback is used when
   the transformer is unavailable (``--no-transformer`` flag).
4. Extract entities with targeted regex patterns:
   - application numbers (slash-delimited planning reference formats),
   - applicant names  (title-led personal names and company suffix patterns).
5. (Optional) Use spaCy NER to enhance name extraction (``--use-ner``).
6. Export per-page results to CSV and JSON.

Usage
-----
    python -m src.pipeline path/to/document.pdf [--output-dir DIR] [--no-transformer]
           [--model MODEL_NAME] [--workers N] [--use-ner] [--dpi INT]

Optional flags
--------------
    --output-dir DIR      Directory for CSV/JSON outputs (default: outputs/)
    --no-transformer      Use keyword-only classification (no internet needed)
    --model NAME          HuggingFace zero-shot model (default: facebook/bart-large-mnli)
    --workers N           Number of parallel OCR workers (default: 1)
    --use-ner             Enable spaCy NER for additional applicant name extraction
    --dpi INT             Rendering DPI for OCR quality (default: 250)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Document category labels
# ---------------------------------------------------------------------------

LABELS = [
    "planning charges register",
    "application for planning permission notice of approval",
    "grant of conditional planning permission letter",
    "notice of approval of details",
    "other planning document",
]

# Friendly names keyed to zero-shot labels
LABEL_TO_CATEGORY: dict[str, str] = {
    "planning charges register": "planning_charges_register",
    "application for planning permission notice of approval": "application_for_planning_permission",
    "grant of conditional planning permission letter": "grant_of_conditional_planning_permission",
    "notice of approval of details": "notice_of_approval_of_details",
    "other planning document": "other",
}

# ---------------------------------------------------------------------------
# Keyword fallback – maps category to trigger keywords
# ---------------------------------------------------------------------------

KEYWORD_RULES: dict[str, list[str]] = {
    "planning_charges_register": [
        "part 3",
        "planning charges",
        "conditions imposed by the following town planning consents",
    ],
    "application_for_planning_permission": [
        "application for planning permission",
        "notice of approval",
        "part i - particulars of application",
        "part ii - particulars of decision",
    ],
    "grant_of_conditional_planning_permission": [
        "grant of conditional planning permission",
        "town and country planning act, 1971",
        "this decision is not a decision under building regulations",
    ],
    "notice_of_approval_of_details": [
        "notice of approval of details",
        "part i – particulars of application",
        "part ii – particulars of decision",
        "approval has been granted",
    ],
}

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Planning application reference formats, e.g. P/00/0759, 02/80/1609, P/98/0964
_APP_NUM_PATTERNS: list[re.Pattern[str]] = [
    # Prefixed with explicit label: "Application Number: P/00/0759"
    re.compile(
        r"Application\s*(?:No\.?|Number)\s*[:\.]?\s*([A-Z]{0,3}/?[\d]{2,4}/[\d]{2,5}(?:/[\d]{1,5})?)",
        re.IGNORECASE,
    ),
    # Bare slash-delimited references: P/00/0759, 02/80/1609, APP/2023/01234
    re.compile(
        r"\b([A-Z]{1,3}/\d{2,4}/\d{3,6}(?:/\d{1,5})?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d{2}/\d{2}/\d{3,6})\b",
    ),
]

# Applicant personal name: "Mr M Dale", "Mrs AM Stephens", "Mr & Mrs J M Doe"
_PERSON_PATTERN = re.compile(
    r"\b((?:Mr\.?|Mrs\.?|Ms\.?|Miss|Dr\.?)(?:\s*&\s*(?:Mr\.?|Mrs\.?|Ms\.?|Miss|Dr\.?))?"
    r"\s+[A-Z][A-Za-z\-']*(?:\s+[A-Z][A-Za-z\-']*){0,3})\b"
)

# Applicant company name: e.g. "My First Company Ltd."
_COMPANY_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z0-9&\-']+(?:\s+[A-Z][A-Za-z0-9&\-']+){0,4}"
    r"\s+(?:Ltd\.?|Limited|PLC|plc|Company|Corp\.?))\b"
)

# Context anchors that signal the applicant field
_APPLICANT_ANCHOR = re.compile(
    r"(?:Applicant|granted\s+to|approval\s+granted\s+to)\s*[:\.]?\s*(.{3,80})",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PageResult:
    """Structured result for a single document page."""

    page_number: int
    page_type: str
    confidence: float
    application_numbers: list[str] = field(default_factory=list)
    applicant_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PDF → image helpers
# ---------------------------------------------------------------------------


def render_pdf_pages(pdf_path: Path, dpi: int = 250) -> list[object]:
    """
    Render each page of *pdf_path* to a PIL Image using PyMuPDF.

    Parameters
    ----------
    pdf_path:
        Path to the source PDF.
    dpi:
        Rendering resolution. Higher values improve OCR on degraded scans
        at the cost of processing time.

    Returns
    -------
    list of PIL.Image
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF and Pillow are required for PDF rendering.\n"
            "Install with: pip install PyMuPDF Pillow"
        ) from exc

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    scale = dpi / 72.0
    images: list[object] = []
    try:
        with fitz.open(str(pdf_path)) as doc:
            for page in doc:
                matrix = fitz.Matrix(scale, scale)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                mode = "RGB" if pix.n < 4 else "RGBA"
                images.append(Image.frombytes(mode, [pix.width, pix.height], pix.samples))
    except Exception as e:
        raise RuntimeError(f"Failed to process PDF {pdf_path}: {e}") from e

    return images


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


def check_tesseract_available() -> None:
    """Raise RuntimeError if Tesseract is not installed or not in PATH."""
    if not shutil.which("tesseract"):
        raise RuntimeError(
            "Tesseract OCR is not installed or not in your PATH.\n"
            "Please install it first: https://github.com/tesseract-ocr/tesseract\n"
            "On Ubuntu/Debian: sudo apt install tesseract-ocr\n"
            "On macOS: brew install tesseract"
        )
    # Optional: check version
    try:
        subprocess.run(["tesseract", "--version"], capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Tesseract is installed but returned an error when called.") from e


def ocr_image(image: object) -> str:
    """
    Run Tesseract OCR on *image* and return the extracted text.

    Uses ``--oem 3 --psm 6`` for full-page OCR with the LSTM engine.
    """
    check_tesseract_available()
    try:
        import pytesseract
    except ImportError as exc:
        raise ImportError(
            "pytesseract is required.\nInstall with: pip install pytesseract"
        ) from exc

    return pytesseract.image_to_string(image, config="--oem 3 --psm 6")


# ---------------------------------------------------------------------------
# Page classification
# ---------------------------------------------------------------------------


def _keyword_classify(text: str) -> tuple[str, float]:
    """Fallback: score each category with a base score for any match."""
    normalised = re.sub(r"\s+", " ", text.lower()).strip()
    scores: dict[str, float] = {}
    for category, keywords in KEYWORD_RULES.items():
        hits = sum(1 for kw in keywords if kw in normalised)
        if hits == 0:
            scores[category] = 0.0
        else:
            # Base score for any match, plus increment for additional hits
            scores[category] = min(1.0, 0.5 + 0.1 * hits)

    best_cat = max(scores, key=lambda k: scores[k])
    best_score = scores[best_cat]
    if best_score == 0.0:
        return "other", 0.0
    return best_cat, round(best_score, 4)


def classify_page(
    text: str,
    zero_shot_pipeline: object | None = None,
) -> tuple[str, float]:
    """
    Classify a single page of OCR text.

    Uses zero-shot NLP when *zero_shot_pipeline* is provided; otherwise falls
    back to deterministic keyword scoring.

    Parameters
    ----------
    text:
        Raw OCR text from the page.
    zero_shot_pipeline:
        A HuggingFace ``pipeline("zero-shot-classification")`` instance, or
        ``None`` to use the keyword fallback.

    Returns
    -------
    (category_name, confidence_score)
    """
    cleaned = " ".join(text.split())[:3000]  # truncate to model max context

    if not cleaned:
        return "other", 0.0

    if zero_shot_pipeline is not None:
        try:
            result = zero_shot_pipeline(cleaned, LABELS, multi_label=False)
            raw_label: str = result["labels"][0]
            score: float = float(result["scores"][0])
            category = LABEL_TO_CATEGORY.get(raw_label, "other")
            return category, round(score, 4)
        except Exception:  # noqa: BLE001
            pass  # fall through to keyword classifier

    return _keyword_classify(text)


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" .,;:\t")


def extract_application_numbers(text: str) -> list[str]:
    """
    Extract unique planning application reference numbers from OCR text.

    Handles formats including:
    - ``P/00/0759``  (prefixed letter + two-digit year + serial)
    - ``02/80/1609`` (two-digit sections)
    - Labelled variants: ``Application No. 02/80/1609``
    """
    candidates: set[str] = set()
    for pattern in _APP_NUM_PATTERNS:
        for match in pattern.findall(text):
            cleaned = re.sub(r"\s+", "", match).upper().strip(".")
            if cleaned:
                candidates.add(cleaned)

    # Remove shorter substrings where a longer, more-specific match exists
    filtered: list[str] = []
    sorted_cands = sorted(candidates, key=len, reverse=True)
    for candidate in sorted_cands:
        if not any(
            other != candidate and other.startswith(candidate.rstrip("/"))
            for other in sorted_cands
        ):
            filtered.append(candidate)

    # Exclude date-like strings: DD/MM/YYYY or similar where the last section
    # is a four-digit year (e.g. "17/07/2000"). Planning references use two-
    # digit year sections and do not have four-digit final segments.
    _DATE_LIKE = re.compile(r"/(?:19|20)\d{2}$")
    filtered = [c for c in filtered if not _DATE_LIKE.search(c)]

    return sorted(filtered)


def extract_applicant_names(text: str, use_ner: bool = False) -> list[str]:
    """
    Extract applicant personal names and company names from OCR text.

    Strategy:
    1. Search anchor phrases (``Applicant:``, ``granted to``) and scan the
       following text for personal/company patterns.
    2. Scan the full page text for person and company patterns as a safety net.
    3. If ``use_ner=True``, additionally run a spaCy NER model and include
       PERSON and ORG entities that aren't already captured.
    """
    candidates: set[str] = set()

    # -- Anchor-guided extraction --
    for anchor_match in _APPLICANT_ANCHOR.finditer(text):
        snippet = anchor_match.group(1)[:150]
        for pattern in (_PERSON_PATTERN, _COMPANY_PATTERN):
            for m in pattern.finditer(snippet):
                name = _normalise(m.group(1))
                if len(name) > 3:
                    candidates.add(name)

    # -- Full-page safety net --
    for pattern in (_PERSON_PATTERN, _COMPANY_PATTERN):
        for m in pattern.finditer(text):
            name = _normalise(m.group(1))
            if len(name) > 3:
                candidates.add(name)

    # -- Optional NER --
    if use_ner:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            # Process first 2000 chars (enough to catch applicant fields)
            doc = nlp(text[:2000])
            for ent in doc.ents:
                if ent.label_ in ("PERSON", "ORG") and len(ent.text) > 3:
                    cleaned = _normalise(ent.text)
                    if cleaned not in candidates:
                        candidates.add(cleaned)
        except ImportError:
            # Silently ignore if spacy not installed
            pass

    # Remove obvious false positives from boilerplate
    _FP_FRAGMENTS = {
        "Town And Country",
        "Town And",
        "North Devon",
        "London Borough",
        "Planning Act",
        "Building Regulations",
        "Secretary Of State",
        "Country Planning",
        "General Development",
    }
    return sorted(
        name
        for name in candidates
        if name.title() not in _FP_FRAGMENTS and len(name.split()) >= 2
    )


# ---------------------------------------------------------------------------
# Per-page orchestration
# ---------------------------------------------------------------------------


def analyse_page(
    page_number: int,
    text: str,
    zero_shot_pipeline: object | None = None,
    use_ner: bool = False,
) -> PageResult:
    """Classify and extract entities from one page of OCR text."""
    page_type, confidence = classify_page(text, zero_shot_pipeline)
    return PageResult(
        page_number=page_number,
        page_type=page_type,
        confidence=confidence,
        application_numbers=extract_application_numbers(text),
        applicant_names=extract_applicant_names(text, use_ner=use_ner),
    )


def analyse_texts(
    texts: list[str],
    zero_shot_pipeline: object | None = None,
    use_ner: bool = False,
) -> list[PageResult]:
    """Analyse a sequence of pre-extracted OCR texts."""
    return [
        analyse_page(i + 1, text, zero_shot_pipeline, use_ner=use_ner)
        for i, text in enumerate(texts)
    ]


def analyse_pdf(
    pdf_path: Path,
    dpi: int = 250,
    use_transformer: bool = True,
    model_name: str = "facebook/bart-large-mnli",
    workers: int = 1,
    use_ner: bool = False,
) -> list[PageResult]:
    """
    End-to-end: render PDF → OCR → classify → extract entities.

    Parameters
    ----------
    pdf_path:
        Path to the input PDF.
    dpi:
        OCR rendering resolution.
    use_transformer:
        When ``True`` (default), loads a zero-shot model for page classification.
    model_name:
        HuggingFace model name for zero-shot classification.
    workers:
        Number of parallel threads for OCR (1 = sequential).
    use_ner:
        Whether to enable spaCy NER for applicant name extraction.
    """
    images = render_pdf_pages(pdf_path, dpi=dpi)

    # OCR in parallel if workers > 1
    texts: list[str] = [None] * len(images)  # type: ignore
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {executor.submit(ocr_image, img): i for i, img in enumerate(images)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            texts[idx] = future.result()

    zsp = None
    if use_transformer:
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore

            zsp = hf_pipeline(
                "zero-shot-classification",
                model=model_name,
            )
        except Exception:  # noqa: BLE001
            print(
                f"[WARNING] Could not load zero-shot model '{model_name}'; "
                "falling back to keyword classifier."
            )

    return analyse_texts(texts, zero_shot_pipeline=zsp, use_ner=use_ner)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def results_to_rows(results: list[PageResult]) -> list[dict]:
    return [
        {
            "page_number": r.page_number,
            "page_type": r.page_type,
            "confidence": r.confidence,
            "application_numbers": "; ".join(r.application_numbers),
            "applicant_names": "; ".join(r.applicant_names),
        }
        for r in results
    ]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(results: list[PageResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(r) for r in results], indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description=(
            "Classify planning-decision PDF pages and extract "
            "application numbers and applicant names."
        ),
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the input PDF.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for output files (default: outputs/).",
    )
    parser.add_argument(
        "--no-transformer",
        action="store_true",
        help="Disable the zero-shot transformer and use keyword fallback only.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/bart-large-mnli",
        help="HuggingFace model name for zero-shot classification (default: facebook/bart-large-mnli). "
             "Use a smaller model like 'typeform/distilbert-base-uncased-mnli' to reduce memory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel OCR workers (default: 1, sequential).",
    )
    parser.add_argument(
        "--use-ner",
        action="store_true",
        help="Enable spaCy NER for additional applicant name extraction (requires spacy and en_core_web_sm).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="DPI for PDF-to-image rendering (default: 250).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    print(f"Processing: {args.pdf_path}")
    results = analyse_pdf(
        args.pdf_path,
        dpi=args.dpi,
        use_transformer=not args.no_transformer,
        model_name=args.model,
        workers=args.workers,
        use_ner=args.use_ner,
    )

    csv_path = args.output_dir / "page_predictions.csv"
    json_path = args.output_dir / "page_predictions.json"

    rows = results_to_rows(results)
    write_csv(rows, csv_path)
    write_json(results, json_path)

    print(f"Saved {len(results)} pages → {csv_path}, {json_path}\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()