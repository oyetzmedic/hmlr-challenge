# Analysis Report – HM Land Registry Data Science Technical Test

## Objective

Build a reproducible pipeline that classifies each page of an anonymised planning-decision PDF into a document-type category and extracts application reference numbers and applicant names.

---

## EDA Observations

The four-page sample document exhibits characteristics typical of historical land registry records:

| Page | Document type | Key signals |
|------|--------------|-------------|
| 1 | Planning charges register | "PART 3 PLANNING CHARGES", tabular layout with reference numbers and dates |
| 2 | Application for planning permission / Notice of approval | Structured form: "APPLICATION FOR PLANNING PERMISSION / NOTICE OF APPROVAL", "Part I/II" headings |
| 3 | Grant of conditional planning permission | Official letter format, "GRANT OF CONDITIONAL PLANNING PERMISSION", NDDC letterhead |
| 4 | Notice of approval of details | Near-identical template to page 2 but headed "NOTICE OF APPROVAL OF DETAILS" |

**OCR quality varies significantly**: page 1 is a faint photocopy with scan artefacts and column misalignment; pages 2 and 4 are cleaner typed forms; page 3 is a darker photocopy with marginal text. Application numbers follow consistent slash-delimited formats (`02/80/1609`, `P/00/0759`, `P/98/0964`) that are reliably extractable by regex even under mild OCR noise. Applicant names appear either in an explicit `Applicant:` field or in natural-language phrasing (`granted to Mr. & Mrs. J M Doe`).

---

## Method Selected

A single **NLP-first OCR pipeline** (one approach as required):

1. **PDF rendering** – PyMuPDF renders each page to a 250 DPI image, improving OCR on degraded scans.
2. **OCR** – Tesseract (LSTM engine, `--psm 6`) extracts raw text.
3. **Page classification** – A zero-shot transformer (`facebook/bart-large-mnli`) classifies each page's text against four human-readable category descriptions. A deterministic keyword-scoring fallback is available for offline or resource-constrained environments (`--no-transformer`).
4. **Entity extraction** – Regex patterns extract:
   - Application numbers (slash-delimited planning references, labelled or bare).
   - Applicant names (title-led personal names via `Mr/Mrs/Ms/Dr`; company suffix patterns for `Ltd/Limited/PLC`), guided by anchor phrases (`Applicant:`, `granted to`).
5. **Output** – Per-page results are written to CSV and JSON.

**Rationale**: No labelled training set is available. Zero-shot classification removes the need to hand-code all category rules while remaining interpretable and easy to audit. Regex extraction is highly reliable for the constrained planning-reference format and well-anchored name fields.

---

## Limitations

- **OCR noise** is the primary failure mode: character substitutions (e.g. `O` → `0`, `l` → `1`) can corrupt application numbers; degraded scans can break name anchors.
- **Regex name extraction** may miss names in atypical layouts (e.g. two-line fields, multiple applicants not separated by a comma) or capture OCR artefacts near the applicant field.
- **Zero-shot confidence** degrades on very short or heavily corrupted pages where the model has little signal to distinguish categories.
- **Template diversity**: unseen council templates with different headings would require either retuning keyword rules or additional zero-shot label descriptions.

---

## Alternate Methods / Future Improvements

| Method | When to use |
|--------|-------------|
| OCR pre-processing (deskew, adaptive thresholding, denoising) | Before OCR on degraded scans to improve recall |
| LayoutLM / Donut (layout-aware transformer) | If a labelled dataset (≥ 50 pages) can be assembled; captures positional structure not available from raw OCR text |
| spaCy/GLiNER NER fine-tuned on planning text | For higher-precision name/reference extraction once sufficient annotated examples exist |
| Confidence-gated human-in-the-loop review | For production deployment; low-confidence pages flagged for manual verification |
| Active learning loop | Incrementally improve the classifier as new document templates are encountered |
