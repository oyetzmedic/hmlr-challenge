# HM Land Registry – Data Science Technical Test

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
