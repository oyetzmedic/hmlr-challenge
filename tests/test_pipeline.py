"""
Unit tests for the HMLR planning-document pipeline.

Run with:  pytest -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import (
    PageResult,
    analyse_texts,
    classify_page,
    extract_applicant_names,
    extract_application_numbers,
    results_to_rows,
    write_csv,
    write_json,
)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassifyPage:
    def test_planning_charges_register(self):
        text = """
        PART 3 PLANNING CHARGES
        Conditions imposed by the following Town Planning Consents:
        Reference number  Nature of consent in brief
        """
        label, score = classify_page(text, zero_shot_pipeline=None)
        assert label == "planning_charges_register"
        assert score > 0.0

    def test_application_for_planning_permission(self):
        text = """
        APPLICATION FOR PLANNING PERMISSION
        NOTICE OF APPROVAL
        Part I - Particulars of Application
        Part II - Particulars of Decision
        """
        label, score = classify_page(text, zero_shot_pipeline=None)
        assert label == "application_for_planning_permission"
        assert score > 0.0

    def test_grant_of_conditional_planning_permission(self):
        text = """
        GRANT OF CONDITIONAL PLANNING PERMISSION
        Town and Country Planning Act, 1971
        This decision is not a decision under Building Regulations
        """
        label, score = classify_page(text, zero_shot_pipeline=None)
        assert label == "grant_of_conditional_planning_permission"
        assert score > 0.0

    def test_notice_of_approval_of_details(self):
        text = """
        NOTICE OF APPROVAL OF DETAILS
        Part I – Particulars of Application
        Part II – Particulars of Decision
        APPROVAL HAS BEEN GRANTED
        """
        label, score = classify_page(text, zero_shot_pipeline=None)
        assert label == "notice_of_approval_of_details"
        assert score > 0.0

    def test_empty_text_returns_other(self):
        label, score = classify_page("", zero_shot_pipeline=None)
        assert label == "other"
        assert score == 0.0

    def test_unknown_text_returns_other(self):
        label, score = classify_page(
            "The quick brown fox jumps over the lazy dog",
            zero_shot_pipeline=None,
        )
        assert label == "other"
        assert score == 0.0

    def test_confidence_between_zero_and_one(self):
        text = "PART 3 PLANNING CHARGES and other planning consents"
        _, score = classify_page(text, zero_shot_pipeline=None)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Application number extraction
# ---------------------------------------------------------------------------


class TestExtractApplicationNumbers:
    def test_prefixed_letter_format(self):
        text = "Application Number: P/00/0759"
        numbers = extract_application_numbers(text)
        assert "P/00/0759" in numbers

    def test_numeric_slash_format(self):
        text = "Application No. 02/80/1609"
        numbers = extract_application_numbers(text)
        assert "02/80/1609" in numbers

    def test_multiple_numbers_on_same_page(self):
        text = (
            "Conditional approval granted to Mr. & Mrs. J M Doe "
            "Application No. 02/80/1609 ... Application No. 02/81/1237."
        )
        numbers = extract_application_numbers(text)
        assert "02/80/1609" in numbers
        assert "02/81/1237" in numbers

    def test_p98_format(self):
        text = "Application Number: P/98/0964"
        numbers = extract_application_numbers(text)
        assert "P/98/0964" in numbers

    def test_no_false_positives_from_plain_dates(self):
        text = "Date of Application: 17/07/2000"
        numbers = extract_application_numbers(text)
        # 17/07/2000 should not be captured as a reference number
        assert "17/07/2000" not in numbers

    def test_returns_sorted_unique(self):
        text = "P/00/0759 and P/00/0759 again"
        numbers = extract_application_numbers(text)
        assert numbers.count("P/00/0759") == 1

    def test_empty_text(self):
        assert extract_application_numbers("") == []


# ---------------------------------------------------------------------------
# Applicant name extraction
# ---------------------------------------------------------------------------


class TestExtractApplicantNames:
    def test_mr_title(self):
        text = "Applicant: Mr M Dale\n138 Roman Road"
        names = extract_applicant_names(text)
        assert any("Mr M Dale" in n for n in names)

    def test_mrs_title(self):
        text = "Applicant: Mrs AM Stephens\n55 Cunnery Road"
        names = extract_applicant_names(text)
        assert any("Mrs AM Stephens" in n for n in names)

    def test_mr_and_mrs(self):
        text = "Conditional approval granted to Mr. & Mrs. J M Doe for 7 houses."
        names = extract_applicant_names(text)
        assert any("Doe" in n for n in names)

    def test_company_name(self):
        text = "Conditional approval granted to My First Company Ltd., dated 7.10.81"
        names = extract_applicant_names(text)
        assert any("My First Company Ltd" in n for n in names)

    def test_empty_text(self):
        assert extract_applicant_names("") == []


# ---------------------------------------------------------------------------
# End-to-end text analysis
# ---------------------------------------------------------------------------


class TestAnalyseTexts:
    SAMPLE_TEXTS = [
        # Page 1 – planning charges register
        (
            "PART 3 PLANNING CHARGES "
            "Conditions imposed by the following Town Planning Consents: "
            "Conditional approval granted to Mr. & Mrs. J M Doe "
            "Application No. 02/80/1609 "
            "Conditional approval granted to My First Company Ltd., "
            "Application No. 02/81/1237."
        ),
        # Page 2 – application for planning permission
        (
            "APPLICATION FOR PLANNING PERMISSION NOTICE OF APPROVAL "
            "Town and Country Planning Act 1990 "
            "Applicant: Mr M Dale, 138 Roman Road AB17 4RU "
            "Part I - Particulars of Application "
            "Application Number: P/00/0759 "
            "Part II - Particulars of Decision "
            "PLANNING PERMISSION HAS BEEN GRANTED"
        ),
        # Page 3 – grant of conditional planning permission
        (
            "NORTH DEVON DISTRICT COUNCIL "
            "GRANT OF CONDITIONAL PLANNING PERMISSION "
            "Town and Country Planning Act, 1971 "
            "This decision is not a decision under Building Regulations"
        ),
        # Page 4 – notice of approval of details
        (
            "NOTICE OF APPROVAL OF DETAILS "
            "Town and Country Planning Act 1990 "
            "Applicant: Mrs AM Stephens, 55 Cunnery Road M66 1EU "
            "Part I – Particulars of Application "
            "Application Number: P/98/0964 "
            "Part II – Particulars of Decision "
            "APPROVAL HAS BEEN GRANTED"
        ),
    ]

    def test_correct_number_of_results(self):
        results = analyse_texts(self.SAMPLE_TEXTS, zero_shot_pipeline=None)
        assert len(results) == 4

    def test_page_numbers_sequential(self):
        results = analyse_texts(self.SAMPLE_TEXTS, zero_shot_pipeline=None)
        assert [r.page_number for r in results] == [1, 2, 3, 4]

    def test_page1_classification(self):
        results = analyse_texts(self.SAMPLE_TEXTS, zero_shot_pipeline=None)
        assert results[0].page_type == "planning_charges_register"

    def test_page2_classification(self):
        results = analyse_texts(self.SAMPLE_TEXTS, zero_shot_pipeline=None)
        assert results[1].page_type == "application_for_planning_permission"

    def test_page3_classification(self):
        results = analyse_texts(self.SAMPLE_TEXTS, zero_shot_pipeline=None)
        assert results[2].page_type == "grant_of_conditional_planning_permission"

    def test_page1_application_numbers(self):
        results = analyse_texts(self.SAMPLE_TEXTS, zero_shot_pipeline=None)
        nums = results[0].application_numbers
        assert "02/80/1609" in nums
        assert "02/81/1237" in nums

    def test_page2_application_number(self):
        results = analyse_texts(self.SAMPLE_TEXTS, zero_shot_pipeline=None)
        assert "P/00/0759" in results[1].application_numbers

    def test_page4_application_number(self):
        results = analyse_texts(self.SAMPLE_TEXTS, zero_shot_pipeline=None)
        assert "P/98/0964" in results[3].application_numbers

    def test_page2_applicant(self):
        results = analyse_texts(self.SAMPLE_TEXTS, zero_shot_pipeline=None)
        assert any("Mr M Dale" in n for n in results[1].applicant_names)

    def test_page4_applicant(self):
        results = analyse_texts(self.SAMPLE_TEXTS, zero_shot_pipeline=None)
        assert any("Mrs AM Stephens" in n for n in results[3].applicant_names)

    def test_all_results_are_PageResult(self):
        results = analyse_texts(self.SAMPLE_TEXTS, zero_shot_pipeline=None)
        assert all(isinstance(r, PageResult) for r in results)

    def test_confidence_values_valid(self):
        results = analyse_texts(self.SAMPLE_TEXTS, zero_shot_pipeline=None)
        for r in results:
            assert 0.0 <= r.confidence <= 1.0


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


class TestOutputHelpers:
    def test_results_to_rows_structure(self):
        result = PageResult(
            page_number=1,
            page_type="planning_charges_register",
            confidence=0.67,
            application_numbers=["02/80/1609"],
            applicant_names=["Mr J M Doe"],
        )
        rows = results_to_rows([result])
        assert len(rows) == 1
        row = rows[0]
        assert row["page_number"] == 1
        assert row["page_type"] == "planning_charges_register"
        assert "02/80/1609" in row["application_numbers"]
        assert "Mr J M Doe" in row["applicant_names"]

    def test_write_csv(self, tmp_path):
        result = PageResult(
            page_number=1,
            page_type="other",
            confidence=0.5,
            application_numbers=[],
            applicant_names=[],
        )
        rows = results_to_rows([result])
        out = tmp_path / "test.csv"
        write_csv(rows, out)
        assert out.exists()
        content = out.read_text()
        assert "page_number" in content

    def test_write_json(self, tmp_path):
        result = PageResult(
            page_number=2,
            page_type="application_for_planning_permission",
            confidence=0.8,
            application_numbers=["P/00/0759"],
            applicant_names=["Mr M Dale"],
        )
        out = tmp_path / "test.json"
        write_json([result], out)
        assert out.exists()
        import json

        data = json.loads(out.read_text())
        assert data[0]["page_number"] == 2
        assert "P/00/0759" in data[0]["application_numbers"]
