"""Unit test suite for EduPro data loading, PII sanitization, cohort assignment, joins, and KPIs."""
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORICAL_COLORS, THEME_COLORS
from src.data_loader import assign_age_cohort, validate_and_process_sheets
from src.charts import apply_edupro_chart_theme, plot_age_category_heatmap
from src.metrics import (
    apply_k_anonymity_matrix,
    compute_category_popularity_index,
    compute_gender_parity_metrics,
    compute_overview_kpis,
    compute_statistical_review_flags,
    filter_analytical_data,
)
from src.ui_components import sanitize_dataframe_for_export


def _relative_luminance(hex_color):
    channels = [int(hex_color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(color_a, color_b):
    luminance_a, luminance_b = sorted(
        (_relative_luminance(color_a), _relative_luminance(color_b)),
        reverse=True,
    )
    return (luminance_a + 0.05) / (luminance_b + 0.05)


def create_sample_raw_sheets():
    """Create a deterministic minimal workbook dictionary for testing."""
    users = pd.DataFrame(
        {
            "UserID": ["U001", "U002", "U003", "U004", "U005"],
            "UserName": ["alice", "bob", "charlie", "dana", "eli"],
            "Age": [16, 22, 30, 40, np.nan],
            "Gender": ["Female", "Male", "Female", "Male", None],
            "Email": ["a@test.com", "b@test.com", "c@test.com", "d@test.com", "e@test.com"],
        }
    )
    teachers = pd.DataFrame(
        {
            "TeacherID": ["TC01", "TC02"],
            "TeacherName": ["Prof X", "Prof Y"],
            "Age": [45, 38],
            "Gender": ["Female", "Male"],
            "Expertise": ["Data Science", "Design"],
            "YearsOfExperience": [10, 5],
            "TeacherRating": [4.5, 4.0],
        }
    )
    courses = pd.DataFrame(
        {
            "CourseID": ["CR01", "CR02"],
            "CourseName": ["Intro Python", "UI Design"],
            "CourseCategory": ["Data Science", "Design"],
            "CourseType": ["Free", "Paid"],
            "CourseLevel": ["Beginner", "Advanced"],
            "CoursePrice": [0.0, 100.0],
            "CourseDuration": [10.0, 20.0],
            "CourseRating": [4.8, 4.2],
        }
    )
    transactions = pd.DataFrame(
        {
            "TransactionID": ["TT01", "TT02", "TT03", "TT04_ORPHAN"],
            "UserID": ["U001", "U002", "U003", "U999_MISSING"],
            "CourseID": ["CR01", "CR01", "CR02", "CR01"],
            "TransactionDate": ["15/01/2025", "20/02/2025", "10/03/2025", "12/03/2025"],
            "Amount": [0.0, 0.0, 100.0, 0.0],
            "PaymentMethod": ["PayPal", "Credit Card", "Bank Transfer", "PayPal"],
            "TeacherID": ["TC01", "TC01", "TC02", "TC01"],
        }
    )
    return {"Users": users, "Teachers": teachers, "Courses": courses, "Transactions": transactions}


def test_mutually_exclusive_age_cohorts():
    ages = pd.Series([15, 17, 18, 25, 26, 35, 36, 44, 45, 65, np.nan, -5, 150])
    cohorts = assign_age_cohort(ages).astype(str).tolist()
    assert cohorts == [
        "Under 18",
        "Under 18",
        "18–25",
        "18–25",
        "26–35",
        "26–35",
        "36–44",
        "36–44",
        "45+",
        "45+",
        "Undisclosed",
        "Undisclosed",
        "Undisclosed",
    ]


def test_pii_stripping_and_join_validation():
    raw_sheets = create_sample_raw_sheets()
    processed, report = validate_and_process_sheets(raw_sheets, source_name="test.xlsx")

    # Verify PII is completely removed
    assert "UserName" not in processed["users"].columns
    assert "Email" not in processed["users"].columns
    assert "TeacherName" not in processed["teachers"].columns

    # Verify orphan transaction is quarantined
    assert report.raw_counts["Transactions"] == 4
    assert report.orphan_user_transactions == 1
    assert report.valid_transactions_count == 3
    assert len(processed["enrollments"]) == 3


def test_kpi_and_cpi_calculations():
    raw_sheets = create_sample_raw_sheets()
    processed, _ = validate_and_process_sheets(raw_sheets)
    enrollments = processed["enrollments"]
    courses = processed["courses"]

    kpis = compute_overview_kpis(enrollments, courses)
    assert kpis["total_enrollments"] == 3
    assert kpis["unique_learners"] == 3
    assert kpis["catalog_courses"] == 2
    assert kpis["free_enrollments"] == 2
    assert kpis["paid_enrollments"] == 1

    # Data Science has 2/3 (66.7%) of enrollments and 1/2 (50%) of courses -> CPI = 1.33
    cpi_df = compute_category_popularity_index(enrollments, courses)
    ds_cpi = cpi_df.loc[cpi_df["CourseCategory"] == "Data Science", "CPI"].iloc[0]
    design_cpi = cpi_df.loc[cpi_df["CourseCategory"] == "Design", "CPI"].iloc[0]
    assert ds_cpi == pytest.approx(1.33, abs=0.01)
    assert design_cpi == pytest.approx(0.67, abs=0.01)


def test_global_filters_handle_categorical_columns():
    processed, _ = validate_and_process_sheets(create_sample_raw_sheets())
    enrollments, catalog = filter_analytical_data(
        enrollments=processed["enrollments"],
        courses=processed["courses"],
        date_range=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-31 23:59:59")),
        age_cohorts=["Under 18", "18–25"],
        genders=["Female", "Male"],
        categories=["Data Science"],
        course_types=["Free"],
        course_levels=["Beginner"],
        teacher_expertise=["Data Science"],
    )

    assert len(enrollments) == 2
    assert set(enrollments["CourseCategory"]) == {"Data Science"}
    assert set(catalog["CourseCategory"]) == {"Data Science"}


def test_k_anonymity_suppresses_small_counts_but_preserves_zeros():
    sample = pd.DataFrame(
        {
            "AgeCohort": ["18–25"] * 4 + ["26–35"] * 5,
            "CourseCategory": ["Design"] * 4 + ["Data Science"] * 5,
            "TransactionID": [f"T{i}" for i in range(9)],
        }
    )

    safe_matrix, suppressed_count = apply_k_anonymity_matrix(
        sample, "AgeCohort", "CourseCategory"
    )

    assert suppressed_count == 1
    assert pd.isna(safe_matrix.loc["18–25", "Design"])
    assert safe_matrix.loc["26–35", "Data Science"] == 5
    assert safe_matrix.loc["18–25", "Data Science"] == 0


def test_chart_theme_and_suppressed_heatmap_cells_render():
    fig = apply_edupro_chart_theme(
        plot_age_category_heatmap(
            pd.DataFrame(
                [[np.nan, 0.0], [5.0, 6.0]],
                index=["18–25", "26–35"],
                columns=["Design", "Data Science"],
            )
        )
    )

    assert fig.layout.title.font.size == 16
    assert list(fig.data[1].text) == ["Suppressed (<5)"]
    assert list(fig.data[2].textfont.color) == [
        THEME_COLORS["text_heading"],
        THEME_COLORS["page"],
        THEME_COLORS["page"],
    ]


def test_plum_and_parchment_palette_meets_contrast_guardrails():
    text_pairs = [
        ("text_heading", "page"),
        ("text_body", "page"),
        ("text_muted", "page"),
        ("text_muted", "surface"),
        ("primary", "page"),
        ("surface", "primary"),
        ("sidebar_text", "sidebar"),
        ("sidebar_muted", "sidebar"),
        ("sidebar_text", "sidebar_raised"),
        ("positive", "surface"),
        ("negative", "surface"),
        ("warning_ink", "warning_tint"),
        ("positive_ink", "positive_tint"),
        ("negative_ink", "negative_tint"),
    ]
    for foreground, background in text_pairs:
        assert _contrast_ratio(
            THEME_COLORS[foreground], THEME_COLORS[background]
        ) >= 4.5, (foreground, background)

    for color in CATEGORICAL_COLORS + [THEME_COLORS["warning"]]:
        assert _contrast_ratio(color, THEME_COLORS["surface"]) >= 3.0, color

    for color_name in ("sidebar_accent", "sidebar_border"):
        assert _contrast_ratio(
            THEME_COLORS[color_name], THEME_COLORS["sidebar"]
        ) >= 3.0, color_name


def test_theme_colors_are_centralized():
    project_root = Path(__file__).resolve().parents[1]
    for relative_path in ("app.py", "src/charts.py", "src/ui_components.py"):
        contents = (project_root / relative_path).read_text(encoding="utf-8")
        assert not re.search(r"(?<!&)#[0-9a-fA-F]{6}\b", contents), relative_path


def test_k_anonymity_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="k must be at least 1"):
        apply_k_anonymity_matrix(pd.DataFrame(), "AgeCohort", "CourseCategory", k=0)


def test_gender_parity_zero_denominator_safety():
    raw_sheets = create_sample_raw_sheets()
    processed, _ = validate_and_process_sheets(raw_sheets)
    enrollments = processed["enrollments"]

    # Design only has 1 Female enrollment and 0 Male enrollments -> Raw_GPR and Adjusted_GPI must be NaN
    with np.errstate(divide="raise", invalid="raise"):
        gpi_df = compute_gender_parity_metrics(enrollments, "CourseCategory")
    design_row = gpi_df[gpi_df["CourseCategory"] == "Design"].iloc[0]
    assert pd.isna(design_row["Raw_GPR"])
    assert pd.isna(design_row["Adjusted_GPI"])
    assert design_row["ParityStatus"] == "Suppressed (n < 5)"


def test_gender_parity_suppresses_small_group_counts():
    enrollments = pd.DataFrame(
        {
            "CourseCategory": ["Small"] * 4,
            "Gender": ["Female", "Female", "Male", "Male"],
            "TransactionID": ["T1", "T2", "T3", "T4"],
        }
    )

    result = compute_gender_parity_metrics(enrollments)
    row = result.iloc[0]

    assert row["ParityStatus"] == "Suppressed (n < 5)"
    assert pd.isna(row["TotalEnrollments"])
    assert pd.isna(row["FemaleSharePct"])
    assert pd.isna(row["Raw_GPR"])
    assert pd.isna(row["Adjusted_GPI"])


def test_export_sanitizer_removes_pii():
    unsafe_df = pd.DataFrame(
        {"UserID": ["U1"], "UserName": ["secret"], "Email": ["s@test.com"], "Enrollments": [5]}
    )
    safe_df = sanitize_dataframe_for_export(unsafe_df)
    assert list(safe_df.columns) == ["UserID", "Enrollments"]


def test_statistical_review_flags_are_aggregate_and_detect_course_amount_deviation():
    enrollments = pd.DataFrame(
        {
            "CourseID": ["C1"] * 8,
            "CourseName": ["Course One"] * 8,
            "CourseCategory": ["Data Science"] * 8,
            "Amount": [100.0] * 7 + [150.0],
            "UserID": [f"U{i}" for i in range(8)],
            "TransactionDate": pd.date_range("2025-01-01", periods=8),
            "TransactionID": [f"T{i}" for i in range(8)],
        }
    )

    results = compute_statistical_review_flags(enrollments)
    flags = results["course_amount_flags"]

    assert len(flags) == 1
    assert flags.iloc[0]["FlaggedTransactions"] == 1
    assert flags.iloc[0]["CourseName"] == "Course One"
    assert not {"UserID", "TransactionID", "CourseID"}.intersection(flags.columns)


def test_statistical_review_ignores_zero_dollar_amounts_and_small_course_samples():
    enrollments = pd.DataFrame(
        {
            "CourseID": ["C1"] * 7 + ["C2"] * 8,
            "CourseName": ["Small Course"] * 7 + ["Free Course"] * 8,
            "CourseCategory": ["Design"] * 15,
            "Amount": [100.0] * 7 + [0.0] * 8,
            "UserID": [f"U{i}" for i in range(15)],
            "TransactionDate": pd.date_range("2025-01-01", periods=15),
        }
    )

    result = compute_statistical_review_flags(enrollments)

    assert result["course_amount_flags"].empty


def test_statistical_review_activity_uses_aggregate_counts_only():
    user_ids = [f"U{i}" for i in range(200)]
    user_ids.extend(["U0"] * 19)
    enrollments = pd.DataFrame(
        {
            "CourseID": ["C1"] * len(user_ids),
            "CourseName": ["Course One"] * len(user_ids),
            "CourseCategory": ["Design"] * len(user_ids),
            "Amount": [0.0] * len(user_ids),
            "UserID": user_ids,
            "TransactionDate": pd.date_range("2025-01-01", periods=len(user_ids)),
        }
    )

    activity = compute_statistical_review_flags(enrollments)["learner_activity"]

    assert activity["learners_above_threshold"] == 1
    assert activity["review_threshold"] == 1
    assert "UserID" not in activity


def test_statistical_review_daily_volume_requires_weekday_baseline():
    dates = pd.date_range("2025-01-01", periods=56, freq="D")
    enrollments = pd.DataFrame(
        {
            "CourseID": ["C1"] * 165,
            "CourseName": ["Course One"] * 165,
            "CourseCategory": ["Design"] * 165,
            "Amount": [0.0] * 165,
            "UserID": [f"U{i}" for i in range(165)],
            "TransactionDate": list(dates) + [dates[10]] * 109,
        }
    )

    daily = compute_statistical_review_flags(enrollments)["daily_volume"]
    flagged = daily.loc[daily["Flagged"]]

    assert len(flagged) == 1
    assert flagged.iloc[0]["Date"] == dates[10]
    assert flagged.iloc[0]["Enrollments"] == 110


def test_csv_export_neutralizes_formula_like_strings_without_changing_numbers():
    unsafe_df = pd.DataFrame(
        {
            "CourseName": ["=HYPERLINK(\"https://example.com\")", "  +SUM(A1:A2)", "@name"],
            "Amount": [-10.0, 0.0, 5.0],
        }
    )

    safe_df = sanitize_dataframe_for_export(unsafe_df)

    assert safe_df["CourseName"].tolist() == [
        "'=HYPERLINK(\"https://example.com\")",
        "'  +SUM(A1:A2)",
        "'@name",
    ]
    assert safe_df["Amount"].tolist() == [-10.0, 0.0, 5.0]
