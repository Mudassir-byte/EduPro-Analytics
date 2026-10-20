"""Robust workbook loading, schema validation, PII sanitization, and relational join verification."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple
import io
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    AGE_COHORT_ORDER,
    COURSE_LEVEL_ORDER,
    PII_COLUMNS,
    REQUIRED_SCHEMA,
)


@dataclass
class DataQualityReport:
    """Structured telemetry report of data ingestion, hygiene, and join integrity."""
    source_name: str
    raw_counts: Dict[str, int] = field(default_factory=dict)
    valid_counts: Dict[str, int] = field(default_factory=dict)
    dropped_pii_columns: Dict[str, list] = field(default_factory=dict)
    missing_values: Dict[str, int] = field(default_factory=dict)
    invalid_age_count: int = 0
    undisclosed_age_count: int = 0
    undisclosed_gender_count: int = 0
    duplicate_transactions_removed: int = 0
    invalid_date_transactions: int = 0
    negative_amount_transactions: int = 0
    orphan_user_transactions: int = 0
    orphan_course_transactions: int = 0
    orphan_teacher_transactions: int = 0
    valid_transactions_count: int = 0
    excluded_transactions_count: int = 0
    date_min: Optional[pd.Timestamp] = None
    date_max: Optional[pd.Timestamp] = None
    unonboarded_users_count: int = 0
    unassigned_courses_count: int = 0


def assign_age_cohort(age_series: pd.Series) -> pd.Series:
    """
    Assign mutually exclusive age cohorts:
    Under 18 (10-17), 18–25, 26–35, 36–44, 45+ (45-100), and Undisclosed.
    """
    numeric_age = pd.to_numeric(age_series, errors="coerce")
    cohorts = pd.Series("Undisclosed", index=age_series.index, dtype="object")

    valid_mask = numeric_age.notna() & (numeric_age >= 10) & (numeric_age <= 100)
    valid_ages = numeric_age[valid_mask]

    conditions = [
        valid_ages < 18,
        (valid_ages >= 18) & (valid_ages <= 25),
        (valid_ages >= 26) & (valid_ages <= 35),
        (valid_ages >= 36) & (valid_ages <= 44),
        valid_ages >= 45,
    ]
    choices = ["Under 18", "18–25", "26–35", "36–44", "45+"]

    cohorts.loc[valid_mask] = np.select(conditions, choices, default="Undisclosed")
    return pd.Categorical(cohorts, categories=AGE_COHORT_ORDER, ordered=True)


def normalize_gender(gender_series: pd.Series) -> pd.Series:
    """Normalize gender strings and categorize missing/blank values as 'Undisclosed'."""
    cleaned = (
        gender_series.astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "null": np.nan})
    )
    mapping = {
        "female": "Female",
        "f": "Female",
        "male": "Male",
        "m": "Male",
        "non-binary": "Non-Binary",
        "nonbinary": "Non-Binary",
        "nb": "Non-Binary",
        "undisclosed": "Undisclosed",
    }
    normalized = cleaned.str.lower().map(mapping).fillna(cleaned.str.title())
    normalized = normalized.fillna("Undisclosed")
    return normalized


def validate_and_process_sheets(
    raw_sheets: Dict[str, pd.DataFrame],
    source_name: str = "EduPro Online Platform.xlsx",
) -> Tuple[Dict[str, pd.DataFrame], DataQualityReport]:
    """
    Validate required sheets/columns, strip PII immediately, normalize types,
    check referential integrity, and build the master enrollment analytical table.
    """
    # 1. Verify required sheets exist
    missing_sheets = set(REQUIRED_SCHEMA.keys()) - set(raw_sheets.keys())
    if missing_sheets:
        raise ValueError(
            f"Malformed workbook: Missing required sheet(s): {', '.join(sorted(missing_sheets))}. "
            f"Found sheets: {', '.join(raw_sheets.keys())}"
        )

    # 2. Verify required columns exist in each sheet
    for sheet_name, req_cols in REQUIRED_SCHEMA.items():
        actual_cols = set(raw_sheets[sheet_name].columns.astype(str).str.strip())
        missing_cols = req_cols - actual_cols
        if missing_cols:
            raise ValueError(
                f"Malformed sheet '{sheet_name}': Missing required column(s): "
                f"{', '.join(sorted(missing_cols))}."
            )

    report = DataQualityReport(source_name=source_name)

    # Copy and strip column whitespace
    users = raw_sheets["Users"].copy()
    teachers = raw_sheets["Teachers"].copy()
    courses = raw_sheets["Courses"].copy()
    transactions = raw_sheets["Transactions"].copy()

    for df in (users, teachers, courses, transactions):
        df.columns = df.columns.astype(str).str.strip()

    report.raw_counts = {
        "Users": len(users),
        "Teachers": len(teachers),
        "Courses": len(courses),
        "Transactions": len(transactions),
    }

    # 3. Privacy Gate: Strip direct PII immediately
    for sheet_label, df_ref in [("Users", users), ("Teachers", teachers)]:
        pii_to_drop = [c for c in PII_COLUMNS.get(sheet_label, []) if c in df_ref.columns]
        df_ref.drop(columns=pii_to_drop, inplace=True, errors="ignore")
        report.dropped_pii_columns[sheet_label] = pii_to_drop

    # 4. Clean & Normalize Users
    users["UserID"] = users["UserID"].astype(str).str.strip()
    users = users.drop_duplicates(subset=["UserID"], keep="first")

    raw_age = pd.to_numeric(users["Age"], errors="coerce")
    invalid_age_mask = raw_age.isna() | (raw_age < 10) | (raw_age > 100)
    report.invalid_age_count = int(invalid_age_mask.sum())
    users["Age"] = raw_age.where(~invalid_age_mask, np.nan)
    users["AgeCohort"] = assign_age_cohort(users["Age"])
    report.undisclosed_age_count = int((users["AgeCohort"] == "Undisclosed").sum())

    users["Gender"] = normalize_gender(users["Gender"])
    report.undisclosed_gender_count = int((users["Gender"] == "Undisclosed").sum())

    # 5. Clean & Normalize Teachers
    teachers["TeacherID"] = teachers["TeacherID"].astype(str).str.strip()
    teachers = teachers.drop_duplicates(subset=["TeacherID"], keep="first")
    teachers["Age"] = pd.to_numeric(teachers["Age"], errors="coerce")
    teachers["Gender"] = normalize_gender(teachers["Gender"])
    teachers["Expertise"] = teachers["Expertise"].astype(str).str.strip().fillna("Unspecified")
    teachers["YearsOfExperience"] = pd.to_numeric(teachers["YearsOfExperience"], errors="coerce")
    teachers["TeacherRating"] = pd.to_numeric(teachers["TeacherRating"], errors="coerce")

    # 6. Clean & Normalize Courses
    courses["CourseID"] = courses["CourseID"].astype(str).str.strip()
    courses = courses.drop_duplicates(subset=["CourseID"], keep="first")
    courses["CourseName"] = courses["CourseName"].astype(str).str.strip()
    courses["CourseCategory"] = courses["CourseCategory"].astype(str).str.strip()
    courses["CourseType"] = courses["CourseType"].astype(str).str.strip().str.title()
    courses["CourseLevel"] = pd.Categorical(
        courses["CourseLevel"].astype(str).str.strip().str.title(),
        categories=COURSE_LEVEL_ORDER,
        ordered=True,
    )
    courses["CoursePrice"] = pd.to_numeric(courses["CoursePrice"], errors="coerce").fillna(0.0)
    courses["CourseDuration"] = pd.to_numeric(courses["CourseDuration"], errors="coerce")
    courses["CourseRating"] = pd.to_numeric(courses["CourseRating"], errors="coerce")

    # 7. Clean & Validate Transactions
    transactions["TransactionID"] = transactions["TransactionID"].astype(str).str.strip()
    transactions["UserID"] = transactions["UserID"].astype(str).str.strip()
    transactions["CourseID"] = transactions["CourseID"].astype(str).str.strip()
    transactions["TeacherID"] = transactions["TeacherID"].astype(str).str.strip()
    transactions["PaymentMethod"] = (
        transactions["PaymentMethod"].astype(str).str.strip().fillna("Unspecified")
    )

    # Deduplicate TransactionID
    before_dedup = len(transactions)
    transactions = transactions.drop_duplicates(subset=["TransactionID"], keep="first")
    report.duplicate_transactions_removed = before_dedup - len(transactions)

    # Parse dates robustly (workbook uses DD/MM/YYYY)
    parsed_dates = pd.to_datetime(transactions["TransactionDate"], dayfirst=True, errors="coerce")
    report.invalid_date_transactions = int(parsed_dates.isna().sum())
    transactions["TransactionDate"] = parsed_dates

    # Validate Amount
    parsed_amount = pd.to_numeric(transactions["Amount"], errors="coerce")
    neg_amount_mask = parsed_amount.isna() | (parsed_amount < 0)
    report.negative_amount_transactions = int(neg_amount_mask.sum())
    transactions["Amount"] = parsed_amount

    # Check Foreign Key Referential Integrity
    valid_user_ids = set(users["UserID"])
    valid_course_ids = set(courses["CourseID"])
    valid_teacher_ids = set(teachers["TeacherID"])

    in_users = transactions["UserID"].isin(valid_user_ids)
    in_courses = transactions["CourseID"].isin(valid_course_ids)
    in_teachers = transactions["TeacherID"].isin(valid_teacher_ids)

    report.orphan_user_transactions = int((~in_users).sum())
    report.orphan_course_transactions = int((~in_courses).sum())
    report.orphan_teacher_transactions = int((~in_teachers).sum())

    valid_tx_mask = (
        in_users
        & in_courses
        & in_teachers
        & transactions["TransactionDate"].notna()
        & (~neg_amount_mask)
    )

    valid_tx = transactions.loc[valid_tx_mask].copy()
    report.valid_transactions_count = len(valid_tx)
    report.excluded_transactions_count = report.raw_counts["Transactions"] - len(valid_tx)

    if not valid_tx.empty:
        report.date_min = valid_tx["TransactionDate"].min()
        report.date_max = valid_tx["TransactionDate"].max()

    # Track missing values across raw tables
    report.missing_values = {
        "Users.Age (Missing/Invalid)": report.invalid_age_count,
        "Users.Gender (Undisclosed)": report.undisclosed_gender_count,
        "Courses.CourseRating (Missing)": int(courses["CourseRating"].isna().sum()),
        "Teachers.TeacherRating (Missing)": int(teachers["TeacherRating"].isna().sum()),
        "Transactions.TransactionDate (Invalid)": report.invalid_date_transactions,
    }

    # Unonboarded users (registered in Users but 0 valid enrollments)
    active_user_ids = set(valid_tx["UserID"])
    report.unonboarded_users_count = len(valid_user_ids - active_user_ids)

    # Unassigned courses (in Courses catalog but 0 valid enrollments)
    active_course_ids = set(valid_tx["CourseID"])
    report.unassigned_courses_count = len(valid_course_ids - active_course_ids)

    # 8. Build Enriched Master Enrollment Table
    enrollments = (
        valid_tx.merge(
            users[["UserID", "Age", "AgeCohort", "Gender"]],
            on="UserID",
            how="inner",
        )
        .merge(
            courses[
                [
                    "CourseID", "CourseName", "CourseCategory", "CourseType",
                    "CourseLevel", "CoursePrice", "CourseDuration", "CourseRating",
                ]
            ],
            on="CourseID",
            how="inner",
        )
        .merge(
            teachers.rename(
                columns={
                    "Age": "TeacherAge",
                    "Gender": "TeacherGender",
                    "Expertise": "TeacherExpertise",
                }
            )[
                [
                    "TeacherID", "TeacherAge", "TeacherGender",
                    "TeacherExpertise", "YearsOfExperience", "TeacherRating",
                ]
            ],
            on="TeacherID",
            how="left",
        )
    )

    enrollments["YearMonth"] = enrollments["TransactionDate"].dt.to_period("M").dt.to_timestamp()
    enrollments["IsZeroDollar"] = enrollments["Amount"] == 0

    report.valid_counts = {
        "Users": len(users),
        "Teachers": len(teachers),
        "Courses": len(courses),
        "Transactions": len(valid_tx),
    }

    processed_data = {
        "users": users,
        "teachers": teachers,
        "courses": courses,
        "transactions": valid_tx,
        "enrollments": enrollments,
    }

    return processed_data, report


@st.cache_data(show_spinner=False)
def load_workbook_from_path(file_path_str: str) -> Tuple[Dict[str, pd.DataFrame], DataQualityReport]:
    """Cached loader for local file path."""
    path = Path(file_path_str)
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found at: {path}")
    raw_sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    return validate_and_process_sheets(raw_sheets, source_name=path.name)


@st.cache_data(show_spinner=False)
def load_workbook_from_bytes(file_bytes: bytes, file_name: str) -> Tuple[Dict[str, pd.DataFrame], DataQualityReport]:
    """Cached loader for uploaded workbook bytes."""
    buffer = io.BytesIO(file_bytes)
    raw_sheets = pd.read_excel(buffer, sheet_name=None, engine="openpyxl")
    return validate_and_process_sheets(raw_sheets, source_name=file_name)
