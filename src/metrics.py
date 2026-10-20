"""Pure analytical calculations, KPIs, Category Popularity Index, Gender Parity Index, and k-anonymity."""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.config import (
    AGE_COHORT_ORDER,
    K_ANONYMITY_THRESHOLD,
    REVIEW_LEARNER_PERCENTILE,
    REVIEW_MIN_DAILY_OBSERVATIONS,
    REVIEW_MIN_PAID_PER_COURSE,
    REVIEW_MIN_WEEKDAY_PEERS,
    REVIEW_MODIFIED_Z_THRESHOLD,
)


def compute_statistical_review_flags(
    enrollments: pd.DataFrame,
) -> Dict[str, object]:
    """Return aggregate-only review flags without changing or excluding records.

    The paid-amount and daily-volume baselines use the full validated input frame.
    Learner activity uses a disclosed upper-percentile review threshold. No learner
    or transaction identifiers are included in the returned results.
    """
    course_columns = [
        "CourseName",
        "CourseCategory",
        "PaidTransactions",
        "FlaggedTransactions",
        "TypicalPaidAmount",
        "MinFlaggedAmount",
        "MaxFlaggedAmount",
        "MaxDeviationPct",
    ]
    daily_columns = [
        "Date",
        "Enrollments",
        "WeekdayPeerMedian",
        "ModifiedZScore",
        "Flagged",
    ]
    empty_result: Dict[str, object] = {
        "course_amount_flags": pd.DataFrame(columns=course_columns),
        "learner_activity": {
            "review_threshold": np.nan,
            "learners_above_threshold": 0,
            "active_learners": 0,
        },
        "daily_volume": pd.DataFrame(columns=daily_columns),
    }
    if enrollments.empty:
        return empty_result

    required = {
        "CourseID", "CourseName", "CourseCategory", "Amount",
        "UserID", "TransactionDate",
    }
    missing = required.difference(enrollments.columns)
    if missing:
        raise ValueError(
            "Statistical review requires enrollment columns: "
            + ", ".join(sorted(missing))
        )

    # Compare paid amounts only within a course; zero-dollar enrollments are a
    # valid, distinct pricing class and are not evaluated as amount outliers.
    paid = enrollments.loc[
        pd.to_numeric(enrollments["Amount"], errors="coerce").gt(0),
        ["CourseID", "CourseName", "CourseCategory", "Amount"],
    ].copy()
    paid["Amount"] = pd.to_numeric(paid["Amount"], errors="coerce")
    paid = paid.loc[np.isfinite(paid["Amount"]) & paid["Amount"].gt(0)]

    amount_flags = []
    for _, course_rows in paid.groupby("CourseID", observed=False, sort=False):
        if len(course_rows) < REVIEW_MIN_PAID_PER_COURSE:
            continue

        amounts = course_rows["Amount"]
        median = float(amounts.median())
        absolute_deviation = (amounts - median).abs()
        mad = float(absolute_deviation.median())
        if mad > 0:
            modified_z = 0.6745 * (amounts - median) / mad
            flagged = modified_z.abs() > REVIEW_MODIFIED_Z_THRESHOLD
        else:
            flagged = absolute_deviation > 0

        flagged_amounts = amounts.loc[flagged]
        if flagged_amounts.empty:
            continue
        amount_flags.append(
            {
                "CourseName": course_rows["CourseName"].iloc[0],
                "CourseCategory": course_rows["CourseCategory"].iloc[0],
                "PaidTransactions": int(len(course_rows)),
                "FlaggedTransactions": int(len(flagged_amounts)),
                "TypicalPaidAmount": median,
                "MinFlaggedAmount": float(flagged_amounts.min()),
                "MaxFlaggedAmount": float(flagged_amounts.max()),
                "MaxDeviationPct": float(
                    ((flagged_amounts - median).abs() / median * 100).max()
                ),
            }
        )

    amount_flags_df = pd.DataFrame(amount_flags, columns=course_columns)
    if not amount_flags_df.empty:
        amount_flags_df = amount_flags_df.sort_values(
            ["FlaggedTransactions", "MaxDeviationPct"],
            ascending=[False, False],
        ).reset_index(drop=True)

    learner_counts = enrollments.groupby("UserID", observed=False).size()
    learner_fence = float(learner_counts.quantile(REVIEW_LEARNER_PERCENTILE))
    learner_activity = {
        "review_threshold": learner_fence,
        "learners_above_threshold": int((learner_counts > learner_fence).sum()),
        "active_learners": int(learner_counts.size),
    }

    transaction_dates = pd.to_datetime(
        enrollments["TransactionDate"], errors="coerce"
    ).dropna().dt.normalize()
    daily_volume = pd.DataFrame(columns=daily_columns)
    if not transaction_dates.empty:
        daily_counts = transaction_dates.value_counts().sort_index()
        calendar = pd.date_range(
            daily_counts.index.min(), daily_counts.index.max(), freq="D"
        )
        daily_volume = (
            daily_counts.reindex(calendar, fill_value=0)
            .rename("Enrollments")
            .rename_axis("Date")
            .reset_index()
        )
        daily_volume["WeekdayPeerMedian"] = np.nan
        daily_volume["ModifiedZScore"] = np.nan
        daily_volume["Flagged"] = False

        if len(daily_volume) >= REVIEW_MIN_DAILY_OBSERVATIONS:
            daily_volume["_Weekday"] = daily_volume["Date"].dt.dayofweek
            for _, weekday_rows in daily_volume.groupby("_Weekday", sort=False):
                if len(weekday_rows) < REVIEW_MIN_WEEKDAY_PEERS:
                    continue
                peer_median = float(weekday_rows["Enrollments"].median())
                deviations = (weekday_rows["Enrollments"] - peer_median).abs()
                mad = float(deviations.median())
                daily_volume.loc[
                    weekday_rows.index, "WeekdayPeerMedian"
                ] = peer_median
                if mad > 0:
                    modified_z = (
                        0.6745
                        * (weekday_rows["Enrollments"] - peer_median)
                        / mad
                    )
                    daily_volume.loc[
                        weekday_rows.index, "ModifiedZScore"
                    ] = modified_z
                    flagged = modified_z.abs() > REVIEW_MODIFIED_Z_THRESHOLD
                else:
                    flagged = deviations > 0
                daily_volume.loc[weekday_rows.index, "Flagged"] = flagged
            daily_volume["Flagged"] = daily_volume["Flagged"].astype(bool)
        daily_volume = daily_volume.drop(columns=["_Weekday"], errors="ignore")
        daily_volume = daily_volume[daily_columns]

    return {
        "course_amount_flags": amount_flags_df,
        "learner_activity": learner_activity,
        "daily_volume": daily_volume,
    }


def filter_analytical_data(
    enrollments: pd.DataFrame,
    courses: pd.DataFrame,
    date_range: Tuple[pd.Timestamp, pd.Timestamp],
    age_cohorts: List[str],
    genders: List[str],
    categories: List[str],
    course_types: List[str],
    course_levels: List[str],
    teacher_expertise: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply global sidebar filters consistently to enrollments and the course catalog."""
    start_date, end_date = date_range
    mask = (
        (enrollments["TransactionDate"] >= start_date)
        & (enrollments["TransactionDate"] <= end_date)
        & (enrollments["AgeCohort"].astype(str).isin(age_cohorts))
        & (enrollments["Gender"].isin(genders))
        & (enrollments["CourseCategory"].isin(categories))
        & (enrollments["CourseType"].isin(course_types))
        & (enrollments["CourseLevel"].astype(str).isin(course_levels))
        & (enrollments["TeacherExpertise"].isin(teacher_expertise))
    )
    filtered_enrollments = enrollments.loc[mask].copy()

    catalog_mask = (
        courses["CourseCategory"].isin(categories)
        & courses["CourseType"].isin(course_types)
        & courses["CourseLevel"].astype(str).isin(course_levels)
    )
    filtered_catalog = courses.loc[catalog_mask].copy()

    return filtered_enrollments, filtered_catalog


def compute_overview_kpis(
    filtered_enrollments: pd.DataFrame,
    filtered_catalog: pd.DataFrame,
) -> Dict[str, float]:
    """Calculate top-level KPI metrics from filtered data."""
    total_enrollments = int(len(filtered_enrollments))
    unique_learners = int(filtered_enrollments["UserID"].nunique()) if total_enrollments > 0 else 0
    catalog_count = int(len(filtered_catalog))
    active_courses_enrolled = (
        int(filtered_enrollments["CourseID"].nunique()) if total_enrollments > 0 else 0
    )

    avg_enrollments_per_learner = (
        round(total_enrollments / unique_learners, 2) if unique_learners > 0 else 0.0
    )

    # Enrollment-weighted average course rating and catalog average rating
    if total_enrollments > 0 and filtered_enrollments["CourseRating"].notna().any():
        avg_course_rating = round(float(filtered_enrollments["CourseRating"].mean()), 2)
    elif catalog_count > 0 and filtered_catalog["CourseRating"].notna().any():
        avg_course_rating = round(float(filtered_catalog["CourseRating"].mean()), 2)
    else:
        avg_course_rating = np.nan

    free_enrollments = (
        int((filtered_enrollments["Amount"] == 0).sum()) if total_enrollments > 0 else 0
    )
    free_share_pct = (
        round((free_enrollments / total_enrollments) * 100, 1) if total_enrollments > 0 else 0.0
    )
    total_tx_value = (
        round(float(filtered_enrollments["Amount"].sum()), 2) if total_enrollments > 0 else 0.0
    )

    return {
        "total_enrollments": total_enrollments,
        "unique_learners": unique_learners,
        "catalog_courses": catalog_count,
        "active_courses_enrolled": active_courses_enrolled,
        "avg_enrollments_per_learner": avg_enrollments_per_learner,
        "avg_course_rating": avg_course_rating,
        "free_enrollments": free_enrollments,
        "paid_enrollments": total_enrollments - free_enrollments,
        "free_share_pct": free_share_pct,
        "total_tx_value": total_tx_value,
    }


def compute_category_popularity_index(
    filtered_enrollments: pd.DataFrame,
    filtered_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the Category Popularity Index (CPI):
    CPI_c = (E_c / E_total) / (K_c / K_total)
    Handles empty categories and zero denominators safely without raising errors.
    """
    if filtered_catalog.empty and filtered_enrollments.empty:
        return pd.DataFrame(
            columns=[
                "CourseCategory", "CatalogCourses", "CatalogSharePct",
                "Enrollments", "EnrollmentSharePct", "EnrollmentsPerCourse",
                "CPI", "AvgCourseRating", "DemandSignal",
            ]
        )

    catalog_grp = (
        filtered_catalog.groupby("CourseCategory", observed=False)
        .agg(
            CatalogCourses=("CourseID", "nunique"),
            AvgCourseRating=("CourseRating", "mean"),
        )
        .reset_index()
    )

    if not filtered_enrollments.empty:
        enroll_grp = (
            filtered_enrollments.groupby("CourseCategory", observed=False)
            .agg(
                Enrollments=("TransactionID", "count"),
                UniqueLearners=("UserID", "nunique"),
                PaidEnrollments=("Amount", lambda s: int((s > 0).sum())),
                TransactionValue=("Amount", "sum"),
            )
            .reset_index()
        )
    else:
        enroll_grp = pd.DataFrame(
            columns=[
                "CourseCategory", "Enrollments",
                "UniqueLearners", "PaidEnrollments", "TransactionValue"
            ]
        )

    merged = pd.merge(catalog_grp, enroll_grp, on="CourseCategory", how="outer").fillna(
        {
            "CatalogCourses": 0,
            "Enrollments": 0,
            "UniqueLearners": 0,
            "PaidEnrollments": 0,
            "TransactionValue": 0.0,
        }
    )

    total_catalog = merged["CatalogCourses"].sum()
    total_enroll = merged["Enrollments"].sum()

    merged["CatalogShare"] = (
        merged["CatalogCourses"] / total_catalog if total_catalog > 0 else 0.0
    )
    merged["EnrollmentShare"] = (
        merged["Enrollments"] / total_enroll if total_enroll > 0 else 0.0
    )

    merged["CatalogSharePct"] = (merged["CatalogShare"] * 100).round(2)
    merged["EnrollmentSharePct"] = (merged["EnrollmentShare"] * 100).round(2)

    merged["EnrollmentsPerCourse"] = np.where(
        merged["CatalogCourses"] > 0,
        (merged["Enrollments"] / merged["CatalogCourses"]).round(1),
        np.nan,
    )

    # Safe division for CPI
    merged["CPI"] = np.where(
        (merged["CatalogShare"] > 0) & (total_enroll > 0),
        (merged["EnrollmentShare"] / merged["CatalogShare"]).round(2),
        np.nan,
    )

    def classify_cpi(val: float) -> str:
        if pd.isna(val):
            return "Insufficient Supply/Demand"
        if val >= 1.15:
            return "High Demand (CPI ≥ 1.15)"
        if val <= 0.85:
            return "Under-Enrolled (CPI ≤ 0.85)"
        return "Balanced (0.85 < CPI < 1.15)"

    merged["DemandSignal"] = merged["CPI"].apply(classify_cpi)
    merged["AvgCourseRating"] = merged["AvgCourseRating"].round(2)

    return merged.sort_values(by=["Enrollments", "CPI"], ascending=[False, False]).reset_index(drop=True)


def compute_gender_parity_metrics(
    filtered_enrollments: pd.DataFrame,
    dimension_col: str = "CourseCategory",
) -> pd.DataFrame:
    """
    Calculate Gender Participation Ratio (GPR) and Adjusted Gender Parity Index (GPI_A)
    across a specified dimension (e.g., CourseCategory or CourseLevel).

    Formulas:
    - Raw GPR_c = F_c / M_c (Female enrollments to Male enrollments in group c)
    - Adjusted GPI_{A,c} = (F_c / F_total) / (M_c / M_total)
      where F_total and M_total are total active Female and Male enrollments in the current view.
    Handles zero denominators safely by returning NaN and labeling status transparently.
    """
    if filtered_enrollments.empty:
        return pd.DataFrame()

    pivot = (
        filtered_enrollments.pivot_table(
            index=dimension_col,
            columns="Gender",
            values="TransactionID",
            aggfunc="count",
            fill_value=0,
            observed=False,
        )
        .reset_index()
    )

    for g_col in ["Female", "Male", "Non-Binary", "Undisclosed"]:
        if g_col not in pivot.columns:
            pivot[g_col] = 0

    pivot["TotalEnrollments"] = (
        pivot["Female"] + pivot["Male"] + pivot["Non-Binary"] + pivot["Undisclosed"]
    )
    pivot = pivot[pivot["TotalEnrollments"] > 0].copy()

    f_total = pivot["Female"].sum()
    m_total = pivot["Male"].sum()

    pivot["FemaleSharePct"] = np.where(
        pivot["TotalEnrollments"] > 0,
        (pivot["Female"] / pivot["TotalEnrollments"] * 100).round(1),
        0.0,
    )
    pivot["MaleSharePct"] = np.where(
        pivot["TotalEnrollments"] > 0,
        (pivot["Male"] / pivot["TotalEnrollments"] * 100).round(1),
        0.0,
    )

    female = pivot["Female"].to_numpy(dtype=float)
    male = pivot["Male"].to_numpy(dtype=float)
    raw_gpr = np.full(len(pivot), np.nan, dtype=float)
    np.divide(female, male, out=raw_gpr, where=male > 0)
    pivot["Raw_GPR"] = np.round(raw_gpr, 2)

    # Adjusted GPI: (F_c / F_total) / (M_c / M_total), without eager zero division.
    female_rate = np.full(len(pivot), np.nan, dtype=float)
    male_rate = np.full(len(pivot), np.nan, dtype=float)
    if f_total > 0:
        np.divide(female, f_total, out=female_rate)
    if m_total > 0:
        np.divide(male, m_total, out=male_rate)

    adjusted_gpi = np.full(len(pivot), np.nan, dtype=float)
    np.divide(
        female_rate,
        male_rate,
        out=adjusted_gpi,
        where=(male_rate > 0) & np.isfinite(female_rate),
    )
    pivot["Adjusted_GPI"] = np.round(adjusted_gpi, 2)

    def interpret_gpi(gpi: float, total: int) -> str:
        if total < K_ANONYMITY_THRESHOLD:
            return f"Suppressed (n < {K_ANONYMITY_THRESHOLD})"
        if pd.isna(gpi):
            return "Undefined (Zero Denominator)"
        if 0.97 <= gpi <= 1.03:
            return "Parity Achieved (0.97–1.03)"
        if gpi < 0.90:
            return "Lower Female Relative Share (<0.90)"
        if gpi < 0.97:
            return "Slight Male Skew (0.90–0.96)"
        if gpi <= 1.10:
            return "Slight Female Skew (1.04–1.10)"
        return "Higher Female Relative Share (>1.10)"

    pivot["ParityStatus"] = [
        interpret_gpi(g, t) for g, t in zip(pivot["Adjusted_GPI"], pivot["TotalEnrollments"])
    ]

    small_groups = pivot["TotalEnrollments"] < K_ANONYMITY_THRESHOLD
    suppressed_columns = [
        "Female",
        "Male",
        "Non-Binary",
        "Undisclosed",
        "TotalEnrollments",
        "FemaleSharePct",
        "MaleSharePct",
        "Raw_GPR",
        "Adjusted_GPI",
    ]
    pivot.loc[small_groups, suppressed_columns] = np.nan

    return pivot.sort_values(by="TotalEnrollments", ascending=False).reset_index(drop=True)


def apply_k_anonymity_matrix(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str = "TransactionID",
    k: int = K_ANONYMITY_THRESHOLD,
) -> Tuple[pd.DataFrame, int]:
    """
    Build a 2D cross-tabulation (e.g., AgeCohort x CourseCategory) and suppress
    non-zero cells with fewer than k observations (1 <= count < k) to prevent individual re-identification.
    Returns the suppressed matrix and the count of suppressed cells.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")
    if df.empty:
        return pd.DataFrame(), 0

    raw_matrix = pd.crosstab(
        index=df[row_col],
        columns=df[col_col],
        values=df[value_col],
        aggfunc="count",
        dropna=False,
    ).fillna(0)

    # Ensure standard age cohort row ordering if applicable
    if row_col == "AgeCohort":
        existing_cohorts = [c for c in AGE_COHORT_ORDER if c in raw_matrix.index]
        raw_matrix = raw_matrix.reindex(existing_cohorts, fill_value=0)

    small_cell_mask = (raw_matrix > 0) & (raw_matrix < k)
    suppressed_count = int(small_cell_mask.sum().sum())

    safe_matrix = raw_matrix.astype(float).where(~small_cell_mask, np.nan)
    return safe_matrix, suppressed_count


def generate_dynamic_observations(
    filtered_enrollments: pd.DataFrame,
    filtered_catalog: pd.DataFrame,
    cpi_df: pd.DataFrame,
    gpi_df: pd.DataFrame,
) -> Dict[str, List[str]]:
    """
    Derive dynamic, defensible findings and clearly separated recommendations
    from the currently filtered dataset.
    """
    findings = []
    recommendations = []

    if filtered_enrollments.empty:
        return {
            "findings": ["No enrollment records match the active filter criteria."],
            "recommendations": ["Broaden the filter selection or reset filters to inspect platform trends."],
        }

    total_e = len(filtered_enrollments)
    unique_u = filtered_enrollments["UserID"].nunique()
    avg_epu = total_e / unique_u if unique_u > 0 else 0.0

    # 1. Age Cohort Concentration Finding
    cohort_counts = (
        filtered_enrollments["AgeCohort"]
        .value_counts()
        .reindex(AGE_COHORT_ORDER, fill_value=0)
    )
    top_cohort = cohort_counts.idxmax()
    top_cohort_cnt = int(cohort_counts.max())
    top_cohort_pct = (top_cohort_cnt / total_e) * 100 if total_e > 0 else 0.0
    zero_cohorts = [c for c in ["36–44", "45+"] if cohort_counts.get(c, 0) == 0]

    findings.append(
        f"**Age Cohort Concentration:** The **{top_cohort}** cohort accounts for the largest share of activity "
        f"with **{top_cohort_cnt:,} enrollments ({top_cohort_pct:.1f}%)**. Active learners average "
        f"**{avg_epu:.2f} enrollments per learner**."
    )
    if zero_cohorts and filtered_enrollments["Age"].notna().any():
        findings.append(
            f"**Mature Learner Gap:** There are currently **0 enrollments** recorded in the "
            f"**{', '.join(zero_cohorts)}** age group(s) within the selected view (active ages span "
            f"**{int(filtered_enrollments['Age'].min())}–{int(filtered_enrollments['Age'].max())}** years)."
        )
        recommendations.append(
            "**Expand Multi-Generational Outreach (WHO Age-Friendly Alignment):** Design executive upskilling "
            "and modular lifelong-learning pathways targeted at mid-career (`36–44`) and mature (`45+`) learners "
            "to diversify platform reach beyond under-35 cohorts."
        )

    # 2. Category Supply vs Demand (CPI) Finding
    valid_cpi = cpi_df.dropna(subset=["CPI"])
    if not valid_cpi.empty:
        highest_cpi_row = valid_cpi.loc[valid_cpi["CPI"].idxmax()]
        lowest_cpi_row = valid_cpi.loc[valid_cpi["CPI"].idxmin()]
        findings.append(
            f"**Category Demand vs. Catalog Supply:** **{highest_cpi_row['CourseCategory']}** leads demand "
            f"with **{int(highest_cpi_row['Enrollments']):,} enrollments** (**CPI = {highest_cpi_row['CPI']:.2f}**), "
            f"whereas **{lowest_cpi_row['CourseCategory']}** has the lowest relative uptake "
            f"(**{int(lowest_cpi_row['Enrollments']):,} enrollments**, **CPI = {lowest_cpi_row['CPI']:.2f}**)."
        )
        if highest_cpi_row["CPI"] >= 1.10:
            recommendations.append(
                f"**Rebalance Catalog Capacity:** Prioritize new syllabus development or section capacity in "
                f"**{highest_cpi_row['CourseCategory']}** (CPI {highest_cpi_row['CPI']:.2f}), where enrollment share "
                f"({highest_cpi_row['EnrollmentSharePct']:.1f}%) outpaces catalog share ({highest_cpi_row['CatalogSharePct']:.1f}%)."
            )

    # 3. Course Level & Free vs Paid Finding
    level_counts = filtered_enrollments["CourseLevel"].value_counts()
    if not level_counts.empty:
        top_level = level_counts.idxmax()
        top_level_pct = (level_counts.max() / total_e) * 100
        free_pct = ((filtered_enrollments["Amount"] == 0).sum() / total_e) * 100
        findings.append(
            f"**Instructional Tier & Pricing Mix:** **{top_level}** courses capture **{top_level_pct:.1f}%** "
            f"of enrollments, and **{free_pct:.1f}%** of all filtered enrollments occur in zero-dollar (`Free`) courses."
        )
        recommendations.append(
            "**Scaffold Intermediate-to-Advanced Progression (Unified Mentor Model):** Pair passive video modules "
            "with structured project milestones and mentor feedback touchpoints to support learners transitioning "
            "between Beginner, Intermediate, and Advanced coursework."
        )

    # 4. Gender Parity Finding
    if not gpi_df.empty and "Adjusted_GPI" in gpi_df.columns:
        valid_gpi = gpi_df.dropna(subset=["Adjusted_GPI"])
        if not valid_gpi.empty:
            min_gpi_row = valid_gpi.loc[valid_gpi["Adjusted_GPI"].idxmin()]
            max_gpi_row = valid_gpi.loc[valid_gpi["Adjusted_GPI"].idxmax()]
            findings.append(
                f"**Gender Participation Balance:** Across categories, the Adjusted Gender Parity Index ranges from "
                f"**{min_gpi_row['Adjusted_GPI']:.2f}** in **{min_gpi_row['CourseCategory']}** to "
                f"**{max_gpi_row['Adjusted_GPI']:.2f}** in **{max_gpi_row['CourseCategory']}** "
                f"(where 0.97–1.03 indicates parity)."
            )
            if min_gpi_row["Adjusted_GPI"] < 0.95 or max_gpi_row["Adjusted_GPI"] > 1.05:
                recommendations.append(
                    f"**Inclusive Curricular Outreach (Toronto PFR Equity Benchmark):** Review course positioning "
                    f"and showcase diverse instructor role models in categories outside the 0.97–1.03 parity band "
                    f"(such as **{min_gpi_row['CourseCategory']}** and **{max_gpi_row['CourseCategory']}**)."
                )

    return {"findings": findings, "recommendations": recommendations}
