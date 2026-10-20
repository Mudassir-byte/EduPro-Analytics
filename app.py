"""
EduPro Online Platform — Learner Demographics & Course Enrollment Analytics Dashboard.
Run locally with: streamlit run app.py
"""
import html
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    AGE_COHORT_DEFINITIONS,
    AGE_COHORT_ORDER,
    COURSE_LEVEL_ORDER,
    COURSE_TYPE_COLORS,
    ENV_WORKBOOK_PATH,
    GENDER_COLORS,
    K_ANONYMITY_THRESHOLD,
    LEVEL_COLORS,
    REVIEW_MIN_DAILY_OBSERVATIONS,
    REVIEW_LEARNER_PERCENTILE,
    REVIEW_MIN_PAID_PER_COURSE,
    REVIEW_MODIFIED_Z_THRESHOLD,
    resolve_default_workbook_path,
)
from src.data_loader import (
    DataQualityReport,
    load_workbook_from_bytes,
    load_workbook_from_path,
)
from src.metrics import (
    apply_k_anonymity_matrix,
    compute_category_popularity_index,
    compute_gender_parity_metrics,
    compute_overview_kpis,
    compute_statistical_review_flags,
    filter_analytical_data,
    generate_dynamic_observations,
)
from src.charts import (
    apply_edupro_chart_theme,
    plot_age_category_heatmap,
    plot_age_cohort_comparison,
    plot_daily_volume_review,
    plot_enrollment_trend,
    plot_gender_parity_lollipop,
    plot_level_distribution_donut,
    plot_supply_vs_demand_cpi,
    plot_top_categories_bar,
)
from src.ui_components import (
    inject_custom_css,
    render_hero_banner,
    render_kpi_card,
    render_safe_csv_download,
)

st.set_page_config(
    page_title="EduPro Analytics | Learner & Enrollment Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()


def render_plotly_chart(fig, config=None) -> None:
    """Render charts with the shared EduPro template instead of Streamlit's default theme."""
    chart_config = {"displaylogo": False, "responsive": True}
    if config:
        chart_config.update(config)
    st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
        config=chart_config,
    )


def load_source_data():
    """Load workbook from configured path or user upload with clear error handling."""
    default_path = resolve_default_workbook_path()

    with st.sidebar.expander("📂 Data Source Configuration", expanded=not default_path.exists()):
        st.caption(
            f"Configure path via `{ENV_WORKBOOK_PATH}` environment variable, local path, or file upload."
        )
        custom_path_str = st.text_input("Workbook Path", value=str(default_path))
        uploaded_file = st.file_uploader("Or upload `EduPro Online Platform.xlsx`", type=["xlsx"])

    try:
        if uploaded_file is not None:
            with st.spinner("Validating uploaded workbook and checking relational integrity..."):
                return load_workbook_from_bytes(uploaded_file.getvalue(), uploaded_file.name)
        else:
            with st.spinner("Loading EduPro workbook and verifying referential integrity..."):
                return load_workbook_from_path(custom_path_str)
    except FileNotFoundError:
        st.error(
            f"### ⚠️ Source Workbook Not Found\n"
            f"Could not locate **`{custom_path_str}`**.\n\n"
            f"**How to resolve:**\n"
            f"1. Place `EduPro Online Platform.xlsx` in the project root directory, **or**\n"
            f"2. Set the `{ENV_WORKBOOK_PATH}` environment variable, **or**\n"
            f"3. Upload `EduPro Online Platform.xlsx` using the **Data Source Configuration** panel in the sidebar.\n\n"
            f"*Note: This application never substitutes fabricated or synthetic fallback data.*"
        )
        st.stop()
    except Exception as exc:
        st.error(
            f"### ❌ Workbook Validation Error\n"
            f"The supplied file could not be validated against the required EduPro schema:\n\n"
            f"`{exc}`"
        )
        st.stop()


def render_sidebar_filters(enrollments: pd.DataFrame, courses: pd.DataFrame):
    """Render global sidebar navigation and interactive filters with Reset affordance."""
    st.sidebar.markdown("## 🎓 EduPro Intelligence")
    st.sidebar.caption("Descriptive Learner & Enrollment Analytics")

    page = st.sidebar.radio(
        "Navigation",
        options=[
            "1. Overview",
            "2. Learner Demographics",
            "3. Course Demand",
            "4. Equity & Insights",
            "5. Data Quality & Methodology",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Global Filters")

    min_date = enrollments["TransactionDate"].min().date()
    max_date = enrollments["TransactionDate"].max().date()

    all_cohorts = [c for c in AGE_COHORT_ORDER if c in enrollments["AgeCohort"].unique() or c in ["Under 18", "18–25", "26–35", "36–44", "45+"]]
    all_genders = sorted(enrollments["Gender"].unique().tolist())
    all_categories = sorted(courses["CourseCategory"].unique().tolist())
    all_types = sorted(courses["CourseType"].unique().tolist())
    all_levels = [lvl for lvl in COURSE_LEVEL_ORDER if lvl in courses["CourseLevel"].unique()]
    all_expertise = sorted(enrollments["TeacherExpertise"].dropna().unique().tolist())

    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith("flt_"):
                del st.session_state[k]
        st.rerun()

    date_selection = st.sidebar.date_input(
        "Transaction Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="flt_dates",
    )
    if isinstance(date_selection, tuple) and len(date_selection) == 2:
        start_ts = pd.Timestamp(date_selection[0])
        end_ts = pd.Timestamp(date_selection[1]) + pd.Timedelta( hours=23, minutes=59, seconds=59)
    else:
        start_ts = pd.Timestamp(min_date)
        end_ts = pd.Timestamp(max_date)

    sel_cohorts = st.sidebar.multiselect(
        "Learner Age Cohort",
        options=all_cohorts,
        default=all_cohorts,
        help="Mutually exclusive age bands: Under 18 (10–17), 18–25, 26–35, 36–44, 45+.",
        key="flt_cohorts",
    )
    sel_genders = st.sidebar.multiselect(
        "Learner Gender",
        options=all_genders,
        default=all_genders,
        key="flt_genders",
    )
    sel_categories = st.sidebar.multiselect(
        "Course Category",
        options=all_categories,
        default=all_categories,
        key="flt_categories",
    )
    sel_types = st.sidebar.multiselect(
        "Course Type (Free / Paid)",
        options=all_types,
        default=all_types,
        key="flt_types",
    )
    sel_levels = st.sidebar.multiselect(
        "Course Level",
        options=all_levels,
        default=all_levels,
        key="flt_levels",
    )
    sel_expertise = st.sidebar.multiselect(
        "Facilitating Teacher Expertise",
        options=all_expertise,
        default=all_expertise,
        key="flt_expertise",
    )

    return page, {
        "date_range": (start_ts, end_ts),
        "age_cohorts": sel_cohorts,
        "genders": sel_genders,
        "categories": sel_categories,
        "course_types": sel_types,
        "course_levels": sel_levels,
        "teacher_expertise": sel_expertise,
    }


def render_data_quality_notice(report: DataQualityReport, filtered_enrollments: pd.DataFrame):
    """Display transparent data-quality and privacy notices when relevant."""
    if report.excluded_transactions_count > 0:
        st.warning(
            f"**Data Quality Notice:** `{report.excluded_transactions_count:,}` invalid or orphaned transaction "
            f"records were excluded during validation (Orphan Users: {report.orphan_user_transactions}, "
            f"Orphan Courses: {report.orphan_course_transactions}, Invalid Dates: {report.invalid_date_transactions})."
        )
    if filtered_enrollments.empty:
        st.info(
            "### 🔎 No Records Match Selected Filters\n"
            "Your current combination of sidebar filters returned **0 enrollment records**. "
            "Note that in the current `EduPro Online Platform.xlsx` snapshot, active learners span ages **15 to 35** "
            "(`Under 18`, `18–25`, and `26–35`). Selecting only `36–44` or `45+` will return an empty view. "
            "Click **🔄 Reset All Filters** in the sidebar to restore the full dataset."
        )
        st.stop()


# ==============================================================================
# PAGE 1: OVERVIEW
# ==============================================================================
def render_overview_page(
    filtered_enrollments: pd.DataFrame,
    filtered_catalog: pd.DataFrame,
    report: DataQualityReport,
):
    kpis = compute_overview_kpis(filtered_enrollments, filtered_catalog)
    cpi_df = compute_category_popularity_index(filtered_enrollments, filtered_catalog)
    gpi_df = compute_gender_parity_metrics(filtered_enrollments, "CourseCategory")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_kpi_card(
            "Valid Enrollments",
            f"{kpis['total_enrollments']:,}",
            f"{kpis['free_share_pct']:.1f}% Free ({kpis['free_enrollments']:,}) | {kpis['paid_enrollments']:,} Paid",
            accent="primary",
        )
    with c2:
        render_kpi_card(
            "Unique Active Learners",
            f"{kpis['unique_learners']:,}",
            f"Of {report.valid_counts['Users']:,} registered users",
            accent="secondary",
        )
    with c3:
        render_kpi_card(
            "Courses in Catalog",
            f"{kpis['catalog_courses']:,}",
            f"{kpis['active_courses_enrolled']:,} courses with active enrollments",
            accent="primary",
        )
    with c4:
        render_kpi_card(
            "Avg Enrollments / Learner",
            f"{kpis['avg_enrollments_per_learner']:.2f}",
            "Active learner engagement density",
            accent="primary",
        )
    with c5:
        rating_str = f"{kpis['avg_course_rating']:.2f} ★" if pd.notna(kpis["avg_course_rating"]) else "N/A"
        render_kpi_card(
            "Avg Course Rating",
            rating_str,
            f"Transaction Value: ${kpis['total_tx_value']:,.2f}",
            accent="secondary",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("ℹ️ Methodology & Metric Definitions (How to Read This Page)", expanded=False):
        st.markdown(
            """
            - **Enrollment Record Definition:** Every valid row in `Transactions` is treated as an enrollment record. Both `$0.00` (Free) and paid transactions are included in enrollment counts. `Amount` is used strictly for labeled transaction-value summaries, never as a proxy for course completion or learning outcomes.
            - **Mutually Exclusive Age Cohorts:** `Under 18` (10–17), `18–25`, `26–35`, `36–44`, and `45+`. This resolves boundary overlap between `36–45` and `45+` in legacy documentation.
            - **Privacy Protection:** Direct identifiers (`UserName`, `Email`, `TeacherName`) are stripped at ingestion.
            """
        )

    col_trend, col_level = st.columns([1.65, 1.0])
    with col_trend:
        freq_choice = st.radio(
            "Trend Aggregation",
            options=["Monthly", "Weekly"],
            horizontal=True,
            label_visibility="collapsed",
        )
        fig_trend = plot_enrollment_trend(filtered_enrollments, freq="W" if freq_choice == "Weekly" else "M")
        render_plotly_chart(fig_trend)

    with col_level:
        fig_level = plot_level_distribution_donut(filtered_enrollments)
        render_plotly_chart(fig_level)

    render_plotly_chart(plot_top_categories_bar(cpi_df))

    # What Stands Out Section
    st.markdown("### 💡 What Stands Out (Filtered View)")
    obs = generate_dynamic_observations(filtered_enrollments, filtered_catalog, cpi_df, gpi_df)
    col_find, col_rec = st.columns(2)
    with col_find:
        findings_html = "".join(
            "<li>"
            + re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html.escape(item))
            + "</li>"
            for item in obs["findings"]
        )
        st.markdown(
            '<div class="insight-box"><h4>&#128202; Calculated Findings (Descriptive)</h4>'
            f"<ul>{findings_html}</ul></div>",
            unsafe_allow_html=True,
        )

    with col_rec:
        recommendations_html = "".join(
            "<li>"
            + re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html.escape(item))
            + "</li>"
            for item in obs["recommendations"]
        )
        st.markdown(
            '<div class="recommendation-box"><h4>&#129517; Program-Planning Recommendations</h4>'
            f"<ul>{recommendations_html}</ul></div>",
            unsafe_allow_html=True,
        )


# ==============================================================================
# PAGE 2: LEARNER DEMOGRAPHICS
# ==============================================================================
def render_demographics_page(
    filtered_enrollments: pd.DataFrame,
    users: pd.DataFrame,
):
    st.markdown("### 👥 Learner Demographics & Subgroup Preferences")

    with st.expander("📌 Age Cohort Definitions & Privacy Safeguards", expanded=False):
        cohort_table = pd.DataFrame(
            [{"Age Cohort": k, "Definition & Life-Stage Context": v} for k, v in AGE_COHORT_DEFINITIONS.items()]
        )
        st.table(cohort_table)
        st.caption(
            f"Privacy Note: Any demographic cross-tabulation cell with fewer than k={K_ANONYMITY_THRESHOLD} "
            f"non-zero observations is automatically suppressed."
        )

    col_age, col_gender = st.columns([1.4, 1.0])
    with col_age:
        render_plotly_chart(plot_age_cohort_comparison(filtered_enrollments, users))

    with col_gender:
        gender_grp = (
            filtered_enrollments.groupby(["AgeCohort", "Gender"], observed=False)
            .agg(Enrollments=("TransactionID", "count"), UniqueLearners=("UserID", "nunique"))
            .reset_index()
        )
        gender_grp = gender_grp[gender_grp["Enrollments"] > 0]
        fig_gen = px.bar(
            gender_grp,
            x="AgeCohort",
            y="Enrollments",
            color="Gender",
            barmode="group",
            text="Enrollments",
            color_discrete_map=GENDER_COLORS,
            custom_data=["UniqueLearners", "Gender"],
            title="Enrollments by Age Cohort & Gender",
        )
        fig_gen.update_traces(
            textposition="outside",
            hovertemplate=(
                "<b>%{x} — %{customdata[1]}</b><br>"
                "Enrollments: <b>%{y:,}</b><br>"
                "Unique Learners: %{customdata[0]:,}<extra></extra>"
            ),
        )
        render_plotly_chart(apply_edupro_chart_theme(fig_gen, height=400))

    # Heatmap: Age Cohort x Course Category
    safe_matrix, suppressed_cells = apply_k_anonymity_matrix(
        filtered_enrollments, "AgeCohort", "CourseCategory"
    )
    if suppressed_cells > 0:
        st.caption(
            f"🔒 Privacy Protection Active: `{suppressed_cells}` small demographic cell(s) with `< {K_ANONYMITY_THRESHOLD}` "
            f"enrollments have been suppressed."
        )
    render_plotly_chart(plot_age_category_heatmap(safe_matrix))

    # Subgroup Drill-Down Table (Aggregate Only)
    st.markdown("#### 📋 Demographic Cohort Summary Table")
    demo_summary = (
        filtered_enrollments.groupby(["AgeCohort", "Gender"], observed=False)
        .agg(
            UniqueLearners=("UserID", "nunique"),
            TotalEnrollments=("TransactionID", "count"),
            FreeEnrollments=("Amount", lambda s: int((s == 0).sum())),
            PaidEnrollments=("Amount", lambda s: int((s > 0).sum())),
            AvgCourseRating=("CourseRating", "mean"),
        )
        .reset_index()
    )
    demo_summary = demo_summary[demo_summary["TotalEnrollments"] > 0].copy()
    demo_summary["AvgEnrollmentsPerLearner"] = (
        demo_summary["TotalEnrollments"] / demo_summary["UniqueLearners"]
    ).round(2)
    demo_summary["AvgCourseRating"] = demo_summary["AvgCourseRating"].round(2)

    # Suppress rows with < K_ANONYMITY_THRESHOLD unique learners
    small_demo_mask = demo_summary["UniqueLearners"] < K_ANONYMITY_THRESHOLD
    if small_demo_mask.any():
        demo_summary.loc[small_demo_mask, ["UniqueLearners", "TotalEnrollments", "FreeEnrollments", "PaidEnrollments"]] = np.nan
        st.caption(f"Rows with fewer than {K_ANONYMITY_THRESHOLD} unique learners are masked for privacy.")

    st.dataframe(demo_summary, use_container_width=True, hide_index=True)
    render_safe_csv_download(demo_summary, "edupro_demographic_summary.csv", key="dl_demo")


# ==============================================================================
# PAGE 3: COURSE DEMAND
# ==============================================================================
def render_course_demand_page(
    filtered_enrollments: pd.DataFrame,
    filtered_catalog: pd.DataFrame,
):
    st.markdown("### 📚 Course Demand & Catalog Supply Analysis")

    with st.expander("📐 Category Popularity Index (CPI) Formula & Methodology", expanded=True):
        st.markdown(
            r"""
            **Category Popularity Index ($\text{CPI}_c$)** measures whether learner enrollment demand in category $c$ outpaces or lags its share of courses in the catalog:
            $$\text{CPI}_c = \frac{\text{Enrollment Share}_c}{\text{Catalog Supply Share}_c} = \frac{E_c / E_{\text{total}}}{K_c / K_{\text{total}}}$$
            - $\text{CPI}_c > 1.00$: Category attracts a **higher share of enrollments** than its share of catalog courses (strong relative demand).
            - $\text{CPI}_c = 1.00$: Demand is **exactly proportional** to catalog supply.
            - $\text{CPI}_c < 1.00$: Category attracts a **lower share of enrollments** relative to catalog supply.
            """
        )

    cpi_df = compute_category_popularity_index(filtered_enrollments, filtered_catalog)
    render_plotly_chart(plot_supply_vs_demand_cpi(cpi_df))

    col_type, col_tier = st.columns(2)
    with col_type:
        type_lvl = (
            filtered_enrollments.groupby(["CourseCategory", "CourseType"], observed=False)
            .agg(Enrollments=("TransactionID", "count"))
            .reset_index()
        )
        fig_type = px.bar(
            type_lvl,
            x="CourseCategory",
            y="Enrollments",
            color="CourseType",
            barmode="stack",
            color_discrete_map=COURSE_TYPE_COLORS,
            title="Enrollments by Category & Course Type (Free vs. Paid)",
        )
        render_plotly_chart(apply_edupro_chart_theme(fig_type, height=390))

    with col_tier:
        cat_lvl = (
            filtered_enrollments.groupby(["CourseCategory", "CourseLevel"], observed=False)
            .agg(Enrollments=("TransactionID", "count"))
            .reset_index()
        )
        fig_lvl = px.bar(
            cat_lvl,
            x="CourseCategory",
            y="Enrollments",
            color="CourseLevel",
            barmode="stack",
            category_orders={"CourseLevel": COURSE_LEVEL_ORDER},
            color_discrete_map=LEVEL_COLORS,
            title="Enrollments by Category & Instructional Level",
        )
        render_plotly_chart(apply_edupro_chart_theme(fig_lvl, height=390))

    # Individual Course Explorer
    st.markdown("#### 🏆 Individual Course Demand & Catalog Explorer")
    course_demand = (
        filtered_catalog.merge(
            filtered_enrollments.groupby("CourseID", observed=False)
            .agg(
                Enrollments=("TransactionID", "count"),
                UniqueLearners=("UserID", "nunique"),
                TransactionValue=("Amount", "sum"),
            )
            .reset_index(),
            on="CourseID",
            how="left",
        )
        .fillna({"Enrollments": 0, "UniqueLearners": 0, "TransactionValue": 0.0})
    )
    course_demand["Enrollments"] = course_demand["Enrollments"].astype(int)
    course_demand["UniqueLearners"] = course_demand["UniqueLearners"].astype(int)
    course_demand["TransactionValue"] = course_demand["TransactionValue"].round(2)
    course_demand = course_demand.sort_values("Enrollments", ascending=False).reset_index(drop=True)

    st.dataframe(
        course_demand[
            [
                "CourseID", "CourseName", "CourseCategory", "CourseType",
                "CourseLevel", "CoursePrice", "CourseDuration", "CourseRating",
                "Enrollments", "UniqueLearners", "TransactionValue",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    render_safe_csv_download(course_demand, "edupro_course_demand.csv", key="dl_courses")

    # Optional Teacher Expertise & Rating Context
    st.markdown("---")
    st.markdown("#### 👨‍🏫 Facilitating Instructor Expertise & Rating Context")
    st.info(
        "**Methodological Caveat:** In `EduPro Online Platform.xlsx`, `TeacherID` is logged per transaction as a "
        "facilitating instructor or mentor assignment. Instructor ratings (`TeacherRating`) and years of experience "
        "provide operational context only and **do not measure or prove student learning outcomes**."
    )
    teacher_grp = (
        filtered_enrollments.groupby(["TeacherID", "TeacherExpertise"], observed=False)
        .agg(
            Enrollments=("TransactionID", "count"),
            DistinctCoursesFacilitated=("CourseID", "nunique"),
            YearsOfExperience=("YearsOfExperience", "first"),
            TeacherRating=("TeacherRating", "first"),
        )
        .reset_index()
    )
    teacher_grp = teacher_grp[teacher_grp["Enrollments"] > 0].sort_values("Enrollments", ascending=False)
    visible_teachers = teacher_grp.loc[
        teacher_grp["Enrollments"] >= K_ANONYMITY_THRESHOLD
    ]
    st.caption(
        f"Instructor summaries with fewer than {K_ANONYMITY_THRESHOLD} enrollments are suppressed."
    )
    st.dataframe(visible_teachers, use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 4: EQUITY & INSIGHTS
# ==============================================================================
def render_equity_page(
    filtered_enrollments: pd.DataFrame,
):
    st.markdown("### ⚖️ Equity, Inclusivity & Representation Insights")

    with st.expander("📐 Gender Parity Index (GPI) Formula & Inclusion Methodology", expanded=True):
        st.markdown(
            r"""
            We report two complementary gender balance metrics across course categories and instructional levels:
            1. **Raw Gender Participation Ratio ($\text{GPR}_c$):**
               $$\text{GPR}_c = \frac{F_c}{M_c}$$
               where $F_c$ and $M_c$ are Female and Male enrollments in segment $c$.
            2. **Adjusted Gender Parity Index ($\text{GPI}_{A,c}$):**
               $$\text{GPI}_{A,c} = \frac{F_c / F_{\text{total}}}{M_c / M_{\text{total}}}$$
               where $F_{\text{total}}$ and $M_{\text{total}}$ are total active Female and Male enrollments in the filtered view.
            - **Included Categories:** `Female` and `Male` form the ratio numerator and denominator; `Non-Binary` and `Undisclosed` counts are displayed transparently alongside in the summary table.
            - **Zero-Denominator Handling:** If $M_c = 0$ or $M_{\text{total}} = 0$, the ratio is safely returned as `NaN` (`Undefined`) rather than dividing by zero.
            - **Target Parity Band:** Aligned with UNESCO equity monitoring standards, $\text{GPI}_{A,c} \in [0.97, 1.03]$ indicates gender parity.
            """
        )

    gpi_cat = compute_gender_parity_metrics(filtered_enrollments, "CourseCategory")

    col_gpi_cat, col_gpi_lvl = st.columns([1.35, 1.0])
    with col_gpi_cat:
        render_plotly_chart(plot_gender_parity_lollipop(gpi_cat, "CourseCategory"))

    with col_gpi_lvl:
        fig_lvl_gen = px.bar(
            filtered_enrollments.groupby(["CourseLevel", "Gender"], observed=False)
            .size()
            .reset_index(name="Enrollments"),
            x="CourseLevel",
            y="Enrollments",
            color="Gender",
            barmode="group",
            text="Enrollments",
            category_orders={"CourseLevel": COURSE_LEVEL_ORDER},
            color_discrete_map=GENDER_COLORS,
            title="Gender Participation Across Course Levels",
        )
        fig_lvl_gen.update_traces(textposition="outside")
        render_plotly_chart(apply_edupro_chart_theme(fig_lvl_gen, height=410))

    st.markdown("#### 📊 Category Gender Parity Telemetry Table")
    st.dataframe(gpi_cat, use_container_width=True, hide_index=True)
    render_safe_csv_download(gpi_cat, "edupro_gender_parity_by_category.csv", key="dl_gpi")

    st.markdown("---")
    st.markdown("#### 🌱 Evidence-Based Observations & Inclusive Design Suggestions")
    st.caption(
        "Note: All observations below are descriptive associations derived from enrollment logs and do not imply causation."
    )
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown(
            """
            <div class="insight-box">
            <h4>🏛️ Public-Sector Equity Alignment (Toronto PFR Framework)</h4>
            <ul>
                <li><b>Zero-Dollar Introductory Access:</b> Free courses account for the majority of platform enrollments, functioning similarly to municipal Welcome Policy fee-subsidy pathways by lowering financial barriers to entry.</li>
                <li><b>Life-Stage Representation Gap:</b> Active learners in the current snapshot are concentrated between ages 15 and 35. Learners aged 36–44 and 45+ are not yet represented in transaction logs.</li>
                <li><b>Accessible Multi-Modal Delivery:</b> Aligning with AODA and WCAG 2.1 AA standards—such as downloadable transcripts, screen-reader compatibility, and keyboard navigation—supports neurodivergent and mature learners.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_e2:
        st.markdown(
            """
            <div class="recommendation-box">
            <h4>🤝 Mentorship & Inclusive Outreach (Unified Mentor Model)</h4>
            <ul>
                <li><b>Closing Category Gender Gaps:</b> In categories where the Adjusted GPI falls below 0.97 or exceeds 1.03, audit course descriptions, prerequisites, and promotional imagery to ensure welcoming entry points for underrepresented groups.</li>
                <li><b>Structured Project Scaffolding:</b> High enrollment in Advanced and Intermediate courses benefits from structured project reviews and mentor office hours so self-directed learners do not stall on complex technical concepts.</li>
                <li><b>Age-Friendly Onboarding:</b> Introduce executive-focused and lifelong-learning tracks with clear prerequisite roadmaps to attract 36–44 and 45+ learners.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# PAGE 5: DATA QUALITY & METHODOLOGY
# ==============================================================================
def render_methodology_page(
    report: DataQualityReport,
    filtered_enrollments: pd.DataFrame,
    cpi_df: pd.DataFrame,
    full_enrollments: pd.DataFrame,
):
    st.markdown("### 🛡️ Data Quality, Validation & Methodology Audit")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            "Source Workbook",
            report.source_name,
            "Verified 4-sheet relational schema",
            accent="primary",
        )
    with c2:
        render_kpi_card(
            "Valid Transactions",
            f"{report.valid_transactions_count:,}",
            f"Excluded / Invalid: {report.excluded_transactions_count:,}",
            accent="secondary",
        )
    with c3:
        render_kpi_card(
            "Join Integrity Rate",
            f"{(report.valid_transactions_count / max(1, report.raw_counts['Transactions'])) * 100:.1f}%",
            "Across Users, Courses, and Teachers",
            accent="primary",
        )
    with c4:
        render_kpi_card(
            "PII Fields Stripped",
            f"{sum(len(v) for v in report.dropped_pii_columns.values())} Columns",
            "UserName, Email, TeacherName removed",
            accent="primary",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("#### 📋 Sheet Ingestion & Join-Integrity Checks")
        audit_rows = [
            {"Check / Metric": "Users Loaded (Unique UserID)", "Count": report.valid_counts["Users"]},
            {"Check / Metric": "Users with ≥1 Valid Enrollment", "Count": report.valid_counts["Users"] - report.unonboarded_users_count},
            {"Check / Metric": "Registered Users with 0 Enrollments", "Count": report.unonboarded_users_count},
            {"Check / Metric": "Courses in Master Catalog", "Count": report.valid_counts["Courses"]},
            {"Check / Metric": "Teachers in Registry", "Count": report.valid_counts["Teachers"]},
            {"Check / Metric": "Raw Transactions Logged", "Count": report.raw_counts["Transactions"]},
            {"Check / Metric": "Duplicate TransactionIDs Removed", "Count": report.duplicate_transactions_removed},
            {"Check / Metric": "Orphan UserID Transactions", "Count": report.orphan_user_transactions},
            {"Check / Metric": "Orphan CourseID Transactions", "Count": report.orphan_course_transactions},
            {"Check / Metric": "Orphan TeacherID Transactions", "Count": report.orphan_teacher_transactions},
            {"Check / Metric": "Valid Transactions Used in Analysis", "Count": report.valid_transactions_count},
        ]
        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)

    with col_s2:
        st.markdown("#### 🔍 Missing-Value & Privacy Telemetry")
        missing_rows = [
            {"Field / Hygiene Check": k, "Affected Records": v}
            for k, v in report.missing_values.items()
        ]
        st.dataframe(pd.DataFrame(missing_rows), use_container_width=True, hide_index=True)

        st.markdown("#### ⚠️ Important Analytical Limitations")
        st.markdown(
            """
            1. **No Completion or Outcome Telemetry:** The workbook does not contain course completion, quiz score, or dropout fields. All counts represent **enrollments** (`Transactions`).
            2. **No Payment Status Field:** Zero-dollar rows correspond to `Free` catalog courses (`CoursePrice = $0.00`) and are included in total enrollments.
            3. **Facilitating Teacher Concentration:** Two instructors (`TC00040` and `TC00042`) are linked to ~60% of transactions across diverse course categories, indicating section facilitation rather than exclusive course authorship.
            """
        )

    st.markdown("---")
    st.markdown("#### 📦 Download Privacy-Safe Aggregate Reports")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        render_safe_csv_download(
            cpi_df,
            filename="edupro_category_popularity_report.csv",
            label="Download Category Demand & CPI Report (CSV)",
            key="dl_method_cpi",
        )
    with col_dl2:
        agg_cohort_cat = (
            filtered_enrollments.groupby(["AgeCohort", "Gender", "CourseCategory", "CourseLevel"], observed=False)
            .agg(
                Enrollments=("TransactionID", "count"),
                UniqueLearners=("UserID", "nunique"),
                TransactionValue=("Amount", "sum"),
            )
            .reset_index()
        )
        agg_cohort_cat = agg_cohort_cat[agg_cohort_cat["Enrollments"] >= K_ANONYMITY_THRESHOLD]
        render_safe_csv_download(
            agg_cohort_cat,
            filename="edupro_filtered_aggregate_enrollments.csv",
            label=f"Download Filtered Multi-Dimensional Aggregate (k≥{K_ANONYMITY_THRESHOLD})",
            key="dl_method_agg",
        )

    st.markdown("---")
    st.markdown("#### 📈 Statistical Review Flags")
    st.caption(
        "These are non-destructive review candidates, not validation failures. "
        "All statistical baselines use the full validated dataset and are independent of sidebar filters."
    )
    review = compute_statistical_review_flags(full_enrollments)
    flag_tab, methods_tab = st.tabs(
        ["🔎 Review candidates", "ℹ️ Methods & limitations"]
    )

    with flag_tab:
        activity = review["learner_activity"]
        learner_count = int(activity["learners_above_threshold"])
        display_learner_count = (
            f"< {K_ANONYMITY_THRESHOLD} (suppressed)"
            if 0 < learner_count < K_ANONYMITY_THRESHOLD
            else f"{learner_count:,}"
        )
        flag_col1, flag_col2, flag_col3 = st.columns(3)
        with flag_col1:
            render_kpi_card(
                "Learners Above Activity Threshold",
                display_learner_count,
                f"Above the {REVIEW_LEARNER_PERCENTILE:.0%} activity percentile "
                f"({activity['review_threshold']:.2f} enrollments)",
                accent="warning",
            )
        with flag_col2:
            course_flag_count = int(
                review["course_amount_flags"]["FlaggedTransactions"].sum()
            )
            render_kpi_card(
                "Course-Peer Amount Flags",
                f"{course_flag_count:,}",
                "Paid transactions; aggregate course results below",
                accent="secondary",
            )
        with flag_col3:
            daily_flags = review["daily_volume"].loc[
                review["daily_volume"]["Flagged"]
            ]
            render_kpi_card(
                "Weekday-Adjusted Volume Flags",
                f"{len(daily_flags):,}",
                "Daily counts compared with same-weekday peers",
                accent="warning",
            )

        render_plotly_chart(
            plot_daily_volume_review(review["daily_volume"]),
            config={"scrollZoom": False},
        )
        if not daily_flags.empty:
            daily_export = daily_flags[
                ["Date", "Enrollments", "WeekdayPeerMedian", "ModifiedZScore"]
            ].copy()
            st.dataframe(
                daily_export,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Date": st.column_config.DateColumn("Date"),
                    "Enrollments": st.column_config.NumberColumn(format="%d"),
                    "WeekdayPeerMedian": st.column_config.NumberColumn(format="%.1f"),
                    "ModifiedZScore": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            render_safe_csv_download(
                daily_export,
                filename="edupro_daily_volume_review.csv",
                label="Download Aggregate Daily Review Flags (CSV)",
                key="dl_review_daily",
            )
        else:
            st.success(
                "No daily-volume review candidates were identified under the current convention."
            )

        amount_flags = review["course_amount_flags"]
        st.markdown("##### Paid Amounts Unusual Relative to Course Peers")
        if amount_flags.empty:
            st.success(
                "No course-peer amount review candidates met the minimum sample and deviation rules."
            )
        else:
            st.dataframe(
                amount_flags,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "TypicalPaidAmount": st.column_config.NumberColumn(format="$%.2f"),
                    "MinFlaggedAmount": st.column_config.NumberColumn(format="$%.2f"),
                    "MaxFlaggedAmount": st.column_config.NumberColumn(format="$%.2f"),
                    "MaxDeviationPct": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )
            render_safe_csv_download(
                amount_flags,
                filename="edupro_course_amount_review.csv",
                label="Download Aggregate Course Amount Flags (CSV)",
                key="dl_review_amount",
            )

    with methods_tab:
        st.markdown(
            f"""
            - **Course-peer paid amounts:** positive amounts are compared within each course using a modified
              z-score (threshold **{REVIEW_MODIFIED_Z_THRESHOLD}**) with at least
              **{REVIEW_MIN_PAID_PER_COURSE}** paid transactions. When peer MAD is zero,
              differing paid amounts are flagged. Zero-dollar enrollments are excluded from this check.
            - **Learner activity:** counts strictly above the **{REVIEW_LEARNER_PERCENTILE:.0%} percentile**
              are included as a review queue, not classified as anomalous. Only an aggregate count is displayed;
              counts from 1 to {K_ANONYMITY_THRESHOLD - 1} are suppressed. Learner identifiers are never shown.
            - **Daily volume:** counts are compared with the same weekday using a modified z-score threshold
              of **{REVIEW_MODIFIED_Z_THRESHOLD}**. At least **{REVIEW_MIN_DAILY_OBSERVATIONS} calendar days**
              and eight observations per weekday are required.
            - Flags are calculated on valid records, are not removed or modified, and are not proof of an error,
              fraud, or cause. No ground-truth outcomes are available; thresholds are conventions and some
              candidates may occur by chance. Dates contain no within-day time-of-day detail.
            """
        )


def main():
    processed_data, report = load_source_data()
    enrollments = processed_data["enrollments"]
    courses = processed_data["courses"]
    users = processed_data["users"]

    page, active_filters = render_sidebar_filters(enrollments, courses)

    filtered_enrollments, filtered_catalog = filter_analytical_data(
        enrollments=enrollments,
        courses=courses,
        date_range=active_filters["date_range"],
        age_cohorts=active_filters["age_cohorts"],
        genders=active_filters["genders"],
        categories=active_filters["categories"],
        course_types=active_filters["course_types"],
        course_levels=active_filters["course_levels"],
        teacher_expertise=active_filters["teacher_expertise"],
    )

    date_str = (
        f"Coverage: {active_filters['date_range'][0].strftime('%b %d, %Y')} – "
        f"{active_filters['date_range'][1].strftime('%b %d, %Y')}"
    )
    rec_str = (
        f"Showing {len(filtered_enrollments):,} of {report.valid_transactions_count:,} Valid Enrollments "
        f"| {filtered_enrollments['UserID'].nunique():,} Active Learners"
    )

    render_hero_banner(
        title="EduPro Learner Demographics & Course Enrollment Intelligence",
        subtitle=(
            "Descriptive analytics for education and program-planning stakeholders — exploring who uses EduPro, "
            "how course preferences differ across demographic segments, and where catalog supply meets learner demand."
        ),
        date_range_str=date_str,
        record_summary_str=rec_str,
    )

    render_data_quality_notice(report, filtered_enrollments)

    if page == "1. Overview":
        render_overview_page(filtered_enrollments, filtered_catalog, report)
    elif page == "2. Learner Demographics":
        render_demographics_page(filtered_enrollments, users)
    elif page == "3. Course Demand":
        render_course_demand_page(filtered_enrollments, filtered_catalog)
    elif page == "4. Equity & Insights":
        render_equity_page(filtered_enrollments)
    elif page == "5. Data Quality & Methodology":
        cpi_df = compute_category_popularity_index(filtered_enrollments, filtered_catalog)
        render_methodology_page(report, filtered_enrollments, cpi_df, enrollments)


if __name__ == "__main__":
    main()
