"""Interactive, accessible Plotly chart builders styled with the EduPro visual identity."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from src.config import (
    AGE_COHORT_ORDER,
    CATEGORICAL_COLORS,
    COURSE_LEVEL_ORDER,
    DIVERGING_COLORS,
    FONT_SANS,
    LEVEL_COLORS,
    SEQUENTIAL_COLORS,
    THEME_COLORS,
)


def register_edupro_plotly_theme() -> None:
    """Register one shared Plotly template for all dashboard charts."""
    axis_style = dict(
        gridcolor=THEME_COLORS["grid"],
        linecolor=THEME_COLORS["axis"],
        zerolinecolor=THEME_COLORS["axis"],
        tickfont=dict(color=THEME_COLORS["text_muted"]),
        title_font=dict(color=THEME_COLORS["text_muted"]),
    )
    pio.templates["edupro"] = go.layout.Template(
        layout=dict(
            font=dict(family=FONT_SANS, size=13, color=THEME_COLORS["text_body"]),
            title=dict(
                font=dict(size=16, color=THEME_COLORS["text_heading"]),
                x=0,
                xanchor="left",
            ),
            paper_bgcolor=THEME_COLORS["surface"],
            plot_bgcolor=THEME_COLORS["surface"],
            colorway=CATEGORICAL_COLORS,
            colorscale=dict(
                sequential=SEQUENTIAL_COLORS,
                sequentialminus=SEQUENTIAL_COLORS,
                diverging=DIVERGING_COLORS,
            ),
            xaxis=axis_style,
            yaxis=axis_style,
            legend=dict(
                font=dict(color=THEME_COLORS["text_body"]),
                bgcolor="rgba(0,0,0,0)",
            ),
            hoverlabel=dict(
                bgcolor=THEME_COLORS["text_heading"],
                bordercolor=THEME_COLORS["text_heading"],
                font=dict(family=FONT_SANS, color=THEME_COLORS["page"]),
            ),
            margin=dict(l=48, r=16, t=56, b=44),
        )
    )
    pio.templates.default = "plotly_white+edupro"


register_edupro_plotly_theme()


def apply_edupro_chart_theme(fig: go.Figure, height: int = 390) -> go.Figure:
    """Apply consistent typography, accessible contrast, subtle gridlines, and hover styling."""
    fig.update_layout(
        template="edupro",
        font=dict(family=FONT_SANS, size=13, color=THEME_COLORS["text_body"]),
        paper_bgcolor=THEME_COLORS["surface"],
        plot_bgcolor=THEME_COLORS["surface"],
        height=height,
        margin=dict(l=48, r=16, t=56, b=44),
        title=dict(
            font=dict(
                size=16,
                color=THEME_COLORS["text_heading"],
                family=FONT_SANS,
            )
        ),
        hoverlabel=dict(
            bgcolor=THEME_COLORS["text_heading"],
            font_size=12,
            font_family=FONT_SANS,
            font_color=THEME_COLORS["page"],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        transition=dict(duration=350, easing="cubic-in-out"),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=THEME_COLORS["axis"],
        tickfont=dict(size=12, color=THEME_COLORS["text_muted"]),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=THEME_COLORS["grid"],
        linecolor=THEME_COLORS["axis"],
        tickfont=dict(size=12, color=THEME_COLORS["text_muted"]),
    )
    return fig


def plot_enrollment_trend(enrollments: pd.DataFrame, freq: str = "M") -> go.Figure:
    """Plot enrollment volume over time split by Free vs. Paid course type."""
    df = enrollments.copy()
    if freq == "W":
        df["Period"] = df["TransactionDate"].dt.to_period("W").apply(lambda r: r.start_time)
        period_label = "Week Starting"
    else:
        df["Period"] = df["TransactionDate"].dt.to_period("M").dt.to_timestamp()
        period_label = "Month"

    grouped = (
        df.groupby(["Period", "CourseType"], observed=False)
        .agg(Enrollments=("TransactionID", "count"), UniqueLearners=("UserID", "nunique"))
        .reset_index()
    )

    fig = px.area(
        grouped,
        x="Period",
        y="Enrollments",
        color="CourseType",
        markers=True,
        color_discrete_map={
            "Free": THEME_COLORS["primary"],
            "Paid": THEME_COLORS["chart_blue"],
        },
        custom_data=["UniqueLearners", "CourseType"],
        title=f"Enrollment Trend Over Time ({period_label})",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[1]} Courses</b><br>"
            f"{period_label}: %{{x|%b %d, %Y}}<br>"
            "Enrollments: <b>%{y:,}</b><br>"
            "Unique Learners: %{customdata[0]:,}<extra></extra>"
        )
    )
    fig.update_layout(xaxis_title=period_label, yaxis_title="Valid Enrollments", legend_title_text="Course Type")
    return apply_edupro_chart_theme(fig, height=380)


def plot_daily_volume_review(daily_volume: pd.DataFrame) -> go.Figure:
    """Plot daily enrollments and highlight weekday-adjusted review candidates."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily_volume["Date"],
            y=daily_volume["Enrollments"],
            mode="lines",
            name="Daily enrollments",
            line=dict(color=THEME_COLORS["primary"], width=2),
            hovertemplate="%{x|%b %d, %Y}<br>Enrollments: %{y:,}<extra></extra>",
        )
    )
    flagged = daily_volume.loc[daily_volume["Flagged"]]
    if not flagged.empty:
        fig.add_trace(
            go.Scatter(
                x=flagged["Date"],
                y=flagged["Enrollments"],
                mode="markers",
                name="Review candidate",
                marker=dict(
                    color=THEME_COLORS["warning"],
                    size=10,
                    symbol="diamond",
                    line=dict(color=THEME_COLORS["text_heading"], width=1),
                ),
                customdata=flagged[["WeekdayPeerMedian", "ModifiedZScore"]],
                hovertemplate=(
                    "%{x|%b %d, %Y}<br>Enrollments: %{y:,}<br>"
                    "Same-weekday median: %{customdata[0]:,.0f}<br>"
                    "Modified z-score: %{customdata[1]:.2f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Daily Enrollment Volume (Compared with the Same Weekday)",
        xaxis_title="Date",
        yaxis_title="Valid enrollments",
    )
    return apply_edupro_chart_theme(fig, height=380)


def plot_top_categories_bar(cpi_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of enrollments by course category with CPI hover context."""
    df = cpi_df.sort_values("Enrollments", ascending=True).copy()
    fig = px.bar(
        df,
        x="Enrollments",
        y="CourseCategory",
        orientation="h",
        text="Enrollments",
        color="CPI",
        color_continuous_scale=DIVERGING_COLORS,
        color_continuous_midpoint=1.0,
        custom_data=["CatalogCourses", "EnrollmentSharePct", "CPI", "AvgCourseRating"],
        title="Enrollments by Course Category (Colored by Popularity Index)",
    )
    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Enrollments: <b>%{x:,}</b> (%{customdata[1]:.1f}%)<br>"
            "Catalog Supply: %{customdata[0]} courses<br>"
            "Popularity Index (CPI): <b>%{customdata[2]:.2f}</b><br>"
            "Avg Course Rating: %{customdata[3]:.2f} ★<extra></extra>"
        ),
    )
    fig.update_layout(
        xaxis_title="Valid Enrollments",
        yaxis_title="",
        coloraxis_colorbar=dict(title="CPI"),
    )
    return apply_edupro_chart_theme(fig, height=420)


def plot_level_distribution_donut(enrollments: pd.DataFrame) -> go.Figure:
    """Donut chart showing enrollment distribution across Beginner, Intermediate, and Advanced levels."""
    grp = (
        enrollments.groupby("CourseLevel", observed=False)
        .agg(Enrollments=("TransactionID", "count"), UniqueCourses=("CourseID", "nunique"))
        .reindex(COURSE_LEVEL_ORDER)
        .fillna(0)
        .reset_index()
    )
    fig = px.pie(
        grp,
        names="CourseLevel",
        values="Enrollments",
        hole=0.58,
        color="CourseLevel",
        color_discrete_map=LEVEL_COLORS,
        custom_data=["UniqueCourses"],
        title="Enrollment Share by Instructional Level",
    )
    fig.update_traces(
        textinfo="percent+label",
        textposition="inside",
        hovertemplate=(
            "<b>%{label} Level</b><br>"
            "Enrollments: <b>%{value:,}</b> (%{percent})<br>"
            "Active Courses: %{customdata[0]}<extra></extra>"
        ),
    )
    return apply_edupro_chart_theme(fig, height=380)


def plot_age_cohort_comparison(enrollments: pd.DataFrame, users: pd.DataFrame) -> go.Figure:
    """Grouped bar comparing Unique Active Learners vs Total Enrollments across all 5 age cohorts."""
    enroll_cohort = (
        enrollments.groupby("AgeCohort", observed=False)
        .agg(
            Enrollments=("TransactionID", "count"),
            ActiveLearners=("UserID", "nunique"),
        )
        .reindex(AGE_COHORT_ORDER)
        .fillna(0)
        .reset_index()
    )
    reg_cohort = (
        users.groupby("AgeCohort", observed=False)["UserID"]
        .nunique()
        .reindex(AGE_COHORT_ORDER)
        .fillna(0)
        .values
    )
    enroll_cohort["RegisteredUsers"] = reg_cohort
    enroll_cohort["EnrollmentsPerLearner"] = np.where(
        enroll_cohort["ActiveLearners"] > 0,
        (enroll_cohort["Enrollments"] / enroll_cohort["ActiveLearners"]).round(2),
        0.0,
    )

    # Filter out Undisclosed if 0 to keep chart clean, but always keep the 5 core cohorts
    display_df = enroll_cohort[
        (enroll_cohort["AgeCohort"] != "Undisclosed") | (enroll_cohort["Enrollments"] > 0)
    ].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Total Enrollments",
            x=display_df["AgeCohort"],
            y=display_df["Enrollments"],
            marker_color=THEME_COLORS["primary"],
            text=display_df["Enrollments"],
            textposition="outside",
            customdata=np.stack([display_df["ActiveLearners"], display_df["EnrollmentsPerLearner"]], axis=-1),
            hovertemplate=(
                "<b>Cohort: %{x}</b><br>"
                "Total Enrollments: <b>%{y:,}</b><br>"
                "Unique Active Learners: %{customdata[0]:,}<br>"
                "Avg Enrollments / Learner: %{customdata[1]:.2f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            name="Unique Active Learners",
            x=display_df["AgeCohort"],
            y=display_df["ActiveLearners"],
            marker_color=THEME_COLORS["chart_blue"],
            text=display_df["ActiveLearners"],
            textposition="outside",
            hovertemplate=(
                "<b>Cohort: %{x}</b><br>"
                "Unique Active Learners: <b>%{y:,}</b><extra></extra>"
            ),
        )
    )
    fig.update_layout(
        barmode="group",
        title="Participation & Enrollment Volume by Mutually Exclusive Age Cohort",
        xaxis_title="Age Cohort",
        yaxis_title="Count",
    )
    return apply_edupro_chart_theme(fig, height=400)


def plot_age_category_heatmap(safe_matrix: pd.DataFrame) -> go.Figure:
    """Interactive heatmap of Age Cohort × Course Category enrollments with k-anonymity protection."""
    # Drop Undisclosed row if all zero/NaN
    mat = safe_matrix.copy()
    if "Undisclosed" in mat.index and mat.loc["Undisclosed"].isna().all() or (
        "Undisclosed" in mat.index and mat.loc["Undisclosed"].fillna(0).sum() == 0
    ):
        mat = mat.drop(index="Undisclosed")

    text_values = mat.map(lambda v: f"{int(v):,}" if pd.notna(v) else "")
    visible_values = [
        (column, index, mat.loc[index, column])
        for index in mat.index
        for column in mat.columns
        if pd.notna(mat.loc[index, column])
    ]
    finite_values = mat.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    max_value = float(finite_values.max()) if finite_values.size else 0.0

    fig = go.Figure(
        data=go.Heatmap(
            z=mat.values,
            x=mat.columns.tolist(),
            y=mat.index.tolist(),
            text=text_values.values,
            texttemplate="",
            colorscale=SEQUENTIAL_COLORS,
            colorbar=dict(title="Enrollments"),
            hoverongaps=False,
            hovertemplate=(
                "Age Cohort: <b>%{y}</b><br>"
                "Course Category: <b>%{x}</b><br>"
                "Enrollments: <b>%{text}</b><extra></extra>"
            ),
        )
    )
    suppressed_cells = [
        (column, index)
        for index in mat.index
        for column in mat.columns
        if pd.isna(mat.loc[index, column])
    ]
    if suppressed_cells:
        fig.add_trace(
            go.Scatter(
                x=[column for column, _ in suppressed_cells],
                y=[index for _, index in suppressed_cells],
                text=["Suppressed (<5)"] * len(suppressed_cells),
                mode="text",
                textfont=dict(size=10, color=THEME_COLORS["text_body"]),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    if visible_values:
        fig.add_trace(
            go.Scatter(
                x=[column for column, _, _ in visible_values],
                y=[index for _, index, _ in visible_values],
                text=[f"{int(value):,}" for _, _, value in visible_values],
                mode="text",
                textfont=dict(
                    size=12,
                    color=[
                        THEME_COLORS["page"]
                        if max_value > 0 and value / max_value >= 0.6
                        else THEME_COLORS["text_heading"]
                        for _, _, value in visible_values
                    ],
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.update_layout(
        title="Age Cohort × Course Category Enrollment Heatmap",
        xaxis_title="Course Category",
        yaxis_title="Learner Age Cohort",
    )
    return apply_edupro_chart_theme(fig, height=430)


def plot_supply_vs_demand_cpi(cpi_df: pd.DataFrame) -> go.Figure:
    """Grouped bar comparing Catalog Supply Share (%) vs. Enrollment Demand Share (%) with CPI line."""
    df = cpi_df.sort_values("CPI", ascending=False).copy()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Catalog Supply Share (%)",
            x=df["CourseCategory"],
            y=df["CatalogSharePct"],
            marker_color=THEME_COLORS["chart_taupe"],
            hovertemplate="<b>%{x}</b><br>Catalog Share: <b>%{y:.1f}%</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Enrollment Demand Share (%)",
            x=df["CourseCategory"],
            y=df["EnrollmentSharePct"],
            marker_color=THEME_COLORS["primary"],
            customdata=df["CPI"],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Enrollment Share: <b>%{y:.1f}%</b><br>"
                "Category Popularity Index (CPI): <b>%{customdata:.2f}</b><extra></extra>"
            ),
        )
    )
    fig.update_layout(
        barmode="group",
        title="Catalog Supply Share (%) vs. Enrollment Demand Share (%)",
        xaxis_title="Course Category",
        yaxis_title="Share of Platform (%)",
    )
    return apply_edupro_chart_theme(fig, height=410)


def plot_gender_parity_lollipop(gpi_df: pd.DataFrame, dimension_col: str = "CourseCategory") -> go.Figure:
    """Plot Adjusted Gender Parity Index (GPI) relative to the 0.97–1.03 UNESCO parity band."""
    df = gpi_df.dropna(subset=["Adjusted_GPI"]).sort_values("Adjusted_GPI", ascending=True).copy()

    colors = [
        THEME_COLORS["primary"]
        if 0.97 <= val <= 1.03
        else (DIVERGING_COLORS[0][1] if val < 0.97 else DIVERGING_COLORS[-1][1])
        for val in df["Adjusted_GPI"]
    ]

    fig = go.Figure()
    # Shaded UNESCO parity zone (0.97 to 1.03)
    fig.add_vrect(
        x0=0.97,
        x1=1.03,
        fillcolor=THEME_COLORS["primary_tint"],
        opacity=0.45,
        layer="below",
        line_width=0,
        annotation_text="Parity Band (0.97–1.03)",
        annotation_position="top left",
    )
    fig.add_vline(
        x=1.0,
        line_dash="dash",
        line_color=THEME_COLORS["primary"],
        line_width=1.5,
    )

    fig.add_trace(
        go.Bar(
            x=df["Adjusted_GPI"],
            y=df[dimension_col],
            orientation="h",
            marker_color=colors,
            text=df["Adjusted_GPI"].apply(lambda v: f"{v:.2f}"),
            textposition="outside",
            customdata=np.stack(
                [df["Female"], df["Male"], df["Raw_GPR"], df["ParityStatus"]],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Adjusted GPI: <b>%{x:.2f}</b><br>"
                "Raw F/M Ratio: %{customdata[2]:.2f}<br>"
                "Female Enrollments: %{customdata[0]:,}<br>"
                "Male Enrollments: %{customdata[1]:,}<br>"
                "Status: %{customdata[3]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Adjusted Gender Parity Index (GPI) by {dimension_col}",
        xaxis_title="Adjusted GPI (1.00 = Exact Parity)",
        yaxis_title="",
        xaxis=dict(range=[0, max(1.35, float(df["Adjusted_GPI"].max()) * 1.15) if not df.empty else 1.35]),
    )
    return apply_edupro_chart_theme(fig, height=410)
