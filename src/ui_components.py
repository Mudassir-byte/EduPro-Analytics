"""Reusable UI components, accessible CSS styling, KPI cards, and privacy-safe CSV download helpers."""
import html
import pandas as pd
import streamlit as st

from src.config import FONT_SANS, FONT_SERIF, PII_COLUMNS, THEME_COLORS


_THEME_CSS = """
.stApp {
    background: var(--edu-page);
    color: var(--edu-text-body);
    color-scheme: light;
    --primary-color: var(--edu-primary);
}
[data-testid="stHeader"] { background: transparent; }
html, body, [class*="css"] {
    font-family: FONT_SANS_TOKEN;
    color: var(--edu-text-body);
}
.block-container {
    max-width: 1400px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
h1, h2, h3, .edupro-hero h1 {
    color: var(--edu-text-heading);
    letter-spacing: -.015em;
}
h1, h2, .edupro-hero h1 { font-family: FONT_SERIF_TOKEN; }
a { color: var(--edu-primary); }
:focus-visible { outline: 3px solid var(--edu-focus); outline-offset: 2px; }
[data-testid="stSidebar"] {
    background: var(--edu-sidebar);
    border-right: 1px solid var(--edu-sidebar-raised);
}
[data-testid="stSidebar"] * { color: var(--edu-sidebar-text) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] input {
    background: var(--edu-sidebar-raised);
    border-color: var(--edu-sidebar-border);
}
[data-testid="stSidebar"] [data-testid="stTextInputRootElement"],
[data-testid="stSidebar"] [data-testid="stDateInputField"],
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [role="group"],
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: var(--edu-sidebar-raised) !important;
    border-color: var(--edu-sidebar-border) !important;
    color: var(--edu-sidebar-text) !important;
}
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-tag] {
    background: var(--edu-primary-tint) !important;
    border-color: var(--edu-primary-tint) !important;
}
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-tag] * {
    color: var(--edu-primary-ink) !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background: var(--edu-sidebar-raised) !important;
    border-color: var(--edu-sidebar-border) !important;
    color: var(--edu-sidebar-text) !important;
}
[data-testid="stSidebar"] :focus-visible { outline-color: var(--edu-sidebar-accent); }
[data-testid="stSidebar"] span[data-baseweb="tag"] {
    background: var(--edu-primary-tint) !important;
    border-color: var(--edu-primary-tint) !important;
}
[data-testid="stSidebar"] span[data-baseweb="tag"] * { color: var(--edu-primary-ink) !important; }
[data-testid="stRadioOption"][data-selected="true"] > div > div {
    background: var(--edu-primary) !important;
    border-color: var(--edu-primary) !important;
}
[data-baseweb="popover"] * { color: var(--edu-text-body); }
[data-baseweb="popover"] [aria-selected="true"] { background: var(--edu-primary-tint); }
input, button, [role="radio"], [role="checkbox"], [role="slider"] {
    accent-color: var(--edu-primary);
}
button[kind="primary"] {
    background: var(--edu-primary);
    border-color: var(--edu-primary);
    color: var(--edu-surface);
}
button[kind="primary"]:hover {
    background: var(--edu-primary-hover);
    border-color: var(--edu-primary-hover);
}
.edupro-hero {
    background:
        radial-gradient(circle at 85% 10%, var(--edu-hero-glow), transparent 45%),
        linear-gradient(135deg, var(--edu-hero-start), var(--edu-hero-end));
    color: var(--edu-page);
    padding: 1.7rem 2rem;
    border-radius: 18px;
    box-shadow: 0 8px 24px var(--edu-shadow);
    margin-bottom: 1.5rem;
    animation: fadeSlideIn .4s ease-out both;
}
.edupro-hero h1 { color: var(--edu-page) !important; font-size: 1.8rem; margin: 0 0 .4rem; }
.edupro-hero p { color: var(--edu-sidebar-text) !important; font-size: .98rem; margin: 0; line-height: 1.55; }
.edupro-badge {
    display: inline-block;
    background: var(--edu-sidebar-raised);
    color: var(--edu-sidebar-text);
    border: 1px solid var(--edu-sidebar-border);
    padding: .25rem .7rem;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 600;
    margin: 0 .5rem .55rem 0;
}
.kpi-card {
    background: var(--edu-surface);
    border: 1px solid var(--edu-border);
    border-top: 3px solid var(--edu-primary);
    border-radius: 14px;
    padding: 1.15rem 1.25rem;
    box-shadow: 0 2px 10px var(--edu-shadow);
    transition: transform .2s ease, box-shadow .2s ease;
    animation: fadeSlideIn .4s ease-out both;
    height: 100%;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 20px var(--edu-shadow); }
.kpi-card.accent-secondary { border-top-color: var(--edu-chart-blue); }
.kpi-card.accent-positive { border-top-color: var(--edu-positive); }
.kpi-card.accent-warning { border-top-color: var(--edu-warning); }
.kpi-card.accent-negative { border-top-color: var(--edu-negative); }
.kpi-label {
    color: var(--edu-text-muted);
    font-size: .78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .045em;
    margin-bottom: .4rem;
}
.kpi-value { color: var(--edu-text-heading); font-size: 1.75rem; font-weight: 700; line-height: 1.2; }
.kpi-sub { color: var(--edu-text-muted); font-size: .8rem; margin-top: .3rem; }
.insight-box, .recommendation-box {
    border-radius: 12px;
    padding: 1.1rem 1.35rem;
    margin-bottom: 1rem;
    color: var(--edu-text-body);
}
.insight-box { background: var(--edu-positive-tint); border: 1px solid var(--edu-border); border-left: 4px solid var(--edu-positive); }
.recommendation-box { background: var(--edu-primary-tint); border: 1px solid var(--edu-border); border-left: 4px solid var(--edu-primary); }
.stButton button, .stDownloadButton button {
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
.stButton button:hover, .stDownloadButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 5px 14px var(--edu-shadow);
    border-color: var(--edu-primary);
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
@media (max-width: 768px) {
    .edupro-hero { padding: 1.25rem; border-radius: 12px; }
    .edupro-hero h1 { font-size: 1.5rem; }
}
@media (prefers-reduced-motion: reduce) {
    .edupro-hero, .kpi-card, .stButton button, .stDownloadButton button {
        animation: none !important;
        transition: none !important;
    }
    .kpi-card:hover, .stButton button:hover, .stDownloadButton button:hover {
        transform: none !important;
    }
}
"""


def _theme_css_vars() -> str:
    """Expose the shared palette as CSS variables without duplicating color values."""
    variables = "".join(
        f"--edu-{name.replace('_', '-')}: {value};"
        for name, value in THEME_COLORS.items()
    )
    variables += (
        f"--primary-color: {THEME_COLORS['primary']};"
    )
    variables += f"--edu-chart-blue: {THEME_COLORS['chart_blue']};"
    css = _THEME_CSS.replace("FONT_SANS_TOKEN", FONT_SANS)
    css = css.replace("FONT_SERIF_TOKEN", FONT_SERIF)
    return f"<style>:root {{{variables}}}{css}</style>"


def inject_custom_css() -> None:
    """Apply the warm Plum & Parchment theme with motion and accessibility support."""
    st.markdown(_theme_css_vars(), unsafe_allow_html=True)


def render_hero_banner(
    title: str,
    subtitle: str,
    date_range_str: str,
    record_summary_str: str,
) -> None:
    """Render the top midnight-blue hero section with date coverage and record telemetry."""
    st.markdown(
        f"""
        <div class="edupro-hero">
            <div>
                <span class="edupro-badge">Descriptive Learner Intelligence</span>
                <span class="edupro-badge">{html.escape(date_range_str)}</span>
                <span class="edupro-badge">{html.escape(record_summary_str)}</span>
            </div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    label: str,
    value: str,
    subtitle: str,
    accent: str = "teal",
) -> None:
    """Render an accessible, rounded KPI card."""
    accent_class = {
        "primary": "",
        "secondary": "accent-secondary",
        "positive": "accent-positive",
        "warning": "accent-warning",
        "negative": "accent-negative",
    }.get(accent, "")
    st.markdown(
        f"""
        <div class="kpi-card {accent_class}">
            <div class="kpi-label">{html.escape(label)}</div>
            <div class="kpi-value">{html.escape(value)}</div>
            <div class="kpi-sub">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sanitize_dataframe_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Remove direct PII and neutralize spreadsheet-formula-like string values."""
    forbidden = set(PII_COLUMNS["Users"] + PII_COLUMNS["Teachers"])
    cols_to_drop = [c for c in df.columns if c in forbidden or "email" in c.lower() or "username" in c.lower()]
    safe_df = df.drop(columns=cols_to_drop, errors="ignore").copy()

    def neutralize_formula(value):
        if isinstance(value, str):
            stripped = value.lstrip(" \t\r\n")
            if value.startswith(("\t", "\r")) or stripped.startswith(("=", "+", "-", "@")):
                return "'" + value
        return value

    for column in safe_df.select_dtypes(include=["object", "string"]).columns:
        safe_df[column] = safe_df[column].map(neutralize_formula)
    return safe_df


def render_safe_csv_download(
    df: pd.DataFrame,
    filename: str,
    label: str = "Download Aggregate CSV (Privacy-Safe)",
    key: str = "csv_download",
) -> None:
    """Render a CSV download button after stripping any personal identifiers."""
    safe_df = sanitize_dataframe_for_export(df)
    csv_bytes = safe_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 {label}",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        key=key,
    )
