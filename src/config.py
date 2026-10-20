"""Configuration constants, theme palette, and methodology definitions for EduPro Analytics."""
import os
from pathlib import Path

# Default workbook lookup paths (configurable via environment variable or UI)
DEFAULT_WORKBOOK_FILENAME = "EduPro Online Platform.xlsx"
ENV_WORKBOOK_PATH = "EDUPRO_WORKBOOK_PATH"

def resolve_default_workbook_path() -> Path:
    """Resolve the workbook path from environment variable or common project locations."""
    env_path = os.environ.get(ENV_WORKBOOK_PATH)
    if env_path:
        return Path(env_path)
    
    candidates = [
        Path.cwd() / DEFAULT_WORKBOOK_FILENAME,
        Path.cwd() / "data" / DEFAULT_WORKBOOK_FILENAME,
        Path(__file__).resolve().parent.parent / DEFAULT_WORKBOOK_FILENAME,
        Path(__file__).resolve().parent.parent / "data" / DEFAULT_WORKBOOK_FILENAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

# Required Schema Definitions
REQUIRED_SCHEMA = {
    "Users": {"UserID", "UserName", "Age", "Gender", "Email"},
    "Teachers": {
        "TeacherID", "TeacherName", "Age", "Gender",
        "Expertise", "YearsOfExperience", "TeacherRating"
    },
    "Courses": {
        "CourseID", "CourseName", "CourseCategory", "CourseType",
        "CourseLevel", "CoursePrice", "CourseDuration", "CourseRating"
    },
    "Transactions": {
        "TransactionID", "UserID", "CourseID", "TransactionDate",
        "Amount", "PaymentMethod", "TeacherID"
    },
}

# Strictly forbidden PII columns that must be dropped immediately upon loading
PII_COLUMNS = {
    "Users": ["UserName", "Email"],
    "Teachers": ["TeacherName"],
}

# Mutually Exclusive Age Cohorts (avoids 36-45 / 45+ boundary overlap)
AGE_COHORT_ORDER = [
    "Under 18",
    "18–25",
    "26–35",
    "36–44",
    "45+",
    "Undisclosed",
]

AGE_COHORT_DEFINITIONS = {
    "Under 18": "Ages 10 to 17 (Secondary Foundation)",
    "18–25": "Ages 18 to 25 (Post-Secondary & Early Career)",
    "26–35": "Ages 26 to 35 (Workforce Upskilling & Advancement)",
    "36–44": "Ages 36 to 44 (Mid-Career Pivot & Managerial Specialization)",
    "45+": "Ages 45 and older (Executive Mastery & Lifelong Learning)",
    "Undisclosed": "Missing, unrecorded, or out-of-range age (<10 or >100)",
}

COURSE_LEVEL_ORDER = ["Beginner", "Intermediate", "Advanced"]
GENDER_STANDARD_ORDER = ["Female", "Male", "Non-Binary", "Undisclosed"]

# Privacy threshold: suppress or aggregate demographic slices smaller than k
K_ANONYMITY_THRESHOLD = 5

# Statistical review conventions; these flag unusual patterns for review only.
REVIEW_MIN_PAID_PER_COURSE = 8
REVIEW_MODIFIED_Z_THRESHOLD = 3.5
REVIEW_LEARNER_PERCENTILE = 0.99
REVIEW_MIN_DAILY_OBSERVATIONS = 56
REVIEW_MIN_WEEKDAY_PEERS = 8

# Visual identity: Plum & Parchment
THEME_COLORS = {
    "page": "#FBF7F0",
    "surface": "#FFFFFF",
    "surface_alt": "#EFE7DA",
    "border": "#E5DCCB",
    "grid": "#E7DFD0",
    "axis": "#CFC6B5",
    "text_heading": "#26203A",
    "text_body": "#3F3A4E",
    "text_muted": "#6B6478",
    "sidebar": "#2A1F3D",
    "sidebar_raised": "#3A2D52",
    "sidebar_border": "#8D7DAA",
    "sidebar_text": "#F3EDE4",
    "sidebar_muted": "#BDB3CC",
    "sidebar_accent": "#C9B3F2",
    "primary": "#5A3E8E",
    "primary_hover": "#47306F",
    "primary_tint": "#E8E0F3",
    "primary_ink": "#3F2A6B",
    "focus": "#5A3E8E",
    "chart_blue": "#0072B2",
    "chart_rose": "#C4679C",
    "chart_taupe": "#8C7B6B",
    "positive": "#2E7D4F",
    "positive_ink": "#1F6B40",
    "positive_tint": "#DFF1E6",
    "warning": "#C27C0E",
    "warning_ink": "#8A5A00",
    "warning_tint": "#FBEBCB",
    "negative": "#B3261E",
    "negative_ink": "#8F1D17",
    "negative_tint": "#FBE4E1",
    "hero_start": "#2A1F3D",
    "hero_end": "#4A3470",
    "hero_glow": "rgba(196, 103, 156, 0.18)",
    "shadow": "rgba(42, 31, 61, 0.07)",
}

CATEGORICAL_COLORS = [
    THEME_COLORS["primary"],
    THEME_COLORS["chart_blue"],
    THEME_COLORS["chart_rose"],
    THEME_COLORS["chart_taupe"],
]
SEQUENTIAL_COLORS = [
    [0.0, "#F4EFF9"],
    [0.2, "#D9CCEB"],
    [0.4, "#B49BD6"],
    [0.6, "#8A68B8"],
    [0.8, "#5A3E8E"],
    [1.0, "#36245C"],
]
DIVERGING_COLORS = [
    [0.0, "#8C510A"],
    [0.25, "#D8B365"],
    [0.5, "#F5F1EA"],
    [0.75, "#7FB3D5"],
    [1.0, "#0B5C99"],
]
FONT_SANS = '"Source Sans Pro", "Segoe UI", system-ui, sans-serif'
FONT_SERIF = 'Georgia, "Iowan Old Style", "Palatino Linotype", serif'

COURSE_TYPE_COLORS = {"Free": CATEGORICAL_COLORS[0], "Paid": CATEGORICAL_COLORS[1]}

LEVEL_COLORS = {
    "Beginner": CATEGORICAL_COLORS[0],
    "Intermediate": CATEGORICAL_COLORS[1],
    "Advanced": CATEGORICAL_COLORS[2],
}

GENDER_COLORS = {
    "Female": CATEGORICAL_COLORS[0],
    "Male": CATEGORICAL_COLORS[1],
    "Non-Binary": CATEGORICAL_COLORS[2],
    "Undisclosed": CATEGORICAL_COLORS[3],
}
