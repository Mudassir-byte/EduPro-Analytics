# EduPro Learner and Course Analytics

A Streamlit dashboard for exploring learner demographics, course demand, catalog
supply, and representation equity from the EduPro Online Platform workbook.

## Features

- Five views: overview, learner demographics, course demand, equity and insights,
  and data quality and methodology.
- A consistent Plum & Parchment visual theme with shared accessible color tokens,
  warm surfaces, aubergine navigation, and coordinated interactive charts.
- Workbook validation, referential-integrity checks, transaction auditing, and
  removal of direct identifiers (`UserName`, `Email`, and `TeacherName`).
- Mutually exclusive learner age cohorts and global filters.
- Category Popularity Index (CPI), gender participation metrics, and
  small-cell suppression for the demographic heatmap.
- Aggregate CSV exports that remove direct identifiers and neutralize
  spreadsheet-formula-like text.
- Non-destructive statistical review flags for paid course-amount deviations,
  high aggregate learner activity, and weekday-adjusted daily volume. Flags are
  separate from data validation, are based on the full validated workbook, and
  are not evidence of errors or fraud.

## Requirements

- Python 3.11 or newer
- The workbook `EduPro Online Platform.xlsx`

## Setup

From the `edupro_analytics` directory, create and activate a virtual environment
and install the pinned application dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Workbook configuration

By default, the app looks for `EduPro Online Platform.xlsx` in the project root
or a `data` subdirectory. You can also set `EDUPRO_WORKBOOK_PATH` to the workbook
location, or select a workbook using the sidebar uploader.

```powershell
$env:EDUPRO_WORKBOOK_PATH = "C:\path\to\EduPro Online Platform.xlsx"
```

The workbook must contain `Users`, `Teachers`, `Courses`, and `Transactions`
sheets with the columns defined in `src/config.py`. The application reports
schema and data validation failures rather than substituting sample data.

## Run the dashboard

From the `edupro_analytics` directory:

```powershell
streamlit run app.py
```

## Run tests

```powershell
python -m pytest tests/test_analytics.py -v
```

## Privacy note

Direct names and email addresses are removed during workbook ingestion and
excluded from CSV exports. The demographic heatmap suppresses cells with fewer
than five observations; small learner-activity counts are similarly suppressed
in the review view. Statistical review flags are aggregate-only and never
exclude or modify records. Small-cell suppression is a display safeguard;
review the source data and any exported aggregates before sharing them
externally.
