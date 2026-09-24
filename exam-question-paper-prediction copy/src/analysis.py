"""
analysis.py
-----------
Dataset-level analytics used by the Home and Dashboard pages.
Every number here is calculated live from the loaded CSV -- nothing
is hard-coded.
"""

import pandas as pd


def dataset_overview_stats(df: pd.DataFrame) -> dict:
    """Return the headline stats shown as cards on Home/Dashboard."""
    return {
        "Total Questions": int(len(df)),
        "Question Papers": int(df["Year"].nunique()) if "Year" in df else 0,
        "Subjects": int(df["Subject"].nunique()) if "Subject" in df else 0,
        "Topics": int(df["Topic"].nunique()) if "Topic" in df else 0,
        "Chapters": int(df["Chapter"].nunique()) if "Chapter" in df else 0,
        "Years": sorted(df["Year"].unique().tolist()) if "Year" in df else [],
    }


def subject_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Subject").size().reset_index(name="Count").sort_values("Count", ascending=False)
    )


def year_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Year").size().reset_index(name="Count").sort_values("Year")
    )


def topic_frequency(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    return (
        df.groupby("Topic").size().reset_index(name="Count")
        .sort_values("Count", ascending=False).head(top_n)
    )


def chapter_frequency(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    return (
        df.groupby("Chapter").size().reset_index(name="Count")
        .sort_values("Count", ascending=False).head(top_n)
    )


def most_repeated_topics(df: pd.DataFrame, min_years: int = 2, top_n: int = 15) -> pd.DataFrame:
    """Topics that have appeared across the most distinct papers/years."""
    grouped = df.groupby("Topic")["Year"].nunique().reset_index(name="Years_Appeared")
    grouped = grouped[grouped["Years_Appeared"] >= 1].sort_values("Years_Appeared", ascending=False)
    return grouped.head(top_n)


def apply_filters(df: pd.DataFrame, year=None, subject=None, topic=None,
                   chapter=None, marks=None) -> pd.DataFrame:
    """Generic multi-field filter used across Dashboard / Question Analysis pages."""
    out = df.copy()
    if year and year != "All":
        out = out[out["Year"] == year]
    if subject and subject != "All":
        out = out[out["Subject"] == subject]
    if topic and topic != "All":
        out = out[out["Topic"] == topic]
    if chapter and chapter != "All":
        out = out[out["Chapter"] == chapter]
    if marks and marks != "All":
        out = out[out["Marks"] == marks]
    return out
