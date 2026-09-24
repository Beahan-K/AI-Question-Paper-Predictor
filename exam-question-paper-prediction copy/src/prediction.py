"""
prediction.py
--------------
Predicts "important" topics for an upcoming exam by combining two
real, data-driven signals:

  1. Historical frequency  -- how often a topic has actually appeared
     across the collected question papers (optionally weighted so
     more recent years count more).
  2. Model confidence      -- how strongly the trained ML classifier
     (Naive Bayes / Logistic Regression / SVM, whichever performed
     best) associates the topic's chapter-level vocabulary with the
     dataset, via the TF-IDF feature space.

The two signals are min-max normalised and blended into a single
"Importance Score" in [0, 100]. This is explicitly a heuristic,
analytical estimate -- NOT a guaranteed prediction of exact exam
questions. Every place this score is shown in the UI must keep that
framing.
"""

import numpy as np
import pandas as pd


def _minmax_normalize(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-9:
        return pd.Series(50.0, index=series.index)  # all equal -> neutral mid score
    return (series - lo) / (hi - lo) * 100.0


def historical_topic_frequency(df: pd.DataFrame, subject: str = None,
                                recent_year_weight: float = 1.5) -> pd.DataFrame:
    """
    Count how often each Topic appears, optionally filtered to one
    Subject, with more recent years weighted slightly higher so the
    score reflects recent trends more than very old papers.

    Returns a dataframe: Topic, Chapter, Frequency (weighted count).
    """
    data = df.copy()
    if subject and subject != "All":
        data = data[data["Subject"] == subject]

    if data.empty:
        return pd.DataFrame(columns=["Topic", "Chapter", "Frequency"])

    years = sorted(data["Year"].unique())
    if len(years) > 1:
        weights = {y: 1.0 + recent_year_weight * (i / (len(years) - 1)) for i, y in enumerate(years)}
    else:
        weights = {years[0]: 1.0}
    data["_weight"] = data["Year"].map(weights)

    grouped = (
        data.groupby(["Topic", "Chapter"])["_weight"]
        .sum()
        .reset_index()
        .rename(columns={"_weight": "Frequency"})
        .sort_values("Frequency", ascending=False)
    )
    return grouped.reset_index(drop=True)


def model_topic_confidence(df: pd.DataFrame, vectorizer, model, subject: str = None) -> pd.DataFrame:
    """
    For each Topic, compute the trained classifier's average predicted
    probability (confidence) for the topic's Chapter label, using the
    TF-IDF vectors of that topic's own historical questions. Topics
    whose vocabulary the model recognises strongly get a higher score.

    Returns a dataframe: Topic, ModelConfidence (0-1).
    """
    data = df.copy()
    if subject and subject != "All":
        data = data[data["Subject"] == subject]

    if data.empty or "Cleaned_Question" not in data.columns:
        return pd.DataFrame(columns=["Topic", "ModelConfidence"])

    has_proba = hasattr(model, "predict_proba")
    class_list = list(getattr(model, "classes_", []))

    rows = []
    for topic, group in data.groupby("Topic"):
        texts = group["Cleaned_Question"].tolist()
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            continue
        X = vectorizer.transform(texts)

        if has_proba and class_list:
            proba = model.predict_proba(X)
            chapter = group["Chapter"].iloc[0]
            if chapter in class_list:
                idx = class_list.index(chapter)
                confidence = float(np.mean(proba[:, idx]))
            else:
                # Chapter wasn't a trained class (rare/filtered) -> use max prob as a proxy
                confidence = float(np.mean(proba.max(axis=1)))
        else:
            # Fall back to decision_function / a neutral score if predict_proba unavailable
            confidence = 0.5

        rows.append({"Topic": topic, "ModelConfidence": confidence})

    return pd.DataFrame(rows)


def predict_important_topics(df: pd.DataFrame, vectorizer, model, subject: str = None,
                              top_n: int = 10, freq_weight: float = 0.6) -> pd.DataFrame:
    """
    Main entry point used by the "Predict Important Topics" page.

    Blends historical frequency and model confidence into a single
    Importance Score (0-100) per topic, and returns the top_n topics
    sorted descending.

    freq_weight controls the blend (0-1): higher -> historical
    frequency matters more; lower -> model confidence matters more.
    """
    freq_df = historical_topic_frequency(df, subject=subject)
    if freq_df.empty:
        return pd.DataFrame(columns=["Topic", "Chapter", "Importance Score"])

    conf_df = model_topic_confidence(df, vectorizer, model, subject=subject)

    merged = freq_df.merge(conf_df, on="Topic", how="left")
    merged["ModelConfidence"] = merged["ModelConfidence"].fillna(merged["ModelConfidence"].mean() if not merged["ModelConfidence"].isna().all() else 0.5)

    merged["FreqScore"] = _minmax_normalize(merged["Frequency"])
    merged["ConfScore"] = merged["ModelConfidence"] * 100.0

    merged["Importance Score"] = (
        freq_weight * merged["FreqScore"] + (1 - freq_weight) * merged["ConfScore"]
    ).round(1)

    merged = merged.sort_values("Importance Score", ascending=False).reset_index(drop=True)
    result = merged[["Topic", "Chapter", "Importance Score", "Frequency"]].head(top_n)
    return result
