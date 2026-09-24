"""
feature_extraction.py
----------------------
TF-IDF feature extraction utilities.

Fits a TfidfVectorizer over the cleaned question text and exposes
helpers to:
  - build/save/load the vectorizer + feature matrix
  - get top keywords overall or per subject/topic
  - transform new (unseen) text using an already-fitted vectorizer
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")


def build_tfidf_matrix(cleaned_texts, max_features: int = 3000, ngram_range=(1, 2)):
    """
    Fit a TfidfVectorizer on the given iterable of cleaned text strings.

    Returns (vectorizer, tfidf_matrix).
    """
    cleaned_texts = list(cleaned_texts)
    if len(cleaned_texts) == 0:
        raise ValueError("Cannot build TF-IDF matrix from an empty dataset.")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=1,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    return vectorizer, tfidf_matrix


def save_vectorizer(vectorizer, path: str = VECTORIZER_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectorizer, path)


def load_vectorizer(path: str = VECTORIZER_PATH):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def top_keywords_overall(vectorizer, tfidf_matrix, top_n: int = 20) -> pd.DataFrame:
    """
    Rank vocabulary terms by their summed TF-IDF weight across the
    whole corpus. Returns a dataframe with columns: Keyword, Score.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
    order = np.argsort(scores)[::-1][:top_n]
    return pd.DataFrame({
        "Keyword": feature_names[order],
        "Score": scores[order],
    })


def top_keywords_for_subset(vectorizer, cleaned_texts_subset, top_n: int = 15) -> pd.DataFrame:
    """
    Compute top TF-IDF keywords for an arbitrary subset of cleaned
    text (e.g. all questions for one subject or one topic), reusing
    an already-fitted vectorizer.
    """
    cleaned_texts_subset = [t for t in cleaned_texts_subset if isinstance(t, str) and t.strip()]
    if not cleaned_texts_subset:
        return pd.DataFrame({"Keyword": [], "Score": []})
    matrix = vectorizer.transform(cleaned_texts_subset)
    feature_names = np.array(vectorizer.get_feature_names_out())
    scores = np.asarray(matrix.sum(axis=0)).ravel()
    order = np.argsort(scores)[::-1][:top_n]
    return pd.DataFrame({
        "Keyword": feature_names[order],
        "Score": scores[order],
    })


def transform_text(vectorizer, cleaned_text: str):
    """Transform a single already-cleaned text string into a TF-IDF vector."""
    return vectorizer.transform([cleaned_text])
