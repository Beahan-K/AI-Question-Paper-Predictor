"""
train_model.py
---------------
Trains and compares three supervised ML models (Multinomial Naive
Bayes, Logistic Regression, Support Vector Machine) that learn to
predict a question's Topic from its TF-IDF features.

This "topic classifier" is the backbone of the prediction pipeline:
by measuring how well a model recognises topic-specific vocabulary,
combined with historical topic frequency, the app estimates which
topics are most "important" for the next exam.

Run directly to (re)train and cache everything to disk:
    python src/train_model.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_dataframe  # noqa: E402
from feature_extraction import build_tfidf_matrix, save_vectorizer  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "questions.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "dataset", "processed_questions.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_classes.pkl")

MIN_SAMPLES_PER_CLASS = 2  # a class needs >=2 rows to be split into train/test


def load_and_prepare_dataset(dataset_path: str = DATASET_PATH) -> pd.DataFrame:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Run dataset/generate_sample_dataset.py or upload a question paper first."
        )
    df = pd.read_csv(dataset_path)
    df = preprocess_dataframe(df, text_column="Question", output_column="Cleaned_Question")
    return df


def _filter_rare_classes(df: pd.DataFrame, label_col: str = "Topic") -> pd.DataFrame:
    """Drop topics that have fewer than MIN_SAMPLES_PER_CLASS examples so
    train_test_split(stratify=...) doesn't fail on tiny classes."""
    counts = df[label_col].value_counts()
    valid_labels = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    return df[df[label_col].isin(valid_labels)].reset_index(drop=True)


def train_and_compare_models(df: pd.DataFrame = None, label_col: str = "Chapter",
                              test_size: float = 0.2, random_state: int = 42):
    """
    Trains Multinomial Naive Bayes, Logistic Regression and SVM on
    TF-IDF features to classify each question's label (Chapter by
    default -- a coarser, better-populated label than Topic, which
    keeps the classifier statistically meaningful even on small demo
    datasets), evaluates all three on a held-out test split, and
    returns everything needed by the app: the fitted vectorizer,
    per-model metrics, confusion matrices, the best model, and its
    name.

    NOTE: `label_col` is configurable so that a richer dataset (e.g.
    university papers with more questions per Topic) can train
    directly on "Topic" instead -- just pass label_col="Topic".
    """
    if df is None:
        df = load_and_prepare_dataset()

    df = _filter_rare_classes(df, label_col)
    if df[label_col].nunique() < 2:
        raise ValueError(
            f"Need at least 2 distinct '{label_col}' classes with "
            f"{MIN_SAMPLES_PER_CLASS}+ questions each to train a classifier."
        )

    vectorizer, X = build_tfidf_matrix(df["Cleaned_Question"])
    y = df[label_col].astype(str)

    # Guard: stratified split needs at least 1 sample per class in the
    # test set, i.e. test_size * n_samples >= n_classes. If the dataset
    # is small/fragmented relative to its class count, grow test_size
    # (capped at 0.5) instead of crashing.
    n_classes = y.nunique()
    n_samples = len(y)
    min_test_size = n_classes / n_samples
    if test_size < min_test_size:
        test_size = min(0.5, min_test_size + 0.05)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "SVM": SVC(kernel="linear", probability=True, random_state=random_state),
    }

    results = {}
    trained_models = {}
    labels_sorted = sorted(y.unique())

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "Confusion Matrix": cm,
            "Labels": labels_sorted,
        }
        trained_models[name] = model

    best_name = max(results, key=lambda n: results[n]["F1 Score"])
    best_model = trained_models[best_name]

    return {
        "vectorizer": vectorizer,
        "results": results,
        "trained_models": trained_models,
        "best_model_name": best_name,
        "best_model": best_model,
        "labels": labels_sorted,
        "df": df,
    }


def save_artifacts(training_output: dict):
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_vectorizer(training_output["vectorizer"])
    joblib.dump({
        "model": training_output["best_model"],
        "model_name": training_output["best_model_name"],
    }, BEST_MODEL_PATH)
    joblib.dump(training_output["results"], METRICS_PATH)
    joblib.dump(training_output["labels"], LABEL_ENCODER_PATH)

    # Persist the cleaned dataset too, so other pages don't have to re-clean it
    training_output["df"].to_csv(PROCESSED_PATH, index=False)


def load_artifacts():
    """Load previously saved model/vectorizer/metrics from disk, if present."""
    if not (os.path.exists(BEST_MODEL_PATH) and os.path.exists(METRICS_PATH)):
        return None
    best = joblib.load(BEST_MODEL_PATH)
    metrics = joblib.load(METRICS_PATH)
    from feature_extraction import load_vectorizer
    vectorizer = load_vectorizer()
    labels = joblib.load(LABEL_ENCODER_PATH) if os.path.exists(LABEL_ENCODER_PATH) else None
    return {
        "best_model": best["model"],
        "best_model_name": best["model_name"],
        "results": metrics,
        "vectorizer": vectorizer,
        "labels": labels,
    }


def train_if_needed(force: bool = False):
    """
    Used by the Streamlit app on startup: load cached model/vectorizer
    if available, otherwise train from scratch and cache the result.
    """
    if not force:
        cached = load_artifacts()
        if cached is not None:
            return cached

    df = load_and_prepare_dataset()
    output = train_and_compare_models(df)
    save_artifacts(output)
    return load_artifacts()


if __name__ == "__main__":
    print("Loading dataset and training models...")
    result = train_if_needed(force=True)
    print(f"Best model: {result['best_model_name']}")
    for name, metrics in result["results"].items():
        print(f"{name}: Accuracy={metrics['Accuracy']:.3f}  F1={metrics['F1 Score']:.3f}")
    print("Artifacts saved to models/")
