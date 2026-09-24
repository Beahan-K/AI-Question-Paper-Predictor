"""
app_utils.py
------------
Shared helpers used by app.py and every page under pages/:
  - consistent path resolution
  - cached dataset loading + preprocessing
  - cached model training/loading
  - shared CSS for a clean academic dashboard look
  - small UI helper (stat cards)
"""

import os
import sys
import streamlit as st
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "questions.csv")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from preprocessing import preprocess_dataframe, nltk_status  # noqa: E402
from train_model import train_if_needed, load_and_prepare_dataset  # noqa: E402
from feature_extraction import build_tfidf_matrix  # noqa: E402


PAGE_CONFIG = dict(
    page_title="Exam Question Paper Prediction",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    }

    .app-header {
        padding: 1.75rem 2rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #0f2340 0%, #1c3d63 60%, #234d7a 100%);
        color: #ffffff;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        margin: 0 0 0.25rem 0;
        font-size: 1.9rem;
        font-weight: 700;
        color: #ffffff;
    }
    .app-header p {
        margin: 0;
        font-size: 1.02rem;
        color: #cfe0f2;
    }

    .stat-card {
        background: #ffffff;
        border: 1px solid #e3e8ee;
        border-radius: 10px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(15, 35, 64, 0.06);
        text-align: left;
    }
    .stat-card .stat-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0f2340;
        line-height: 1.1;
    }
    .stat-card .stat-label {
        font-size: 0.85rem;
        color: #5a6c7d;
        margin-top: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .info-box {
        background: #f4f8fc;
        border-left: 4px solid #234d7a;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        font-size: 0.93rem;
        color: #2a3b4d;
    }
    .warn-box {
        background: #fff8e6;
        border-left: 4px solid #d9a441;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        font-size: 0.93rem;
        color: #5c4a1a;
    }

    section[data-testid="stSidebar"] {
        background-color: #f7f9fb;
    }

    div[data-testid="stMetricValue"] {
        color: #0f2340;
    }
</style>
"""


def apply_page_style(title: str, subtitle: str = ""):
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    header_html = f"""
    <div class="app-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def stat_card(label: str, value):
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def get_dataset(dataset_path: str = DATASET_PATH):
    """Load + clean the dataset. Cached so every page shares one copy."""
    if not os.path.exists(dataset_path):
        return None
    df = load_and_prepare_dataset(dataset_path)
    return df


@st.cache_resource(show_spinner=False)
def get_trained_artifacts(_dataset_signature: str):
    """
    Load cached model/vectorizer from disk, training fresh if none
    exists yet. `_dataset_signature` (e.g. row count + mtime) is only
    used as a cache key so Streamlit retrains automatically if the
    underlying dataset changes.
    """
    return train_if_needed()


def get_dataset_signature(dataset_path: str = DATASET_PATH) -> str:
    if not os.path.exists(dataset_path):
        return "no-dataset"
    stat = os.stat(dataset_path)
    return f"{stat.st_size}-{stat.st_mtime}"


def build_live_tfidf(df: pd.DataFrame):
    """Build a TF-IDF matrix aligned with df's current row order (for similarity search etc.)."""
    vectorizer, matrix = build_tfidf_matrix(df["Cleaned_Question"])
    return vectorizer, matrix


def require_dataset():
    """Call at the top of every page: stop with a friendly message if no dataset is loaded."""
    df = get_dataset()
    if df is None or df.empty:
        st.error(
            "No dataset found or dataset is empty. Please make sure "
            "`dataset/questions.csv` exists (run "
            "`python dataset/generate_sample_dataset.py` for a sample), "
            "or upload a question paper via the **Upload Question Paper** page."
        )
        st.stop()
    return df
