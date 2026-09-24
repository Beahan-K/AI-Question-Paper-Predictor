"""
app.py
------
Home page / entry point for the Exam Question Paper Prediction
Streamlit application.

Run with:
    streamlit run app.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st
from app_utils import (
    PAGE_CONFIG, apply_page_style, stat_card, get_dataset,
    get_dataset_signature, get_trained_artifacts, require_dataset,
)
from analysis import dataset_overview_stats

st.set_page_config(**PAGE_CONFIG)
apply_page_style(
    "Exam Question Paper Prediction",
    "AI-powered Question Paper Trend Analysis & Important Topic Prediction",
)

df = get_dataset()

if df is None or df.empty:
    st.error(
        "No dataset found. Place a CSV at `dataset/questions.csv` "
        "(columns: ID, Year, Subject, Question_No, Chapter, Question, Topic, Marks), "
        "or run `python dataset/generate_sample_dataset.py` to generate a sample dataset."
    )
    st.stop()

# --- Introduction -----------------------------------------------------
st.markdown(
    """
    <div class="info-box">
    <b>Exam Question Paper Prediction Using Machine Learning</b> is an intelligent
    system that analyzes historical examination papers using Natural Language
    Processing and Machine Learning techniques. The system identifies frequently
    occurring topics, analyzes question trends, finds similar questions, and
    predicts topics that may be important based on historical patterns.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="warn-box" style="margin-top:0.75rem;">
    ⚠️ Predictions are based on historical patterns and should be treated as
    analytical guidance, not guaranteed examination questions.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### ")

# --- Dataset stat cards -------------------------------------------------
stats = dataset_overview_stats(df)
c1, c2, c3, c4 = st.columns(4)
with c1:
    stat_card("Total Questions", stats["Total Questions"])
with c2:
    stat_card("Question Papers", stats["Question Papers"])
with c3:
    stat_card("Subjects", stats["Subjects"])
with c4:
    stat_card("Topics", stats["Topics"])

st.markdown("### ")

# --- How it works --------------------------------------------------------
left, right = st.columns([1.3, 1])

with left:
    st.subheader("How the system works")
    st.markdown(
        """
1. **Dataset** — Previous-year question papers are stored in a structured CSV
   (`Year`, `Subject`, `Chapter`, `Question`, `Topic`, `Marks`).
2. **NLP Preprocessing** — Every question is lowercased, cleaned of punctuation
   and numbers, tokenized, stripped of stop-words, and lemmatized.
3. **TF-IDF Feature Extraction** — Cleaned text is converted into numerical
   TF-IDF vectors that capture which words matter most in each question.
4. **Machine Learning** — Multinomial Naive Bayes, Logistic Regression, and
   SVM are trained on those vectors and compared on accuracy, precision,
   recall, and F1 score.
5. **Prediction** — The best-performing model's confidence is blended with
   historical topic frequency to produce an importance score for each topic.
6. **Dashboard** — Interactive charts and tables let you explore trends,
   repeated questions, and keyword importance.
        """
    )

with right:
    st.subheader("Technologies used")
    st.markdown(
        """
- **Python** · Pandas · NumPy
- **NLTK** — tokenization, stop-words, lemmatization
- **Scikit-learn** — TF-IDF, Naive Bayes, Logistic Regression, SVM
- **Streamlit** — interactive web interface
- **Matplotlib / Plotly / WordCloud** — visualization
- **pdfplumber** — PDF question-paper extraction
        """
    )
    st.subheader("Dataset information")
    st.markdown(
        f"""
- Years covered: **{', '.join(str(y) for y in stats['Years'])}**
- Subjects: **{stats['Subjects']}**
- Distinct chapters: **{stats['Chapters']}**
- Distinct topics: **{stats['Topics']}**
        """
    )

st.markdown("### ")
st.info(
    "👉 Use the **sidebar** to navigate to the Dashboard, Question Analysis, "
    "Predict Topics, Model Performance, TF-IDF Analysis, or Upload Question "
    "Paper pages. Click **Predict Important Topics** in the sidebar to get "
    "started right away.",
    icon="🧭",
)

# Warm up (train once, cache for every page) so the first click on any
# page doesn't have to wait for training.
with st.spinner("Preparing ML models (first run only)..."):
    get_trained_artifacts(get_dataset_signature())
