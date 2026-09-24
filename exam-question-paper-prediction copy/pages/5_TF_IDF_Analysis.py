"""
5_TF_IDF_Analysis.py
-----------------------
Educational page explaining the TF-IDF pipeline and showing the
top TF-IDF keywords, computed live from the dataset.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import streamlit as st
import plotly.express as px

from app_utils import PAGE_CONFIG, apply_page_style, require_dataset, build_live_tfidf
from feature_extraction import top_keywords_overall, top_keywords_for_subset
from preprocessing import nltk_status

st.set_page_config(**PAGE_CONFIG)
apply_page_style("TF-IDF Analysis", "How raw question text becomes numerical features the model can learn from")

df = require_dataset()

st.subheader("Pipeline")
st.markdown(
    """
    ```
    Question Text
          |
          v
    Text Cleaning        (lowercase, remove punctuation/numbers)
          |
          v
    Tokenization          (split into words)
          |
          v
    Stop-word Removal + Lemmatization
          |
          v
    TF-IDF                (Term Frequency x Inverse Document Frequency)
          |
          v
    Numerical Feature Vector
    ```
    """
)

if not nltk_status():
    st.warning(
        "NLTK tokenizer/lemmatizer data could not be downloaded in this environment "
        "(likely no internet access), so a built-in fallback stop-word list is being "
        "used instead. Preprocessing still runs correctly, just without lemmatization.",
        icon="ℹ️",
    )

st.markdown("### ")
st.subheader("Example: before vs. after cleaning")
sample = df.sample(min(3, len(df)), random_state=1)[["Question", "Cleaned_Question"]]
st.dataframe(sample, use_container_width=True, hide_index=True)

st.markdown("### ")
st.subheader("Top TF-IDF Keywords")

scope = st.radio("Scope", ["Whole dataset"] + sorted(df["Subject"].unique().tolist()), horizontal=True)
scoped_df = df if scope == "Whole dataset" else df[df["Subject"] == scope]

if len(scoped_df) < 2:
    st.info("Not enough questions in this scope to compute TF-IDF keywords.")
else:
    vectorizer, matrix = build_live_tfidf(scoped_df)
    kw = top_keywords_overall(vectorizer, matrix, top_n=20)

    col_a, col_b = st.columns([1, 1.4])
    with col_a:
        st.dataframe(kw, use_container_width=True, hide_index=True)
    with col_b:
        fig = px.bar(kw.sort_values("Score"), x="Score", y="Keyword", orientation="h",
                     title=f"Top Keywords — {scope}")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("### ")
with st.expander("What does TF-IDF actually measure?"):
    st.markdown(
        """
- **Term Frequency (TF)** — how often a word appears within a single question.
- **Inverse Document Frequency (IDF)** — how rare that word is across *all*
  questions. Common words like "the" or "which" get a low IDF; subject-specific
  words like "electrostatics" or "photosynthesis" get a high IDF.
- **TF-IDF score = TF × IDF** — high for words that appear often in a
  particular question but rarely elsewhere, which makes them good signals of
  what that question (and, in aggregate, that topic) is really about.

This project also uses **bigrams** (2-word phrases, e.g. "electric field") in
addition to single words, so short domain phrases aren't lost.
        """
    )
