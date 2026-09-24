"""
2_Question_Analysis.py
------------------------
Search previous questions, filter by metadata, and detect
identical/highly-similar questions using TF-IDF cosine similarity.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import streamlit as st

from app_utils import PAGE_CONFIG, apply_page_style, require_dataset, build_live_tfidf
from analysis import apply_filters
from similarity import search_questions, most_similar_to_query, find_similar_question_pairs

st.set_page_config(**PAGE_CONFIG)
apply_page_style("Question Analysis", "Search, filter, and detect repeated or highly similar questions")

df = require_dataset()

tab_search, tab_repeated = st.tabs(["🔍 Search Questions", "🔁 Repeated / Similar Questions"])

# ---------------------------------------------------------------------
with tab_search:
    st.subheader("Search Questions")
    query = st.text_input("Search", placeholder="e.g. electromagnetic, genetics, thermodynamics ...")

    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        sel_year = st.selectbox("Year", ["All"] + sorted(df["Year"].unique().tolist()), key="qa_year")
    with f2:
        sel_subject = st.selectbox("Subject", ["All"] + sorted(df["Subject"].unique().tolist()), key="qa_subject")
    with f3:
        sel_topic = st.selectbox("Topic", ["All"] + sorted(df["Topic"].unique().tolist()), key="qa_topic")
    with f4:
        sel_chapter = st.selectbox("Chapter", ["All"] + sorted(df["Chapter"].unique().tolist()), key="qa_chapter")
    with f5:
        sel_marks = st.selectbox("Marks", ["All"] + sorted(df["Marks"].unique().tolist()), key="qa_marks")

    filtered = apply_filters(df, year=sel_year, subject=sel_subject, topic=sel_topic,
                              chapter=sel_chapter, marks=sel_marks)

    search_clicked = st.button("Search", type="primary")

    use_semantic = st.checkbox(
        "Use smart (TF-IDF similarity) search instead of exact text match",
        value=False,
        help="Finds conceptually similar questions even if the exact words differ.",
    )

    if search_clicked or query:
        if query.strip():
            if use_semantic and len(filtered) >= 2:
                vectorizer, matrix = build_live_tfidf(filtered)
                results = most_similar_to_query(filtered, vectorizer, matrix, query, top_n=25)
                display_cols = ["Year", "Subject", "Chapter", "Topic", "Question", "Marks", "Similarity"]
            else:
                results = search_questions(filtered, query)
                display_cols = ["Year", "Subject", "Chapter", "Topic", "Question", "Marks"]

            st.caption(f"Found **{len(results)}** matching question(s).")
            if len(results) > 0:
                st.dataframe(results[display_cols], use_container_width=True, hide_index=True)
            else:
                st.info("No matching questions found. Try a different keyword.")
        else:
            st.dataframe(
                filtered[["Year", "Subject", "Chapter", "Topic", "Question", "Marks"]],
                use_container_width=True, hide_index=True,
            )
    else:
        st.dataframe(
            filtered[["Year", "Subject", "Chapter", "Topic", "Question", "Marks"]].head(50),
            use_container_width=True, hide_index=True,
        )
        st.caption(f"Showing first 50 of {len(filtered)} questions. Enter a search term or click Search.")

# ---------------------------------------------------------------------
with tab_repeated:
    st.subheader("Repeated / Highly Similar Questions")
    st.markdown(
        "Uses **TF-IDF + cosine similarity** to flag question pairs that are "
        "identical or highly similar in wording — a proxy for 'repeated' questions "
        "across papers."
    )

    scope = st.radio(
        "Scope", ["Whole dataset", "Filter by subject"], horizontal=True, key="rep_scope"
    )
    if scope == "Filter by subject":
        subj = st.selectbox("Subject", sorted(df["Subject"].unique().tolist()), key="rep_subject")
        scoped_df = df[df["Subject"] == subj].reset_index(drop=True)
    else:
        scoped_df = df.reset_index(drop=True)

    threshold_pct = st.slider("Minimum similarity (%)", min_value=40, max_value=99, value=60, step=1)

    if len(scoped_df) < 2:
        st.info("Not enough questions in this scope to compare.")
    else:
        if len(scoped_df) > 600:
            st.warning(
                f"{len(scoped_df)} questions selected — comparing every pair can be slow. "
                "Consider filtering by subject."
            )
        with st.spinner("Computing pairwise similarity..."):
            vectorizer, matrix = build_live_tfidf(scoped_df)
            pairs = find_similar_question_pairs(
                scoped_df, vectorizer, matrix, threshold=threshold_pct / 100.0, max_pairs=100
            )

        if pairs.empty:
            st.success(f"No question pairs found with similarity ≥ {threshold_pct}%.")
        else:
            st.caption(f"Found **{len(pairs)}** similar pair(s) (showing up to 100, most similar first).")
            for _, row in pairs.iterrows():
                with st.container(border=True):
                    st.markdown(f"**Question A** ({row['Year_A']}, {row['Subject_A']}): {row['Question_A']}")
                    st.markdown(f"**Question B** ({row['Year_B']}, {row['Subject_B']}): {row['Question_B']}")
                    st.progress(min(int(row["Similarity"]), 100) / 100.0,
                                text=f"Similarity: {row['Similarity']}%")
