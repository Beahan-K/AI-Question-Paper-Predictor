"""
1_Dashboard.py
---------------
Analytics dashboard: dataset statistics, filters, and charts, all
calculated live from the CSV dataset.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from app_utils import PAGE_CONFIG, apply_page_style, stat_card, require_dataset, build_live_tfidf
from analysis import (
    dataset_overview_stats, subject_distribution, year_distribution,
    topic_frequency, chapter_frequency, most_repeated_topics, apply_filters,
)
from feature_extraction import top_keywords_overall

st.set_page_config(**PAGE_CONFIG)
apply_page_style("Dashboard", "Live analytics calculated directly from the question dataset")

df = require_dataset()

# --- Filters --------------------------------------------------------------
st.subheader("Filters")
f1, f2, f3, f4 = st.columns(4)
with f1:
    year_opt = ["All"] + sorted(df["Year"].unique().tolist())
    sel_year = st.selectbox("Year", year_opt)
with f2:
    subj_opt = ["All"] + sorted(df["Subject"].unique().tolist())
    sel_subject = st.selectbox("Subject", subj_opt)
with f3:
    topic_opt = ["All"] + sorted(df["Topic"].unique().tolist())
    sel_topic = st.selectbox("Topic", topic_opt)
with f4:
    chapter_opt = ["All"] + sorted(df["Chapter"].unique().tolist())
    sel_chapter = st.selectbox("Chapter", chapter_opt)

filtered = apply_filters(df, year=sel_year, subject=sel_subject, topic=sel_topic, chapter=sel_chapter)

if filtered.empty:
    st.warning("No questions match the selected filters.")
    st.stop()

st.caption(f"Showing **{len(filtered)}** of {len(df)} questions after filters.")

# --- Stat cards -------------------------------------------------------------
stats = dataset_overview_stats(filtered)
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

# --- Charts -----------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Subject Distribution", "Year Distribution", "Topic Frequency",
    "Chapter Frequency", "Most Repeated Topics", "Top Keywords",
])

with tab1:
    sd = subject_distribution(filtered)
    fig = px.bar(sd, x="Subject", y="Count", color="Subject",
                 title="Subject-wise Question Distribution", text="Count")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    yd = year_distribution(filtered)
    yd["Year"] = yd["Year"].astype(str)
    fig = px.bar(yd, x="Year", y="Count", title="Year-wise Question Distribution", text="Count")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    tf = topic_frequency(filtered, top_n=15)
    fig = px.bar(tf.sort_values("Count"), x="Count", y="Topic", orientation="h",
                 title="Topic Frequency (Top 15)")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    cf = chapter_frequency(filtered, top_n=15)
    fig = px.bar(cf.sort_values("Count"), x="Count", y="Chapter", orientation="h",
                 title="Chapter Frequency (Top 15)")
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    mrt = most_repeated_topics(filtered, top_n=15)
    fig = px.bar(mrt.sort_values("Years_Appeared"), x="Years_Appeared", y="Topic",
                 orientation="h", title="Most Repeated Topics (by number of papers/years appeared in)")
    st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.markdown("TF-IDF keyword importance, computed live from the (filtered) question text.")
    if len(filtered) >= 2:
        vectorizer, matrix = build_live_tfidf(filtered)
        kw = top_keywords_overall(vectorizer, matrix, top_n=25)

        col_a, col_b = st.columns([1, 1.2])
        with col_a:
            st.dataframe(kw, use_container_width=True, hide_index=True)
        with col_b:
            freqs = dict(zip(kw["Keyword"], kw["Score"]))
            if freqs:
                wc = WordCloud(width=700, height=420, background_color="white",
                                colormap="Blues").generate_from_frequencies(freqs)
                fig_wc, ax = plt.subplots(figsize=(7, 4.2))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig_wc)
    else:
        st.info("Need at least 2 questions in the current filter to compute keywords.")

st.markdown("### ")
with st.expander("View filtered data table"):
    st.dataframe(
        filtered[["ID", "Year", "Subject", "Chapter", "Topic", "Question", "Marks"]],
        use_container_width=True, hide_index=True,
    )
