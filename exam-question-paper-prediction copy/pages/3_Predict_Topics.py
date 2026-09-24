"""
3_Predict_Topics.py
---------------------
Predict Important Topics page: lets the user pick a subject and see
the topics the system estimates are most important, based on a real
blend of historical frequency and trained-model confidence.
"""

import os
import sys
import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import streamlit as st
import plotly.express as px

from app_utils import (
    PAGE_CONFIG, apply_page_style, require_dataset,
    get_trained_artifacts, get_dataset_signature,
)
from prediction import predict_important_topics

st.set_page_config(**PAGE_CONFIG)
apply_page_style("Predict Important Topics", "Historical-frequency + ML-confidence based topic importance estimation")

df = require_dataset()

with st.spinner("Loading trained model..."):
    artifacts = get_trained_artifacts(get_dataset_signature())

st.markdown(
    """
    <div class="warn-box">
    ⚠️ These are <b>importance / confidence scores</b> derived from historical
    patterns and model analysis — <b>not</b> a guarantee that a topic will
    appear in the next examination.
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("### ")

c1, c2, c3 = st.columns(3)
with c1:
    subject = st.selectbox("Subject", ["All"] + sorted(df["Subject"].unique().tolist()))
with c2:
    next_year = datetime.datetime.now().year + 1
    year_choice = st.selectbox("Year", [next_year] + sorted(df["Year"].unique().tolist(), reverse=True))
with c3:
    top_n = st.slider("Number of topics", min_value=5, max_value=25, value=10)

blend = st.slider(
    "Weighting: Historical frequency ⟷ Model confidence", min_value=0, max_value=100, value=60,
    help="100 = rely fully on how often a topic has appeared historically. "
         "0 = rely fully on the trained ML model's confidence in the topic's vocabulary.",
)
freq_weight = blend / 100.0

predict_clicked = st.button("🔮 Predict Important Topics", type="primary")

if predict_clicked:
    with st.spinner("Analyzing historical frequency and model confidence..."):
        predictions = predict_important_topics(
            df, artifacts["vectorizer"], artifacts["best_model"],
            subject=subject, top_n=top_n, freq_weight=freq_weight,
        )

    if predictions.empty:
        st.warning("Not enough data to generate predictions for this selection.")
    else:
        st.success(f"Predicted important topics for **{subject}** — {year_choice} (using model: {artifacts['best_model_name']})")

        st.subheader("Predicted Important Topics")
        for i, row in predictions.reset_index(drop=True).iterrows():
            with st.container(border=True):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**{i + 1}. {row['Topic']}**")
                    st.caption(f"Chapter: {row['Chapter']}  ·  Historical occurrences (weighted): {row['Frequency']:.1f}")
                with col_b:
                    st.metric("Importance Score", f"{row['Importance Score']:.0f}%")
                st.progress(min(row["Importance Score"], 100) / 100.0)

        st.markdown("### ")
        fig = px.bar(
            predictions.sort_values("Importance Score"),
            x="Importance Score", y="Topic", orientation="h",
            title="Importance Score by Topic", text="Importance Score",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Importance Score = weighted blend of historical topic frequency "
            f"({int(freq_weight * 100)}%) and trained-model confidence "
            f"({int((1 - freq_weight) * 100)}%), min-max normalized to 0–100."
        )
else:
    st.info("Choose a subject and click **Predict Important Topics** to generate results.")
