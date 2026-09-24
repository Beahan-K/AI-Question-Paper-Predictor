"""
4_Model_Performance.py
------------------------
Displays real accuracy/precision/recall/F1 and confusion matrices for
all three trained models, computed on a held-out test split -- never
hard-coded.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app_utils import PAGE_CONFIG, apply_page_style, require_dataset, get_trained_artifacts, get_dataset_signature
from train_model import train_if_needed

st.set_page_config(**PAGE_CONFIG)
apply_page_style("Model Performance", "Real accuracy, precision, recall, and F1 score — calculated after training, never invented")

df = require_dataset()

col_retrain, _ = st.columns([1, 3])
with col_retrain:
    retrain = st.button("🔄 Retrain models now")

if retrain:
    with st.spinner("Retraining Naive Bayes, Logistic Regression, and SVM..."):
        artifacts = train_if_needed(force=True)
    st.success("Models retrained.")
else:
    with st.spinner("Loading trained models..."):
        artifacts = get_trained_artifacts(get_dataset_signature())

results = artifacts["results"]

# --- Comparison table -------------------------------------------------
rows = []
for name, metrics in results.items():
    rows.append({
        "Model": name,
        "Accuracy": round(metrics["Accuracy"], 3),
        "Precision": round(metrics["Precision"], 3),
        "Recall": round(metrics["Recall"], 3),
        "F1 Score": round(metrics["F1 Score"], 3),
    })
comparison_df = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)

st.subheader("Model Comparison")
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

fig = px.bar(
    comparison_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
    x="Model", y="Score", color="Metric", barmode="group",
    title="Model Comparison across Metrics",
)
st.plotly_chart(fig, use_container_width=True)

# --- Best model ----------------------------------------------------------
best_name = artifacts["best_model_name"]
best_f1 = results[best_name]["F1 Score"]

st.markdown("### ")
st.markdown(
    f"""
    <div class="info-box">
    🏆 <b>Best Performing Model</b><br>
    <span style="font-size:1.3rem; font-weight:700; color:#0f2340;">{best_name}</span><br>
    F1 Score: <b>{best_f1:.2f}</b>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Confusion matrices -----------------------------------------------------
st.markdown("### ")
st.subheader("Confusion Matrices")
st.caption("Rows = actual class, Columns = predicted class (classifier target: Chapter).")

model_names = list(results.keys())
tabs = st.tabs(model_names)
for tab, name in zip(tabs, model_names):
    with tab:
        cm = results[name]["Confusion Matrix"]
        labels = results[name]["Labels"]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=labels, y=labels, colorscale="Blues", showscale=True,
        ))
        fig_cm.update_layout(
            title=f"{name} — Confusion Matrix",
            xaxis_title="Predicted", yaxis_title="Actual",
            height=max(500, 22 * len(labels)),
        )
        st.plotly_chart(fig_cm, use_container_width=True)
