"""
6_Upload_Paper.py
--------------------
Upload a PDF question paper, extract questions with pdfplumber,
preview and tag them, then optionally append them to the dataset
(never silently overwriting the original file -- a timestamped
backup of the previous dataset is kept in dataset/raw/).
"""

import os
import sys
import shutil
import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import streamlit as st
import pandas as pd

from app_utils import PAGE_CONFIG, apply_page_style, get_dataset, DATASET_PATH
from pdf_extraction import (
    pdfplumber_available, extract_text_from_pdf, detect_questions, build_preview_dataframe,
)

st.set_page_config(**PAGE_CONFIG)
apply_page_style("Upload Question Paper", "Extract questions from a PDF and add them to the dataset")

if not pdfplumber_available():
    st.error(
        "The `pdfplumber` package is not installed in this environment. "
        "Install it with `pip install pdfplumber` and restart the app to use this page."
    )
    st.stop()

st.markdown(
    """
    <div class="info-box">
    Upload a scanned or digital PDF question paper. The system extracts text with
    <b>pdfplumber</b>, detects individual numbered questions, and shows a preview
    table for you to review and tag (Subject / Chapter / Topic / Marks) before
    anything is added to the dataset.
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("### ")

uploaded_file = st.file_uploader("Upload PDF question paper", type=["pdf"])

if "extracted_questions_df" not in st.session_state:
    st.session_state.extracted_questions_df = None

if uploaded_file is not None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        year = st.number_input("Year", min_value=1990, max_value=2100,
                                value=datetime.datetime.now().year, step=1)
    with c2:
        existing_df = get_dataset()
        subj_options = sorted(existing_df["Subject"].unique().tolist()) if existing_df is not None else []
        subject = st.selectbox("Subject", subj_options + ["Other (type below)"])
        if subject == "Other (type below)":
            subject = st.text_input("Enter subject name", value="")
    with c3:
        chapter = st.text_input("Default Chapter (optional)", value="Unknown")
    with c4:
        marks = st.number_input("Default marks per question", min_value=0, max_value=100, value=4, step=1)

    extract_clicked = st.button("📄 Extract Questions from PDF", type="primary")

    if extract_clicked:
        with st.spinner("Extracting text from PDF..."):
            try:
                raw_text = extract_text_from_pdf(uploaded_file.getvalue())
            except Exception as e:
                st.error(f"Failed to read PDF: {e}")
                raw_text = ""

        if not raw_text.strip():
            st.warning(
                "No extractable text was found in this PDF. It may be a scanned "
                "image without an OCR text layer -- pdfplumber can only read "
                "text-based PDFs."
            )
        else:
            questions = detect_questions(raw_text)
            if not questions:
                st.warning("Could not detect individual questions in the extracted text.")
            else:
                preview_df = build_preview_dataframe(
                    questions, year=int(year), subject=subject or "Unknown",
                    chapter=chapter or "Unknown", marks=int(marks),
                )
                st.session_state.extracted_questions_df = preview_df
                st.success(f"Extracted {len(preview_df)} question(s). Review them below before saving.")

if st.session_state.extracted_questions_df is not None:
    st.markdown("### Preview extracted questions")
    edited_df = st.data_editor(
        st.session_state.extracted_questions_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "ID": st.column_config.NumberColumn(disabled=True),
        },
        key="preview_editor",
    )

    save_clicked = st.button("💾 Add these questions to the dataset")

    if save_clicked:
        existing_df = get_dataset()
        if existing_df is None:
            base_columns = ["ID", "Year", "Subject", "Question_No", "Chapter", "Question", "Topic", "Marks"]
            existing_raw = pd.DataFrame(columns=base_columns)
        else:
            existing_raw = pd.read_csv(DATASET_PATH)

        # Never silently overwrite -- back up the current dataset first.
        os.makedirs(os.path.join(os.path.dirname(DATASET_PATH), "raw"), exist_ok=True)
        backup_path = os.path.join(
            os.path.dirname(DATASET_PATH), "raw",
            f"questions_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        if os.path.exists(DATASET_PATH):
            shutil.copy(DATASET_PATH, backup_path)

        new_rows = edited_df.copy()
        next_id = (existing_raw["ID"].max() + 1) if (len(existing_raw) and existing_raw["ID"].notna().any()) else 1
        new_rows["ID"] = range(int(next_id), int(next_id) + len(new_rows))

        combined = pd.concat([existing_raw, new_rows], ignore_index=True)
        combined.to_csv(DATASET_PATH, index=False)

        st.success(
            f"Added {len(new_rows)} question(s) to the dataset "
            f"(backup of the previous dataset saved to `{os.path.relpath(backup_path)}`)."
        )
        st.info("Reload the app (or navigate to another page and back) to see updated stats and retrained models.")
        st.session_state.extracted_questions_df = None
        st.cache_data.clear()
        st.cache_resource.clear()
