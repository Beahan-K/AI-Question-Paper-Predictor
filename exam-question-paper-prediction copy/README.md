# Exam Question Paper Prediction Using Machine Learning

An AI-powered web application that analyzes previous-year examination
question papers using **Natural Language Processing (NLP)**, **TF-IDF**,
and **Machine Learning** to identify frequently asked topics, repeated
questions, important keywords, year-wise trends, and to estimate which
topics are likely to be important in the next examination.

> ⚠️ **Disclaimer:** Predictions are based on historical patterns and
> are shown as analytical *importance/confidence scores*. This system
> does **not** guarantee the exact questions that will appear in any
> future examination.

---

## 1. Problem Statement

Students preparing for competitive/academic exams often have to sift
through years of past papers by hand to identify which topics are
repeatedly tested. This project automates that analysis using NLP and
supervised machine learning, turning unstructured question text into
structured, data-driven trend insights.

## 2. Objective

- Analyze a structured dataset of previous-year exam questions.
- Preprocess question text using standard NLP techniques.
- Convert text into numerical features using TF-IDF.
- Train and compare multiple ML classifiers.
- Use historical frequency + model confidence to estimate topic
  importance for an upcoming paper.
- Present everything through an interactive, multi-page dashboard.

## 3. Features

- 📊 **Dashboard** — live dataset statistics and interactive charts
  (subject/year/topic/chapter distribution, most repeated topics, top
  keywords), all calculated from the CSV — nothing hard-coded.
- 🔍 **Question Analysis** — keyword search (exact or TF-IDF "smart"
  search), multi-field filtering, and detection of repeated/near-duplicate
  questions via cosine similarity.
- 🔮 **Predict Important Topics** — pick a subject, and get a ranked list
  of topics with an Importance Score blending historical frequency and
  trained-model confidence.
- 📈 **Model Performance** — real accuracy / precision / recall / F1 for
  Naive Bayes, Logistic Regression, and SVM, plus confusion matrices and
  the automatically-selected best model.
- 🧮 **TF-IDF Analysis** — an educational walkthrough of the TF-IDF
  pipeline with live top-keyword extraction and a word cloud.
- 📄 **Upload Question Paper** — upload a PDF, extract questions with
  `pdfplumber`, review/tag them, and append to the dataset (the original
  CSV is always backed up first, never overwritten silently).

## 4. Technology Stack

| Layer            | Tools |
|-------------------|-------|
| Backend / ML       | Python, Pandas, NumPy, Scikit-learn, Joblib |
| NLP                | NLTK (tokenization, stop-words, lemmatization) |
| Feature Extraction | TF-IDF (unigrams + bigrams) |
| ML Models          | Multinomial Naive Bayes, Logistic Regression, SVM |
| Frontend           | Streamlit (multi-page app) |
| Visualization      | Plotly, Matplotlib, WordCloud |
| PDF Processing     | pdfplumber |

## 5. System Architecture

```
                    USER
                     |
                     v
              Streamlit UI
                     |
        +------------+------------+
        |                         |
        v                         v
   Dataset Analysis         Prediction Input
        |                         |
        v                         v
  NLP Preprocessing         Text Preprocessing
        |                         |
        v                         v
      TF-IDF <--------------------+
        |
        v
   ML Prediction Model
        |
        v
 Topic / Trend Prediction
        |
        v
 Visualization Dashboard
```

## 6. Dataset Description

The prototype dataset is a **synthetically generated, NEET-style sample**
(`dataset/generate_sample_dataset.py`) covering Physics, Chemistry, and
Biology across two example papers (2019, 2022) — ~360 rows — so the app
runs end-to-end out of the box. It uses a generic schema so real datasets
(including university-level Machine Learning question papers) can be
swapped in without changing any application code:

```
ID, Year, Subject, Question_No, Chapter, Question, Topic, Marks
```

To regenerate the sample dataset:

```bash
python dataset/generate_sample_dataset.py
```

To use your own data, replace `dataset/questions.csv` with a CSV using
the same column headers (or use the **Upload Question Paper** page to
append PDF-extracted questions).

## 7. Methodology

1. **Data Cleaning** — handle missing values, blank/invalid rows, and
   duplicate questions (`src/preprocessing.py`).
2. **NLP Preprocessing** — lowercase → remove punctuation/numbers →
   tokenize → remove stop-words → lemmatize.
3. **TF-IDF Feature Extraction** — unigrams + bigrams, up to 3000
   features (`src/feature_extraction.py`).
4. **Model Training** — Multinomial Naive Bayes, Logistic Regression,
   and SVM are trained on an 80/20 stratified split to classify each
   question's **Chapter** (a coarser, better-populated label than raw
   Topic, which keeps evaluation statistically meaningful even on small
   datasets — see `src/train_model.py` for how to switch to Topic-level
   training on a larger dataset).
5. **Model Evaluation** — accuracy, precision, recall, F1, and confusion
   matrices are computed on the held-out test set; the highest-F1 model
   is selected automatically.
6. **Prediction** — `src/prediction.py` blends recency-weighted historical
   topic frequency with the trained model's confidence in each topic's
   vocabulary into a single 0–100 Importance Score.
7. **Similarity Detection** — `src/similarity.py` uses TF-IDF cosine
   similarity to flag repeated/near-duplicate questions.

## 8. Installation

```bash
# 1. Clone / unzip the project, then move into it
cd exam-question-paper-prediction

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate the sample dataset (skip if you already have dataset/questions.csv)
python dataset/generate_sample_dataset.py
```

The first time NLTK preprocessing runs, it will attempt to download the
`punkt`, `stopwords`, and `wordnet` data packages automatically. If your
machine has no internet access at that point, the app automatically
falls back to a built-in stop-word list so it still runs correctly
(lemmatization is simply skipped in that case).

## 9. Usage

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`)
in your browser. Use the sidebar to move between Home, Dashboard,
Question Analysis, Predict Topics, Model Performance, TF-IDF Analysis,
and Upload Question Paper.

On first run, the app automatically trains all three ML models from
`dataset/questions.csv` and caches the vectorizer + best model to
`models/`. Subsequent runs load the cached artifacts instantly; use the
**"Retrain models now"** button on the Model Performance page (or delete
the files in `models/`) to force retraining, e.g. after updating the
dataset.

## 10. Model Evaluation (example)

Actual numbers are calculated live and will vary slightly with the
dataset and split — no results are hard-coded. On the bundled sample
dataset, a typical run looks like:

| Model               | Accuracy | Precision | Recall | F1 Score |
|----------------------|---------:|----------:|-------:|---------:|
| Naive Bayes          |    ~0.53 |     ~0.55 |  ~0.53 |    ~0.54 |
| Logistic Regression  |    ~0.71 |     ~0.72 |  ~0.71 |    ~0.71 |
| SVM                  |    ~0.77 |     ~0.79 |  ~0.77 |    ~0.76 |

## 11. Screenshots

_Add screenshots of the Home, Dashboard, Predict Topics, and Model
Performance pages here once you've run the app locally._

## 12. Future Enhancements

- Swap in a real university Machine Learning question-paper dataset
  (schema already supports `Unit` / `Question_Type` via the generic
  columns described in the project spec).
- Add OCR support (e.g. Tesseract) for scanned PDF question papers with
  no text layer.
- Add authentication and multi-user dataset management.
- Persist trained models per-dataset-version instead of overwriting.
- Add topic-level (rather than chapter-level) classification once more
  data is available per topic.

## 13. Project File Structure

```
exam-question-paper-prediction/
│
├── dataset/
│   ├── questions.csv                 # main dataset (generic schema)
│   ├── generate_sample_dataset.py    # builds the sample NEET dataset
│   ├── processed_questions.csv       # cached cleaned dataset (auto-generated)
│   └── raw/                          # timestamped backups from PDF uploads
│
├── models/                           # cached vectorizer + best model (auto-generated)
│   ├── tfidf_vectorizer.pkl
│   └── best_model.pkl
│
├── src/
│   ├── app_utils.py                  # shared Streamlit helpers/caching/styling
│   ├── preprocessing.py              # NLP cleaning pipeline
│   ├── pdf_extraction.py             # PDF -> question extraction
│   ├── feature_extraction.py         # TF-IDF utilities
│   ├── train_model.py                # model training + comparison
│   ├── prediction.py                 # topic importance prediction
│   ├── similarity.py                 # repeated/similar question detection
│   └── analysis.py                   # dataset statistics for the dashboard
│
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Question_Analysis.py
│   ├── 3_Predict_Topics.py
│   ├── 4_Model_Performance.py
│   ├── 5_TF_IDF_Analysis.py
│   └── 6_Upload_Paper.py
│
├── app.py                            # Home page / entry point
├── requirements.txt
├── README.md
└── .gitignore
```

## 14. Team Members

_Add your name(s) and roll number(s) / registration details here._

---

*This is an academic Machine Learning project. All predictions are
analytical estimates based on historical data and should be used as
study guidance, not as a guarantee of future exam content.*
