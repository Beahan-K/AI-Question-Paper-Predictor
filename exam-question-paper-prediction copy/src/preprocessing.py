"""
preprocessing.py
-----------------
Reusable NLP preprocessing utilities for the Exam Question Paper
Prediction project.

Pipeline implemented in `preprocess_text`:
    1. Lowercase conversion
    2. Punctuation removal
    3. Number removal
    4. Tokenization
    5. Stop-word removal
    6. Lemmatization
    7. Return cleaned text (a single space-joined string)

The module tries to use NLTK (as required by the project spec). If the
required NLTK data packages are not yet downloaded, it will attempt to
download them automatically. If that also fails (e.g. no internet
access on the machine running the app), it silently falls back to a
small built-in English stop-word list and a no-op lemmatizer, so the
application never crashes just because NLTK data is missing.
"""

import re
import string
import pandas as pd

# ---------------------------------------------------------------------
# Try to set up NLTK. Fall back gracefully if it is unavailable.
# ---------------------------------------------------------------------
_NLTK_READY = False
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    def _ensure_nltk_data():
        packages = [
            ("tokenizers/punkt", "punkt"),
            ("tokenizers/punkt_tab", "punkt_tab"),
            ("corpora/stopwords", "stopwords"),
            ("corpora/wordnet", "wordnet"),
            ("corpora/omw-1.4", "omw-1.4"),
        ]
        for path, pkg in packages:
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass

    _ensure_nltk_data()

    _STOPWORDS = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()
    _NLTK_READY = True
except Exception:
    _NLTK_READY = False

# ---------------------------------------------------------------------
# Fallback stop-word list (used only if NLTK/its data is unavailable)
# ---------------------------------------------------------------------
_FALLBACK_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "you", "your", "yours", "yourself",
    "yourselves", "is", "was", "were", "been", "being",
}


def _simple_tokenize(text: str):
    return text.split()


def preprocess_text(text) -> str:
    """
    Clean a single piece of question text.

    Steps: lowercase -> remove punctuation -> remove numbers ->
    tokenize -> remove stop words -> lemmatize -> rejoin.

    Handles None / NaN / empty input gracefully by returning "".
    """
    if text is None:
        return ""
    if isinstance(text, float) and pd.isna(text):
        return ""
    text = str(text).strip()
    if not text:
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 3. Remove numbers (standalone digits) -- keep alphabetic tokens
    text = re.sub(r"\d+", " ", text)

    # collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    # 4. Tokenize
    if _NLTK_READY:
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = _simple_tokenize(text)
    else:
        tokens = _simple_tokenize(text)

    # 5. Remove stop words (+ any leftover single-character tokens)
    stop_set = _STOPWORDS if _NLTK_READY else _FALLBACK_STOPWORDS
    tokens = [t for t in tokens if t not in stop_set and len(t) > 1]

    # 6. Lemmatize
    if _NLTK_READY:
        try:
            tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
        except Exception:
            pass

    # 7. Return cleaned text
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame, text_column: str = "Question",
                          output_column: str = "Cleaned_Question") -> pd.DataFrame:
    """
    Apply preprocessing to an entire dataframe of questions, and handle
    common data-quality issues:
      - missing values in the text column
      - empty questions after cleaning
      - exact duplicate questions
      - completely invalid / blank rows

    Returns a new dataframe with an added `output_column`, duplicates
    dropped, and invalid rows removed. Does not mutate the input df.
    """
    df = df.copy()

    # Handle missing values in required columns
    required_cols = ["Question"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Dataset is missing required column: '{col}'")

    # Drop rows where the question text itself is null/blank
    df[text_column] = df[text_column].astype(str)
    df = df[df[text_column].str.strip().ne("")]
    df = df[df[text_column].str.lower().ne("nan")]

    # Fill other missing metadata with sensible defaults instead of dropping
    for col, default in [("Chapter", "Unknown"), ("Topic", "Unknown"),
                          ("Subject", "Unknown"), ("Marks", 0), ("Year", 0),
                          ("Question_No", 0)]:
        if col in df.columns:
            df[col] = df[col].fillna(default)
        else:
            df[col] = default

    # Remove exact duplicate questions (keep first occurrence)
    df = df.drop_duplicates(subset=[text_column], keep="first")

    # Clean text
    df[output_column] = df[text_column].apply(preprocess_text)

    # Drop rows that became empty after cleaning (e.g. question was just numbers/punctuation)
    df = df[df[output_column].str.strip().ne("")]

    df = df.reset_index(drop=True)
    if "ID" not in df.columns or df["ID"].isna().any():
        df["ID"] = range(1, len(df) + 1)

    return df


def nltk_status() -> bool:
    """Expose whether NLTK + its data loaded successfully (used by the UI)."""
    return _NLTK_READY
