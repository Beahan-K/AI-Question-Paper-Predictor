"""
similarity.py
--------------
Finds repeated / highly similar questions across the dataset using
TF-IDF + cosine similarity, so the "Question Analysis" page can show
which questions have effectively been asked more than once.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_question_pairs(df: pd.DataFrame, vectorizer, tfidf_matrix,
                                 threshold: float = 0.6, max_pairs: int = 100) -> pd.DataFrame:
    """
    Compute pairwise cosine similarity between all questions in `df`
    (whose rows must align 1:1 with the rows of `tfidf_matrix`) and
    return pairs above `threshold`, sorted by similarity descending.

    Returns columns: Question_A, Question_B, Year_A, Year_B,
    Subject_A, Subject_B, Similarity (0-100, %).
    """
    n = tfidf_matrix.shape[0]
    if n < 2:
        return pd.DataFrame(columns=[
            "Question_A", "Question_B", "Year_A", "Year_B",
            "Subject_A", "Subject_B", "Similarity"
        ])

    sim_matrix = cosine_similarity(tfidf_matrix)
    df = df.reset_index(drop=True)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            score = sim_matrix[i, j]
            if score >= threshold:
                pairs.append({
                    "Question_A": df.loc[i, "Question"],
                    "Question_B": df.loc[j, "Question"],
                    "Year_A": df.loc[i, "Year"],
                    "Year_B": df.loc[j, "Year"],
                    "Subject_A": df.loc[i, "Subject"],
                    "Subject_B": df.loc[j, "Subject"],
                    "Similarity": round(float(score) * 100, 1),
                })

    result = pd.DataFrame(pairs)
    if result.empty:
        return result
    result = result.sort_values("Similarity", ascending=False).head(max_pairs).reset_index(drop=True)
    return result


def search_questions(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Simple case-insensitive substring search across question text."""
    if not query or not query.strip():
        return df
    mask = df["Question"].astype(str).str.contains(query, case=False, na=False, regex=False)
    return df[mask]


def most_similar_to_query(df: pd.DataFrame, vectorizer, tfidf_matrix, query: str,
                           top_n: int = 10) -> pd.DataFrame:
    """
    Given a free-text query, clean it with the same pipeline used to
    build tfidf_matrix's vocabulary, and return the top_n most similar
    historical questions with their similarity score.
    """
    from preprocessing import preprocess_text

    cleaned_query = preprocess_text(query)
    if not cleaned_query:
        return pd.DataFrame(columns=list(df.columns) + ["Similarity"])

    query_vec = vectorizer.transform([cleaned_query])
    scores = cosine_similarity(query_vec, tfidf_matrix).ravel()

    result = df.reset_index(drop=True).copy()
    result["Similarity"] = (scores * 100).round(1)
    result = result.sort_values("Similarity", ascending=False).head(top_n)
    return result[result["Similarity"] > 0].reset_index(drop=True)
