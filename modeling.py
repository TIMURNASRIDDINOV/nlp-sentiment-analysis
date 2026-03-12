"""From-scratch Bag of Words (BoW) and TF-IDF text vectorization.

Implements both algorithms step-by-step with full intermediate results,
designed for educational demonstration of the underlying math.

Formulas implemented:
    BoW:      BoW(t, d) = count of term t in document d
    TF:       TF(t, d)  = count(t, d) / |d|
    DF:       DF(t)     = number of documents containing t
    IDF:      IDF(t)    = log(N / DF(t)) + 1
    TF-IDF:   TFIDF(t, d) = TF(t, d) × IDF(t)
    Cosine:   cos(A, B) = (A · B) / (‖A‖ × ‖B‖)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Stop-word lists (Uzbek + English)
# ---------------------------------------------------------------------------
UZ_STOP_WORDS = {
    "va", "bu", "u", "o", "bilan", "uchun", "da", "ham", "esa", "bir",
    "har", "hamma", "shu", "shunday", "bunday", "qanday", "nima", "kim",
    "qayerda", "qachon", "agar", "lekin", "ammo", "yoki", "chunki",
    "ning", "ga", "dan", "ni", "dagi", "dek", "kabi", "orqali",
    "keyin", "oldin", "shuning", "lar",
    "bo'lgan", "bo'ladi", "bo'lib", "bo'lsa", "o'z", "o'sha",
    "ko'p", "ko'ra", "to'g'ri", "bo'yicha", "qo'shma",
}

EN_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "can", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "not", "only",
    "so", "than", "too", "very", "just", "because", "but", "and", "or",
    "if", "while", "it", "its", "this", "that", "these", "those", "i",
    "me", "my", "we", "our", "you", "your", "he", "him", "his", "she",
    "her", "they", "them", "their", "what", "which", "who", "about", "up",
    "down",
}

STOP_WORDS = UZ_STOP_WORDS | EN_STOP_WORDS


# ---------------------------------------------------------------------------
# Step 1 – Tokenization & preprocessing
# ---------------------------------------------------------------------------
def _normalize_apostrophes(text: str) -> str:
    """Normalize curly/typographic apostrophes to straight ones.

    Uzbek Latin uses apostrophe in digraphs: o', g', sh, ch.
    Texts may contain \u2018 \u2019 \u02BB \u02BC or the standard '.
    """
    return re.sub(r"[\u2018\u2019\u02BB\u02BC\u0060\u00B4]", "'", text)


def tokenize(
    text: str,
    lowercase: bool = True,
    remove_stopwords: bool = True,
) -> List[str]:
    """Split text into word tokens with optional lowercasing and stop-word removal.

    Handles Uzbek apostrophe-containing words like O'zbekiston, bo'lgan, g'arb.
    """
    text = _normalize_apostrophes(text)
    if lowercase:
        text = text.lower()
    tokens = re.findall(r"\b\w+(?:['\u2019]\w+)*\b", text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


# ---------------------------------------------------------------------------
# Step 2 – Vocabulary
# ---------------------------------------------------------------------------
def build_vocabulary(tokenized_docs: List[List[str]]) -> List[str]:
    """Build a sorted vocabulary from all tokens across documents."""
    return sorted({t for doc in tokenized_docs for t in doc})


# ---------------------------------------------------------------------------
# Step 3 – Bag of Words (raw term counts)
# ---------------------------------------------------------------------------
def compute_bow_matrix(
    tokenized_docs: List[List[str]],
    vocabulary: List[str],
) -> List[List[int]]:
    """BoW(t, d) = count of term t in document d."""
    matrix = []
    for tokens in tokenized_docs:
        counter = Counter(tokens)
        matrix.append([counter.get(term, 0) for term in vocabulary])
    return matrix


# ---------------------------------------------------------------------------
# Step 4 – Term Frequency (TF)
# ---------------------------------------------------------------------------
def compute_tf_matrix(
    tokenized_docs: List[List[str]],
    vocabulary: List[str],
) -> List[List[float]]:
    """TF(t, d) = count(t, d) / total_tokens(d)."""
    matrix = []
    for tokens in tokenized_docs:
        counter = Counter(tokens)
        total = len(tokens) if tokens else 1
        matrix.append([round(counter.get(t, 0) / total, 4) for t in vocabulary])
    return matrix


# ---------------------------------------------------------------------------
# Step 5 – Document Frequency (DF) & Inverse Document Frequency (IDF)
# ---------------------------------------------------------------------------
def compute_df(
    tokenized_docs: List[List[str]],
    vocabulary: List[str],
) -> Dict[str, int]:
    """DF(t) = number of documents that contain term t."""
    return {
        t: sum(1 for doc in tokenized_docs if t in set(doc))
        for t in vocabulary
    }


def compute_idf(df: Dict[str, int], n_docs: int) -> Dict[str, float]:
    """IDF(t) = log(N / DF(t)) + 1  (standard smoothing)."""
    return {t: round(math.log(n_docs / count) + 1, 4) for t, count in df.items()}


# ---------------------------------------------------------------------------
# Step 6 – TF-IDF matrix
# ---------------------------------------------------------------------------
def compute_tfidf_matrix(
    tf_matrix: List[List[float]],
    idf: Dict[str, float],
    vocabulary: List[str],
) -> List[List[float]]:
    """TF-IDF(t, d) = TF(t, d) × IDF(t)."""
    matrix = []
    for tf_row in tf_matrix:
        matrix.append([
            round(tf * idf[vocabulary[j]], 4) for j, tf in enumerate(tf_row)
        ])
    return matrix


# ---------------------------------------------------------------------------
# Step 7 – L2 normalization
# ---------------------------------------------------------------------------
def l2_normalize(matrix: List[List[float]]) -> List[List[float]]:
    """Normalize each row vector to unit length: v / ‖v‖."""
    result = []
    for row in matrix:
        norm = math.sqrt(sum(v ** 2 for v in row))
        if norm > 0:
            result.append([round(v / norm, 4) for v in row])
        else:
            result.append(row[:])
    return result


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------
def cosine_similarity_matrix(matrix: List[List[float]]) -> List[List[float]]:
    """Pairwise cosine similarity: cos(A, B) = (A·B) / (‖A‖×‖B‖)."""
    n = len(matrix)
    sim: List[List[float]] = []
    for i in range(n):
        row: List[float] = []
        for j in range(n):
            dot = sum(a * b for a, b in zip(matrix[i], matrix[j]))
            ni = math.sqrt(sum(a ** 2 for a in matrix[i]))
            nj = math.sqrt(sum(b ** 2 for b in matrix[j]))
            row.append(round(dot / (ni * nj), 4) if ni > 0 and nj > 0 else 0.0)
        sim.append(row)
    return sim


# ---------------------------------------------------------------------------
# Public API – full pipeline with intermediate steps
# ---------------------------------------------------------------------------
def analyze_modeling(
    documents: List[str],
    method: str = "tfidf",
    lowercase: bool = True,
    remove_stopwords: bool = True,
) -> Dict[str, Any]:
    """Run the full BoW or TF-IDF pipeline and return every intermediate step.

    Returns a dict suitable for direct JSON serialisation with keys:
        method, features, data, steps, similarity_matrix, documents
    """
    tokenized = [tokenize(doc, lowercase, remove_stopwords) for doc in documents]
    vocabulary = build_vocabulary(tokenized)
    bow = compute_bow_matrix(tokenized, vocabulary)

    steps: Dict[str, Any] = {
        "original_texts": documents,
        "tokenized_docs": tokenized,
        "vocabulary": vocabulary,
        "bow_matrix": bow,
    }

    if method == "bow":
        final_matrix = bow
        float_matrix = [[float(v) for v in row] for row in bow]
    else:
        tf = compute_tf_matrix(tokenized, vocabulary)
        df_vals = compute_df(tokenized, vocabulary)
        idf_vals = compute_idf(df_vals, len(documents))
        tfidf_raw = compute_tfidf_matrix(tf, idf_vals, vocabulary)
        tfidf_norm = l2_normalize(tfidf_raw)

        steps["tf_matrix"] = tf
        steps["df"] = df_vals
        steps["idf"] = idf_vals
        steps["tfidf_raw"] = tfidf_raw
        steps["tfidf_normalized"] = tfidf_norm

        final_matrix = tfidf_norm
        float_matrix = tfidf_norm

    similarity = cosine_similarity_matrix(float_matrix)

    data = [
        {"text": doc, "vector": final_matrix[i]}
        for i, doc in enumerate(documents)
    ]

    return {
        "method": method,
        "features": vocabulary,
        "data": data,
        "steps": steps,
        "similarity_matrix": similarity,
        "documents": documents,
    }
