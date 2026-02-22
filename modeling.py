"""Modeling engine for BoW and TF-IDF vectorization.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import List, Dict, Any

class TextModeler:
    """Helper to perform BoW and TF-IDF analysis."""

    def __init__(self, method: str = "tfidf"):
        self.method = method.lower()
        if self.method == "bow":
            self.vectorizer = CountVectorizer()
        else:
            self.vectorizer = TfidfVectorizer()

    def fit_transform(self, documents: List[str]) -> Dict[str, Any]:
        """Transform documents into a list-of-dicts representation.
        
        Returns:
            A dictionary containing the vocabulary (features) and 
            the matrix values per document.
        """
        if not documents:
            return {"features": [], "data": []}

        # Perform vectorization
        matrix = self.vectorizer.fit_transform(documents)
        features = self.vectorizer.get_feature_names_out().tolist()
        
        # Convert to a DataFrame for easy manipulation
        df = pd.DataFrame(matrix.toarray(), columns=features)
        
        # Prepare response data: list of documents, each with its vector
        data = []
        for i, doc in enumerate(documents):
            # We round values for readability
            row_values = df.iloc[i].round(4).tolist()
            data.append({
                "text": doc,
                "vector": row_values
            })

        return {
            "method": self.method,
            "features": features,
            "data": data
        }

def analyze_modeling(documents: List[str], method: str = "tfidf") -> Dict[str, Any]:
    """One-off modeling analysis."""
    modeler = TextModeler(method=method)
    return modeler.fit_transform(documents)
