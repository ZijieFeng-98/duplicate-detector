"""
Utilities for applying trained classifiers to duplicate pairs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd


class PairClassifier:
    """Wrapper around a scikit-learn classifier and scaler."""

    def __init__(self, model_path: Path, scaler_path: Path, threshold: float = 0.5):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.threshold = threshold

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Compute classifier probability for each row."""
        features = df.copy()
        for col in ["Cosine_Similarity", "SSIM", "Hamming_Distance"]:
            features[col] = pd.to_numeric(features.get(col, 0), errors="coerce").fillna(0)
        X = features[["Cosine_Similarity", "SSIM", "Hamming_Distance"]].to_numpy()
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        return pd.Series(probs, index=df.index, dtype=float)

    def apply_threshold(self, scores: pd.Series) -> pd.Series:
        """Return boolean mask for scores >= threshold."""
        return scores >= self.threshold


def load_pair_classifier(model_path: Optional[Path], scaler_path: Optional[Path], threshold: float) -> Optional[PairClassifier]:
    """Instantiate classifier if both paths exist."""
    if not model_path or not scaler_path:
        return None
    if not model_path.exists() or not scaler_path.exists():
        return None
    return PairClassifier(model_path, scaler_path, threshold)

