"""
tfidf_pipeline.py — Build and persist the TF-IDF feature matrices.

Pipeline:
  1. Load cleaned dataset
  2. Stratified train/test split (80/20)
  3. Fit TfidfVectorizer on training text
  4. Transform train + test → sparse matrices
  5. Persist all artifacts (vectorizer, X, y, keys) with joblib

Usage:
    python src/tfidf_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from config import (
    CLEANED_DATASET_PATH,
    MAX_TFIDF_FEATURES,
    NGRAM_RANGE,
    SUBLINEAR_TF,
    RANDOM_STATE,
    TEST_SIZE,
    TFIDF_VECTORIZER_PATH,
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_TEST_PATH,
    SPLIT_KEYS_TRAIN_PATH,
    SPLIT_KEYS_TEST_PATH,
)


def create_stratification_bins(y: pd.Series, n_bins: int = 5) -> pd.Series:
    """
    Bin continuous story-point values into quantile-based strata for
    stratified splitting.  Falls back to coarser bins if any quantile
    bin is too small.
    """
    try:
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        # If too few unique values, use simpler binning
        bins = pd.cut(y, bins=min(n_bins, y.nunique()), labels=False)
    return bins


def build_tfidf_pipeline():
    """Main pipeline: split → fit → transform → persist."""

    # ── 1. Load cleaned data ──────────────────────────────────────
    if not CLEANED_DATASET_PATH.exists():
        print(f"  [X] Cleaned dataset not found at {CLEANED_DATASET_PATH}")
        print("    Run `python src/clean_data.py` first.")
        sys.exit(1)

    df = pd.read_csv(CLEANED_DATASET_PATH)
    print(f"  Loaded {len(df):,} rows from {CLEANED_DATASET_PATH.name}")

    X_text = df["text"]
    y      = df["storypoint"].astype(np.float64)
    keys   = df["issuekey"]

    # ── 2. Stratified train/test split ────────────────────────────
    strata = create_stratification_bins(y)

    X_train_text, X_test_text, y_train, y_test, keys_train, keys_test = (
        train_test_split(
            X_text, y, keys,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=strata,
        )
    )

    print(f"  Train set: {len(X_train_text):,} rows")
    print(f"  Test set:  {len(X_test_text):,} rows")

    # ── 3. Fit TfidfVectorizer on training data ───────────────────
    print(f"\n  Fitting TfidfVectorizer (max_features={MAX_TFIDF_FEATURES}, "
          f"ngrams={NGRAM_RANGE}, sublinear_tf={SUBLINEAR_TF}) ...")

    vectorizer = TfidfVectorizer(
        max_features=MAX_TFIDF_FEATURES,
        stop_words="english",
        ngram_range=NGRAM_RANGE,
        sublinear_tf=SUBLINEAR_TF,
        dtype=np.float64,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf  = vectorizer.transform(X_test_text)

    print(f"  X_train shape: {X_train_tfidf.shape}  (sparse, "
          f"{X_train_tfidf.nnz:,} non-zero)")
    print(f"  X_test  shape: {X_test_tfidf.shape}  (sparse, "
          f"{X_test_tfidf.nnz:,} non-zero)")

    # ── 4. Report top features by IDF ─────────────────────────────
    feature_names = vectorizer.get_feature_names_out()
    idf_scores    = vectorizer.idf_
    top_idx       = np.argsort(idf_scores)[::-1][:20]

    print(f"\n  Top 20 features by IDF score:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"    {rank:2d}. {feature_names[idx]:30s}  IDF={idf_scores[idx]:.3f}")

    # Also show bottom 20 (most common terms that survived stop-word filter)
    bottom_idx = np.argsort(idf_scores)[:20]
    print(f"\n  Bottom 20 features by IDF (most frequent):")
    for rank, idx in enumerate(bottom_idx, 1):
        print(f"    {rank:2d}. {feature_names[idx]:30s}  IDF={idf_scores[idx]:.3f}")

    # ── 5. Persist artifacts ──────────────────────────────────────
    print(f"\n  Saving artifacts to {X_TRAIN_PATH.parent} ...")

    joblib.dump(vectorizer,    TFIDF_VECTORIZER_PATH)
    joblib.dump(X_train_tfidf, X_TRAIN_PATH)
    joblib.dump(X_test_tfidf,  X_TEST_PATH)
    joblib.dump(y_train.values, Y_TRAIN_PATH)
    joblib.dump(y_test.values,  Y_TEST_PATH)
    joblib.dump(keys_train.values, SPLIT_KEYS_TRAIN_PATH)
    joblib.dump(keys_test.values,  SPLIT_KEYS_TEST_PATH)

    print(f"  [+] tfidf_vectorizer.joblib  ({TFIDF_VECTORIZER_PATH.stat().st_size / 1024:.0f} KB)")
    print(f"  [+] X_train_tfidf.joblib     ({X_TRAIN_PATH.stat().st_size / 1024:.0f} KB)")
    print(f"  [+] X_test_tfidf.joblib      ({X_TEST_PATH.stat().st_size / 1024:.0f} KB)")
    print(f"  [+] y_train.joblib")
    print(f"  [+] y_test.joblib")
    print(f"  [+] keys_train.joblib")
    print(f"  [+] keys_test.joblib")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" TF-IDF Pipeline Complete")
    print("=" * 60)
    print(f"  Vocabulary size:  {len(feature_names):,}")
    print(f"  Train matrix:     {X_train_tfidf.shape}")
    print(f"  Test matrix:      {X_test_tfidf.shape}")
    print(f"  y_train range:    [{y_train.min():.0f}, {y_train.max():.0f}]")
    print(f"  y_test  range:    [{y_test.min():.0f}, {y_test.max():.0f}]")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print(" Federated Agile Effort Estimation")
    print(" TF-IDF Vectorization Pipeline")
    print("=" * 60)

    build_tfidf_pipeline()
