"""
test_pipeline.py — Smoke tests for the data pipeline.

Run with:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np
import joblib

from config import (
    CLEANED_DATASET_PATH,
    STORYPOINT_CAP,
    MAX_TFIDF_FEATURES,
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_TEST_PATH,
    TFIDF_VECTORIZER_PATH,
    SPLIT_KEYS_TRAIN_PATH,
    SPLIT_KEYS_TEST_PATH,
)


# ════════════════════════════════════════════════════════════════
#  Cleaned Dataset Tests
# ════════════════════════════════════════════════════════════════

class TestCleanedDataset:
    """Tests for data/processed/cleaned_dataset.csv."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        if not CLEANED_DATASET_PATH.exists():
            pytest.skip("Cleaned dataset not found. Run clean_data.py first.")
        self.df = pd.read_csv(CLEANED_DATASET_PATH)

    def test_cleaned_data_exists(self):
        """Cleaned CSV exists and is loadable."""
        assert CLEANED_DATASET_PATH.exists()
        assert len(self.df) > 0

    def test_expected_columns(self):
        """Must have the required columns."""
        required = {"issuekey", "title", "description", "storypoint", "project_id", "text"}
        assert required.issubset(set(self.df.columns)), (
            f"Missing columns: {required - set(self.df.columns)}"
        )

    def test_no_nulls_in_text(self):
        """The 'text' column must have no NaN values."""
        assert self.df["text"].isna().sum() == 0

    def test_no_nulls_in_storypoint(self):
        """The 'storypoint' column must have no NaN values."""
        assert self.df["storypoint"].isna().sum() == 0

    def test_storypoint_cap(self):
        """All story points must be ≤ STORYPOINT_CAP."""
        assert self.df["storypoint"].max() <= STORYPOINT_CAP

    def test_storypoint_positive(self):
        """All story points must be > 0."""
        assert self.df["storypoint"].min() > 0

    def test_no_duplicate_issuekeys(self):
        """No duplicate issuekeys in the cleaned data."""
        assert self.df["issuekey"].duplicated().sum() == 0

    def test_text_is_lowercase(self):
        """All text should be lowercased."""
        sample = self.df["text"].head(100)
        assert all(t == t.lower() for t in sample)

    def test_project_id_present(self):
        """project_id column must be present and non-null."""
        assert "project_id" in self.df.columns
        assert self.df["project_id"].isna().sum() == 0


# ════════════════════════════════════════════════════════════════
#  TF-IDF Feature Tests
# ════════════════════════════════════════════════════════════════

class TestTfidfFeatures:
    """Tests for the TF-IDF feature artifacts in data/features/."""

    @pytest.fixture(autouse=True)
    def load_artifacts(self):
        if not X_TRAIN_PATH.exists():
            pytest.skip("TF-IDF features not found. Run tfidf_pipeline.py first.")
        self.X_train = joblib.load(X_TRAIN_PATH)
        self.X_test  = joblib.load(X_TEST_PATH)
        self.y_train = joblib.load(Y_TRAIN_PATH)
        self.y_test  = joblib.load(Y_TEST_PATH)

    def test_tfidf_shapes(self):
        """Feature matrices should have MAX_TFIDF_FEATURES columns."""
        assert self.X_train.shape[1] == MAX_TFIDF_FEATURES
        assert self.X_test.shape[1]  == MAX_TFIDF_FEATURES

    def test_train_test_row_counts(self):
        """X and y must have matching row counts."""
        assert self.X_train.shape[0] == len(self.y_train)
        assert self.X_test.shape[0]  == len(self.y_test)

    def test_no_nan_in_features(self):
        """TF-IDF matrices should not contain NaN."""
        # Sparse matrices: convert a small slice to dense for checking
        dense_sample = self.X_train[:100].toarray()
        assert not np.isnan(dense_sample).any()

    def test_vectorizer_loadable(self):
        """The fitted TfidfVectorizer must load and have vocabulary."""
        vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        assert hasattr(vectorizer, "vocabulary_")
        assert len(vectorizer.vocabulary_) == MAX_TFIDF_FEATURES

    def test_train_test_no_overlap(self):
        """No issuekey leakage between train and test sets."""
        if not SPLIT_KEYS_TRAIN_PATH.exists():
            pytest.skip("Split keys not saved.")
        keys_train = set(joblib.load(SPLIT_KEYS_TRAIN_PATH))
        keys_test  = set(joblib.load(SPLIT_KEYS_TEST_PATH))
        overlap = keys_train & keys_test
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping issuekeys!"

    def test_y_values_capped(self):
        """All y values should respect the storypoint cap."""
        assert self.y_train.max() <= STORYPOINT_CAP
        assert self.y_test.max()  <= STORYPOINT_CAP
        assert self.y_train.min() > 0
        assert self.y_test.min()  > 0
