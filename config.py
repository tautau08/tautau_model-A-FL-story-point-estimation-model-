"""
config.py — Centralized configuration for the Federated Agile Effort Estimation project.

All paths, hyperparameters, and constants are defined here so that every
module in the pipeline reads from a single source of truth.
"""

from pathlib import Path

# ============================================================
# Directory Layout
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DATA_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR       = PROJECT_ROOT / "data" / "features"
BASELINE_MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
FEDERATED_DATA_DIR = PROJECT_ROOT / "data" / "federated"

# Ensure directories exist on import
for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, BASELINE_MODEL_DIR, FEDERATED_DATA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Dataset Configuration
# ============================================================
# The 16 open-source JIRA projects from the Choetkiertikul et al. benchmark.
# Each key is the short project slug; value is the CSV filename on GitHub.
PROJECTS = [
    "appceleratorstudio",
    "aptanastudio",
    "bamboo",
    "clover",
    "datamanagement",
    "duracloud",
    "jirasoftware",
    "mesos",
    "moodle",
    "mule",
    "mulestudio",
    "springxd",
    "talenddataquality",
    "talendesb",
    "titanium",
    "usergrid",
]

# Primary and fallback GitHub raw URLs for downloading the per-project CSVs.
# Each URL template takes a project slug via .format(project=...).
DATASET_URL_TEMPLATES = [
    # Primary: morakotch fork — IEEE TSE2018 dataset directory
    "https://raw.githubusercontent.com/morakotch/datasets/master/storypoint/IEEE%20TSE2018/dataset/{project}.csv",
    # Fallback 1: morakotch fork — Deep-SE data directory
    "https://raw.githubusercontent.com/morakotch/datasets/master/storypoint/IEEE%20TSE2018/Deep-SE/data/{project}.csv",
]

# Cleaned / processed data file
CLEANED_DATASET_PATH = PROCESSED_DATA_DIR / "cleaned_dataset.csv"

# ============================================================
# Preprocessing Hyperparameters
# ============================================================
STORYPOINT_CAP = 100          # Max SP value (Khattab et al. threshold)
RANDOM_STATE   = 42           # Reproducibility seed
TEST_SIZE      = 0.2          # 80/20 train-test split

# ============================================================
# TF-IDF Vectorizer Settings
# ============================================================
MAX_TFIDF_FEATURES = 5000     # Vocabulary size cap
NGRAM_RANGE        = (1, 2)   # Unigrams + bigrams
SUBLINEAR_TF       = True     # Apply log normalization to TF

# ============================================================
# Feature Artifact Paths
# ============================================================
TFIDF_VECTORIZER_PATH = FEATURES_DIR / "tfidf_vectorizer.joblib"
X_TRAIN_PATH          = FEATURES_DIR / "X_train_tfidf.joblib"
X_TEST_PATH           = FEATURES_DIR / "X_test_tfidf.joblib"
Y_TRAIN_PATH          = FEATURES_DIR / "y_train.joblib"
Y_TEST_PATH           = FEATURES_DIR / "y_test.joblib"
SPLIT_KEYS_TRAIN_PATH = FEATURES_DIR / "keys_train.joblib"
SPLIT_KEYS_TEST_PATH  = FEATURES_DIR / "keys_test.joblib"

# ============================================================
# Baseline Model Artifact Paths
# ============================================================
MLP_PATH              = BASELINE_MODEL_DIR / "mlp_regressor.joblib"
LSVR_PATH             = BASELINE_MODEL_DIR / "linear_svr.joblib"
RF_PATH               = BASELINE_MODEL_DIR / "random_forest.joblib"
LSTM_MODEL_PATH       = BASELINE_MODEL_DIR / "lstm_model.keras"
META_LEARNER_PATH     = BASELINE_MODEL_DIR / "meta_learner.joblib"
BASELINE_METRICS_PATH = BASELINE_MODEL_DIR / "baseline_metrics.json"
