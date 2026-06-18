"""
metric_validation.py - Statistical validation of Phase 1 Centralized Baseline metrics.
"""

import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, median_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    Y_TRAIN_PATH,
    Y_TEST_PATH,
    X_TEST_PATH,
    MLP_PATH,
    LSVR_PATH,
    RF_PATH,
    LSTM_MODEL_PATH,
    META_LEARNER_PATH,
)

from src.train_baseline import sparse_to_dense_f32

def main():
    print("=" * 60)
    print(" Phase 1 Metric Validation & Dataset Scale Analysis")
    print("=" * 60)

    # 1. Dataset Scale Analysis
    print("\n[1] Dataset Scale Analysis (True Story Points)")
    y_train = joblib.load(Y_TRAIN_PATH)
    y_test = joblib.load(Y_TEST_PATH)
    y_all = np.concatenate([y_train, y_test])

    print(f"  Total Samples:   {len(y_all):,}")
    print(f"  Minimum:         {np.min(y_all):.2f}")
    print(f"  Maximum:         {np.max(y_all):.2f}")
    print(f"  Mean:            {np.mean(y_all):.2f}")
    print(f"  Median:          {np.median(y_all):.2f}")
    print(f"  95th Percentile: {np.percentile(y_all, 95):.2f}")

    # 2. Model Evaluation
    print("\n[2] Median Absolute Error (MdAE) Calculation")
    print("  Loading Phase 1 models...")
    X_test_sparse = joblib.load(X_TEST_PATH)
    X_test_dense = sparse_to_dense_f32(X_test_sparse)
    X_test_3d = X_test_dense.reshape(X_test_dense.shape[0], 1, X_test_dense.shape[1])

    mlp = joblib.load(MLP_PATH)
    lsvr = joblib.load(LSVR_PATH)
    rf = joblib.load(RF_PATH)
    
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    lstm = tf.keras.models.load_model(LSTM_MODEL_PATH)
    meta_learner = joblib.load(META_LEARNER_PATH)

    print("  Generating predictions...")
    test_preds = np.zeros((X_test_sparse.shape[0], 4), dtype=np.float64)
    test_preds[:, 0] = mlp.predict(X_test_sparse)
    test_preds[:, 1] = lsvr.predict(X_test_sparse)
    test_preds[:, 2] = rf.predict(X_test_sparse)
    test_preds[:, 3] = lstm.predict(X_test_3d, batch_size=512, verbose=0).flatten()

    y_pred_ensemble = meta_learner.predict(test_preds)

    mae = mean_absolute_error(y_test, y_pred_ensemble)
    mdae = median_absolute_error(y_test, y_pred_ensemble)

    print(f"\n  Final Centralized Ensemble Metrics:")
    print(f"  Mean Absolute Error (MAE):     {mae:.4f}")
    print(f"  Median Absolute Error (MdAE):  {mdae:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
