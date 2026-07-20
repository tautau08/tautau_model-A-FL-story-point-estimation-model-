"""
evaluate_all_phase4.py -- Comprehensive Phase 4 Final Evaluation

This script evaluates all 16 clients sequentially using their test datasets
against the Phase 4 Split-Federation architecture:
  - Global DL Embeddings (LSTM + MLP)
  - Local Personalized StackingRegressor Ensembles
"""

import os
import sys
import gc
import json
from pathlib import Path
import warnings

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging_level = "ERROR"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    FEDERATED_DATA_DIR,
    TFIDF_VECTORIZER_PATH,
    PHASE4_MODEL_DIR,
    PHASE4_GLOBAL_MLP_PATH,
    PHASE4_GLOBAL_LSTM_PATH,
    PROJECTS,
)
from src.CentralizedKhattab_phase1 import sparse_to_dense_f32

def build_feature_extractor(model):
    """Build a Functional API sub-model to extract penultimate layer output.
    Uses layers[:-1] to drop the final Dense(1) output layer.
    """
    inp = tf.keras.layers.Input(shape=model.input_shape[1:])
    x = inp
    for layer in model.layers[:-1]:
        x = layer(x)
    return tf.keras.Model(inputs=inp, outputs=x)

def main():
    print("=" * 70)
    print(" Phase 4 Final Comprehensive Evaluation (16 Clients)")
    print(" Architecture: Split-Federation (Global DL + Local ML)")
    print("=" * 70)
    print("\n[+] Loading global assets...")

    # 1. Load Vectorizer
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)

    # 2. Load Global Models
    tf.get_logger().setLevel("ERROR")
    global_mlp = tf.keras.models.load_model(PHASE4_GLOBAL_MLP_PATH, compile=False)
    global_lstm = tf.keras.models.load_model(PHASE4_GLOBAL_LSTM_PATH, compile=False)

    # Rebuild extractors
    mlp_ext = build_feature_extractor(global_mlp)
    lstm_ext = build_feature_extractor(global_lstm)

    results = []
    total_test_samples = 0

    print("\n[+] Evaluating clients...")
    
    # Table header
    print(f"\n  {'Client ID':<10} | {'Project Name':<20} | {'Test N':<8} | {'MAE':<8} | {'RMSE':<8}")
    print("-" * 65)

    for cid in range(16):
        project_name = PROJECTS[cid] if cid < len(PROJECTS) else f"Unknown_{cid}"
        
        client_dir = FEDERATED_DATA_DIR / f"client_{cid}"
        test_path = client_dir / "test.csv"
        
        local_model_dir = PHASE4_MODEL_DIR / f"client_{cid}"
        ensemble_path = local_model_dir / "local_ensemble.joblib"
        scaler_path = local_model_dir / "y_scaler.joblib"

        if not test_path.exists() or not ensemble_path.exists():
            print(f"  {cid:<10} | {project_name:<20} | {'SKIPPED':<8} | {'-':<8} | {'-':<8}")
            continue

        # Load Data
        df = pd.read_csv(test_path)
        y_test_raw = df["storypoint"].values.astype(np.float64)
        n_samples = len(y_test_raw)
        total_test_samples += n_samples

        X_sparse = vectorizer.transform(df["text"])
        X_dense = sparse_to_dense_f32(X_sparse)

        # Extract Embeddings
        X_3d = X_dense.reshape(X_dense.shape[0], 1, X_dense.shape[1])
        lstm_emb = lstm_ext.predict(X_3d, batch_size=512, verbose=0)
        mlp_emb = mlp_ext.predict(X_dense, batch_size=512, verbose=0)
        X_embeddings = np.concatenate([lstm_emb, mlp_emb], axis=1)

        # Load Local Ensemble & Predict
        ensemble = joblib.load(ensemble_path)
        y_scaler = joblib.load(scaler_path)

        y_pred_scaled = ensemble.predict(X_embeddings)
        y_pred_raw = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        # Compute Metrics
        mae = mean_absolute_error(y_test_raw, y_pred_raw)
        rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_raw))

        results.append({
            "client_id": cid,
            "project_name": project_name,
            "n_samples": n_samples,
            "mae": mae,
            "rmse": rmse
        })

        print(f"  {cid:<10} | {project_name:<20} | {n_samples:<8} | {mae:<8.4f} | {rmse:<8.4f}")

        # Cleanup RAM
        del df, X_sparse, X_dense, X_3d, lstm_emb, mlp_emb, X_embeddings, ensemble, y_scaler
        gc.collect()

    print("-" * 65)

    # Calculate Aggregates
    if results:
        # Macro Average (equal weight per client)
        macro_mae = np.mean([r["mae"] for r in results])
        macro_rmse = np.mean([r["rmse"] for r in results])

        # Weighted Average (equal weight per sample)
        weighted_mae = np.sum([r["mae"] * r["n_samples"] for r in results]) / total_test_samples
        weighted_rmse = np.sum([r["rmse"] * r["n_samples"] for r in results]) / total_test_samples

        print(f"\n[+] Final Aggregated Metrics (16 Clients, {total_test_samples} Total Test Samples)")
        print(f"  Macro Average MAE     : {macro_mae:.4f}")
        print(f"  Macro Average RMSE    : {macro_rmse:.4f}")
        print(f"  Weighted Average MAE  : {weighted_mae:.4f}")
        print(f"  Weighted Average RMSE : {weighted_rmse:.4f}")
        print("=" * 70)

        # Optionally append to metrics JSON
        eval_metrics = {
            "macro_mae": float(macro_mae),
            "macro_rmse": float(macro_rmse),
            "weighted_mae": float(weighted_mae),
            "weighted_rmse": float(weighted_rmse),
            "total_test_samples": int(total_test_samples),
            "client_results": results
        }
        
        eval_path = PHASE4_MODEL_DIR / "phase4_evaluation_all.json"
        with open(eval_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        print(f"\n  Detailed JSON saved to: {eval_path}")

if __name__ == "__main__":
    main()
