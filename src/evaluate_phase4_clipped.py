"""
evaluate_phase4_clipped.py -- Stage A sanity check (see plan: i-need-a-proper-jaunty-moore)

Re-runs the exact same Phase 4 evaluation as evaluate_all_phase4.py against the
ALREADY-SAVED artifacts (no retraining), but clips predictions to the valid
label domain [1, STORYPOINT_CAP] before computing MAE/RMSE.

Rationale: StandardScaler.inverse_transform on a z-score fit to a heavy-tailed
per-client distribution (e.g. moodle: mean=15.54, std=21.65) can produce raw
predictions that are negative or far above the label cap. This clips PREDICTIONS
only -- it does not touch training labels/data, so it does not conflict with the
thesis's full-spectrum-regression (no filtering) design commitment.

Prints a side-by-side comparison against the existing (unclipped) baseline
metrics in models/phase4_personalized/phase4_evaluation_all.json, and saves its
own results to models/phase4_personalized/phase4_evaluation_clipped.json so the
original baseline file is never overwritten.
"""

import os
import sys
import gc
import json
from pathlib import Path
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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
    STORYPOINT_CAP,
)
from src.CentralizedKhattab_phase1 import sparse_to_dense_f32


def build_feature_extractor(model):
    """Build a Functional API sub-model to extract penultimate layer output."""
    inp = tf.keras.layers.Input(shape=model.input_shape[1:])
    x = inp
    for layer in model.layers[:-1]:
        x = layer(x)
    return tf.keras.Model(inputs=inp, outputs=x)


def main():
    print("=" * 78)
    print(" Phase 4 Stage A: Prediction-Clipping Sanity Check (no retraining)")
    print(f" Clipping predictions to [1, {STORYPOINT_CAP}] before scoring")
    print("=" * 78)
    print("\n[+] Loading global assets...")

    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)

    tf.get_logger().setLevel("ERROR")
    global_mlp = tf.keras.models.load_model(PHASE4_GLOBAL_MLP_PATH, compile=False)
    global_lstm = tf.keras.models.load_model(PHASE4_GLOBAL_LSTM_PATH, compile=False)

    mlp_ext = build_feature_extractor(global_mlp)
    lstm_ext = build_feature_extractor(global_lstm)

    results = []
    total_test_samples = 0

    print("\n[+] Evaluating clients (unclipped vs clipped)...")
    header = (
        f"\n  {'Client':<8} | {'Project':<20} | {'N':<6} | "
        f"{'MAE (raw)':<10} | {'MAE (clip)':<11} | {'RMSE (raw)':<11} | {'RMSE (clip)':<11} | {'#clipped':<8}"
    )
    print(header)
    print("-" * len(header))

    for cid in range(16):
        project_name = PROJECTS[cid] if cid < len(PROJECTS) else f"Unknown_{cid}"

        client_dir = FEDERATED_DATA_DIR / f"client_{cid}"
        test_path = client_dir / "test.csv"

        local_model_dir = PHASE4_MODEL_DIR / f"client_{cid}"
        ensemble_path = local_model_dir / "local_ensemble.joblib"
        scaler_path = local_model_dir / "y_scaler.joblib"

        if not test_path.exists() or not ensemble_path.exists():
            print(f"  {cid:<8} | {project_name:<20} | SKIPPED (missing artifacts)")
            continue

        df = pd.read_csv(test_path)
        y_test_raw = df["storypoint"].values.astype(np.float64)
        n_samples = len(y_test_raw)
        total_test_samples += n_samples

        X_sparse = vectorizer.transform(df["text"])
        X_dense = sparse_to_dense_f32(X_sparse)

        X_3d = X_dense.reshape(X_dense.shape[0], 1, X_dense.shape[1])
        lstm_emb = lstm_ext.predict(X_3d, batch_size=512, verbose=0)
        mlp_emb = mlp_ext.predict(X_dense, batch_size=512, verbose=0)
        X_embeddings = np.concatenate([lstm_emb, mlp_emb], axis=1)

        ensemble = joblib.load(ensemble_path)
        y_scaler = joblib.load(scaler_path)

        y_pred_scaled = ensemble.predict(X_embeddings)
        # [Phase 5] y_scaler now operates in log1p space -- invert scale, then expm1.
        y_pred_log = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_pred_raw = np.expm1(y_pred_log)

        y_pred_clipped = np.clip(y_pred_raw, 1.0, float(STORYPOINT_CAP))
        n_clipped = int(np.sum(y_pred_raw != y_pred_clipped))

        mae_raw = mean_absolute_error(y_test_raw, y_pred_raw)
        rmse_raw = np.sqrt(mean_squared_error(y_test_raw, y_pred_raw))
        mae_clip = mean_absolute_error(y_test_raw, y_pred_clipped)
        rmse_clip = np.sqrt(mean_squared_error(y_test_raw, y_pred_clipped))

        results.append({
            "client_id": cid,
            "project_name": project_name,
            "n_samples": n_samples,
            "mae_raw": mae_raw,
            "rmse_raw": rmse_raw,
            "mae_clipped": mae_clip,
            "rmse_clipped": rmse_clip,
            "n_predictions_clipped": n_clipped,
            "pred_min_raw": float(np.min(y_pred_raw)),
            "pred_max_raw": float(np.max(y_pred_raw)),
        })

        print(
            f"  {cid:<8} | {project_name:<20} | {n_samples:<6} | "
            f"{mae_raw:<10.4f} | {mae_clip:<11.4f} | {rmse_raw:<11.4f} | {rmse_clip:<11.4f} | {n_clipped:<8}"
        )

        del df, X_sparse, X_dense, X_3d, lstm_emb, mlp_emb, X_embeddings, ensemble, y_scaler
        gc.collect()

    print("-" * len(header))

    if results:
        macro_mae_raw = np.mean([r["mae_raw"] for r in results])
        macro_rmse_raw = np.mean([r["rmse_raw"] for r in results])
        macro_mae_clip = np.mean([r["mae_clipped"] for r in results])
        macro_rmse_clip = np.mean([r["rmse_clipped"] for r in results])

        weighted_mae_raw = np.sum([r["mae_raw"] * r["n_samples"] for r in results]) / total_test_samples
        weighted_rmse_raw = np.sum([r["rmse_raw"] * r["n_samples"] for r in results]) / total_test_samples
        weighted_mae_clip = np.sum([r["mae_clipped"] * r["n_samples"] for r in results]) / total_test_samples
        weighted_rmse_clip = np.sum([r["rmse_clipped"] * r["n_samples"] for r in results]) / total_test_samples

        print(f"\n[+] Aggregate comparison ({total_test_samples} total test samples)")
        print(f"  {'Metric':<22} | {'Unclipped':<12} | {'Clipped':<12} | {'Delta':<10}")
        print("-" * 62)
        print(f"  {'Macro MAE':<22} | {macro_mae_raw:<12.4f} | {macro_mae_clip:<12.4f} | {macro_mae_clip - macro_mae_raw:<+10.4f}")
        print(f"  {'Macro RMSE':<22} | {macro_rmse_raw:<12.4f} | {macro_rmse_clip:<12.4f} | {macro_rmse_clip - macro_rmse_raw:<+10.4f}")
        print(f"  {'Weighted MAE':<22} | {weighted_mae_raw:<12.4f} | {weighted_mae_clip:<12.4f} | {weighted_mae_clip - weighted_mae_raw:<+10.4f}")
        print(f"  {'Weighted RMSE':<22} | {weighted_rmse_raw:<12.4f} | {weighted_rmse_clip:<12.4f} | {weighted_rmse_clip - weighted_rmse_raw:<+10.4f}")
        print("=" * 78)

        total_clipped = sum(r["n_predictions_clipped"] for r in results)
        print(f"\n  Total predictions clipped: {total_clipped} / {total_test_samples}")

        out = {
            "macro_mae_raw": float(macro_mae_raw),
            "macro_rmse_raw": float(macro_rmse_raw),
            "macro_mae_clipped": float(macro_mae_clip),
            "macro_rmse_clipped": float(macro_rmse_clip),
            "weighted_mae_raw": float(weighted_mae_raw),
            "weighted_rmse_raw": float(weighted_rmse_raw),
            "weighted_mae_clipped": float(weighted_mae_clip),
            "weighted_rmse_clipped": float(weighted_rmse_clip),
            "total_test_samples": int(total_test_samples),
            "total_predictions_clipped": int(total_clipped),
            "client_results": results,
        }

        out_path = PHASE4_MODEL_DIR / "phase4_evaluation_clipped.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Detailed JSON saved to: {out_path}")


if __name__ == "__main__":
    main()
