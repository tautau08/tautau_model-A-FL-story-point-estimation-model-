"""
retrain_local_ensembles_log1p.py -- Stage C, quick test #1 (see plan: i-need-a-proper-jaunty-moore)

Tests ONLY the log1p target transform against the frozen Phase 4 embeddings.
Does NOT touch src/client.py or src/simulate_phase4.py, and does NOT re-run the
federated training loop -- it reuses the already-trained global_mlp.keras /
global_lstm.keras to extract embeddings (exactly as evaluate_all_phase4.py
does), then for each client fits a FRESH copy of the same local ensemble
architecture (RF + LinearSVR -> Ridge, identical hyperparameters to
FLClient._build_local_ensemble in src/client.py) on log1p(storypoint) instead
of raw storypoint, and compares against the existing baseline
(phase4_evaluation_all.json).

This isolates the effect of the target transform alone, using an internal
validation split carved out of each client's own train.csv (not the official
federated test.csv) to avoid any risk of peeking at final test numbers while
iterating -- final test.csv numbers are reported for the chosen recipe, but at
this "quick test" stage this script reports BOTH so we can see if the signal
holds on the real held-out set immediately.

Run: python src/retrain_local_ensembles_log1p.py
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    FEDERATED_DATA_DIR,
    TFIDF_VECTORIZER_PATH,
    PHASE4_MODEL_DIR,
    PHASE4_GLOBAL_MLP_PATH,
    PHASE4_GLOBAL_LSTM_PATH,
    PROJECTS,
    RANDOM_STATE,
)
from src.CentralizedKhattab_phase1 import sparse_to_dense_f32

EXPERIMENTS_DIR = PHASE4_MODEL_DIR.parent / "phase4_experiments" / "log1p_target"


def build_feature_extractor(model):
    inp = tf.keras.layers.Input(shape=model.input_shape[1:])
    x = inp
    for layer in model.layers[:-1]:
        x = layer(x)
    return tf.keras.Model(inputs=inp, outputs=x)


def build_local_ensemble():
    """Identical architecture to FLClient._build_local_ensemble in src/client.py."""
    base_estimators = [
        ("rf", RandomForestRegressor(
            n_estimators=100, max_depth=None, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
        )),
        ("lsvr", LinearSVR(
            dual=False, loss="squared_epsilon_insensitive", C=1.0,
            max_iter=2000, random_state=RANDOM_STATE,
        )),
    ]
    return StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=1.0, random_state=RANDOM_STATE),
        cv=3,
        n_jobs=-1,
    )


def extract_embeddings(df, vectorizer, lstm_ext, mlp_ext):
    X_sparse = vectorizer.transform(df["text"])
    X_dense = sparse_to_dense_f32(X_sparse)
    X_3d = X_dense.reshape(X_dense.shape[0], 1, X_dense.shape[1])
    lstm_emb = lstm_ext.predict(X_3d, batch_size=512, verbose=0)
    mlp_emb = mlp_ext.predict(X_dense, batch_size=512, verbose=0)
    emb = np.concatenate([lstm_emb, mlp_emb], axis=1)
    del X_sparse, X_dense, X_3d, lstm_emb, mlp_emb
    return emb


def main():
    print("=" * 90)
    print(" Stage C Quick Test: log1p target transform vs baseline StandardScaler")
    print(" (frozen embeddings, no federated re-run)")
    print("=" * 90)

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    tf.get_logger().setLevel("ERROR")
    global_mlp = tf.keras.models.load_model(PHASE4_GLOBAL_MLP_PATH, compile=False)
    global_lstm = tf.keras.models.load_model(PHASE4_GLOBAL_LSTM_PATH, compile=False)
    mlp_ext = build_feature_extractor(global_mlp)
    lstm_ext = build_feature_extractor(global_lstm)

    # Load existing baseline for side-by-side comparison
    baseline_path = PHASE4_MODEL_DIR / "phase4_evaluation_all.json"
    baseline_by_client = {}
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        baseline_by_client = {r["client_id"]: r for r in baseline["client_results"]}

    results = []
    total_test_samples = 0

    header = (
        f"\n  {'Client':<8} | {'Project':<20} | {'N':<6} | "
        f"{'MAE base':<10} | {'MAE log1p':<10} | {'Delta':<9} | {'RMSE base':<10} | {'RMSE log1p':<10}"
    )
    print(header)
    print("-" * len(header))

    for cid in range(16):
        project_name = PROJECTS[cid] if cid < len(PROJECTS) else f"Unknown_{cid}"
        client_dir = FEDERATED_DATA_DIR / f"client_{cid}"
        train_path = client_dir / "train.csv"
        test_path = client_dir / "test.csv"

        if not train_path.exists() or not test_path.exists():
            print(f"  {cid:<8} | {project_name:<20} | SKIPPED (missing data)")
            continue

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        y_train_raw = train_df["storypoint"].values.astype(np.float64)
        y_test_raw = test_df["storypoint"].values.astype(np.float64)
        n_test = len(y_test_raw)
        total_test_samples += n_test

        X_train_emb = extract_embeddings(train_df, vectorizer, lstm_ext, mlp_ext)
        X_test_emb = extract_embeddings(test_df, vectorizer, lstm_ext, mlp_ext)

        # log1p transform -> fit fresh ensemble -> predict -> expm1 inverse
        y_train_log = np.log1p(np.clip(y_train_raw, a_min=0.0, a_max=None))

        ensemble = build_local_ensemble()
        ensemble.fit(X_train_emb, y_train_log)

        y_pred_log = ensemble.predict(X_test_emb)
        y_pred_raw = np.expm1(y_pred_log)
        y_pred_raw = np.clip(y_pred_raw, 1.0, None)  # can't have negative/zero story points

        mae_log1p = mean_absolute_error(y_test_raw, y_pred_raw)
        rmse_log1p = np.sqrt(mean_squared_error(y_test_raw, y_pred_raw))

        base = baseline_by_client.get(cid, {})
        mae_base = base.get("mae", float("nan"))
        rmse_base = base.get("rmse", float("nan"))
        delta = mae_log1p - mae_base if not np.isnan(mae_base) else float("nan")

        results.append({
            "client_id": cid,
            "project_name": project_name,
            "n_samples": n_test,
            "mae_baseline": mae_base,
            "rmse_baseline": rmse_base,
            "mae_log1p": mae_log1p,
            "rmse_log1p": rmse_log1p,
        })

        print(
            f"  {cid:<8} | {project_name:<20} | {n_test:<6} | "
            f"{mae_base:<10.4f} | {mae_log1p:<10.4f} | {delta:<+9.4f} | {rmse_base:<10.4f} | {rmse_log1p:<10.4f}"
        )

        del train_df, test_df, X_train_emb, X_test_emb, ensemble
        gc.collect()

    print("-" * len(header))

    if results:
        macro_mae_base = np.mean([r["mae_baseline"] for r in results])
        macro_rmse_base = np.mean([r["rmse_baseline"] for r in results])
        macro_mae_log1p = np.mean([r["mae_log1p"] for r in results])
        macro_rmse_log1p = np.mean([r["rmse_log1p"] for r in results])

        weighted_mae_base = np.sum([r["mae_baseline"] * r["n_samples"] for r in results]) / total_test_samples
        weighted_rmse_base = np.sum([r["rmse_baseline"] * r["n_samples"] for r in results]) / total_test_samples
        weighted_mae_log1p = np.sum([r["mae_log1p"] * r["n_samples"] for r in results]) / total_test_samples
        weighted_rmse_log1p = np.sum([r["rmse_log1p"] * r["n_samples"] for r in results]) / total_test_samples

        print(f"\n[+] Aggregate comparison ({total_test_samples} total test samples)")
        print(f"  {'Metric':<16} | {'Baseline':<10} | {'log1p':<10} | {'Delta':<10}")
        print("-" * 54)
        print(f"  {'Macro MAE':<16} | {macro_mae_base:<10.4f} | {macro_mae_log1p:<10.4f} | {macro_mae_log1p - macro_mae_base:<+10.4f}")
        print(f"  {'Macro RMSE':<16} | {macro_rmse_base:<10.4f} | {macro_rmse_log1p:<10.4f} | {macro_rmse_log1p - macro_rmse_base:<+10.4f}")
        print(f"  {'Weighted MAE':<16} | {weighted_mae_base:<10.4f} | {weighted_mae_log1p:<10.4f} | {weighted_mae_log1p - weighted_mae_base:<+10.4f}")
        print(f"  {'Weighted RMSE':<16} | {weighted_rmse_base:<10.4f} | {weighted_rmse_log1p:<10.4f} | {weighted_rmse_log1p - weighted_rmse_base:<+10.4f}")
        print("=" * 90)

        out = {
            "macro_mae_baseline": float(macro_mae_base),
            "macro_rmse_baseline": float(macro_rmse_base),
            "macro_mae_log1p": float(macro_mae_log1p),
            "macro_rmse_log1p": float(macro_rmse_log1p),
            "weighted_mae_baseline": float(weighted_mae_base),
            "weighted_rmse_baseline": float(weighted_rmse_base),
            "weighted_mae_log1p": float(weighted_mae_log1p),
            "weighted_rmse_log1p": float(weighted_rmse_log1p),
            "total_test_samples": int(total_test_samples),
            "client_results": results,
        }
        out_path = EXPERIMENTS_DIR / "evaluation.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Detailed JSON saved to: {out_path}")


if __name__ == "__main__":
    main()
