"""
simulate_phase3.py -- Single-process Flower simulation for Phase 3 (FedProx).

Designed for Google Colab's 12.7 GB system RAM constraint.  Replaces the
17-process architecture (1 server + 16 client subprocesses) with Flower's
Virtual Client Engine (VCE), which shares a single TensorFlow / Python
runtime across all virtual clients.

Estimated RAM: ~2-3 GB (vs. 12+ GB with 17 processes).

Usage (via run_colab.py, or directly):
    python src/simulate_phase3.py
"""

import subprocess
import sys
import os
import logging
import json
from pathlib import Path
import warnings

# ── Suppress noisy logs BEFORE any heavy imports ──────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
logging.getLogger("flwr").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import flwr as fl
import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Optional, Tuple

# Enable GPU memory growth so the main process doesn't pre-allocate all
# VRAM, leaving room for Ray worker processes to use fractional GPU slices.
_gpus = tf.config.list_physical_devices("GPU")
for _gpu in _gpus:
    tf.config.experimental.set_memory_growth(_gpu, True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    X_TEST_PATH,
    Y_TEST_PATH,
    MAX_TFIDF_FEATURES,
    FEDPROX_MLP_PATH,
    FEDPROX_LSTM_PATH,
    PHASE3_METRICS_PATH,
)
from src.CentralizedKhattab_phase1 import (
    get_sklearn_base_learners,
    build_keras_lstm,
    sparse_to_dense_f32,
)
from src.client import client_fn

# =====================================================================
# Configuration
# =====================================================================
NUM_CLIENTS = 16
NUM_ROUNDS = 10
# 1/16 of the T4 GPU per client — allows all 16 virtual clients to
# share the GPU, though only 8 are active per round (fraction_fit=0.5).
CLIENT_RESOURCES = {"num_cpus": 1, "num_gpus": 0.0625}


# =====================================================================
# Server-side helpers (mirrored from server_phase3.py)
# =====================================================================

def _extract_initial_parameters(mlp, lstm):
    """Extract a flat list of numpy arrays from initialized models (excluding LSVR)."""
    params = []
    # MLP: coefs_ then intercepts_
    params.extend(mlp.coefs_)
    params.extend(mlp.intercepts_)
    # LSTM
    params.extend(lstm.get_weights())
    return params


# Store round results for final summary
_round_results = []


def get_evaluate_fn(mlp, lstm):
    """Returns an evaluate function for the server (evaluating MLP and LSTM only)."""
    X_test_sparse = joblib.load(X_TEST_PATH)
    y_test = joblib.load(Y_TEST_PATH)
    X_test_dense = sparse_to_dense_f32(X_test_sparse)

    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        idx = 0

        # 1. MLP
        n_layers = len(mlp.hidden_layer_sizes) + 1
        mlp.coefs_ = list(parameters[idx : idx + n_layers])
        idx += n_layers
        mlp.intercepts_ = list(parameters[idx : idx + n_layers])
        idx += n_layers

        # 2. LSTM
        lstm.set_weights(parameters[idx:])

        # --- Generate Predictions ---
        test_preds = np.zeros((X_test_sparse.shape[0], 2), dtype=np.float64)

        # Federated MLP
        test_preds[:, 0] = mlp.predict(X_test_sparse)

        # Federated LSTM
        X_test_3d = X_test_dense.reshape(X_test_dense.shape[0], 1, X_test_dense.shape[1])
        test_preds[:, 1] = lstm.predict(X_test_3d, batch_size=512, verbose=0).flatten()

        # Simple average ensemble for the federated deep models
        y_pred_ensemble = np.mean(test_preds, axis=1)

        mae = mean_absolute_error(y_test, y_pred_ensemble)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))

        label = "INIT" if server_round == 0 else f"R{server_round}"
        _round_results.append((label, mae, rmse))
        print(f"    [{label}]  MAE={mae:.4f}  RMSE={rmse:.4f}")

        return float(rmse), {"mae": float(mae), "rmse": float(rmse)}

    return evaluate


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 60)
    print(" Federated Agile Effort Estimation")
    print(" Phase 3 -- Flower Simulation (FedProx)")
    print(" Runtime: Single-process  •  Colab-optimized")
    print("=" * 60)

    # ── Step 1: Partition data ─────────────────────────────────────────
    print("\n  [1/5] Partitioning data by project ...")
    result = subprocess.run([sys.executable, "src/partition_data.py"], check=False)
    if result.returncode != 0:
        raise RuntimeError("Partitioning failed.")
    print("         Done.")

    # ── Step 2: Initialize model parameter shapes ─────────────────────
    print("\n  [2/5] Initializing model parameter shapes ...")
    X_test_sparse = joblib.load(X_TEST_PATH)
    y_test = joblib.load(Y_TEST_PATH)

    sklearn_models = get_sklearn_base_learners()
    mlp = sklearn_models["MLP"]
    lstm = build_keras_lstm(MAX_TFIDF_FEATURES)

    mlp.fit(X_test_sparse[:20], y_test[:20])

    init_params = _extract_initial_parameters(mlp, lstm)
    total_params = sum(p.size for p in init_params)
    print(f"         {len(init_params)} arrays, {total_params:,} total parameters")

    # ── Step 3: Configure FedProx strategy ─────────────────────────────
    print("  [3/5] Configuring FedProx strategy ...")
    print("         fraction_fit=0.5 (8 clients/round)")
    print("         proximal_mu=0.1")
    print(f"         min_available_clients={NUM_CLIENTS}")

    strategy = fl.server.strategy.FedProx(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        min_fit_clients=8,
        min_evaluate_clients=8,
        min_available_clients=NUM_CLIENTS,
        proximal_mu=0.1,
        evaluate_fn=get_evaluate_fn(mlp, lstm),
        initial_parameters=fl.common.ndarrays_to_parameters(init_params),
    )

    # ── Step 4: Run Flower simulation ──────────────────────────────────
    print(f"  [4/5] Starting simulation ({NUM_CLIENTS} virtual clients, "
          f"{NUM_ROUNDS} rounds) ...")
    print(f"         Client resources: {CLIENT_RESOURCES}")
    print("\n  -- Round-by-round evaluation (centralized test set) --")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources=CLIENT_RESOURCES,
        ray_init_args={"include_dashboard": False},
    )

    # ── Step 5: Results summary + save ─────────────────────────────────
    print("\n  [5/5] Simulation complete!")
    print("\n" + "=" * 60)
    print(" Phase 3 Results Summary")
    print("=" * 60)
    print(f"  {'Round':<8s} {'MAE':>10s} {'RMSE':>10s}")
    print(f"  {'-'*8:<8s} {'-'*10:>10s} {'-'*10:>10s}")
    for label, mae, rmse in _round_results:
        print(f"  {label:<8s} {mae:>10.4f} {rmse:>10.4f}")

    # Phase 1 comparison
    p1_mae, p1_rmse = 3.7739, 8.0838
    if _round_results:
        final_mae = _round_results[-1][1]
        final_rmse = _round_results[-1][2]
        delta_mae = final_mae - p1_mae
        delta_rmse = final_rmse - p1_rmse
        pct_mae = (delta_mae / p1_mae) * 100
        pct_rmse = (delta_rmse / p1_rmse) * 100
        print(f"\n  Phase 1 Centralized Baseline:")
        print(f"  {'P1':8s} {p1_mae:>10.4f} {p1_rmse:>10.4f}")
        print(f"\n  Client Drift (Final Round vs Phase 1):")
        print(f"    MAE  delta: {delta_mae:+.4f} ({pct_mae:+.1f}%)")
        print(f"    RMSE delta: {delta_rmse:+.4f} ({pct_rmse:+.1f}%)")

        # ---- Save Metrics to JSON ----
        metrics_dict = {
            "rounds": {},
            "final_ensemble": {"MAE": final_mae, "RMSE": final_rmse},
            "client_drift_vs_phase1": {
                "MAE_delta": delta_mae,
                "MAE_degradation_pct": pct_mae,
                "RMSE_delta": delta_rmse,
                "RMSE_degradation_pct": pct_rmse
            }
        }
        for label, m, r in _round_results:
            metrics_dict["rounds"][label] = {"MAE": m, "RMSE": r}

        with open(PHASE3_METRICS_PATH, "w") as f:
            json.dump(metrics_dict, f, indent=2)

    # ---- Save Models ----
    print("\n  Saving Phase 3 global models to disk ...")
    joblib.dump(mlp, FEDPROX_MLP_PATH)
    lstm.save(FEDPROX_LSTM_PATH)
    print(f"         Saved MLP to {FEDPROX_MLP_PATH.name}")
    print(f"         Saved LSTM to {FEDPROX_LSTM_PATH.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
