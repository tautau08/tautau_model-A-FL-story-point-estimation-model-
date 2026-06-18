"""
server.py -- Flower server orchestrator for Phase 2 FL Baseline.

Initializes `FedAvg` with explicit initial_parameters, runs for 3 rounds,
and evaluates the aggregated global weights for the continuous models
(MLP, LSTM) against the global centralized test set.
"""

import sys
import os
import logging
from pathlib import Path
import warnings

# Suppress noisy logs BEFORE any imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
logging.getLogger("flwr").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import flwr as fl
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    X_TEST_PATH,
    Y_TEST_PATH,
    MAX_TFIDF_FEATURES,
)
from src.train_baseline import get_sklearn_base_learners, build_keras_lstm, sparse_to_dense_f32


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


def main():
    print("=" * 60)
    print(" Federated Agile Effort Estimation")
    print(" Phase 2 -- Flower Server (FedAvg)")
    print("=" * 60)

    # ---- Build models on the server side for shape init ----
    print("\n  [1/4] Initializing model parameter shapes ...")
    X_test_sparse = joblib.load(X_TEST_PATH)
    y_test = joblib.load(Y_TEST_PATH)

    sklearn_models = get_sklearn_base_learners()
    mlp = sklearn_models["MLP"]
    lstm = build_keras_lstm(MAX_TFIDF_FEATURES)

    mlp.fit(X_test_sparse[:20], y_test[:20])

    init_params = _extract_initial_parameters(mlp, lstm)
    total_params = sum(p.size for p in init_params)
    print(f"         {len(init_params)} arrays, {total_params:,} total parameters")

    # ---- Configure FedAvg ----
    print("  [2/4] Configuring FedAvg strategy ...")
    print("         fraction_fit=0.19 (~3 clients/round)")
    print("         min_available_clients=16")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.19,
        fraction_evaluate=0.19,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=16,
        evaluate_fn=get_evaluate_fn(mlp, lstm),
        initial_parameters=fl.common.ndarrays_to_parameters(init_params),
    )

    # ---- Start Server ----
    print("  [3/4] Waiting for 16 clients to connect ...")
    print("\n  -- Round-by-round evaluation (centralized test set) --")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    # ---- Print Final Summary ----
    print("\n  [4/4] Simulation complete!")
    print("\n" + "=" * 60)
    print(" Phase 2 Results Summary")
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
    print("=" * 60)


if __name__ == "__main__":
    main()
