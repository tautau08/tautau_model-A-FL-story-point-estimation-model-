"""
simulate_phase4.py -- Sequential Flower-free simulation for Phase 4.

Architecture: Personalized Federated Ensemble (Split-Federation)
  - Server aggregates ONLY deep model weights (Keras MLP + Keras LSTM)
  - Each client trains a local StackingRegressor on deep embeddings
  - Evaluation uses the local ensemble for personalized predictions

Runtime: Sequential single-process (Windows GPU safe)
"""

import subprocess
import sys
import os
import gc
import json
import logging
import random
from pathlib import Path
import warnings

# -- Suppress noisy logs BEFORE any heavy imports ----------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import tensorflow as tf

# Enable GPU memory growth to avoid pre-allocating all VRAM
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    MAX_TFIDF_FEATURES,
    PHASE4_MODEL_DIR,
    PHASE4_METRICS_PATH,
    RANDOM_STATE,
)
from src.client import FLClient
from src.CentralizedKhattab_phase1 import build_keras_lstm

# =====================================================================
# Configuration
# =====================================================================
NUM_CLIENTS = 16
NUM_ROUNDS = 10
FRACTION_FIT = 0.5       # 50% of clients per round = 8
FRACTION_EVAL = 0.5      # 50% of clients per round = 8
PROXIMAL_MU = 0.1        # [Phase 4 — FedProx] Proximal regularization strength

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# =====================================================================
# Server-side helpers
# =====================================================================

def _build_keras_mlp(n_features):
    """Build the same Keras MLP as the client uses."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def extract_global_weights(mlp, lstm):
    """Get the global weight arrays [mlp_weights..., lstm_weights...]."""
    params = []
    params.extend(mlp.get_weights())
    params.extend(lstm.get_weights())
    return params


def fedavg_aggregate(results):
    """Weighted FedAvg aggregation."""
    total_samples = sum(n for n, _ in results)
    agg = [np.zeros_like(w) for w in results[0][1]]

    for n_samples, weights in results:
        weight_factor = n_samples / total_samples
        for i, w in enumerate(weights):
            agg[i] += w * weight_factor

    return agg


def set_global_weights(mlp, lstm, params):
    """Push aggregated weights back into server-side model copies."""
    n_mlp = len(mlp.get_weights())
    mlp.set_weights(params[:n_mlp])
    lstm.set_weights(params[n_mlp:])


# =====================================================================
# Main simulation loop
# =====================================================================

def main():
    print()
    print("=" * 60)
    print(" Federated Agile Effort Estimation")
    print(" Phase 4 -- Personalized Federated Ensemble")
    print(" Architecture: Split-Federation (Global DL + Local ML)")
    print(" Runtime: Sequential single-process (Windows GPU safe)")
    print("=" * 60)

    # -- Step 1: Partition data -----------------------------------------
    print("\n  [1/4] Partitioning data by project ...")
    result = subprocess.run([sys.executable, "src/partition_data.py"], check=False)
    if result.returncode != 0:
        raise RuntimeError("Partitioning failed.")
    print("         Done.")

    # -- Step 2: Initialize global models ───────────────────────────────
    print("\n  [2/4] Initializing global deep models ...")
    global_mlp = _build_keras_mlp(MAX_TFIDF_FEATURES)
    global_lstm = build_keras_lstm(MAX_TFIDF_FEATURES)

    global_params = extract_global_weights(global_mlp, global_lstm)
    total_p = sum(p.size for p in global_params)
    n_mlp_p = sum(p.size for p in global_mlp.get_weights())
    n_lstm_p = sum(p.size for p in global_lstm.get_weights())
    print(f"         {len(global_params)} arrays, {total_p:,} total parameters")
    print(f"         MLP: {n_mlp_p:,} params  |  LSTM: {n_lstm_p:,} params")

    # -- Step 3: Configure ──────────────────────────────────────────────
    n_fit = int(NUM_CLIENTS * FRACTION_FIT)
    n_eval = int(NUM_CLIENTS * FRACTION_EVAL)
    print(f"\n  [3/4] Configuration:")
    print(f"         FedProx (mu={PROXIMAL_MU}) + Split-Federation")
    print(f"         {NUM_ROUNDS} rounds, {n_fit} clients/round (fit)")
    print(f"         {n_eval} clients/round (evaluate)")

    all_cids = list(range(NUM_CLIENTS))
    round_metrics = []

    # -- Step 4: Training loop ──────────────────────────────────────────
    print(f"\n  [4/4] Starting training loop ...\n")
    print(f"  {'Round':>6s}  {'Fit OK':>6s}  {'Eval OK':>7s}  "
          f"{'Avg MAE':>10s}  {'Avg RMSE':>10s}")
    print(f"  {'─'*6:>6s}  {'─'*6:>6s}  {'─'*7:>7s}  "
          f"{'─'*10:>10s}  {'─'*10:>10s}")

    for rnd in range(1, NUM_ROUNDS + 1):
        # ── FIT PHASE ──────────────────────────────────────────────
        fit_cids = sorted(random.sample(all_cids, n_fit))
        fit_results = []  

        client_config = {
            "proximal_mu": PROXIMAL_MU,
            "current_round": rnd
        }

        for cid in fit_cids:
            try:
                client = FLClient(str(cid))
                updated_weights, n_samples, _ = client.fit(
                    global_params, config=client_config
                )
                fit_results.append((n_samples, updated_weights))
            except Exception as e:
                print(f"    ⚠ Client {cid} fit failed: {e}")
            finally:
                if 'client' in locals():
                    del client
                tf.keras.backend.clear_session()  # CRITICAL: Clears GPU VRAM memory leak
                gc.collect()

        # ── AGGREGATION (FedAvg) ───────────────────────────────────
        n_fit_ok = len(fit_results)
        if fit_results:
            global_params = fedavg_aggregate(fit_results)
            set_global_weights(global_mlp, global_lstm, global_params)

        del fit_results
        gc.collect()

        # ── EVALUATE PHASE ─────────────────────────────────────────
        eval_cids = sorted(random.sample(all_cids, n_eval))
        eval_maes = []
        eval_rmses = []

        for cid in eval_cids:
            try:
                client = FLClient(str(cid))
                loss, n_samples, metrics = client.evaluate(
                    global_params, config={}
                )
                eval_maes.append(metrics["mae"])
                eval_rmses.append(metrics["rmse"])
            except Exception as e:
                print(f"    ⚠ Client {cid} eval failed: {e}")
            finally:
                if 'client' in locals():
                    del client
                tf.keras.backend.clear_session()  # CRITICAL: Clears GPU VRAM memory leak
                gc.collect()

        # ── Log round results ──────────────────────────────────────
        avg_mae = np.mean(eval_maes) if eval_maes else float("nan")
        avg_rmse = np.mean(eval_rmses) if eval_rmses else float("nan")
        n_eval_ok = len(eval_maes)

        round_metrics.append({
            "round": rnd,
            "mae": float(avg_mae),
            "rmse": float(avg_rmse),
            "fit_clients": n_fit_ok,
            "eval_clients": n_eval_ok,
        })

        print(f"  R{rnd:>4d}  {n_fit_ok:>6d}  {n_eval_ok:>7d}  "
              f"{avg_mae:>10.4f}  {avg_rmse:>10.4f}")

        del eval_maes, eval_rmses
        gc.collect()

    # ── Final summary ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(" Phase 4 — Training Complete!")
    print("=" * 60)

    # Phase 1 comparison baseline values
    p1_mae, p1_rmse = 3.7739, 8.0838
    if round_metrics:
        final = round_metrics[-1]
        best = min(round_metrics, key=lambda x: x["mae"])
        delta_mae = final["mae"] - p1_mae
        pct_mae = (delta_mae / p1_mae) * 100

        print(f"\n  Final Round (R{final['round']}):")
        print(f"    MAE  = {final['mae']:.4f}")
        print(f"    RMSE = {final['rmse']:.4f}")
        print(f"\n  Best Round (R{best['round']}):")
        print(f"    MAE  = {best['mae']:.4f}")
        print(f"    RMSE = {best['rmse']:.4f}")
        print(f"\n  Phase 1 Centralized Baseline:")
        print(f"    MAE  = {p1_mae:.4f}  |  RMSE = {p1_rmse:.4f}")
        print(f"\n  Drift (Final vs Phase 1):")
        print(f"    MAE delta: {delta_mae:+.4f} ({pct_mae:+.1f}%)")

    # ── Save metrics ──
    PHASE4_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics_out = {
        "architecture": "Split-Federation (Global DL + Local StackingRegressor)",
        "strategy": f"FedProx (mu={PROXIMAL_MU})",
        "num_clients": NUM_CLIENTS,
        "num_rounds": NUM_ROUNDS,
        "rounds": round_metrics,
    }
    if round_metrics:
        metrics_out["final"] = round_metrics[-1]
        metrics_out["best"] = min(round_metrics, key=lambda x: x["mae"])

    with open(PHASE4_METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n  Metrics saved to {PHASE4_METRICS_PATH}")

    # ── Save global models ──
    global_mlp_path = PHASE4_MODEL_DIR / "global_mlp.keras"
    global_lstm_path = PHASE4_MODEL_DIR / "global_lstm.keras"
    global_mlp.save(global_mlp_path)
    global_lstm.save(global_lstm_path)
    print(f"  Global MLP  → {global_mlp_path}")
    print(f"  Global LSTM → {global_lstm_path}")

    print("\n  Local ensembles saved per-client at:")
    print("    models/phase4_personalized/client_{cid}/local_ensemble.joblib")
    print("=" * 60)


if __name__ == "__main__":
    main()