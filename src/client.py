"""
client.py -- Flower client for Phase 3 Federated Learning (FedProx).

Memory-efficient implementation: data is loaded on-demand inside fit() and
evaluate(), then immediately released.

Supports two execution modes:
  1. Simulation (Colab):  via client_fn() factory — called by
     fl.simulation.start_simulation() in simulate_phase3.py
  2. Standalone (Windows): via main() — launched as a separate process
     by run_phase3.ps1

Usage (standalone):
    python src/client.py --client_id 0
"""

import argparse
import gc
import sys
import os
import logging
from pathlib import Path
import warnings

# Suppress ALL noisy output before any imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
logging.getLogger("flwr").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import flwr as fl
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    FEDERATED_DATA_DIR,
    TFIDF_VECTORIZER_PATH,
    MAX_TFIDF_FEATURES,
)
from src.CentralizedKhattab_phase1 import get_sklearn_base_learners, build_keras_lstm, sparse_to_dense_f32


class FLClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.client_dir = FEDERATED_DATA_DIR / f"client_{client_id}"

        # Store paths — do NOT load data into RAM yet
        self.train_path = self.client_dir / "train.csv"
        self.test_path = self.client_dir / "test.csv"

        # Load lightweight global vectorizer (shared, ~1 MB)
        self.vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)

        # Instantiate base models (weights are tiny until data arrives)
        sklearn_models = get_sklearn_base_learners()
        self.mlp = sklearn_models["MLP"]
        self.mlp.warm_start = True
        self.lsvr = sklearn_models["LSVR"]
        self.lstm = build_keras_lstm(MAX_TFIDF_FEATURES)

        # Initialize Scikit-learn weight shapes with a minimal data slice
        self._initialize_sklearn_models()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _initialize_sklearn_models(self):
        """Fit on a tiny slice (20 rows) just to create weight arrays."""
        chunk = pd.read_csv(self.train_path, nrows=20)
        X_small = self.vectorizer.transform(chunk["text"])
        y_small = chunk["storypoint"].values
        self.mlp.fit(X_small, y_small)
        self.lsvr.fit(X_small, y_small)
        del chunk, X_small, y_small
        gc.collect()

    def _load_train_data(self):
        """Load training data on demand and return (X_sparse, X_dense, y)."""
        df = pd.read_csv(self.train_path)
        X_sparse = self.vectorizer.transform(df["text"])
        y = df["storypoint"].values
        X_dense = sparse_to_dense_f32(X_sparse)
        del df
        gc.collect()
        return X_sparse, X_dense, y

    def _load_test_data(self):
        """Load test data on demand and return (X_sparse, X_dense, y)."""
        df = pd.read_csv(self.test_path)
        X_sparse = self.vectorizer.transform(df["text"])
        y = df["storypoint"].values
        X_dense = sparse_to_dense_f32(X_sparse)
        del df
        gc.collect()
        return X_sparse, X_dense, y

    def _release_memory(self):
        """Free data arrays and trigger garbage collection."""
        gc.collect()

    # ------------------------------------------------------------------
    # Flower interface
    # ------------------------------------------------------------------

    def get_parameters(self, config):
        """Extract parameters from MLP and LSTM (LSVR is local-only)."""
        params = []
        params.extend(self.mlp.coefs_)
        params.extend(self.mlp.intercepts_)
        params.extend(self.lstm.get_weights())
        return params

    def set_parameters(self, parameters):
        """Inject parameters into MLP and LSTM."""
        idx = 0
        n_layers = len(self.mlp.hidden_layer_sizes) + 1
        self.mlp.coefs_ = list(parameters[idx : idx + n_layers])
        idx += n_layers
        self.mlp.intercepts_ = list(parameters[idx : idx + n_layers])
        idx += n_layers
        self.lstm.set_weights(parameters[idx:])

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Load training data into RAM only for the duration of fit()
        X_train_sparse, X_train_dense, y_train = self._load_train_data()
        n_samples = len(y_train)

        self.mlp.fit(X_train_sparse, y_train)

        # LSVR trains locally but does not exchange parameters globally
        self.lsvr.fit(X_train_sparse, y_train)

        X_3d = X_train_dense.reshape(X_train_dense.shape[0], 1, X_train_dense.shape[1])
        self.lstm.fit(X_3d, y_train, epochs=1, batch_size=512, verbose=0)

        result = self.get_parameters(config={})

        # Immediately free all training data
        del X_train_sparse, X_train_dense, y_train, X_3d
        self._release_memory()

        return result, n_samples, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # Load test data into RAM only for the duration of evaluate()
        X_test_sparse, X_test_dense, y_test = self._load_test_data()
        n_samples = len(y_test)

        X_3d = X_test_dense.reshape(X_test_dense.shape[0], 1, X_test_dense.shape[1])
        y_pred = self.lstm.predict(X_3d, batch_size=512, verbose=0).flatten()
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Immediately free all test data
        del X_test_sparse, X_test_dense, y_test, X_3d, y_pred
        self._release_memory()

        return float(rmse), n_samples, {"mae": float(mae), "rmse": float(rmse)}


# ======================================================================
# Simulation factory (used by src/simulate_phase3.py)
# ======================================================================

def client_fn(cid: str) -> fl.client.Client:
    """Factory function for Flower's simulation engine.

    Creates a fresh FLClient for the given client ID.  Called on-demand
    by the simulation engine — clients are not kept alive between rounds,
    so memory is naturally reclaimed after each fit()/evaluate() cycle.
    """
    return FLClient(cid).to_client()


# ======================================================================
# Standalone mode (used by run_phase3.ps1 on Windows)
# ======================================================================

def main():
    """Launch a single FL client as a standalone process.

    This mode is used by the Windows multi-process launcher (run_phase3.ps1).
    For Colab, use simulate_phase3.py instead — it avoids the 17-process
    overhead that exceeds Colab's 12.7 GB system RAM limit.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=str, required=True)
    args = parser.parse_args()

    client = FLClient(args.client_id)
    fl.client.start_client(server_address="127.0.0.1:8082", client=client.to_client())

if __name__ == "__main__":
    main()
