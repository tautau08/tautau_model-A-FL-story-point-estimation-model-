"""
client.py -- Flower client for Phase 2 Federated Learning.

Usage:
    python src/client.py --client_id 0
"""

import argparse
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
        client_dir = FEDERATED_DATA_DIR / f"client_{client_id}"
        
        # Load local data
        self.train_df = pd.read_csv(client_dir / "train.csv")
        self.test_df = pd.read_csv(client_dir / "test.csv")
        
        # Load global vectorizer
        self.vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        
        # Transform local data
        self.X_train_sparse = self.vectorizer.transform(self.train_df["text"])
        self.X_test_sparse = self.vectorizer.transform(self.test_df["text"])
        self.y_train = self.train_df["storypoint"].values
        self.y_test = self.test_df["storypoint"].values
        
        # Dense data for LSTM
        self.X_train_dense = sparse_to_dense_f32(self.X_train_sparse)
        self.X_test_dense = sparse_to_dense_f32(self.X_test_sparse)
        
        # Instantiate base models
        sklearn_models = get_sklearn_base_learners()
        self.mlp = sklearn_models["MLP"]
        self.mlp.warm_start = True
        self.lsvr = sklearn_models["LSVR"]
        self.lstm = build_keras_lstm(MAX_TFIDF_FEATURES)
        
        # Initialize Scikit-learn models so they have weight shapes
        self._initialize_sklearn_models()

    def _initialize_sklearn_models(self):
        """Fit on 20 samples to initialize weight shapes."""
        self.mlp.fit(self.X_train_sparse[:20], self.y_train[:20])
        self.lsvr.fit(self.X_train_sparse[:20], self.y_train[:20])

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
        
        self.mlp.fit(self.X_train_sparse, self.y_train)
        
        # LSVR trains locally but does not exchange parameters globally
        self.lsvr.fit(self.X_train_sparse, self.y_train)
        
        X_3d = self.X_train_dense.reshape(self.X_train_dense.shape[0], 1, self.X_train_dense.shape[1])
        self.lstm.fit(X_3d, self.y_train, epochs=1, batch_size=512, verbose=0)
        
        return self.get_parameters(config={}), len(self.y_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        X_3d = self.X_test_dense.reshape(self.X_test_dense.shape[0], 1, self.X_test_dense.shape[1])
        y_pred = self.lstm.predict(X_3d, batch_size=512, verbose=0).flatten()
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        return float(rmse), len(self.y_test), {"mae": float(mae), "rmse": float(rmse)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=str, required=True)
    args = parser.parse_args()
    
    client = FLClient(args.client_id)
    fl.client.start_client(server_address="127.0.0.1:8082", client=client.to_client())

if __name__ == "__main__":
    main()
