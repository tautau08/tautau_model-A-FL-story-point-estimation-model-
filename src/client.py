"""
client.py -- Flower client for Split-Federation Learning (Phase 4).

Architecture: Personalized Federated Ensemble
  - GLOBAL (aggregated by Flower server): LSTM + MLP deep feature extractors
  - LOCAL  (never sent to server):        StackingRegressor(RF + LinearSVR, final=Ridge)

The deep models are trained federally via FedProx and produce embeddings.
The local ensemble trains on those embeddings using each client's own data,
creating a personalized predictor that captures project-specific patterns.

Supports two execution modes:
  1. Simulation (Colab):  via client_fn() factory — called by
     fl.simulation.start_simulation()
  2. Standalone (Windows): via main() — launched as a separate process

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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import Ridge
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    FEDERATED_DATA_DIR,
    TFIDF_VECTORIZER_PATH,
    MAX_TFIDF_FEATURES,
    PHASE4_MODEL_DIR,
    RANDOM_STATE,
)
from src.CentralizedKhattab_phase1 import build_keras_lstm, sparse_to_dense_f32


# ======================================================================
# Phase 4: Personalized Federated Ensemble Client (Split-Federation)
# ======================================================================

class FLClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.client_dir = FEDERATED_DATA_DIR / f"client_{client_id}"

        # Store paths — do NOT load data into RAM yet
        self.train_path = self.client_dir / "train.csv"
        self.test_path = self.client_dir / "test.csv"

        # ── Local ensemble persistence directory ──
        self.local_model_dir = PHASE4_MODEL_DIR / f"client_{client_id}"
        self.local_model_dir.mkdir(parents=True, exist_ok=True)
        self.ensemble_path = self.local_model_dir / "local_ensemble.joblib"
        self.scaler_path = self.local_model_dir / "y_scaler.joblib"

        # Load lightweight global vectorizer (shared, ~1 MB)
        self.vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)

        # ── Global deep models (LSTM + MLP) ──
        # MLP: build a Keras MLP that mirrors the sklearn (128, 64) architecture
        # but as a proper Keras model so we can extract intermediate features.
        self.lstm = build_keras_lstm(MAX_TFIDF_FEATURES)
        self.mlp = self._build_keras_mlp(MAX_TFIDF_FEATURES)

        # Force-build by calling with dummy data so that model.input/output
        # are initialized (required by Keras 3 for sub-model extraction).
        _dummy_dense = np.zeros((1, MAX_TFIDF_FEATURES), dtype=np.float32)
        self.mlp(_dummy_dense)
        _dummy_3d = _dummy_dense.reshape(1, 1, MAX_TFIDF_FEATURES)
        self.lstm(_dummy_3d)
        del _dummy_dense, _dummy_3d

        # ── StandardScaler for y (story points) ──
        self.y_scaler = StandardScaler()
        self._fit_y_scaler()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_keras_mlp(self, n_features):
        """Build a Keras MLP matching the Khattab architecture (128 -> 64 -> 1).

        Using Keras instead of sklearn MLPRegressor so we can:
        1. Extract intermediate layer activations as embeddings
        2. Properly serialize/deserialize weights for Flower aggregation
        """
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

    def _build_feature_extractor(self, model, layer_index=-1):
        """Build a sub-model that outputs the penultimate layer activations.

        For LSTM: returns the Dense(32, relu) output → 32-dim embedding
        For MLP:  returns the Dense(64, relu) output → 64-dim embedding

        Uses Functional API rebuild to avoid Keras 3 'never been called' errors.
        """
        # Create a fresh input matching the model's input shape
        inp = tf.keras.layers.Input(shape=model.input_shape[1:])
        x = inp
        for layer in model.layers[:layer_index]:
            x = layer(x)
        return tf.keras.Model(inputs=inp, outputs=x)

    def _fit_y_scaler(self):
        """Fit StandardScaler on training y values for this client."""
        if self.scaler_path.exists():
            self.y_scaler = joblib.load(self.scaler_path)
        else:
            df = pd.read_csv(self.train_path)
            y = df["storypoint"].values.reshape(-1, 1)
            self.y_scaler.fit(y)
            joblib.dump(self.y_scaler, self.scaler_path)
            del df, y
            gc.collect()

    def _load_train_data(self):
        """Load training data on demand and return (X_sparse, X_dense, y_raw)."""
        df = pd.read_csv(self.train_path)
        X_sparse = self.vectorizer.transform(df["text"])
        y = df["storypoint"].values.astype(np.float64)
        X_dense = sparse_to_dense_f32(X_sparse)
        del df
        gc.collect()
        return X_sparse, X_dense, y

    def _load_test_data(self):
        """Load test data on demand and return (X_sparse, X_dense, y_raw)."""
        df = pd.read_csv(self.test_path)
        X_sparse = self.vectorizer.transform(df["text"])
        y = df["storypoint"].values.astype(np.float64)
        X_dense = sparse_to_dense_f32(X_sparse)
        del df
        gc.collect()
        return X_sparse, X_dense, y

    def _extract_embeddings(self, X_dense):
        """Pass X through LSTM and MLP feature extractors and concatenate.

        Returns:
            X_embeddings: np.ndarray of shape (n_samples, 32 + 64) = (n, 96)
        """
        # LSTM expects (samples, 1, features)
        X_3d = X_dense.reshape(X_dense.shape[0], 1, X_dense.shape[1])

        lstm_extractor = self._build_feature_extractor(self.lstm)
        lstm_emb = lstm_extractor.predict(X_3d, batch_size=512, verbose=0)

        mlp_extractor = self._build_feature_extractor(self.mlp)
        mlp_emb = mlp_extractor.predict(X_dense, batch_size=512, verbose=0)

        # Concatenate: (n, 32) + (n, 64) = (n, 96)
        X_embeddings = np.concatenate([lstm_emb, mlp_emb], axis=1)

        del X_3d, lstm_extractor, mlp_extractor, lstm_emb, mlp_emb
        gc.collect()

        return X_embeddings

    def _build_local_ensemble(self):
        """Build a fresh StackingRegressor: RF + LinearSVR → Ridge meta-learner."""
        base_estimators = [
            ("rf", RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=0,
            )),
            ("lsvr", LinearSVR(
                dual=False,
                loss="squared_epsilon_insensitive",
                C=1.0,
                max_iter=2000,
                random_state=RANDOM_STATE,
            )),
        ]

        ensemble = StackingRegressor(
            estimators=base_estimators,
            final_estimator=Ridge(alpha=1.0, random_state=RANDOM_STATE),
            cv=3,
            n_jobs=-1,
        )
        return ensemble

    def _release_memory(self):
        """Free data arrays and trigger garbage collection."""
        gc.collect()

    # ------------------------------------------------------------------
    # [Phase 4 — FedProx] Custom proximal training step
    # ------------------------------------------------------------------

    def _fedprox_train_step(self, model, X, y, global_weights, mu, batch_size=512):
        """[Phase 4 — FedProx] One epoch of training with proximal penalty.

        Instead of model.fit(), we run a manual GradientTape loop that adds
        the FedProx proximal term to the standard MSE loss:

            L_total = MSE(y_true, y_pred) + (mu / 2) * Σ ||w_local - w_global||²

        This penalizes the local model for drifting too far from the global
        server weights, stabilizing convergence under Non-IID data.

        Args:
            model:          Keras model (LSTM or MLP)
            X:              Training features (dense numpy array)
            y:              Training targets (scaled)
            global_weights: Snapshot of server weights BEFORE local training
            mu:             Proximal regularization strength (default 0.1)
            batch_size:     Mini-batch size
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=len(y)).batch(batch_size)

        optimizer = model.optimizer

        for X_batch, y_batch in dataset:
            with tf.GradientTape() as tape:
                # Standard forward pass
                y_pred = model(X_batch, training=True)
                mse_loss = tf.reduce_mean(tf.square(y_batch - tf.squeeze(y_pred)))

                # [FedProx] Proximal penalty: (mu / 2) * Σ ||w - w_global||²
                prox_term = tf.constant(0.0, dtype=tf.float32)
                for w_local, w_global in zip(model.trainable_weights, global_weights):
                    prox_term += tf.reduce_sum(tf.square(w_local - w_global))
                prox_term = (mu / 2.0) * prox_term

                total_loss = mse_loss + prox_term

            gradients = tape.gradient(total_loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    # ------------------------------------------------------------------
    # Flower interface — Phase 4: Split-Federation + FedProx
    # ------------------------------------------------------------------

    def get_parameters(self, config):
        """Extract ONLY the deep learning weights (LSTM + MLP).

        The local ensemble (StackingRegressor) is NEVER sent to the server.
        Network boundary: only continuous deep model weights cross the wire.
        """
        params = []
        # MLP weights (Keras model: 3 Dense layers → 6 arrays: 3 kernels + 3 biases)
        params.extend(self.mlp.get_weights())
        # LSTM weights
        params.extend(self.lstm.get_weights())
        return params

    def set_parameters(self, parameters):
        """Inject global deep learning weights into local LSTM + MLP.

        The parameter list is structured as:
          [mlp_weights..., lstm_weights...]
        """
        n_mlp_arrays = len(self.mlp.get_weights())
        self.mlp.set_weights(parameters[:n_mlp_arrays])
        self.lstm.set_weights(parameters[n_mlp_arrays:])

    def fit(self, parameters, config):
        """[Phase 4 — FedProx + Split-Federation] fit:
           update DL → train DL with proximal penalty → extract embeddings → train local ensemble.

        Execution Flow:
          1. set_parameters: inject global LSTM/MLP weights
          2. Snapshot global weights (for FedProx proximal term)
          3. Train LSTM and MLP with FedProx custom loop (1 epoch each)
          4. Extract embeddings from the updated LSTM/MLP
          5. Load or initialize the local StackingRegressor
          6. Train ensemble on (X_embeddings, y_scaled)
          7. Save ensemble to disk
          8. Return ONLY deep learning weights to the server
        """
        self.set_parameters(parameters)

        # ── [Phase 4 — FedProx] Extract mu from server config ──
        mu = config.get("proximal_mu", 0.1)

        # ── [Phase 4 — FedProx] Snapshot global weights BEFORE local training ──
        # These are the server's weights that we just injected via set_parameters().
        # The proximal penalty will penalize our local weights for drifting from these.
        lstm_global_weights = [tf.constant(w) for w in self.lstm.get_weights()]
        mlp_global_weights  = [tf.constant(w) for w in self.mlp.get_weights()]

        # ── Load training data ──
        X_train_sparse, X_train_dense, y_train_raw = self._load_train_data()
        n_samples = len(y_train_raw)

        # ── Scale y for training ──
        y_train_scaled = self.y_scaler.transform(
            y_train_raw.reshape(-1, 1)
        ).flatten()

        # ── Step 1: Train deep models with FedProx proximal penalty (1 epoch) ──
        # [Phase 4 — FedProx] LSTM training with proximal term
        X_3d = X_train_dense.reshape(X_train_dense.shape[0], 1, X_train_dense.shape[1])
        self._fedprox_train_step(
            model=self.lstm,
            X=X_3d.astype(np.float32),
            y=y_train_scaled.astype(np.float32),
            global_weights=lstm_global_weights,
            mu=mu,
            batch_size=512,
        )
        del X_3d

        # [Phase 4 — FedProx] MLP training with proximal term
        self._fedprox_train_step(
            model=self.mlp,
            X=X_train_dense.astype(np.float32),
            y=y_train_scaled.astype(np.float32),
            global_weights=mlp_global_weights,
            mu=mu,
            batch_size=512,
        )

        # ── Step 2: Extract embeddings from updated deep models ──
        X_embeddings = self._extract_embeddings(X_train_dense)

        # ── Step 3: Load or build local ensemble ──
        if self.ensemble_path.exists():
            ensemble = joblib.load(self.ensemble_path)
        else:
            ensemble = self._build_local_ensemble()

        # ── Step 4: Train ensemble on embeddings ──
        ensemble.fit(X_embeddings, y_train_scaled)

        # ── Step 5: Save ensemble to disk ──
        joblib.dump(ensemble, self.ensemble_path)

        # ── Step 6: Return ONLY deep learning weights ──
        result = self.get_parameters(config={})

        # ── Cleanup ──
        del X_train_sparse, X_train_dense, y_train_raw, y_train_scaled
        del X_embeddings, ensemble
        self._release_memory()

        return result, n_samples, {}

    def evaluate(self, parameters, config):
        """Phase 4 evaluate: update DL → extract embeddings → predict via local ensemble.

        Execution Flow:
          1. set_parameters: inject global LSTM/MLP weights
          2. Extract embeddings from test data
          3. Load the client's local ensemble
          4. Predict: y_pred_scaled = ensemble.predict(X_embeddings)
          5. CRITICAL: inverse_transform both y_pred and y_test back to raw SP scale
          6. Compute MAE and RMSE on the real story point scale
        """
        self.set_parameters(parameters)

        # ── Load test data ──
        X_test_sparse, X_test_dense, y_test_raw = self._load_test_data()
        n_samples = len(y_test_raw)

        # ── Extract embeddings ──
        X_embeddings = self._extract_embeddings(X_test_dense)

        # ── Load local ensemble ──
        if not self.ensemble_path.exists():
            # Fallback: if evaluate is called before fit (e.g., initial eval),
            # use raw LSTM predictions as a baseline
            X_3d = X_test_dense.reshape(X_test_dense.shape[0], 1, X_test_dense.shape[1])
            y_pred_scaled = self.lstm.predict(X_3d, batch_size=512, verbose=0).flatten()
            del X_3d
        else:
            ensemble = joblib.load(self.ensemble_path)
            y_pred_scaled = ensemble.predict(X_embeddings)
            del ensemble

        # ── CRITICAL: Inverse transform to real story point scale ──
        y_pred_raw = self.y_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).flatten()

        # ── Compute metrics on the REAL scale [1, 100] ──
        mae = mean_absolute_error(y_test_raw, y_pred_raw)
        rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_raw))

        # ── Cleanup ──
        del X_test_sparse, X_test_dense, y_test_raw
        del X_embeddings, y_pred_scaled, y_pred_raw
        self._release_memory()

        return float(rmse), n_samples, {"mae": float(mae), "rmse": float(rmse)}


# ======================================================================
# Simulation factory (used by simulation script)
# ======================================================================

def client_fn(cid: str) -> fl.client.Client:
    """Factory function for Flower's simulation engine.

    Creates a fresh FLClient for the given client ID.  Called on-demand
    by the simulation engine — clients are not kept alive between rounds,
    so memory is naturally reclaimed after each fit()/evaluate() cycle.
    """
    return FLClient(cid).to_client()


# ======================================================================
# Standalone mode (used by run scripts on Windows)
# ======================================================================

def main():
    """Launch a single FL client as a standalone process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=str, required=True)
    args = parser.parse_args()

    client = FLClient(args.client_id)
    fl.client.start_client(server_address="127.0.0.1:8082", client=client.to_client())

if __name__ == "__main__":
    main()


# ######################################################################
# HISTORICAL RECORD — Phase 2/3 Client Logic (Commented Out)
# ######################################################################
#
# The code below is the original FLClient implementation used in:
#   - Phase 2: Vanilla FedAvg
#   - Phase 3: FedProx (simulation mode)
#
# Key differences from Phase 4:
#   - get_parameters/set_parameters exchanged sklearn MLP coefs_ + intercepts_
#     (sklearn MLPRegressor) plus Keras LSTM weights
#   - No local ensemble; LSVR trained locally but not aggregated
#   - No StandardScaler on y; raw story points used directly
#   - No embedding extraction; models predicted directly
#
# ──────────────────────────────────────────────────────────────────────
#
# class FLClient(fl.client.NumPyClient):
#     """Phase 2/3 client: federated sklearn MLP + Keras LSTM."""
#
#     def __init__(self, client_id):
#         self.client_id = client_id
#         self.client_dir = FEDERATED_DATA_DIR / f"client_{client_id}"
#         self.train_path = self.client_dir / "train.csv"
#         self.test_path = self.client_dir / "test.csv"
#         self.vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
#         sklearn_models = get_sklearn_base_learners()
#         self.mlp = sklearn_models["MLP"]
#         self.mlp.warm_start = True
#         self.lsvr = sklearn_models["LSVR"]
#         self.lstm = build_keras_lstm(MAX_TFIDF_FEATURES)
#         self._initialize_sklearn_models()
#
#     def _initialize_sklearn_models(self):
#         """Fit on a tiny slice (20 rows) just to create weight arrays."""
#         chunk = pd.read_csv(self.train_path, nrows=20)
#         X_small = self.vectorizer.transform(chunk["text"])
#         y_small = chunk["storypoint"].values
#         self.mlp.fit(X_small, y_small)
#         self.lsvr.fit(X_small, y_small)
#         del chunk, X_small, y_small
#         gc.collect()
#
#     def get_parameters(self, config):
#         """Extract parameters from MLP and LSTM (LSVR is local-only)."""
#         params = []
#         params.extend(self.mlp.coefs_)        # sklearn MLP weight matrices
#         params.extend(self.mlp.intercepts_)   # sklearn MLP bias vectors
#         params.extend(self.lstm.get_weights()) # Keras LSTM weights
#         return params
#
#     def set_parameters(self, parameters):
#         """Inject parameters into MLP and LSTM."""
#         idx = 0
#         n_layers = len(self.mlp.hidden_layer_sizes) + 1
#         self.mlp.coefs_ = list(parameters[idx : idx + n_layers])
#         idx += n_layers
#         self.mlp.intercepts_ = list(parameters[idx : idx + n_layers])
#         idx += n_layers
#         self.lstm.set_weights(parameters[idx:])
#
#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         X_train_sparse, X_train_dense, y_train = self._load_train_data()
#         n_samples = len(y_train)
#         self.mlp.fit(X_train_sparse, y_train)
#         self.lsvr.fit(X_train_sparse, y_train)
#         X_3d = X_train_dense.reshape(X_train_dense.shape[0], 1, X_train_dense.shape[1])
#         self.lstm.fit(X_3d, y_train, epochs=1, batch_size=512, verbose=0)
#         result = self.get_parameters(config={})
#         del X_train_sparse, X_train_dense, y_train, X_3d
#         self._release_memory()
#         return result, n_samples, {}
#
#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         X_test_sparse, X_test_dense, y_test = self._load_test_data()
#         n_samples = len(y_test)
#         X_3d = X_test_dense.reshape(X_test_dense.shape[0], 1, X_test_dense.shape[1])
#         y_pred = self.lstm.predict(X_3d, batch_size=512, verbose=0).flatten()
#         mae = mean_absolute_error(y_test, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         del X_test_sparse, X_test_dense, y_test, X_3d, y_pred
#         self._release_memory()
#         return float(rmse), n_samples, {"mae": float(mae), "rmse": float(rmse)}
#
# ──────────────────────────────────────────────────────────────────────
