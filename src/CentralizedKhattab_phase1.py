"""
train_baseline.py -- Centralized Khattab ensemble baseline.

Architecture (stacking ensemble):
  Base learners:
    1. MLPRegressor          (scikit-learn)
    2. SVR (RBF kernel)      (scikit-learn)
    3. RandomForestRegressor (scikit-learn)
    4. LSTM (Keras)           (TensorFlow / Keras)

  Meta-learner:
    LinearRegression on base-learner out-of-fold predictions.

Pipeline:
  1. Load TF-IDF artifacts from data/features/
  2. Generate out-of-fold (OOF) predictions via K-fold CV for stacking
  3. Train each base learner on the full training set
  4. Fit the meta-learner on OOF predictions
  5. Evaluate on held-out test set
  6. Persist all artifacts to models/baseline/

Usage:
    python src/train_baseline.py
"""

import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import numpy as np
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from config import (
    RANDOM_STATE,
    MAX_TFIDF_FEATURES,
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_TEST_PATH,
    MLP_PATH,
    LSVR_PATH,
    RF_PATH,
    LSTM_MODEL_PATH,
    META_LEARNER_PATH,
    BASELINE_METRICS_PATH,
    PHASE1_MODEL_DIR,
)

# ================================================================
#  Constants
# ================================================================
N_FOLDS = 3          # K-fold for OOF stacking
BATCH_SIZE = 512     # For LSTM dense conversion batches


# ================================================================
#  Helpers
# ================================================================

def sparse_to_dense_f32(X_sparse):
    """Convert sparse matrix to dense float32 (memory-efficient)."""
    return np.asarray(X_sparse.todense(), dtype=np.float32)


def reshape_for_lstm(X_dense):
    """Reshape 2D (samples, features) -> 3D (samples, 1, features) for LSTM."""
    return X_dense.reshape(X_dense.shape[0], 1, X_dense.shape[1])


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def build_keras_lstm(n_features: int):
    """
    Build a small Keras Sequential model with an LSTM layer.

    Architecture:
      Input -> LSTM(64) -> Dense(32, relu) -> Dense(1, linear)

    Returns the compiled model.
    """
    # Lazy import so TF doesn't load unless needed
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, n_features)),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_keras_lstm(X_train_dense, y_train, X_val_dense=None, y_val=None,
                     epochs=30, batch_size=BATCH_SIZE, verbose=0):
    """Train a Keras LSTM and return the fitted model."""
    import tensorflow as tf
    tf.random.set_seed(RANDOM_STATE)

    n_features = X_train_dense.shape[1]
    model = build_keras_lstm(n_features)

    X_3d = reshape_for_lstm(X_train_dense)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if X_val_dense is not None else "loss",
            patience=5,
            restore_best_weights=True,
        ),
    ]

    validation_data = None
    if X_val_dense is not None and y_val is not None:
        validation_data = (reshape_for_lstm(X_val_dense), y_val)

    model.fit(
        X_3d, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
    )
    return model


def predict_keras_lstm(model, X_dense):
    """Predict with a Keras LSTM model (handles 2D->3D reshape)."""
    X_3d = reshape_for_lstm(X_dense)
    return model.predict(X_3d, batch_size=BATCH_SIZE, verbose=0).flatten()


# ================================================================
#  Base Learner Definitions
# ================================================================

def get_sklearn_base_learners():
    """Return a dict of {name: estimator} for the sklearn base learners."""
    return {
        "MLP": MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
            verbose=False,
        ),
        "LSVR": LinearSVR(
            dual=False,
            loss="squared_epsilon_insensitive",
            C=1.0,
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "RF": RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        ),
    }


# ================================================================
#  Stacking: Out-of-Fold Predictions
# ================================================================

def generate_oof_predictions(X_train_sparse, y_train):
    """
    Generate out-of-fold (OOF) predictions for all base learners.

    Uses K-fold CV: for each fold, train each base learner on (K-1) folds,
    predict on the held-out fold.  This produces un-biased predictions
    that the meta-learner can train on.

    Returns:
        oof_preds: np.ndarray of shape (n_train, 4) -- one column per base learner
        base_names: list of base learner names in column order
    """
    n_train = X_train_sparse.shape[0]
    base_names = ["MLP", "LSVR", "RF", "LSTM"]
    oof_preds = np.zeros((n_train, len(base_names)), dtype=np.float64)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_sparse), 1):
        print(f"\n  -- Fold {fold_idx}/{N_FOLDS} --")
        t0 = time.time()

        X_tr_sparse = X_train_sparse[train_idx]
        X_va_sparse = X_train_sparse[val_idx]
        y_tr = y_train[train_idx]
        y_va = y_train[val_idx]

        # Dense copies for LSTM (sklearn learners handle sparse fine)
        X_tr_dense = sparse_to_dense_f32(X_tr_sparse)
        X_va_dense = sparse_to_dense_f32(X_va_sparse)

        # --- Scikit-learn base learners ---
        sklearn_models = get_sklearn_base_learners()

        for i, (name, model) in enumerate(sklearn_models.items()):
            print(f"    Training {name} ...", end=" ", flush=True)
            ts = time.time()
            model.fit(X_tr_sparse, y_tr)
            preds = model.predict(X_va_sparse)
            oof_preds[val_idx, i] = preds
            print(f"done ({time.time() - ts:.1f}s)")

        # --- LSTM ---
        print(f"    Training LSTM ...", end=" ", flush=True)
        ts = time.time()
        lstm_model = train_keras_lstm(
            X_tr_dense, y_tr,
            X_val_dense=X_va_dense, y_val=y_va,
            epochs=30, verbose=0,
        )
        lstm_preds = predict_keras_lstm(lstm_model, X_va_dense)
        oof_preds[val_idx, 3] = lstm_preds
        print(f"done ({time.time() - ts:.1f}s)")

        # Clean up dense arrays to free memory
        del X_tr_dense, X_va_dense
        elapsed = time.time() - t0
        print(f"    Fold {fold_idx} total: {elapsed:.1f}s")

    # Report per-base OOF scores
    print("\n  -- Out-of-Fold Scores --")
    for i, name in enumerate(base_names):
        mae_val = mean_absolute_error(y_train, oof_preds[:, i])
        rmse_val = rmse(y_train, oof_preds[:, i])
        print(f"    {name:6s}  MAE={mae_val:.4f}  RMSE={rmse_val:.4f}")

    return oof_preds, base_names


# ================================================================
#  Main Training Pipeline
# ================================================================

def main():
    print("=" * 60)
    print(" Federated Agile Effort Estimation")
    print(" Phase 1 -- Centralized Khattab Ensemble Baseline")
    print("=" * 60)

    # -- 1. Load TF-IDF artifacts --------------------------------
    print("\n  Loading TF-IDF artifacts ...")
    for p in (X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH):
        if not p.exists():
            print(f"  [X] Missing: {p}")
            print("    Run `python src/tfidf_pipeline.py` first.")
            sys.exit(1)

    X_train_sparse = joblib.load(X_TRAIN_PATH)
    X_test_sparse  = joblib.load(X_TEST_PATH)
    y_train        = joblib.load(Y_TRAIN_PATH)
    y_test         = joblib.load(Y_TEST_PATH)

    print(f"    X_train: {X_train_sparse.shape}  ({X_train_sparse.nnz:,} nnz)")
    print(f"    X_test:  {X_test_sparse.shape}  ({X_test_sparse.nnz:,} nnz)")
    print(f"    y_train: {y_train.shape}  range=[{y_train.min():.0f}, {y_train.max():.0f}]")
    print(f"    y_test:  {y_test.shape}  range=[{y_test.min():.0f}, {y_test.max():.0f}]")

    # -- 2. Generate OOF predictions for stacking ----------------
    print("\n  Generating out-of-fold predictions for stacking ...")
    oof_preds, base_names = generate_oof_predictions(X_train_sparse, y_train)

    # -- 3. Fit meta-learner on OOF predictions ------------------
    print("\n  Fitting meta-learner (LinearRegression) on OOF predictions ...")
    meta_learner = LinearRegression()
    meta_learner.fit(oof_preds, y_train)

    print(f"    Meta-learner coefficients: {dict(zip(base_names, meta_learner.coef_))}")
    print(f"    Meta-learner intercept:    {meta_learner.intercept_:.4f}")

    # -- 4. Re-train all base learners on FULL training set ------
    print("\n  Re-training base learners on full training set ...")

    X_train_dense = sparse_to_dense_f32(X_train_sparse)
    X_test_dense  = sparse_to_dense_f32(X_test_sparse)

    # Scikit-learn models
    sklearn_models = get_sklearn_base_learners()
    trained_sklearn = {}

    for name, model in sklearn_models.items():
        print(f"    Training {name} (full) ...", end=" ", flush=True)
        ts = time.time()
        model.fit(X_train_sparse, y_train)
        trained_sklearn[name] = model
        print(f"done ({time.time() - ts:.1f}s)")

    # LSTM on full training set
    print(f"    Training LSTM (full) ...", end=" ", flush=True)
    ts = time.time()
    lstm_final = train_keras_lstm(
        X_train_dense, y_train,
        epochs=30, verbose=0,
    )
    print(f"done ({time.time() - ts:.1f}s)")

    # -- 5. Generate test predictions ----------------------------
    print("\n  Generating test-set predictions ...")

    test_preds = np.zeros((X_test_sparse.shape[0], len(base_names)), dtype=np.float64)

    for i, (name, model) in enumerate(trained_sklearn.items()):
        test_preds[:, i] = model.predict(X_test_sparse)

    test_preds[:, 3] = predict_keras_lstm(lstm_final, X_test_dense)

    # Meta-learner ensemble prediction
    y_pred_ensemble = meta_learner.predict(test_preds)

    # -- 6. Evaluate ---------------------------------------------
    print("\n" + "=" * 60)
    print(" Test Set Results")
    print("=" * 60)

    metrics = {"base_learners": {}, "ensemble": {}}

    for i, name in enumerate(base_names):
        mae_val = mean_absolute_error(y_test, test_preds[:, i])
        rmse_val = rmse(y_test, test_preds[:, i])
        print(f"    {name:6s}  MAE={mae_val:.4f}  RMSE={rmse_val:.4f}")
        metrics["base_learners"][name] = {"MAE": round(mae_val, 4), "RMSE": round(rmse_val, 4)}

    # Ensemble
    ens_mae = mean_absolute_error(y_test, y_pred_ensemble)
    ens_rmse = rmse(y_test, y_pred_ensemble)
    print(f"    {'ENSEM':6s}  MAE={ens_mae:.4f}  RMSE={ens_rmse:.4f}  <- stacking ensemble")
    metrics["ensemble"] = {"MAE": round(ens_mae, 4), "RMSE": round(ens_rmse, 4)}

    # Store meta-learner info
    metrics["meta_learner"] = {
        "coefficients": dict(zip(base_names, [round(c, 4) for c in meta_learner.coef_])),
        "intercept": round(float(meta_learner.intercept_), 4),
    }

    print("=" * 60)

    # -- 7. Save artifacts ---------------------------------------
    print(f"\n  Saving artifacts to {BASELINE_MODEL_DIR} ...")

    joblib.dump(trained_sklearn["MLP"], MLP_PATH)
    print(f"    [+] {MLP_PATH.name}")

    joblib.dump(trained_sklearn["LSVR"], LSVR_PATH)
    print(f"    [+] {LSVR_PATH.name}")

    joblib.dump(trained_sklearn["RF"], RF_PATH)
    print(f"    [+] {RF_PATH.name}")

    lstm_final.save(str(LSTM_MODEL_PATH))
    print(f"    [+] {LSTM_MODEL_PATH.name}")

    joblib.dump(meta_learner, META_LEARNER_PATH)
    print(f"    [+] {META_LEARNER_PATH.name}")

    with open(BASELINE_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"    [+] {BASELINE_METRICS_PATH.name}")

    # -- Final summary -------------------------------------------
    print("\n" + "=" * 60)
    print(" Phase 1 Baseline Complete")
    print("=" * 60)
    print(f"  Ensemble MAE:  {ens_mae:.4f}")
    print(f"  Ensemble RMSE: {ens_rmse:.4f}")
    print(f"  Artifacts:     {BASELINE_MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
