import sys
import os
from pathlib import Path
import joblib
import numpy as np

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MAX_TFIDF_FEATURES
from src.CentralizedKhattab_phase1 import build_keras_lstm

def _build_keras_mlp(n_features):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def main():
    mlp = _build_keras_mlp(MAX_TFIDF_FEATURES)
    lstm = build_keras_lstm(MAX_TFIDF_FEATURES)
    
    params = []
    params.extend(mlp.get_weights())
    params.extend(lstm.get_weights())
    
    # Save as .npz
    out_path = Path(__file__).resolve().parent.parent / "models" / "initial_weights.npz"
    np.savez(out_path, *params)
    print(f"Saved {len(params)} arrays to {out_path}")

if __name__ == "__main__":
    main()
