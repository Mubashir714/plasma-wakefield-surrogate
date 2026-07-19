"""
train_surrogate.py
===================

Trains two surrogate models against the simulated dataset:

1. Scalar surrogate: (kp_sigma_z, Q_hat) -> peak_Ez
   Compares a Random Forest and a small MLP neural network.

2. Full-profile surrogate: (kp_sigma_z, Q_hat) -> Ez_hat(xi_hat) vector
   A single MLP with multi-output regression.

Saves trained models (joblib) to models/, and a metrics summary to
results/metrics.json.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def _filter_valid(df: pd.DataFrame):
    """Drop samples that hit (near) wave-breaking; not useful as clean training targets."""
    return df[df["broke"] == 0].reset_index(drop=True)


def train_scalar_surrogates():
    scalar_path = os.path.join(DATA_DIR, "dataset_scalar.csv")
    df = _filter_valid(pd.read_csv(scalar_path))

    X = df[["kp_sigma_z", "Q_hat"]].values
    y = df["peak_Ez"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    results = {}

    # --- Random Forest ---
    t0 = time.perf_counter()
    rf = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=0)
    rf.fit(X_train, y_train)
    rf_train_time = time.perf_counter() - t0
    rf_pred = rf.predict(X_test)
    results["random_forest"] = {
        "r2": r2_score(y_test, rf_pred),
        "mae": mean_absolute_error(y_test, rf_pred),
        "train_time_sec": rf_train_time,
    }

    # --- MLP neural network ---
    t0 = time.perf_counter()
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu",
                        max_iter=3000, random_state=0, early_stopping=True)
    mlp.fit(X_train_s, y_train)
    mlp_train_time = time.perf_counter() - t0
    mlp_pred = mlp.predict(X_test_s)
    results["mlp"] = {
        "r2": r2_score(y_test, mlp_pred),
        "mae": mean_absolute_error(y_test, mlp_pred),
        "train_time_sec": mlp_train_time,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf, os.path.join(MODEL_DIR, "scalar_random_forest.joblib"))
    joblib.dump({"model": mlp, "scaler": scaler}, os.path.join(MODEL_DIR, "scalar_mlp.joblib"))

    # stash test data for later plotting
    np.savez(os.path.join(RESULTS_DIR, "scalar_test_predictions.npz"),
              X_test=X_test, y_test=y_test, rf_pred=rf_pred, mlp_pred=mlp_pred)

    return results


def train_profile_surrogate():
    profile_path = os.path.join(DATA_DIR, "dataset_profile.npz")
    npz = np.load(profile_path)
    kp_sigma_z, Q_hat, xi_grid, profiles = (npz["kp_sigma_z"], npz["Q_hat"],
                                              npz["xi_grid"], npz["profiles"])

    X = np.stack([kp_sigma_z, Q_hat], axis=1)
    y = profiles

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    t0 = time.perf_counter()
    mlp = MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu",
                        max_iter=4000, random_state=0, early_stopping=True)
    mlp.fit(X_train_s, y_train)
    train_time = time.perf_counter() - t0

    y_pred = mlp.predict(X_test_s)
    r2 = r2_score(y_test, y_pred, multioutput="variance_weighted")
    mae = mean_absolute_error(y_test, y_pred)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": mlp, "scaler": scaler, "xi_grid": xi_grid},
                os.path.join(MODEL_DIR, "profile_mlp.joblib"))

    np.savez(os.path.join(RESULTS_DIR, "profile_test_predictions.npz"),
              X_test=X_test, y_test=y_test, y_pred=y_pred, xi_grid=xi_grid)

    return {"r2": r2, "mae": mae, "train_time_sec": train_time}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Training scalar (peak field) surrogates...")
    scalar_results = train_scalar_surrogates()
    for name, m in scalar_results.items():
        print(f"  {name:15s}  R2={m['r2']:.4f}   MAE={m['mae']:.4f}   train_time={m['train_time_sec']:.2f}s")

    print("\nTraining full-profile surrogate...")
    profile_results = train_profile_surrogate()
    print(f"  profile_mlp     R2={profile_results['r2']:.4f}   MAE={profile_results['mae']:.4f}   "
          f"train_time={profile_results['train_time_sec']:.2f}s")

    all_results = {"scalar": scalar_results, "profile": profile_results}
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved metrics to {os.path.join(RESULTS_DIR, 'metrics.json')}")


if __name__ == "__main__":
    main()
