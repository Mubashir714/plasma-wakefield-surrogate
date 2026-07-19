"""
dataset.py
==========

Builds the training dataset for the surrogate model(s) by sampling the
(kp_sigma_z, Q_hat) parameter space and running the nonlinear quasi-static
fluid simulation (`physics.nonlinear_wakefield`) at each sample point.

Two dataset variants are produced:

  * `dataset_scalar.csv`  -> (kp_sigma_z, Q_hat) -> peak_Ez, peak_position, broke
  * `dataset_profile.npz` -> (kp_sigma_z, Q_hat) -> full Ez_hat(xi_hat) profile
                             resampled onto a fixed xi_hat grid, for the
                             full-profile surrogate.

Running this file directly regenerates both dataset files under data/.
"""

import os
import numpy as np
import pandas as pd

from physics import nonlinear_wakefield

# Fixed grid the profile surrogate will be trained/evaluated on
PROFILE_XI_GRID = np.linspace(-8.0, 0.0, 120)  # behind the beam only

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def sample_parameter_space(n_samples: int, seed: int = 42):
    """Latin-hypercube-ish random sampling of (kp_sigma_z, Q_hat)."""
    rng = np.random.default_rng(seed)
    kp_sigma_z = rng.uniform(0.4, 3.0, size=n_samples)
    Q_hat = rng.uniform(0.05, 1.6, size=n_samples)  # up to mildly nonlinear regime
    return kp_sigma_z, Q_hat


def build_datasets(n_samples: int = 1200, seed: int = 42, verbose: bool = True):
    kp_sigma_z_arr, Q_hat_arr = sample_parameter_space(n_samples, seed)

    scalar_rows = []
    profile_rows = []

    for i, (kp_sigma_z, Q_hat) in enumerate(zip(kp_sigma_z_arr, Q_hat_arr)):
        result = nonlinear_wakefield(kp_sigma_z, Q_hat)

        scalar_rows.append({
            "kp_sigma_z": kp_sigma_z,
            "Q_hat": Q_hat,
            "peak_Ez": result.peak_Ez,
            "peak_position": result.peak_position,
            "broke": int(result.broke),
        })

        # Resample the Ez_hat(xi_hat) profile onto the fixed grid for ML use
        profile_interp = np.interp(PROFILE_XI_GRID, result.xi_hat, result.Ez_hat)
        profile_rows.append(profile_interp)

        if verbose and (i + 1) % 200 == 0:
            print(f"  simulated {i + 1}/{n_samples} parameter points...")

    scalar_df = pd.DataFrame(scalar_rows)
    profile_arr = np.array(profile_rows)

    os.makedirs(DATA_DIR, exist_ok=True)
    scalar_path = os.path.join(DATA_DIR, "dataset_scalar.csv")
    profile_path = os.path.join(DATA_DIR, "dataset_profile.npz")

    scalar_df.to_csv(scalar_path, index=False)
    np.savez(profile_path,
             kp_sigma_z=kp_sigma_z_arr,
             Q_hat=Q_hat_arr,
             xi_grid=PROFILE_XI_GRID,
             profiles=profile_arr)

    if verbose:
        n_broke = scalar_df["broke"].sum()
        print(f"\nSaved {len(scalar_df)} samples.")
        print(f"  -> {scalar_path}")
        print(f"  -> {profile_path}")
        print(f"  {n_broke} samples hit near-wave-breaking (excluded from surrogate training).")

    return scalar_df, profile_arr


if __name__ == "__main__":
    build_datasets()
