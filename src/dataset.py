"""
dataset.py
==========

Builds the training dataset for the surrogate model(s) by sampling the
(kp_sigma_z, Q_hat) parameter space and running a REAL particle-in-cell
simulation (FBPIC, see fbpic_sim.py) at each sample point.

NOTE ON DATASET SIZE: each sample is now a genuine PIC simulation
(~3-6 seconds on CPU for the small grids used here), not a closed-form/ODE
evaluation (~milliseconds). The sample count is deliberately much smaller
than an analytical-model dataset would use as a result. See README.md for
guidance on scaling this up (more samples, finer resolution, GPU) on a more
capable machine.

Two dataset variants are produced (same file names/format as before, so
train_surrogate.py needs no changes):

  * `dataset_scalar.csv`  -> (kp_sigma_z, Q_hat) -> peak_Ez, peak_position, broke
  * `dataset_profile.npz` -> (kp_sigma_z, Q_hat) -> full Ez_hat(xi_hat) profile
                             resampled onto a fixed xi_hat grid
"""

import os
import time
import numpy as np
import pandas as pd

from fbpic_sim import run_fbpic_wakefield

# Fixed grid the profile surrogate will be trained/evaluated on
PROFILE_XI_GRID = np.linspace(-8.0, 0.0, 120)  # behind the beam only

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def sample_parameter_space(n_samples: int, seed: int = 42):
    """Random sampling of (kp_sigma_z, Q_hat), kept within a range where the
    small FBPIC grids used here remain numerically well-resolved."""
    rng = np.random.default_rng(seed)
    kp_sigma_z = rng.uniform(0.4, 2.0, size=n_samples)
    Q_hat = rng.uniform(0.05, 1.1, size=n_samples)
    return kp_sigma_z, Q_hat


def build_datasets(n_samples: int = 150, seed: int = 42, verbose: bool = True,
                    start_idx: int = 0, end_idx: int = None, checkpoint_every: int = 5):
    """Build (or resume) the dataset.

    Supports partial/chunked generation via start_idx/end_idx so a long run
    can be split across multiple shorter invocations without losing
    progress: results are checkpointed to disk every `checkpoint_every`
    samples, and a resumed call with the same seed/n_samples reuses any
    already-computed rows found in the existing CSV.
    """
    kp_sigma_z_arr, Q_hat_arr = sample_parameter_space(n_samples, seed)
    if end_idx is None:
        end_idx = n_samples

    os.makedirs(DATA_DIR, exist_ok=True)
    scalar_path = os.path.join(DATA_DIR, "dataset_scalar.csv")
    profile_path = os.path.join(DATA_DIR, "dataset_profile.npz")

    # Resume from existing checkpoint if present
    if os.path.exists(scalar_path):
        existing_df = pd.read_csv(scalar_path)
        existing_profiles = np.load(profile_path)["profiles"] if os.path.exists(profile_path) else np.zeros((0, len(PROFILE_XI_GRID)))
        n_done = len(existing_df)
    else:
        existing_df = pd.DataFrame(columns=["kp_sigma_z", "Q_hat", "peak_Ez", "peak_position", "broke", "run_time_sec"])
        existing_profiles = np.zeros((0, len(PROFILE_XI_GRID)))
        n_done = 0

    scalar_rows = existing_df.to_dict("records")
    profile_rows = list(existing_profiles)

    actual_start = max(start_idx, n_done)
    t_start = time.time()
    for i in range(actual_start, min(end_idx, n_samples)):
        kp_sigma_z, Q_hat = kp_sigma_z_arr[i], Q_hat_arr[i]
        result = run_fbpic_wakefield(kp_sigma_z, Q_hat)

        scalar_rows.append({
            "kp_sigma_z": kp_sigma_z,
            "Q_hat": Q_hat,
            "peak_Ez": result.peak_Ez,
            "peak_position": result.peak_position,
            "broke": int(result.broke),
            "run_time_sec": result.run_time_sec,
        })
        profile_interp = np.interp(PROFILE_XI_GRID, result.xi_hat, result.Ez_hat)
        profile_rows.append(profile_interp)

        done_count = i + 1 - actual_start
        if verbose and (i + 1) % 5 == 0:
            elapsed = time.time() - t_start
            print(f"  simulated sample {i + 1}/{n_samples}  (this batch: {done_count} in {elapsed/60:.1f} min)")

        if (i + 1) % checkpoint_every == 0 or (i + 1) == min(end_idx, n_samples):
            pd.DataFrame(scalar_rows).to_csv(scalar_path, index=False)
            np.savez(profile_path, kp_sigma_z=kp_sigma_z_arr[:len(scalar_rows)],
                     Q_hat=Q_hat_arr[:len(scalar_rows)], xi_grid=PROFILE_XI_GRID,
                     profiles=np.array(profile_rows))

    scalar_df = pd.DataFrame(scalar_rows)
    profile_arr = np.array(profile_rows)

    if verbose:
        n_broke = scalar_df["broke"].sum() if len(scalar_df) else 0
        print(f"\nDataset now has {len(scalar_df)}/{n_samples} samples "
              f"({n_broke} unstable/excluded).")
        if len(scalar_df) < n_samples:
            print(f"  Run again (same n_samples/seed) to continue from sample {len(scalar_df)}.")
        else:
            print(f"  -> {scalar_path}\n  -> {profile_path}")

    return scalar_df, profile_arr


if __name__ == "__main__":
    build_datasets()
