"""
benchmark.py
============

Compares wall-clock time of:
  (a) running a real FBPIC particle-in-cell simulation
  (b) querying the trained scalar surrogate model

for a batch of parameter points, to quantify the speedup a surrogate offers
inside e.g. an optimization or parameter-scan loop (exactly the kind of loop
used in beam-loading / transformer-ratio optimization studies). Because each
simulation run now costs several real seconds (vs. milliseconds for the toy
analytical model), n_queries is kept small by default.
"""

import os
import time
import json
import joblib
import numpy as np

from fbpic_sim import run_fbpic_wakefield
from dataset import sample_parameter_space

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def run_benchmark(n_queries: int = 15, seed: int = 123):
    kp_sigma_z_arr, Q_hat_arr = sample_parameter_space(n_queries, seed)

    # --- Time the full simulation ---
    t0 = time.perf_counter()
    sim_peaks = []
    for kp_sigma_z, Q_hat in zip(kp_sigma_z_arr, Q_hat_arr):
        result = run_fbpic_wakefield(kp_sigma_z, Q_hat)
        sim_peaks.append(result.peak_Ez)
    sim_time = time.perf_counter() - t0

    # --- Time the surrogate (Random Forest scalar model) ---
    rf = joblib.load(os.path.join(MODEL_DIR, "scalar_random_forest.joblib"))
    X = np.stack([kp_sigma_z_arr, Q_hat_arr], axis=1)

    t0 = time.perf_counter()
    surrogate_peaks = rf.predict(X)
    surrogate_time = time.perf_counter() - t0

    speedup = sim_time / surrogate_time if surrogate_time > 0 else float("inf")

    summary = {
        "n_queries": n_queries,
        "simulation_total_sec": sim_time,
        "simulation_per_query_ms": 1000 * sim_time / n_queries,
        "surrogate_total_sec": surrogate_time,
        "surrogate_per_query_ms": 1000 * surrogate_time / n_queries,
        "speedup_factor": speedup,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "benchmark.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Simulation: {sim_time:.4f}s total  ({summary['simulation_per_query_ms']:.3f} ms/query)")
    print(f"Surrogate:  {surrogate_time:.4f}s total  ({summary['surrogate_per_query_ms']:.5f} ms/query)")
    print(f"Speedup:    {speedup:.0f}x")

    return summary


if __name__ == "__main__":
    run_benchmark()
