"""
main.py
=======

Runs the full pipeline end to end:
  1. Generate the simulation dataset (src/dataset.py)
  2. Train the surrogate models (src/train_surrogate.py)
  3. Benchmark simulation vs surrogate runtime (src/benchmark.py)
  4. Generate all figures (src/visualize.py)

Usage:
    python main.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import dataset
import train_surrogate
import benchmark
import visualize


def run_all():
    print("=" * 60)
    print("STEP 1/4: Generating simulation dataset")
    print("=" * 60)
    dataset.build_datasets(n_samples=1200)

    print("\n" + "=" * 60)
    print("STEP 2/4: Training surrogate models")
    print("=" * 60)
    train_surrogate.main()

    print("\n" + "=" * 60)
    print("STEP 3/4: Benchmarking simulation vs surrogate")
    print("=" * 60)
    benchmark.run_benchmark(n_queries=200)

    print("\n" + "=" * 60)
    print("STEP 4/4: Generating figures")
    print("=" * 60)
    visualize.main()

    print("\nPipeline complete. See figures/, models/, and results/.")


if __name__ == "__main__":
    run_all()
