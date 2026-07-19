# Plasma Wakefield Surrogate Modeling

A small computational-science project combining **plasma physics simulation**
with **machine learning surrogates**, built as a proof of concept for
accelerating parameter scans in plasma wakefield accelerator (PWFA) design.

> This is a prototype built with a simplified 1D analytical/fluid model,
> **not** a replacement for full quasi-static or 3D particle-in-cell (PIC)
> codes like QuickPIC. The goal is to demonstrate the surrogate-modeling
> approach end to end, on a system small enough to simulate thousands of
> times in seconds.

## Motivation

Full PIC simulations of plasma wakefield accelerators are physically
accurate but computationally expensive — a single run can take many CPU/GPU-hours.
This is a major bottleneck for tasks like:

- optimizing beam-loading / transformer ratio over a parameter space,
- scanning bunch length, charge, and plasma density for a target design,
- iterative design loops that need many evaluations.

A common approach in computational science is to train a fast **surrogate
model** — a machine learning model trained on a set of simulation
results — that approximates the simulation's output nearly instantly,
at the cost of some accuracy. This project builds a small, self-contained
version of that idea:

1. Simulate a simplified 1D plasma wakefield model across a range of
   (bunch length, bunch charge) combinations.
2. Train ML surrogates to predict the resulting wakefield from those two
   inputs alone.
3. Benchmark the speedup and visualize both the underlying physics and the
   surrogate's accuracy.

## Physics background

The simulation (`src/physics.py`) uses the standard **quasi-static**
formulation used throughout the PWFA literature (and by full PIC codes such
as QuickPIC): the drive beam is treated as rigid, and the plasma response is
evolved as a function of the co-moving coordinate `xi = z - c*t` rather than
real time. Two models are implemented:

- **Linear theory** (closed form): valid when the beam density is much
  lower than the plasma density. Produces the classic Gaussian-damped
  cosine wake.
- **Nonlinear fluid model** (RK4-integrated ODEs): valid at higher beam
  charge, where the wake steepens — the 1D precursor to the "blowout"
  regime studied in full 2D/3D PIC simulations.

All quantities are in normalized plasma units (`k_p`, `omega_p`), consistent
with standard PWFA papers (e.g. Rosenzweig 1988, Whittum 1997).

## What's actually simulated vs. what's ML

| Component | What it is |
|---|---|
| `src/physics.py` | **Real physics simulation** — numerically integrates the nonlinear cold-fluid quasi-static equations (RK4), and the closed-form linear solution for validation. |
| `src/dataset.py` | Runs the simulation ~1200 times across a sampled parameter grid to build the training set. |
| `src/train_surrogate.py` | Trains ML models (Random Forest, MLP neural nets) to predict simulation output from inputs alone — **no physics inside the ML model itself.** |
| `src/benchmark.py` | Times simulation vs. surrogate inference. |
| `src/visualize.py` | All plots. |

## Results (this run)

- Surrogate for peak wakefield amplitude: **R² ≈ 0.87–0.89** (Random Forest / MLP)
- Full-profile surrogate (predicts the entire wakefield shape, not just the peak): **R² ≈ 0.96**
- Surrogate inference is **~500x faster** than running the full simulation
  (see `results/benchmark.json` and `figures/06_benchmark_speedup.png`)

Exact numbers regenerate slightly differently each run (random sampling) —
see `results/metrics.json` after running the pipeline.

## Repository structure

```
plasma-wakefield-surrogate/
├── README.md
├── requirements.txt
├── main.py                  # run the full pipeline end-to-end
├── src/
│   ├── physics.py            # linear + nonlinear wakefield simulation
│   ├── dataset.py             # parameter sampling + dataset generation
│   ├── train_surrogate.py     # trains scalar + full-profile surrogates
│   ├── benchmark.py           # simulation vs. surrogate runtime
│   └── visualize.py           # generates all figures
├── data/                      # generated datasets (csv + npz)
├── models/                    # trained surrogate models (joblib)
├── results/                   # metrics.json, benchmark.json, test predictions
└── figures/                   # all output plots
```

## Setup & usage

```bash
git clone <this-repo-url>
cd plasma-wakefield-surrogate
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

python main.py     # runs the full pipeline: simulate -> train -> benchmark -> plot
```

Or run each stage independently (useful for iterating on one piece):

```bash
cd src
python dataset.py            # regenerate data/
python train_surrogate.py    # regenerate models/ and results/metrics.json
python benchmark.py          # regenerate results/benchmark.json
python visualize.py          # regenerate figures/
```

## Figures produced

1. `01_wakefield_linear_vs_nonlinear.png` — sanity check: nonlinear simulation matches closed-form linear theory in the low-charge limit.
2. `02_nonlinear_steepening.png` — wake steepening as beam charge increases.
3. `03_phase_space_density.png` — beam density, plasma density response, and plasma electron velocity along the co-moving coordinate.
4. `04_parity_scalar_surrogate.png` — surrogate-predicted vs. simulated peak wakefield (Random Forest & MLP).
5. `05_profile_surrogate_examples.png` — full wakefield shape: surrogate prediction vs. ground-truth simulation, for several test cases.
6. `06_benchmark_speedup.png` — runtime comparison, simulation vs. surrogate.

## Honest limitations / next steps

- This uses a **simplified 1D model**, not real QuickPIC/PIC output. A natural
  next step would be to regenerate the dataset from actual QuickPIC runs
  (2D/3D, full beam dynamics, ion motion, etc.) and retrain the surrogate on
  that — the ML pipeline here (`train_surrogate.py`) would transfer directly.
- The nonlinear model excludes samples that hit (near) wave-breaking, since
  the 1D fluid model isn't well-posed there; a full PIC treatment (like
  QuickPIC) handles this regime natively, which is part of why it's needed.
- Only two input parameters (bunch length, bunch charge) are varied here;
  a real design study would also vary plasma density, bunch radius/shape,
  and possibly asymmetric or hollow-channel profiles.

## Motivation for this project

Built as a demonstration of interest in computational plasma physics /
accelerator science research, connecting a CS/ML background to
simulation-driven physics research (surrogate modeling for expensive
simulations, and quasi-static PIC-style formulations).
