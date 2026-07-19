# Plasma Wakefield Surrogate Modeling

A small computational-science project combining **real particle-in-cell (PIC)
plasma physics simulation** with **machine learning surrogates**, built as a
proof of concept for accelerating parameter scans in plasma wakefield
accelerator (PWFA) design.

> **This version trains on real PIC simulation data**, generated with
> [FBPIC](https://github.com/fbpic/fbpic) — a published, open-source,
> quasi-cylindrical particle-in-cell code used in real accelerator physics
> research. It is not QuickPIC (the quasi-static code used in Dr. Su's own
> work), but it solves the same underlying physics — a relativistic beam
> driving a wake in a plasma — with genuine macroparticle and field data,
> not an analytical approximation. See "Honest limitations" below for
> exactly what that does and doesn't mean.

## Motivation

Full PIC simulations of plasma wakefield accelerators are physically
accurate but computationally expensive. This is a major bottleneck for
tasks like optimizing beam-loading / transformer ratio over a parameter
space, scanning bunch length/charge/density for a target design, or running
iterative design loops that need many evaluations.

A common approach in computational science is to train a fast **surrogate
model** — a machine learning model trained on a set of simulation
results — that approximates the simulation's output nearly instantly, at
the cost of some accuracy. This project builds a small, self-contained,
*and real* version of that idea:

1. Simulate a plasma wakefield (real FBPIC PIC runs) across a range of
   (bunch length, bunch charge) combinations.
2. Train ML surrogates to predict the resulting wakefield from those two
   inputs alone.
3. Benchmark the speedup and visualize both the underlying PIC physics and
   the surrogate's accuracy.

## Physics background

`src/fbpic_sim.py` maps two normalized parameters — `kp_sigma_z` (bunch
length in units of the plasma wavenumber) and `Q_hat` (beam-to-plasma
density ratio) — onto physical units for a fixed reference plasma density
(n0 = 1x10^24 m^-3, i.e. 1x10^18 cm^-3, a typical PWFA experimental density),
then runs a real FBPIC simulation: a relativistic Gaussian electron bunch
driving a wake through a preformed plasma. The on-axis longitudinal field
Ez(z) is extracted and re-normalized back into the same dimensionless units
used throughout the project, so the ML pipeline doesn't care whether the
data came from an analytical model or a real simulation.

`src/physics.py` still contains the original closed-form **linear wakefield
theory** (Rosenzweig 1988) — kept as an independent analytical check, used
in figure 1 to confirm the real PIC simulation reproduces the expected
linear-regime wake shape.

## What's actually simulated vs. what's ML

| Component | What it is |
|---|---|
| `src/fbpic_sim.py` | **Real PIC simulation** — wraps FBPIC, runs an actual relativistic-beam-in-plasma simulation, extracts genuine on-axis field data. |
| `src/physics.py` | Closed-form linear wakefield theory, used only as an independent sanity check (fig. 1). |
| `src/dataset.py` | Runs FBPIC across a sampled parameter grid to build the training set (with checkpointing, since each sample costs several real seconds of compute). |
| `src/train_surrogate.py` | Trains ML models (Random Forest, MLP) to predict simulation output from inputs alone — **no physics inside the ML model itself.** |
| `src/benchmark.py` | Times real FBPIC simulation vs. surrogate inference. |
| `src/visualize.py` | All plots, including real PIC diagnostics (2D charge-density map, macroparticle phase space). |

## Results (this run)

- **Dataset: 100 real FBPIC simulations** (0 numerically unstable / excluded)
- Scalar surrogate (peak wakefield amplitude):
  - **Random Forest: R-squared = 0.968**, MAE = 0.030
  - **MLP: R-squared = 0.40** — noticeably worse than the Random Forest. With only
    ~80 training points, the MLP doesn't have enough data to outperform a
    bagged tree ensemble. This is a genuine, honest result of training on a
    *small real dataset* rather than a bug — see "Honest limitations."
- **Full-profile surrogate** (predicts the entire wakefield shape, not just
  the peak): **R-squared = 0.984**
- **Surrogate inference is ~4,400x faster** than running the real FBPIC
  simulation (5.2 sec/query for the simulation vs. 1.2 ms/query for the
  surrogate — see `results/benchmark.json` and
  `figures/06_benchmark_speedup.png`)

Exact numbers will vary slightly if you regenerate the dataset (different
random parameter samples) — see `results/metrics.json` after running.

## Repository structure

```
plasma-wakefield-surrogate/
├── README.md
├── requirements.txt
├── main.py                  # run the full pipeline end-to-end
├── src/
│   ├── physics.py             # closed-form linear theory (sanity check only)
│   ├── fbpic_sim.py            # real FBPIC simulation wrapper + unit mapping
│   ├── dataset.py              # parameter sampling + dataset generation (checkpointed)
│   ├── train_surrogate.py      # trains scalar + full-profile surrogates
│   ├── benchmark.py            # real simulation vs. surrogate runtime
│   └── visualize.py            # generates all figures
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
```

**Important:** FBPIC needs a working FFT backend. If `pip install fbpic`
doesn't pull in a working FFT library automatically, also run:
```bash
pip install pyfftw
```

Then run the full pipeline:
```bash
python main.py     # runs the full pipeline: simulate -> train -> benchmark -> plot
```

**A note on runtime:** unlike a toy analytical model, each dataset sample
here is a real PIC simulation (~5 seconds on a single CPU core with the
small grid sizes used in this project). Generating the full 100-sample
dataset from scratch takes roughly 8-10 minutes. `dataset.py` checkpoints
progress to `data/` every 5 samples, so if you need to stop and resume
(e.g. on a slower machine), just re-run the same command — it picks up
where it left off.

Or run each stage independently:
```bash
cd src
python dataset.py            # regenerate data/ (real FBPIC runs -- the slow step)
python train_surrogate.py    # regenerate models/ and results/metrics.json
python benchmark.py          # regenerate results/benchmark.json
python visualize.py          # regenerate figures/ (also runs a few extra FBPIC calls)
```

## Figures produced

1. `01_wakefield_linear_vs_nonlinear.png` — sanity check: the real FBPIC simulation matches closed-form linear theory in the low-charge limit.
2. `02_nonlinear_steepening.png` — real FBPIC-simulated wake steepening as beam charge increases.
3. `03_phase_space_density.png` — **genuine PIC diagnostics**: a 2D charge-density map (the canonical PWFA "bubble" picture) and a macroparticle longitudinal phase-space scatter plot, both pulled directly from a real FBPIC run.
4. `04_parity_scalar_surrogate.png` — surrogate-predicted vs. simulated peak wakefield (Random Forest & MLP), on real held-out PIC data.
5. `05_profile_surrogate_examples.png` — full wakefield shape: surrogate prediction vs. ground-truth PIC simulation, for several held-out test cases.
6. `06_benchmark_speedup.png` — runtime comparison, real FBPIC simulation vs. surrogate (~4,400x).

## Honest limitations / next steps

- **FBPIC is a real PIC code, but it is not QuickPIC.** FBPIC solves the
  full explicit electromagnetic PIC equations in quasi-cylindrical (2D
  azimuthal-mode-expansion) geometry; QuickPIC uses the quasi-static
  approximation (evolving the plasma as a function of the co-moving
  coordinate rather than real time), which is what makes it fast enough for
  the large 3D parameter scans her group runs. The data here is genuine PIC
  output, but a like-for-like comparison to her actual QuickPIC results
  would require rerunning with QuickPIC itself.
- **Small dataset, small grids.** Each simulation here uses a deliberately
  small grid (Nz=300, Nr=40) and short run (60 timesteps) so 100 samples
  finish in minutes on a single CPU core. Real design studies would use
  finer resolution, longer propagation distances, and far more samples —
  which is exactly the kind of scan a surrogate model is meant to make
  cheaper.
- **The MLP underperforms Random Forest here specifically because the
  dataset is small** (~80 training points after the train/test split).
  This would very likely flip with a larger dataset — worth revisiting if
  this project is extended with more compute (GPU, cluster, or more time).
- Only two input parameters (bunch length, bunch charge) are varied; a real
  design study would also vary plasma density, bunch radius/shape, and
  asymmetric or hollow-channel profiles.
- **Scaling this up:** the same `dataset.py` script works unchanged with
  larger `n_samples`, finer grids (`Nz`, `Nr` in `fbpic_sim.py`), or a GPU
  (FBPIC supports CUDA -- set `use_cuda=True`) -- all straightforward next
  steps if this becomes an actual lab project.

## Motivation for this project

Built as a demonstration of interest in computational plasma physics /
accelerator science research, connecting a CS/ML background to
simulation-driven physics research (surrogate modeling for expensive
simulations, applied here to real PIC data rather than a toy analytical
stand-in).
