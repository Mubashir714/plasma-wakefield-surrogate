"""
visualize.py
============

Generates all figures for the project, saved to figures/:

  01_wakefield_linear_vs_nonlinear.png   - physics sanity check
  02_nonlinear_steepening.png            - wake steepening vs beam charge
  03_phase_space_density.png             - plasma electron phase space + density
  04_parity_scalar_surrogate.png         - surrogate vs simulation, peak field
  05_profile_surrogate_examples.png      - full-profile surrogate vs ground truth
  06_benchmark_speedup.png               - simulation vs surrogate runtime
"""

import os
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics import linear_wakefield, nonlinear_wakefield

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


def fig01_linear_vs_nonlinear():
    kp_sigma_z, Q_hat = 1.0, 0.15  # small Q_hat -> linear regime should match well
    lin = linear_wakefield(kp_sigma_z, Q_hat)
    nl = nonlinear_wakefield(kp_sigma_z, Q_hat)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lin.xi_hat, lin.Ez_hat, label="Linear theory (closed form)", lw=2)
    ax.plot(nl.xi_hat, nl.Ez_hat, "--", label="Nonlinear fluid simulation", lw=2)
    ax.axvspan(-3 * kp_sigma_z, 3 * kp_sigma_z, color="grey", alpha=0.15, label="Drive beam region")
    ax.set_xlabel(r"$\hat{\xi} = k_p \xi$")
    ax.set_ylabel(r"$\hat{E}_z$ (normalized wakefield)")
    ax.set_title(f"Wakefield check: linear theory vs simulation\n"
                 f"($k_p\\sigma_z$={kp_sigma_z}, $\\hat{{Q}}$={Q_hat}, low-charge regime)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "01_wakefield_linear_vs_nonlinear.png"))
    plt.close(fig)


def fig02_nonlinear_steepening():
    kp_sigma_z = 1.0
    Q_values = [0.1, 0.4, 0.8, 1.2]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for Q in Q_values:
        res = nonlinear_wakefield(kp_sigma_z, Q)
        label = f"$\\hat{{Q}}$={Q}" + ("  (near wave-breaking)" if res.broke else "")
        ax.plot(res.xi_hat, res.Ez_hat, label=label, lw=2)
    ax.set_xlabel(r"$\hat{\xi} = k_p \xi$")
    ax.set_ylabel(r"$\hat{E}_z$")
    ax.set_title("Wake steepening with increasing beam charge\n"
                 "(1D precursor to the nonlinear 'blowout' regime seen in full PIC)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "02_nonlinear_steepening.png"))
    plt.close(fig)


def fig03_phase_space_density():
    kp_sigma_z, Q_hat = 1.0, 1.0
    res = nonlinear_wakefield(kp_sigma_z, Q_hat)

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

    axes[0].plot(res.xi_hat, res.n_beam_hat, color="tab:red")
    axes[0].set_ylabel(r"$\hat{n}_{beam}$")
    axes[0].set_title("Drive beam density profile")

    axes[1].plot(res.xi_hat, res.n_hat, color="tab:blue")
    axes[1].set_ylabel(r"$\hat{n}_e$")
    axes[1].set_title("Plasma electron density response (this is what a real PIC code resolves in 2D/3D)")

    axes[2].plot(res.xi_hat, res.v_hat, color="tab:green")
    axes[2].set_ylabel(r"$\hat{v}_z$")
    axes[2].set_xlabel(r"$\hat{\xi} = k_p \xi$")
    axes[2].set_title("Plasma electron longitudinal velocity (fluid 'phase space' slice)")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "03_phase_space_density.png"))
    plt.close(fig)


def fig04_parity_scalar_surrogate():
    data = np.load(os.path.join(RESULTS_DIR, "scalar_test_predictions.npz"))
    y_test, rf_pred, mlp_pred = data["y_test"], data["rf_pred"], data["mlp_pred"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharex=True, sharey=True)
    lims = [min(y_test.min(), rf_pred.min(), mlp_pred.min()),
            max(y_test.max(), rf_pred.max(), mlp_pred.max())]

    for ax, pred, name in zip(axes, [rf_pred, mlp_pred], ["Random Forest", "MLP Neural Net"]):
        ax.scatter(y_test, pred, s=18, alpha=0.6, edgecolor="none")
        ax.plot(lims, lims, "k--", lw=1, label="ideal")
        ax.set_xlabel("True peak $\\hat{E}_z$ (simulation)")
        ax.set_title(name)
        ax.set_xlim(lims); ax.set_ylim(lims)
    axes[0].set_ylabel("Surrogate-predicted peak $\\hat{E}_z$")
    fig.suptitle("Scalar surrogate: predicted vs. simulated peak wakefield")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "04_parity_scalar_surrogate.png"))
    plt.close(fig)


def fig05_profile_surrogate_examples():
    data = np.load(os.path.join(RESULTS_DIR, "profile_test_predictions.npz"))
    y_test, y_pred, xi_grid = data["y_test"], data["y_pred"], data["xi_grid"]

    idxs = np.random.default_rng(0).choice(len(y_test), size=4, replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
    for ax, idx in zip(axes.flat, idxs):
        ax.plot(xi_grid, y_test[idx], label="Simulation (ground truth)", lw=2)
        ax.plot(xi_grid, y_pred[idx], "--", label="Surrogate prediction", lw=2)
        ax.set_title(f"Test sample #{idx}")
        ax.legend(fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\hat{\xi}$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\hat{E}_z$")
    fig.suptitle("Full-profile surrogate: predicted vs. simulated wakefield shape")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "05_profile_surrogate_examples.png"))
    plt.close(fig)


def fig06_benchmark_speedup():
    with open(os.path.join(RESULTS_DIR, "benchmark.json")) as f:
        bench = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))

    axes[0].bar(["Simulation", "Surrogate"],
                [bench["simulation_per_query_ms"], bench["surrogate_per_query_ms"]],
                color=["tab:red", "tab:blue"])
    axes[0].set_ylabel("ms per query")
    axes[0].set_yscale("log")
    axes[0].set_title("Per-query runtime (log scale)")

    axes[1].text(0.5, 0.6, f"{bench['speedup_factor']:.0f}x", fontsize=42,
                 ha="center", va="center", weight="bold", color="tab:blue")
    axes[1].text(0.5, 0.25, "faster than running\nthe full simulation", fontsize=12, ha="center")
    axes[1].axis("off")

    fig.suptitle(f"Surrogate speedup over {bench['n_queries']} evaluations")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "06_benchmark_speedup.png"))
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    fig01_linear_vs_nonlinear()
    fig02_nonlinear_steepening()
    fig03_phase_space_density()
    fig04_parity_scalar_surrogate()
    fig05_profile_surrogate_examples()
    fig06_benchmark_speedup()
    print(f"Saved 6 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
