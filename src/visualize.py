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

from physics import linear_wakefield
from fbpic_sim import run_fbpic_wakefield

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
    pic = run_fbpic_wakefield(kp_sigma_z, Q_hat)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lin.xi_hat, lin.Ez_hat, label="Linear theory (closed form)", lw=2)
    ax.plot(pic.xi_hat, pic.Ez_hat, "--", label="FBPIC (real PIC simulation)", lw=2)
    ax.axvspan(-3 * kp_sigma_z, 3 * kp_sigma_z, color="grey", alpha=0.15, label="Drive beam region")
    ax.set_xlabel(r"$\hat{\xi} = k_p \xi$")
    ax.set_ylabel(r"$\hat{E}_z$ (normalized wakefield)")
    ax.set_title(f"Wakefield check: linear theory vs real PIC simulation\n"
                 f"($k_p\\sigma_z$={kp_sigma_z}, $\\hat{{Q}}$={Q_hat}, low-charge regime)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "01_wakefield_linear_vs_nonlinear.png"))
    plt.close(fig)


def fig02_nonlinear_steepening():
    kp_sigma_z = 1.0
    Q_values = [0.1, 0.4, 0.8, 1.1]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for Q in Q_values:
        res = run_fbpic_wakefield(kp_sigma_z, Q)
        label = f"$\\hat{{Q}}$={Q}" + ("  (unstable run)" if res.broke else "")
        ax.plot(res.xi_hat, res.Ez_hat, label=label, lw=2)
    ax.set_xlabel(r"$\hat{\xi} = k_p \xi$")
    ax.set_ylabel(r"$\hat{E}_z$")
    ax.set_title("Wake steepening with increasing beam charge\n"
                 "(real FBPIC simulations — precursor to the nonlinear 'blowout' regime)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "02_nonlinear_steepening.png"))
    plt.close(fig)


def fig03_phase_space_density():
    """Uses a real FBPIC run and pulls genuine PIC diagnostics: a 2D charge
    density map (the canonical PWFA 'bubble' picture) and a macroparticle
    phase-space scatter plot."""
    from fbpic_sim import kp_of, N0_REFERENCE, E_CHARGE, M_ELECTRON, C_LIGHT
    from fbpic.main import Simulation
    from fbpic.lpa_utils.bunch import add_particle_bunch_gaussian

    kp_sigma_z, Q_hat = 1.0, 0.8
    n0 = N0_REFERENCE
    kp, wp = kp_of(n0)
    sigma_z = kp_sigma_z / kp
    sig_r = 5e-6
    n_b_peak = Q_hat * n0
    N_particles = n_b_peak * (2 * np.pi) ** 1.5 * sig_r ** 2 * sigma_z

    lambda_p = 2 * np.pi / kp
    Lz, Nz, Nr, Nm = 8 * lambda_p, 300, 40, 2
    rmax = 6 / kp
    zmin, zmax = -0.7 * Lz, 0.3 * Lz
    dt = (Lz / Nz) / C_LIGHT * 0.9

    sim = Simulation(Nz, zmax, Nr, rmax, Nm, dt,
                      p_zmin=zmin, p_zmax=zmax, p_rmin=0, p_rmax=rmax,
                      p_nz=2, p_nr=2, p_nt=4, n_e=n0,
                      zmin=zmin, use_cuda=False,
                      boundaries={"z": "open", "r": "reflective"})
    add_particle_bunch_gaussian(sim, -E_CHARGE, M_ELECTRON,
                                 sig_r=sig_r, sig_z=sigma_z, n_emit=0.,
                                 gamma0=300, sig_gamma=0.,
                                 n_physical_particles=N_particles, n_macroparticles=4000,
                                 zf=zmax * 0.5, boost=None)
    sim.step(60, show_progress=False)

    interp0 = sim.fld.interp[0]
    z_um = interp0.z * 1e6
    r_um = interp0.r * 1e6
    rho = interp0.rho.real

    plasma = sim.ptcl[0]
    beam = sim.ptcl[1]
    rng = np.random.default_rng(0)
    sub = rng.choice(len(plasma.z), size=min(4000, len(plasma.z)), replace=False)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8))

    im = axes[0].pcolormesh(z_um, r_um, rho.T, shading="auto", cmap="RdBu_r")
    axes[0].set_xlabel(r"$z\ (\mu m)$")
    axes[0].set_ylabel(r"$r\ (\mu m)$")
    axes[0].set_title("Real FBPIC charge density (the PWFA 'bubble' — a genuine PIC diagnostic)")
    fig.colorbar(im, ax=axes[0], label="charge density")

    axes[1].scatter((plasma.z[sub]) * 1e6, plasma.uz[sub], s=1, alpha=0.4,
                     color="tab:blue", label="plasma electrons")
    axes[1].scatter(beam.z * 1e6, beam.uz, s=1, alpha=0.6,
                     color="tab:red", label="drive beam")
    axes[1].set_xlabel(r"$z\ (\mu m)$")
    axes[1].set_ylabel(r"$u_z = \gamma v_z / c$")
    axes[1].set_title("Macroparticle longitudinal phase space (real PIC output)")
    axes[1].legend(markerscale=8)

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
