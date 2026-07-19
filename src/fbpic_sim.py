"""
fbpic_sim.py
============

Wraps FBPIC (https://github.com/fbpic/fbpic) — a real, published,
quasi-cylindrical particle-in-cell code — to generate *actual* kinetic PIC
wakefield data, in place of the simplified analytical/fluid model in
physics.py.

FBPIC is a full electromagnetic PIC code (not quasi-static like QuickPIC),
but it solves the same underlying physics (a relativistic beam driving a
wake in a plasma) and produces genuine macroparticle + field data. This
module:

  1. Maps our normalized parameters (kp_sigma_z, Q_hat) onto physical units
     (plasma density, bunch length, bunch charge) for a fixed reference
     plasma density n0.
  2. Builds and runs a small FBPIC simulation for those physical parameters.
  3. Extracts the on-axis longitudinal wakefield Ez(xi) behind the bunch,
     re-normalizes it back into the same normalized units used throughout
     the rest of this project, so it is a drop-in replacement for
     `physics.nonlinear_wakefield`.

Because each call runs a real (if small) PIC simulation, this is
computationally far more expensive per sample than the analytical model
(~seconds per run vs. milliseconds) — see README.md for the resulting
dataset-size tradeoff.
"""

from dataclasses import dataclass
import numpy as np

# Physical constants (SI)
E_CHARGE = 1.602176634e-19
M_ELECTRON = 9.1093837015e-31
EPS0 = 8.8541878128e-12
C_LIGHT = 2.99792458e8

# Reference plasma density used for all runs in this project.
# (1e24 m^-3 = 1e18 cm^-3, a typical PWFA experimental density.)
N0_REFERENCE = 1e24


@dataclass
class FBPICWakefieldResult:
    xi_hat: np.ndarray
    Ez_hat: np.ndarray
    n_beam_hat: np.ndarray
    peak_Ez: float
    peak_position: float
    broke: bool          # True if the run looked numerically unstable / unusable
    n0: float
    sigma_z_m: float
    N_particles: float
    run_time_sec: float


def kp_of(n0: float):
    """Plasma wavenumber and frequency for density n0 (SI units)."""
    wp = np.sqrt(n0 * E_CHARGE ** 2 / (EPS0 * M_ELECTRON))
    return wp / C_LIGHT, wp


def run_fbpic_wakefield(kp_sigma_z: float, Q_hat: float, n0: float = N0_REFERENCE,
                         sig_r: float = 5e-6, n_steps: int = 60, gamma0: float = 300,
                         n_macroparticles: int = 4000, Nz: int = 300, Nr: int = 40,
                         Nm: int = 2) -> FBPICWakefieldResult:
    """Run one real FBPIC simulation and return the wake in normalized units.

    kp_sigma_z : normalized bunch length (k_p * sigma_z)
    Q_hat      : beam-to-plasma peak density ratio (n_b_peak / n0)
    """
    import time
    from fbpic.main import Simulation
    from fbpic.lpa_utils.bunch import add_particle_bunch_gaussian

    kp, wp = kp_of(n0)
    sigma_z = kp_sigma_z / kp
    n_b_peak = Q_hat * n0
    # Peak density of a 3D Gaussian bunch: n(0,0) = N / ((2*pi)^1.5 * sig_r^2 * sig_z)
    N_particles = n_b_peak * (2 * np.pi) ** 1.5 * sig_r ** 2 * sigma_z

    lambda_p = 2 * np.pi / kp
    Lz = 8 * lambda_p
    rmax = 6 / kp
    zmin = -0.7 * Lz
    zmax = 0.3 * Lz
    dt = (Lz / Nz) / C_LIGHT * 0.9

    broke = False
    t0 = time.time()
    try:
        sim = Simulation(Nz, zmax, Nr, rmax, Nm, dt,
                          p_zmin=zmin, p_zmax=zmax, p_rmin=0, p_rmax=rmax,
                          p_nz=2, p_nr=2, p_nt=4, n_e=n0,
                          zmin=zmin, use_cuda=False,
                          boundaries={"z": "open", "r": "reflective"})

        add_particle_bunch_gaussian(sim, -E_CHARGE, M_ELECTRON,
                                     sig_r=sig_r, sig_z=sigma_z, n_emit=0.,
                                     gamma0=gamma0, sig_gamma=0.,
                                     n_physical_particles=N_particles,
                                     n_macroparticles=n_macroparticles,
                                     zf=zmax * 0.5, boost=None)

        sim.step(n_steps, show_progress=False)

        interp0 = sim.fld.interp[0]
        z = interp0.z
        Ez_axis = interp0.Ez[:, 0].real
        n_beam_axis = np.abs(interp0.rho[:, 0].real) / E_CHARGE  # rough density proxy

        zbeam = float(sim.ptcl[1].z.mean())
        xi = z - zbeam
        xi_hat = kp * xi
        Ez_hat = (E_CHARGE * Ez_axis) / (M_ELECTRON * C_LIGHT * wp)
        n_beam_hat = n_beam_axis / n0

        if not np.all(np.isfinite(Ez_hat)) or np.max(np.abs(Ez_hat)) > 50:
            broke = True

    except Exception as exc:  # pragma: no cover - defensive: some parameter
        # combinations can be numerically unstable at low resolution.
        broke = True
        xi_hat = np.linspace(-8, 0, 50)
        Ez_hat = np.zeros_like(xi_hat)
        n_beam_hat = np.zeros_like(xi_hat)

    run_time = time.time() - t0

    order = np.argsort(xi_hat)
    xi_hat, Ez_hat, n_beam_hat = xi_hat[order], Ez_hat[order], n_beam_hat[order]

    behind = xi_hat <= 0
    if np.any(behind) and not broke:
        idx = int(np.argmax(np.abs(Ez_hat[behind])))
        peak_Ez = float(Ez_hat[behind][idx])
        peak_position = float(xi_hat[behind][idx])
    else:
        peak_Ez, peak_position = 0.0, 0.0

    return FBPICWakefieldResult(xi_hat, Ez_hat, n_beam_hat, peak_Ez, peak_position,
                                 broke, n0, sigma_z, N_particles, run_time)
