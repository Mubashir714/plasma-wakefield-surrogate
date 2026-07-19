"""
physics.py
==========

Simplified plasma wakefield physics, implemented in normalized (dimensionless)
plasma units, following the standard 1D quasi-static beam-driven wakefield
formulation used throughout the PWFA literature (e.g. Rosenzweig 1988;
Whittum 1997). This is the same *quasi-static* framing used by full PIC codes
such as QuickPIC (the beam is treated as rigid over the short timescale of
the plasma response, and the plasma is evolved as a function of the
co-moving coordinate xi = z - c*t rather than real time).

Two models are implemented:

1. `linear_wakefield(xi_hat, kp_sigma_z, Q_hat)`
   Closed-form solution of the *linearized* fluid equations for a Gaussian
   drive bunch. Valid when Q_hat << 1 (beam density << plasma density).

2. `nonlinear_wakefield(xi_hat, kp_sigma_z, Q_hat)`
   Numerical (RK4) integration of the full nonlinear 1D quasi-static
   cold-fluid equations. This captures wake steepening at larger Q_hat,
   which is the 1D precursor to the "blowout regime" studied with full
   2D/3D PIC codes.

All quantities are normalized:
    xi_hat        = k_p * xi                (co-moving position)
    n_hat          = n_e / n0                (plasma electron density)
    v_hat          = v_z / c                 (plasma electron velocity)
    Ez_hat         = e * Ez / (m_e * c * wp)  (longitudinal field)
    Q_hat          = n_b_peak / n0            (beam-to-plasma density ratio)
    kp_sigma_z     = k_p * sigma_z            (normalized bunch length)

This is a teaching/prototyping model, not a replacement for full PIC.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class WakefieldResult:
    xi_hat: np.ndarray      # co-moving coordinate grid (normalized)
    n_hat: np.ndarray       # plasma electron density (normalized)
    v_hat: np.ndarray       # plasma electron longitudinal velocity (normalized)
    Ez_hat: np.ndarray      # longitudinal wakefield (normalized)
    n_beam_hat: np.ndarray  # drive beam density profile (normalized)
    peak_Ez: float          # max |Ez_hat| behind the beam (accelerating field)
    peak_position: float    # xi_hat location of that peak
    broke: bool              # True if the simulation hit (near) wave-breaking


def gaussian_beam_profile(xi_hat: np.ndarray, Q_hat: float, kp_sigma_z: float) -> np.ndarray:
    """Normalized Gaussian drive-beam density profile n_b_hat(xi_hat).

    The beam is centered at xi_hat = 0, with its head at xi_hat > 0 and
    its wake trailing at xi_hat < 0 (standard PWFA convention: beam moves
    in +z, xi = z - ct, so structures behind the beam appear at xi < 0).
    """
    return Q_hat * np.exp(-(xi_hat ** 2) / (2.0 * kp_sigma_z ** 2))


def linear_wakefield(kp_sigma_z: float, Q_hat: float, xi_range: float = 12.0,
                      n_points: int = 600) -> WakefieldResult:
    """Closed-form linear wakefield behind a Gaussian drive bunch.

    Standard result (e.g. Rosenzweig 1988): convolving the Gaussian beam
    charge profile with the plasma's impulse response (a cosine at the
    plasma wavenumber) gives a Gaussian-damped cosine wake:

        Ez_hat(xi_hat) = -Q_hat * sqrt(pi/2) * kp_sigma_z
                          * exp(-kp_sigma_z^2 / 2) * cos(xi_hat)   for xi_hat < ~0 (behind the beam)

    valid only in the linear regime Q_hat << 1.
    """
    xi_hat = np.linspace(-xi_range, xi_range, n_points)
    n_beam_hat = gaussian_beam_profile(xi_hat, Q_hat, kp_sigma_z)

    amplitude = Q_hat * np.sqrt(np.pi / 2.0) * kp_sigma_z * np.exp(-(kp_sigma_z ** 2) / 2.0)
    Ez_hat = np.where(xi_hat <= 0, -amplitude * np.cos(xi_hat), -amplitude * np.cos(xi_hat) * np.exp(-((xi_hat) ** 2) / (2 * kp_sigma_z ** 2) * 0.0))
    # Ahead of the beam center the linear response is negligible (causality-like damping);
    # we keep the model simple and only report/trust xi_hat <= 0 (behind the drive beam).

    v_hat = np.gradient(-Ez_hat, xi_hat)  # not physically integrated; for linear model we don't need it
    n_hat = np.ones_like(xi_hat)

    behind = xi_hat <= 0
    idx = np.argmax(np.abs(Ez_hat[behind]))
    peak_Ez = float(Ez_hat[behind][idx])
    peak_position = float(xi_hat[behind][idx])

    return WakefieldResult(xi_hat, n_hat, v_hat, Ez_hat, n_beam_hat,
                            peak_Ez=peak_Ez, peak_position=peak_position, broke=False)


def nonlinear_wakefield(kp_sigma_z: float, Q_hat: float, xi_range: float = 12.0,
                         n_points: int = 4000) -> WakefieldResult:
    """RK4 integration of the nonlinear 1D quasi-static cold-fluid equations.

    Governing ODEs (co-moving coordinate xi_hat plays the role of "time"):

        d(v_hat)/d(xi_hat)  = -Ez_hat / (1 - v_hat)
        d(Ez_hat)/d(xi_hat) = n_beam_hat(xi_hat) + n_hat - 1
        n_hat                = 1 / (1 - v_hat)          [flux conservation]

    Integration proceeds from the head of the beam (xi_hat = +xi_range,
    undisturbed plasma: v_hat=0, Ez_hat=0) backward to the tail (xi_hat = -xi_range).
    For small Q_hat this reduces to the linear solution above; for larger
    Q_hat the wake steepens (nonlinear precursor to wave-breaking / blowout).
    """
    xi_hat = np.linspace(xi_range, -xi_range, n_points)  # integrate head -> tail
    dxi = xi_hat[1] - xi_hat[0]  # negative step

    v = np.zeros(n_points)
    Ez = np.zeros(n_points)
    n_e = np.ones(n_points)

    broke = False

    def derivs(v_, Ez_, xi_):
        denom = 1.0 - v_
        if abs(denom) < 1e-3:
            denom = np.sign(denom) * 1e-3 if denom != 0 else 1e-3
        dv = -Ez_ / denom
        n_here = 1.0 / denom
        dEz = gaussian_beam_profile(np.array([xi_]), Q_hat, kp_sigma_z)[0] + n_here - 1.0
        return dv, dEz, n_here

    for i in range(n_points - 1):
        xi_i = xi_hat[i]
        v_i, Ez_i = v[i], Ez[i]

        if abs(1.0 - v_i) < 5e-3:
            broke = True
            v[i + 1:] = v_i
            Ez[i + 1:] = Ez_i
            n_e[i + 1:] = n_e[i]
            break

        k1v, k1E, _ = derivs(v_i, Ez_i, xi_i)
        k2v, k2E, _ = derivs(v_i + 0.5 * dxi * k1v, Ez_i + 0.5 * dxi * k1E, xi_i + 0.5 * dxi)
        k3v, k3E, _ = derivs(v_i + 0.5 * dxi * k2v, Ez_i + 0.5 * dxi * k2E, xi_i + 0.5 * dxi)
        k4v, k4E, _ = derivs(v_i + dxi * k3v, Ez_i + dxi * k3E, xi_i + dxi)

        v[i + 1] = v_i + (dxi / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        Ez[i + 1] = Ez_i + (dxi / 6.0) * (k1E + 2 * k2E + 2 * k3E + k4E)
        _, _, n_e[i + 1] = derivs(v[i + 1], Ez[i + 1], xi_hat[i + 1])

    # re-sort ascending in xi_hat for downstream plotting/dataset use
    order = np.argsort(xi_hat)
    xi_hat_sorted = xi_hat[order]
    v_sorted = v[order]
    Ez_sorted = Ez[order]
    n_sorted = n_e[order]
    n_beam_hat = gaussian_beam_profile(xi_hat_sorted, Q_hat, kp_sigma_z)

    behind = xi_hat_sorted <= 0
    if np.any(behind):
        idx = np.argmax(np.abs(Ez_sorted[behind]))
        peak_Ez = float(Ez_sorted[behind][idx])
        peak_position = float(xi_hat_sorted[behind][idx])
    else:
        peak_Ez, peak_position = 0.0, 0.0

    return WakefieldResult(xi_hat_sorted, n_sorted, v_sorted, Ez_sorted, n_beam_hat,
                            peak_Ez=peak_Ez, peak_position=peak_position, broke=broke)
