"""Sensitivity metrics for the PKPD model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# =============================================================================
# DATA CONTAINER
# =============================================================================

@dataclass
class SensitivityHistory:
    """Full trajectory of states and parameter sensitivities.

    Attributes
    ----------
    t : np.ndarray, shape (N+1,)
        Time grid.
    states : np.ndarray, shape (N+1, 3)
        PK states [Ad, Ac, Ap] at each time step.
    S_states : np.ndarray, shape (N+1, 3, 9)
        Jacobian ∂x/∂p at each time step  (column-major: S[:, i, j] = ∂x_i/∂p_j).
    E : np.ndarray, shape (N+1,)
        PD output (blood pressure via Emax) at each time step.
    dEdp : np.ndarray, shape (N+1, 9)
        Jacobian ∂E/∂p at each time step.
    """
    t: np.ndarray
    states: np.ndarray
    S_states: np.ndarray
    E: np.ndarray
    dEdp: np.ndarray


# =============================================================================
# METRICS
# =============================================================================

def compute_normalized_sensitivity(
    history: SensitivityHistory,
    p_val: np.ndarray,
) -> np.ndarray:
    """Compute dimensionless normalized sensitivity for all outputs and parameters.

    Formula:  norm_sens[t, i, j] = (p_j / scale_i) * (∂output_i/∂p_j)(t)

    where scale_i = max|output_i(t)| over the trajectory.
    Entries where scale_i < 1e-12 are left as NaN.

    Parameters
    ----------
    history : SensitivityHistory
    p_val : np.ndarray, shape (9,)
        Nominal parameter values in canonical PARAM_NAMES order.

    Returns
    -------
    np.ndarray, shape (N+1, 4, 9)
        Normalized sensitivity for outputs [Ad, Ac, Ap, E].
    """
    n_t = len(history.t)
    n_params = len(p_val)

    # Raw Jacobian for all 4 outputs: shape (N+1, 4, 9)
    raw = np.concatenate(
        [history.S_states, history.dEdp[:, np.newaxis, :]],
        axis=1,
    )

    # Output trajectories for scaling: shape (N+1, 4)
    outputs = np.column_stack([history.states, history.E])

    norm_sens = np.full((n_t, 4, n_params), np.nan)
    for i in range(4):
        scale = np.max(np.abs(outputs[:, i]))
        if scale < 1e-12:
            continue
        for j in range(n_params):
            norm_sens[:, i, j] = (p_val[j] / scale) * raw[:, i, j]

    return norm_sens


def compute_l2_norms(
    history: SensitivityHistory,
    p_val: np.ndarray,
) -> np.ndarray:
    """Compute L2-integrated sensitivity norms for all (output, parameter) pairs.

    Formula:  L2[i, j] = sqrt( ∫₀ᵀ (∂output_i/∂p_j)² dt )

    Uses raw (non-normalized) Jacobians — units are preserved.

    Parameters
    ----------
    history : SensitivityHistory
    p_val : np.ndarray, shape (9,)
        Nominal parameter values (unused in computation, kept for API consistency).

    Returns
    -------
    np.ndarray, shape (4, 9)
        L2 norms for outputs [Ad, Ac, Ap, E] × parameters.
    """
    n_params = len(p_val)

    # Raw Jacobian: shape (N+1, 4, 9)
    raw = np.concatenate(
        [history.S_states, history.dEdp[:, np.newaxis, :]],
        axis=1,
    )

    l2 = np.zeros((4, n_params))
    for i in range(4):
        for j in range(n_params):
            l2[i, j] = np.sqrt(np.trapezoid(raw[:, i, j] ** 2, history.t))

    return l2
