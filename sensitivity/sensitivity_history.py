from dataclasses import dataclass

import numpy as np

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