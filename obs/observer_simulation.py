"""
Luenberger Observer Simulation for Extended PKPD System

This script implements an observer to estimate:
- x1 = Ad (depot compartment)
- x2 = Ac (central compartment)
- x3 = Ap (peripheral compartment)
- x4 = Vc*(EC50 + Cendo) (parameter embedding)

From blood pressure observations using the measurement equation:
g(y) = (E0 - Emax) * Vc * EC50 / (y - Emax) = x2 + x4

Observer dynamics: x_hat_dot = (A - LC) x_hat + B*u + L*g(y)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.signal import place_poles
from scipy.linalg import solve_continuous_are
from scipy.interpolate import interp1d
from typing import Dict, Tuple, Optional, Union, List

from pkpd import NorepinephrinePKPD
from utils.datatools import (
    load_observations,
    load_injections,
    load_resimulated_trajectories
)
from utils.plots import (
    plot_observer_individual_states,
    plot_observer_summary,
    plot_observer_estimation_error
)
from opti.postprocessing import load_optimized_parameters


# =============================================================================
# SYSTEM MATRICES
# =============================================================================

def build_system_matrices(params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build state-space matrices A, B, C from PKPD parameters.

    Extended state: x = [Ad, Ac, Ap, Vc*(EC50+Cendo)]^T

    Args:
        params: Dictionary with keys k_a, k_12, k_21, k_el

    Returns:
        A (4x4), B (4x1), C (1x4) matrices
    """
    k_a = params['k_a']
    k_12 = params['k_12']
    k_21 = params['k_21']
    k_el = params['k_el']

    # State matrix (4x4)
    A = np.array([
        [-k_a,              0,      0,    0],
        [k_a,  -(k_12 + k_el),  k_21,    0],
        [0,             k_12, -k_21,    0],
        [0,                0,     0,    0]  # x4 is constant
    ])

    # Input matrix (4x1) - injection affects Ad
    B = np.array([[1], [0], [0], [0]])

    # Output matrix (1x4) - we observe x2 + x4
    C = np.array([[0, 1, 0, 1]])

    return A, B, C


def compute_observability_matrix(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Compute the observability matrix O = [C; CA; CA^2; CA^3].

    Args:
        A: State matrix (n x n)
        C: Output matrix (1 x n)

    Returns:
        Observability matrix (n x n)
    """
    n = A.shape[0]
    O = np.zeros((n, n))
    CA_power = C.copy()

    for i in range(n):
        O[i, :] = CA_power
        CA_power = CA_power @ A

    return O


def check_observability(A: np.ndarray, C: np.ndarray) -> Tuple[bool, int]:
    """Check if the system (A, C) is observable.

    Args:
        A: State matrix
        C: Output matrix

    Returns:
        (is_observable, rank) tuple
    """
    O = compute_observability_matrix(A, C)
    rank = np.linalg.matrix_rank(O)
    n = A.shape[0]
    return rank == n, rank


def compute_observer_gain(A: np.ndarray, C: np.ndarray,
                          desired_poles: list) -> np.ndarray:
    """Compute observer gain L via pole placement.

    For observer: x_hat_dot = (A - LC) x_hat + ...
    We need eigenvalues of (A - LC) to be the desired poles.

    Using duality: place poles for (A^T, C^T) then L = K^T

    Args:
        A: State matrix (n x n)
        C: Output matrix (1 x n)
        desired_poles: List of n desired eigenvalues (must be negative)

    Returns:
        Observer gain L (n x 1)
    """
    # Pole placement for dual system (A^T, C^T)
    result = place_poles(A.T, C.T, desired_poles)
    K = result.gain_matrix
    L = K.T
    return L


def compute_kalman_gain(A: np.ndarray, C: np.ndarray,
                        Q: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute steady-state Kalman gain via Riccati equation.

    Solves the Continuous Algebraic Riccati Equation (CARE):
        A Σ + Σ A^T - Σ C^T R^{-1} C Σ + Q = 0

    Then computes optimal gain: L = Σ C^T R^{-1}

    Args:
        A: State matrix (4x4)
        C: Output matrix (1x4)
        Q: Process noise covariance (4x4) - higher = less trust in model
        R: Measurement noise covariance (1x1) - higher = less trust in measurements

    Returns:
        L: Kalman gain (4x1)
        Sigma: Steady-state error covariance (4x4)
    """
    # scipy's solve_continuous_are solves: A^T X + X A - X B R^{-1} B^T X + Q = 0
    # For observer (dual problem), we use A.T and C.T
    Sigma = solve_continuous_are(A.T, C.T, Q, R)

    # Kalman gain: L = Σ C^T R^{-1}
    L = Sigma @ C.T @ np.linalg.inv(R)

    return L, Sigma


# =============================================================================
# OBSERVER SIMULATION
# =============================================================================

def create_bp_interpolator(bp_observations: list,
                           method: str = 'linear') -> callable:
    """Create an interpolator for blood pressure observations.

    Args:
        bp_observations: List of (time, bp_value) tuples
        method: Interpolation method ('linear', 'nearest', 'zero')

    Returns:
        Interpolator function: t -> bp(t)
    """
    times = np.array([obs[0] for obs in bp_observations])
    values = np.array([obs[1] for obs in bp_observations])

    # Sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    values = values[sort_idx]

    if method == 'zero':
        # Zero-order hold (previous value)
        interp_func = interp1d(times, values, kind='previous',
                               bounds_error=False, fill_value=(values[0], values[-1]))
    else:
        interp_func = interp1d(times, values, kind=method,
                               bounds_error=False, fill_value=(values[0], values[-1]))

    return interp_func


def simulate_observer(A: np.ndarray, B: np.ndarray, C: np.ndarray, L: np.ndarray,
                      pkpd_model: NorepinephrinePKPD,
                      bp_interp: callable,
                      t_span: np.ndarray,
                      x_hat_0: np.ndarray,
                      patient_id: int,
                      dt: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate Luenberger observer dynamics using implicit Euler.

    Observer: x_hat_dot = (A - LC) x_hat + B*u + L*g(y)

    Implicit Euler scheme:
        x[k+1] = x[k] + dt * (A_obs @ x[k+1] + f[k+1])
        => (I - dt*A_obs) @ x[k+1] = x[k] + dt * f[k+1]

    Args:
        A, B, C: System matrices
        L: Observer gain (4x1)
        pkpd_model: PKPD model instance (for INOR and g(y) computation)
        bp_interp: Blood pressure interpolator function
        t_span: Time points for simulation
        x_hat_0: Initial observer state (4,)
        patient_id: Patient ID for INOR lookup
        dt: Integration time step (unused, uses t_span spacing)

    Returns:
        t_out: Time array
        x_hat_out: Estimated states (4 x n_times)
    """
    A_obs = A - L @ C  # Observer dynamics matrix
    I = np.eye(4)

    n_steps = len(t_span)
    x_hat = np.zeros((4, n_steps))
    x_hat[:, 0] = x_hat_0

    for i in range(n_steps - 1):
        t_next = t_span[i+1]
        dt_i = t_next - t_span[i]

        # Get blood pressure at next time step (implicit)
        y_next = bp_interp(t_next)

        # Compute g(y) = x2 + x4 at next time
        g_y_next = pkpd_model.bp_obs_to_x_variables(y_next)

        # Get input u(t) at next time step
        u_next = pkpd_model.INOR(t_next, patient_id)

        # Forcing term: f = B*u + L*g(y)
        f_next = B.flatten() * u_next + L.flatten() * g_y_next

        # Implicit Euler: (I - dt*A_obs) @ x[k+1] = x[k] + dt * f[k+1]
        lhs_matrix = I - dt_i * A_obs
        rhs_vector = x_hat[:, i] + dt_i * f_next

        x_hat[:, i+1] = np.linalg.solve(lhs_matrix, rhs_vector)

    return t_span, x_hat


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(x_true: np.ndarray, x_hat: np.ndarray,
                    state_names: list) -> Dict[str, Dict[str, float]]:
    """Compute comparison metrics between true and estimated states.

    Args:
        x_true: True states (4 x n_times)
        x_hat: Estimated states (4 x n_times)
        state_names: Names of states ['Ad', 'Ac', 'Ap', 'x4']

    Returns:
        Dictionary with MAE, RMSE, final_error for each state
    """
    metrics = {}

    for i, name in enumerate(state_names):
        error = x_hat[i, :] - x_true[i, :]
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error**2))
        final_error = np.abs(error[-1])

        # Relative metrics (avoid division by zero)
        x_true_mean = np.mean(np.abs(x_true[i, :]))
        if x_true_mean > 1e-10:
            rel_mae = mae / x_true_mean * 100
            rel_rmse = rmse / x_true_mean * 100
        else:
            rel_mae = np.nan
            rel_rmse = np.nan

        metrics[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'final_error': final_error,
            'relative_MAE_%': rel_mae,
            'relative_RMSE_%': rel_rmse
        }

    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("LUENBERGER OBSERVER SIMULATION")
    print("="*70)

    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    PATIENT_ID = 1
    TARGET_DIR = 'opti'  # 'opti', 'opti-constrained', or 'standalone'
    RES_DIR = 'results'

    # Gain computation mode: 'kalman' or 'pole_placement'
    GAIN_MODE = 'kalman'

    # --- Kalman filter parameters (used if GAIN_MODE == 'kalman') ---
    # Process noise covariance Q diagonal (4x4)
    # Higher values = less trust in model for that state
    Q_DIAG = [0, 0, 0, 0.1]  # x4 is constant, so low uncertainty

    # Measurement noise covariance R (scalar)
    # Higher value = less trust in BP measurements
    R_VALUE = 5.0

    # --- Pole placement parameters (used if GAIN_MODE == 'pole_placement') ---
    # Desired observer poles (must be negative for stability)
    DESIRED_POLES = [-0.1, -0.2, -0.3, -0.4]

    # Initial observer state:
    #   None        -> zeros [0, 0, 0, 0]
    #   "true"      -> use true initial conditions from loaded trajectories
    #   [list]      -> custom values [x1_0, x2_0, x3_0, x4_0]
    X_HAT_0: Union[None, str, List[float]] = None

    # Simulation time step for observer integration
    DT_OBSERVER = 0.1  # seconds

    # -------------------------------------------------------------------------
    # 1. Load parameters and data
    # -------------------------------------------------------------------------
    print(f"\nPatient ID: {PATIENT_ID}")
    print(f"Target directory: {TARGET_DIR}")

    # Load parameters
    if TARGET_DIR == 'standalone':
        # Use default parameters from PKPD model
        print("\nUsing default PKPD parameters (standalone mode)")
        pkpd_model = NorepinephrinePKPD()
        params = {
            'k_a': pkpd_model.k_a,
            'k_12': pkpd_model.k_12,
            'k_21': pkpd_model.k_21,
            'k_el': pkpd_model.k_el,
            'V_c': pkpd_model.V_c,
            'C_endo': pkpd_model.C_endo,
            'EC_50': pkpd_model.EC_50,
            'E_0': pkpd_model.E_0,
            'E_max': pkpd_model.E_max
        }
    else:
        # Load optimized parameters
        print(f"\nLoading optimized parameters from {TARGET_DIR}...")
        params = load_optimized_parameters(PATIENT_ID, RES_DIR, TARGET_DIR)

    print("\nLoaded parameters:")
    for key in ['k_a', 'k_12', 'k_21', 'k_el', 'V_c', 'C_endo', 'EC_50', 'E_0', 'E_max']:
        print(f"  {key}: {params[key]:.6f}")

    # Create PKPD model with parameters
    injections_dict = load_injections([PATIENT_ID])
    pkpd_model = NorepinephrinePKPD(injections_dict)

    # Update model parameters
    pkpd_model.k_a = params['k_a']
    pkpd_model.k_12 = params['k_12']
    pkpd_model.k_21 = params['k_21']
    pkpd_model.k_el = params['k_el']
    pkpd_model.V_c = params['V_c']
    pkpd_model.C_endo = params['C_endo']
    pkpd_model.EC_50 = params['EC_50']
    pkpd_model.E_0 = params['E_0']
    pkpd_model.E_max = params['E_max']

    # Load true trajectories
    print("\nLoading true trajectories...")
    t_true, Ad_true, Ac_true, Ap_true, _, _ = load_resimulated_trajectories(
        PATIENT_ID, RES_DIR, TARGET_DIR
    )

    # Compute true x4
    x4_true_value = pkpd_model.x4()
    x4_true = np.full_like(t_true, x4_true_value)

    # Stack true states
    x_true = np.vstack([Ad_true, Ac_true, Ap_true, x4_true])
    print(f"  Loaded {len(t_true)} time points")
    print(f"  Time range: [{t_true[0]:.1f}, {t_true[-1]:.1f}] s")
    print(f"  True x4 = Vc*(EC50 + Cendo) = {x4_true_value:.4f}")

    # Load blood pressure observations
    print("\nLoading blood pressure observations...")
    observations = load_observations([PATIENT_ID])
    bp_obs = observations[PATIENT_ID]['blood_pressure']
    print(f"  Loaded {len(bp_obs)} blood pressure measurements")

    # Create BP interpolator
    bp_interp = create_bp_interpolator(bp_obs, method='linear')

    # -------------------------------------------------------------------------
    # 2. Build system matrices
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("SYSTEM MATRICES")
    print("-"*70)

    A, B, C = build_system_matrices(params)

    print("\nA matrix:")
    print(A)
    print("\nB matrix:", B.T)
    print("C matrix:", C)

    # -------------------------------------------------------------------------
    # 3. Compute observer gain L
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("OBSERVER GAIN DESIGN")
    print("-"*70)

    if GAIN_MODE == 'kalman':
        Q = np.diag(Q_DIAG)
        R = np.array([[R_VALUE]])
        L, Sigma = compute_kalman_gain(A, C, Q, R)
        print(f"\nKalman filter gain (Q_diag={Q_DIAG}, R={R_VALUE})")
        print(f"\nSteady-state error covariance Σ:")
        print(Sigma)
    else:
        print(f"\nPole placement (desired poles: {DESIRED_POLES})")
        L = compute_observer_gain(A, C, DESIRED_POLES)

    print(f"\nObserver gain L:")
    print(L)

    # Verify eigenvalues
    A_obs = A - L @ C
    eigenvalues = np.linalg.eigvals(A_obs)
    print(f"\nEigenvalues of (A - LC):")
    for ev in eigenvalues:
        print(f"  {ev:.6f} (stable: {np.real(ev) < 0})")

    # -------------------------------------------------------------------------
    # 4. Simulate observer
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("OBSERVER SIMULATION")
    print("-"*70)

    # Resolve initial condition
    if X_HAT_0 is None:
        x_hat_0 = np.zeros(4)
        print("\nInitial observer state: zeros [0, 0, 0, 0]")
    elif X_HAT_0 == "true":
        x_hat_0 = np.array([Ad_true[0], Ac_true[0], Ap_true[0], x4_true_value])
        print(f"\nInitial observer state: true ICs {x_hat_0}")
    else:
        x_hat_0 = np.array(X_HAT_0)
        print(f"\nInitial observer state: custom {x_hat_0}")

    print(f"Simulation time step: {DT_OBSERVER} s")
    print("\nRunning observer simulation...")

    t_obs, x_hat = simulate_observer(
        A, B, C, L, pkpd_model, bp_interp,
        t_true, x_hat_0, PATIENT_ID, DT_OBSERVER
    )

    print("  Done!")

    # -------------------------------------------------------------------------
    # 5. Compute metrics
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("ESTIMATION METRICS")
    print("-"*70)

    state_names = ['Ad', 'Ac', 'Ap', 'x4']
    metrics = compute_metrics(x_true, x_hat, state_names)

    print(f"\n{'State':<8} {'MAE':>12} {'RMSE':>12} {'Final Err':>12} {'Rel MAE %':>12}")
    print("-" * 60)
    for name in state_names:
        m = metrics[name]
        print(f"{name:<8} {m['MAE']:>12.4f} {m['RMSE']:>12.4f} "
              f"{m['final_error']:>12.4f} {m['relative_MAE_%']:>12.2f}")

    # -------------------------------------------------------------------------
    # 6. Generate plots
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("GENERATING PLOTS")
    print("-"*70)

    output_dir = f'{RES_DIR}/patient_{PATIENT_ID}/obs'
    os.makedirs(output_dir, exist_ok=True)

    units = ['nmol', 'nmol', 'nmol', 'nmol']

    print(f"\nSaving plots to: {output_dir}/")

    # Individual state plots
    plot_observer_individual_states(t_obs, x_true, x_hat, state_names, units,
                                    PATIENT_ID, output_dir)
    print("  - Individual state plots saved")

    # Summary plot
    plot_observer_summary(t_obs, x_true, x_hat, state_names, PATIENT_ID, output_dir)
    print("  - Summary plot saved")

    # Error plot
    plot_observer_estimation_error(t_obs, x_true, x_hat, state_names, PATIENT_ID, output_dir)
    print("  - Error plot saved")

    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
