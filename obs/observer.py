"""
Luenberger Observer Simulation for Extended PKPD System

This script implements an observer to estimate:
- x1 = Ad (depot compartment)
- x2 = Ac (central compartment)
- x3 = Ap (peripheral compartment)
- x4 = Vc*(EC50 + Cendo) (parameter embedding)

Observer dynamics: x_hat_dot = (A - LC) x_hat + B*u + L*y
where y = C @ x_true (validation mode using true output)

Gain L computed via steady-state Kalman filter (Riccati equation).
"""

import sys
import os
from typing import Dict, Tuple, Union, List

import numpy as np
from scipy.linalg import solve_continuous_are

from pkpd import NorepinephrinePKPD
from utils.datatools import get_patient_ids, load_injections, load_resimulated_trajectories
from utils.plots import (
    plot_observer_individual_states,
    plot_observer_summary,
    plot_observer_estimation_error
)
from opti.postprocessing import load_optimized_parameters

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


# =============================================================================
# KALMAN GAIN
# =============================================================================

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

def simulate_observer(A: np.ndarray, B: np.ndarray, C: np.ndarray, L: np.ndarray,
                      pkpd_model: NorepinephrinePKPD,
                      t_span: np.ndarray,
                      x_hat_0: np.ndarray,
                      x_true: np.ndarray,
                      patient_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate Luenberger observer dynamics using implicit Euler.

    Observer: x_hat_dot = (A - LC) x_hat + B*u + L*y
    where y = C @ x_true (validation mode).

    Implicit Euler scheme:
        x[k+1] = x[k] + dt * (A_obs @ x[k+1] + f[k+1])
        => (I - dt*A_obs) @ x[k+1] = x[k] + dt * f[k+1]

    Args:
        A, B, C: System matrices
        L: Observer gain (4x1)
        pkpd_model: PKPD model instance (for INOR)
        t_span: Time points for simulation
        x_hat_0: Initial observer state (4,)
        x_true: True states (4 x n) for computing output y = C @ x_true
        patient_id: Patient ID for INOR lookup

    Returns:
        t_out: Time array
        x_hat_out: Estimated states (4 x n_times)
    """
    I = np.eye(4)

    n_steps = len(t_span)
    x_hat = np.zeros((4, n_steps))
    x_hat[:, 0] = x_hat_0

    for i in range(n_steps - 1):
        t_next = t_span[i+1]
        dt_i = t_next - t_span[i]

        # True output: y = C @ x_true
        y_next = (C @ x_true[:, i+1]).item()

        # Input u(t) = INOR(t)
        u_next = pkpd_model.INOR(t_next, patient_id)

        # Implicit Euler: (I - dt*A_obs) @ x[k+1] = x[k] + dt * f[k+1]
        lhs_matrix = I - dt_i * (A - L @ C)
        rhs_vector = x_hat[:, i] + dt_i * (B.flatten() * u_next + L.flatten() * y_next)

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
    print("LUENBERGER OBSERVER SIMULATION (Kalman Gain)")
    print("="*70)

    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    patients = 'all'  # int for 1 patient, list for multiple patients, 'all' for every patient
    parameter_mode = 'population'  # 'population' or 'optimized'
    
    parameters_dir = 'opti-constrained' if parameter_mode == 'optimized' else ''  # 'opti', 'opti-constrained', or 'standalone'
    res_dir = 'results'

    # Kalman filter parameters
    # Process noise covariance Q diagonal (4x4)
    # Higher values = less trust in model for that state
    process_noises_vars = [1, 1, 1, 1]

    # Measurement noise covariance R (scalar)
    # Higher value = less trust in measurements
    measure_noise_var = 5.0

    # Initial observer state:
    #   None        -> zeros [0, 0, 0, 0]
    #   "true"      -> use true initial conditions from loaded trajectories
    #   [list]      -> custom values [x1_0, x2_0, x3_0, x4_0]
    initial_conditions: Union[None, str, List[float]] = [50, 5, 4.6, 150]

    # -------------------------------------------------------------------------
    # 1. Load parameters and data
    # -------------------------------------------------------------------------
    patients_ids = get_patient_ids(patients)
    print(f"\nPatient ID: {patients_ids}")

    # Load input function
    innjections = load_injections(patients_ids)
    pkpd_model = NorepinephrinePKPD(innjections)
    
    for p_id in patients_ids:
        print("\n" + "="*50)
        print(f"PATIENT {p_id}")
        print("="*50 + "\n")

        # Load parameters
        if parameter_mode == 'population':
            print("\nUsing default PKPD parameters (population mode)")
            print("Target directory: standalone")
            params = pkpd_model.get_parameters()
        else:
            print(f"\nLoading optimized parameters from {parameters_dir}...")
            print(f"Target directory: {parameters_dir}")
            params = load_optimized_parameters(p_id, res_dir, parameters_dir)
            pkpd_model.set_parameters(params)

        print("\nLoaded parameters:")
        for key in ['k_a', 'k_12', 'k_21', 'k_el', 'V_c', 'C_endo', 'EC_50', 'E_0', 'E_max']:
            print(f"  {key}: {params[key]:.6f}")

        # Load true trajectories
        print("\nLoading true trajectories...")
        t_true, a_d_true, a_c_true, a_p_true, _ = load_resimulated_trajectories(
            p_id, res_dir, 'opti-constrained'
        )

        # Compute true x4
        print("\nLoading true parameters...")
        true_parameters = load_optimized_parameters(p_id, res_dir, 'opti-constrained')
        temp_pkpd = NorepinephrinePKPD()
        temp_pkpd.set_parameters(true_parameters)
        x4_true_value = temp_pkpd.x4()
        x4_true = np.full_like(t_true, x4_true_value)

        # Stack true states
        x_true = np.vstack([a_d_true, a_c_true, a_p_true, x4_true])
        print(f"  Loaded {len(t_true)} time points")
        print(f"  Time range: [{t_true[0]:.1f}, {t_true[-1]:.1f}] s")
        print(f"  True x4 = Vc*(EC50 + Cendo) = {x4_true_value:.4f}")

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
        # 3. Compute Kalman gain L
        # -------------------------------------------------------------------------
        print("\n" + "-"*70)
        print("KALMAN GAIN DESIGN")
        print("-"*70)

        Q = np.diag(process_noises_vars)
        R = np.array([[measure_noise_var]])
        L, Sigma = compute_kalman_gain(A, C, Q, R)

        print(f"\nKalman filter (Q_diag={process_noises_vars}, R={measure_noise_var})")
        print("\nSteady-state error covariance Σ:")
        print(Sigma)
        print("\nObserver gain L:")
        print(L)

        # Verify eigenvalues
        A_obs = A - L @ C
        eigenvalues = np.linalg.eigvals(A_obs)
        print("\nEigenvalues of (A - LC):")
        for ev in eigenvalues:
            print(f"  {ev:.6f} (stable: {np.real(ev) < 0})")

        # -------------------------------------------------------------------------
        # 4. Simulate observer
        # -------------------------------------------------------------------------
        print("\n" + "-"*70)
        print("OBSERVER SIMULATION")
        print("-"*70)

        # Resolve initial condition
        if initial_conditions is None:
            x_hat_0 = np.zeros(4)
            print("\nInitial observer state: zeros [0, 0, 0, 0]")
        elif initial_conditions == "true":
            x_hat_0 = np.array([a_d_true[0], a_c_true[0], a_p_true[0], x4_true_value])
            print(f"\nInitial observer state: true ICs {x_hat_0}")
        else:
            x_hat_0 = np.array(initial_conditions)
            print(f"\nInitial observer state: custom {x_hat_0}")

        print("\nRunning observer simulation (using y = C @ x_true)...")

        t_obs, x_hat = simulate_observer(
            A, B, C, L, pkpd_model,
            t_true, x_hat_0, x_true, p_id
        )


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

        output_dir = f'{res_dir}/patient_{p_id}/obs'
        os.makedirs(output_dir, exist_ok=True)

        units = ['nmol', 'nmol', 'nmol', 'nmol']

        print(f"\nSaving plots to: {output_dir}/")

        # Individual state plots
        plot_observer_individual_states(t_obs, x_true, x_hat, state_names, units,
                                        p_id, output_dir)
        print("  - Individual state plots saved")

        # Summary plot
        plot_observer_summary(t_obs, x_true, x_hat, state_names, p_id, output_dir)
        print("  - Summary plot saved")

        # Error plot
        plot_observer_estimation_error(t_obs, x_true, x_hat, state_names, p_id, output_dir)
        print("  - Error plot saved")
        
        print(f"  \nPatient {p_id} done!")
        

    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
