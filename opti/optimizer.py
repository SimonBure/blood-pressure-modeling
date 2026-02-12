"""
Optimization functions for PKPD parameter estimation.

This module contains all functions related to setting up and solving the
CasADi optimization problem for PKPD parameter estimation.
"""

import casadi as ca
import numpy as np
from typing import Dict, Tuple
from opti.config import OptimizationConfig, PhysiologicalConstants
from opti.results import OptimizationResult
from opti.cost_functions import EmaxBPCost


def setup_optimization_variables(opti: ca.Opti,
                                config: OptimizationConfig,
                                physio: PhysiologicalConstants) -> Dict:
    """Setup optimization parameter variables.

    Args:
        opti: CasADi Opti object.
        config: Optimization configuration.
        physio: Physiological constants for fixed values.

    Returns:
        Dictionary of parameter variables (or fixed values).
    """
    # PK parameters (always optimized)
    params = {
        'C_endo': opti.variable(),
        'k_a': opti.variable(),
        'V_c': opti.variable(),
        'k_12': opti.variable(),
        'k_21': opti.variable(),
        'k_el': opti.variable(),
    }

    # PD Emax parameters
    params['E_0'] = opti.variable()
    params['E_max'] = opti.variable()
    params['EC_50'] = opti.variable()

    return params


def setup_state_variables(opti: ca.Opti,
                          N: int,
                          config: OptimizationConfig) -> Dict:
    """Setup state trajectory variables.

    Args:
        opti: CasADi Opti object.
        N: Number of intervals.
        config: Optimization configuration.

    Returns:
        Dictionary of state variables.
    """
    # PK states (always needed)
    states = {
        'Ad': opti.variable(N + 1),
        'Ac': opti.variable(N + 1),
        'Ap': opti.variable(N + 1),
    }

    return states


def apply_initial_conditions(opti: ca.Opti,
                             states: Dict,
                             physio: PhysiologicalConstants,
                             config: OptimizationConfig) -> None:
    """Apply initial conditions to state variables.

    Args:
        opti: CasADi Opti object.
        states: Dictionary of state variables.
        physio: Physiological constants.
        config: Optimization configuration.
    """
    opti.subject_to(states['Ad'][0] == physio.Ad_0)
    opti.subject_to(states['Ac'][0] == physio.Ac_0)
    opti.subject_to(states['Ap'][0] == physio.Ap_0)


def apply_parameter_bounds(opti: ca.Opti,
                           params: Dict,
                           config: OptimizationConfig) -> None:
    """Apply parameter bounds (physical constraints).

    Args:
        opti: CasADi Opti object.
        params: Dictionary of parameter variables.
        config: Optimization configuration.
    """
    # Apply bounds from config
    for param_name, bounds in config.param_bounds.items():
        if param_name not in params:
            continue

        lower, upper = bounds
        param_var = params[param_name]

        # Check if it's a CasADi variable (not a fixed value)
        if isinstance(param_var, (ca.MX, ca.SX)):
            if lower is not None:
                if lower == 0:
                    opti.subject_to(param_var >= lower)
                else:
                    opti.subject_to(param_var > lower)
            if upper is not None:
                opti.subject_to(param_var <= upper)


def apply_parameter_constraints(opti: ca.Opti,
                                params: Dict,
                                config: OptimizationConfig,
                                patient_e0: float | None = None) -> None:
    """Apply hard equality constraints on parameters.

    Args:
        opti: CasADi Opti object.
        params: Dictionary of parameter variables.
        config: Optimization configuration.
        patient_e0: Patient-specific E0_indiv value to constrain E_0 to.
    """
    if config.use_e0_constraint:
        # Only apply constraint if E_0 is a variable (not a fixed value)
        if 'E_0' in params and isinstance(params['E_0'], (ca.MX, ca.SX)):
            if patient_e0 is not None:
                print(f"    Applying hard constraint: E_0 == {patient_e0:.2f} mmHg")
                opti.subject_to(params['E_0'] == patient_e0)
            else:
                print("    WARNING: use_e0_constraint=True but patient_e0=None, constraint not applied")


def apply_positivity_constraints(opti: ca.Opti, N: int, states: Dict) -> None:
    """
    Apply > 0 constraints to PK states.
    
    :param opti: CasADi Opti object.
    :type opti: ca.Opti
    :param N: Number of intervals.
    :type N: int
    :param states: Dictionary of states.
    :type states: Dict
    """
    for k in range(N):
        opti.subject_to(states['Ad'][k] >= 0)
        opti.subject_to(states['Ac'][k] >= 0)
        opti.subject_to(states['Ap'][k] >= 0)


def apply_dynamics_constraints(opti: ca.Opti,
                               N: int,
                               dt_values: np.ndarray,
                               params: Dict,
                               states: Dict,
                               inor_values: np.ndarray,
                               config: OptimizationConfig) -> None:
    """Apply PK and PD dynamics constraints.

    Args:
        opti: CasADi Opti object.
        N: Number of intervals.
        dt_values: Array of time step sizes.
        params: Dictionary of parameters.
        states: Dictionary of states.
        inor_values: Precomputed injection rates.
        config: Optimization configuration.
    """
    print("  Building dynamics constraints...")

    for k in range(N):
        dt = dt_values[k]

        # PK equations (Euler implicit at k+1) - ALWAYS
        # dAd/dt = -k_a * Ad + INOR
        opti.subject_to(
            states['Ad'][k+1] == states['Ad'][k] + dt * (
                -params['k_a'] * states['Ad'][k+1] + inor_values[k+1]
            )
        )

        # dAc/dt = k_a * Ad - (k_12 + k_el) * Ac + k_21 * Ap
        opti.subject_to(
            states['Ac'][k+1] == states['Ac'][k] + dt * (
                params['k_a'] * states['Ad'][k+1] -
                (params['k_12'] + params['k_el']) * states['Ac'][k+1] +
                params['k_21'] * states['Ap'][k+1]
            )
        )

        # dAp/dt = k_12 * Ac - k_21 * Ap
        opti.subject_to(
            states['Ap'][k+1] == states['Ap'][k] + dt * (
                params['k_12'] * states['Ac'][k+1] -
                params['k_21'] * states['Ap'][k+1]
            )
        )


def configure_solver(opti: ca.Opti, config: OptimizationConfig) -> None:
    """Configure IPOPT solver with options.

    Args:
        opti: CasADi Opti object.
        config: Optimization configuration.
    """
    opts = {
        'ipopt.max_iter': config.ipopt_max_iter,
        'ipopt.tol': config.ipopt_tol,
        'ipopt.acceptable_tol': config.ipopt_acceptable_tol,
        'ipopt.acceptable_iter': config.ipopt_acceptable_iter,
        'ipopt.print_level': config.ipopt_print_level,
        'print_time': False
    }
    opti.solver('ipopt', opts)


def set_initial_guess(opti: ca.Opti,
                     params: Dict,
                     states: Dict,
                     physio: PhysiologicalConstants,
                     Ad_data: np.ndarray,
                     Ac_data: np.ndarray,
                     Ap_data: np.ndarray,
                     E_data: np.ndarray,
                     N: int,
                     config: OptimizationConfig,
                     patient_e0: float | None = None) -> None:
    """Set initial guess for parameters and states.

    Args:
        opti: CasADi Opti object.
        params: Dictionary of parameter variables.
        states: Dictionary of state variables.
        physio: Physiological constants.
        Ad_data, Ac_data, Ap_data, E_data: State data for initialization.
        N: Number of intervals.
        config: Optimization configuration.
        patient_e0: Patient-specific baseline E0 (from E0_indiv). If None, uses physio.E_0.
    """
    print("  Setting initial guesses...")

    # Initialize PK parameters
    opti.set_initial(params['C_endo'], physio.C_endo)
    opti.set_initial(params['k_a'], physio.k_a)
    opti.set_initial(params['V_c'], physio.V_c)
    opti.set_initial(params['k_12'], physio.k_12)
    opti.set_initial(params['k_21'], physio.k_21)
    opti.set_initial(params['k_el'], physio.k_el)
    
    # Use patient-specific E0 if available, otherwise use default
    e0_initial = patient_e0 if patient_e0 is not None else physio.E_0
    opti.set_initial(params['E_0'], e0_initial)
    opti.set_initial(params['E_max'], physio.E_max)
    opti.set_initial(params['EC_50'], physio.EC_50)

    # Initialize state trajectories with data
    opti.set_initial(states['Ad'], Ad_data)
    opti.set_initial(states['Ac'], Ac_data)
    opti.set_initial(states['Ap'], Ap_data)


def solve_optimization(opti: ca.Opti) -> Tuple[ca.OptiSol, bool]:
    """Solve the optimization problem.

    Args:
        opti: CasADi Opti object.

    Returns:
        Tuple of (solution, converged flag).
    """
    print("  Solving optimization problem...")
    try:
        sol = opti.solve()
        print("  ✓ Optimization converged successfully!")
        converged = True
    except RuntimeError as e:
        error_msg = str(e)
        if 'return_status is' in error_msg:
            status = error_msg.split('return_status is')[1].strip().strip("'")
            print(f"  ⚠ Solver did not fully converge: {status}")
        else:
            print(f"  ⚠ Solver error: {error_msg}")
        print("  ⚠ Using best solution found (may still be reasonable)...")
        sol = opti.debug
        converged = False

    return sol, converged


def extract_solution(sol: ca.OptiSol,
                    params: Dict,
                    states: Dict,
                    times: np.ndarray,
                    config: OptimizationConfig) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Extract optimized parameters and trajectories from solution.

    Args:
        sol: CasADi solution object.
        params: Dictionary of parameter variables.
        states: Dictionary of state variables.
        times: Array of time points.
        config: Optimization configuration.

    Returns:
        Tuple of (params_opt dict, trajectories dict).
    """
    # Extract parameters
    params_opt = {}
    for param_name, param_var in params.items():
        if isinstance(param_var, (ca.MX, ca.SX)):
            params_opt[param_name] = float(sol.value(param_var))
        else:
            # Fixed parameter
            params_opt[param_name] = float(param_var)

    # Extract state trajectories
    trajectories = {
        'times': times,
        'Ad': sol.value(states['Ad']),
        'Ac': sol.value(states['Ac']),
        'Ap': sol.value(states['Ap']),
    }

    return params_opt, trajectories


def optimize_patient_parameters(times: np.ndarray,
                                BP_obs: np.ndarray,
                                inor_values: np.ndarray,
                                Ad_data: np.ndarray,
                                Ac_data: np.ndarray,
                                Ap_data: np.ndarray,
                                E_data: np.ndarray,
                                config: OptimizationConfig,
                                patient_e0: float | None = None) -> OptimizationResult:
    """Run CasADi optimization to estimate PKPD parameters.

    Args:
        times: Array of time points.
        BP_obs: Interpolated blood pressure observations (BP-only, no concentration).
        inor_values: Precomputed injection rates.
        Ad_data, Ac_data, Ap_data: State data for initialization.
        E_data: Emax state for initialization.
        config: Optimization configuration.
        patient_e0: Patient-specific baseline E0 (from E0_indiv). If None, uses default.

    Returns:
        OptimizationResult dataclass with params, trajectories, cost, etc.
    """
    import time
    start_time = time.time()

    opti = ca.Opti()
    N = len(times) - 1
    dt_values = np.diff(times)

    print(f"  Setting up optimization problem: N={N} intervals, "
          f"dt range=[{dt_values.min():.2f}, {dt_values.max():.2f}]s")

    # Get physiological constants
    physio = PhysiologicalConstants()

    # Setup optimization variables
    params = setup_optimization_variables(opti, config, physio)
    states = setup_state_variables(opti, N, config)

    # Apply constraints
    # apply_initial_conditions(opti, states, physio, config)
    apply_parameter_bounds(opti, params, config)
    apply_parameter_constraints(opti, params, config, patient_e0)
    apply_positivity_constraints(opti, N, states)
    apply_dynamics_constraints(opti, N, dt_values, params, states, inor_values, config)

    # Build cost function (BP-only, no concentration term)
    print("  Building cost function on BP: obs VS model...")

    cost_fn = EmaxBPCost()
    opti_vars = {**params, **states}
    observations = {'BP_obs': BP_obs}
    cost = cost_fn.compute(opti_vars, observations)

    opti.minimize(cost)

    # Configure solver
    configure_solver(opti, config)

    # Set initial guess
    set_initial_guess(opti, params, states, physio,
                     Ad_data, Ac_data, Ap_data, E_data, N, config, patient_e0)

    # Solve optimization
    sol, converged = solve_optimization(opti)

    # Extract solution
    params_opt, trajectories = extract_solution(sol, params, states, times, config)
    cost_value = float(sol.value(cost))

    solve_time = time.time() - start_time

    # Create result object
    result = OptimizationResult(
        params=params_opt,
        trajectories=trajectories,
        cost=cost_value,
        converged=converged,
        iterations=-1,  # IPOPT doesn't easily expose iteration count
        solve_time=solve_time
    )

    return result
