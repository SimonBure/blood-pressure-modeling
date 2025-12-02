import casadi as ca
import numpy as np
from typing import Dict, Tuple

from .config import OptimizationConfig, PhysiologicalConstants
from .results import OptimizationResult
from .cost_functions import create_cost_function
from ..utils import (
    load_observations,
    load_injections,
    load_patient_e0_indiv,
    save_optimal_parameters,
    print_optimization_results
)
from ..utils.plots import (
    plot_optimization_results,
    plot_pkpd_vs_casadi_trajectories,
    plot_injection_verification
)
from ..pkpd import NorepinephrinePKPD


def precompute_injection_rates(patient_id: int,
                               times: np.ndarray,
                               injections_dict: Dict) -> np.ndarray:
    """Precompute INOR injection rates at each time point.

    Args:
        patient_id: Patient ID.
        times: Array of time points.
        injections_dict: Dictionary of injection protocols.

    Returns:
        Array of injection rates at each time point.
    """
    model = NorepinephrinePKPD(injections_dict)
    inor_values = np.array([model.INOR(t, patient_id) for t in times])
    return inor_values


def compute_equilibrium_blood_pressure(times: np.ndarray,
                                       inor_values: np.ndarray,
                                       params: Dict[str, float]) -> np.ndarray:
    """Compute equilibrium blood pressure over time using Emax model.

    The equilibrium is computed using the stationary INOR value at each time point,
    representing the theoretical steady-state blood pressure if the current injection
    rate were maintained indefinitely.

    Formula:
    E_eq = E_0 + (E_max - E_0) * C_eq / (C_eq + EC_50)
    where C_eq = C_endo + I* / (k_el * Vc)

    Args:
        times: Array of time points.
        inor_values: Array of INOR injection rates at each time point (stationary values).
        params: Dictionary containing optimized parameters (E_0, E_max, EC_50, C_endo, k_el, V_c).

    Returns:
        Array of equilibrium blood pressure values at each time point.
    """
    # Extract parameters
    E_0 = params['E_0']
    E_max = params['E_max']
    EC_50 = params['EC_50']
    C_endo = params['C_endo']
    k_el = params['k_el']
    V_c = params['V_c']

    # Compute equilibrium concentration
    # At equilibrium: dAc/dt = 0, which gives Ac_eq = I* / k_el
    # Therefore: C_eq = C_endo + Ac_eq / Vc = C_endo + I* / (k_el * Vc)
    C_eq = C_endo + inor_values / (k_el * V_c)

    # Compute equilibrium blood pressure using Emax model
    E_eq = E_0 + (E_max - E_0) * C_eq / (C_eq + EC_50)

    return E_eq


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

    # PD Emax parameters (conditional)
    if config.cost_function_mode in ['emax', 'both']:
        params['E_0'] = opti.variable()
        params['E_max'] = opti.variable()
        params['EC_50'] = opti.variable()
    else:
        # Fixed values when not optimizing Emax
        params['E_0'] = physio.E_0
        params['E_max'] = physio.E_max
        params['EC_50'] = physio.EC_50

    # PD Windkessel parameters (conditional)
    if config.cost_function_mode in ['windkessel', 'both']:
        params['omega'] = opti.variable()
        params['zeta'] = opti.variable()
        params['nu'] = opti.variable()
    else:
        # Fixed values when not optimizing Windkessel
        params['omega'] = physio.omega
        params['zeta'] = physio.zeta
        params['nu'] = physio.nu

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

    # PD Windkessel states (only if windkessel mode)
    if config.cost_function_mode in ['windkessel', 'both']:
        states['E'] = opti.variable(N + 1)
        states['dEdt'] = opti.variable(N + 1)

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

    if config.cost_function_mode in ['windkessel', 'both']:
        opti.subject_to(states['E'][0] == physio.E_0_init)
        opti.subject_to(states['dEdt'][0] == physio.dEdt_0)


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
                print(f"    WARNING: use_e0_constraint=True but patient_e0=None, constraint not applied")


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

        # Windkessel equations (only if windkessel mode)
        if config.cost_function_mode in ['windkessel', 'both']:
            # Compute concentration at k+1 (needed for PD)
            Cc_kp1 = params['C_endo'] + states['Ac'][k+1] / params['V_c']

            # dE/dt = dEdt
            opti.subject_to(
                states['E'][k+1] == states['E'][k] + dt * states['dEdt'][k+1]
            )

            # d²E/dt² = nu * Cc - 2*zeta*omega*dEdt - omega²*E
            opti.subject_to(
                states['dEdt'][k+1] == states['dEdt'][k] + dt * (
                    params['nu'] * Cc_kp1 -
                    2 * params['zeta'] * params['omega'] * states['dEdt'][k+1] -
                    params['omega']**2 * states['E'][k+1]
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

    # Initialize PD parameters (only if they are variables)
    if config.cost_function_mode in ['emax', 'both']:
        # Use patient-specific E0 if available, otherwise use default
        e0_initial = patient_e0 if patient_e0 is not None else physio.E_0
        opti.set_initial(params['E_0'], e0_initial)
        opti.set_initial(params['E_max'], physio.E_max)
        opti.set_initial(params['EC_50'], physio.EC_50)

    if config.cost_function_mode in ['windkessel', 'both']:
        opti.set_initial(params['omega'], physio.omega)
        opti.set_initial(params['zeta'], physio.zeta)
        opti.set_initial(params['nu'], physio.nu)

    # Initialize state trajectories with data
    opti.set_initial(states['Ad'], Ad_data)
    opti.set_initial(states['Ac'], Ac_data)
    opti.set_initial(states['Ap'], Ap_data)

    if config.cost_function_mode in ['windkessel', 'both']:
        opti.set_initial(states['E'], E_data)
        opti.set_initial(states['dEdt'], np.zeros(N+1))


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

    if config.cost_function_mode in ['windkessel', 'both']:
        trajectories['E'] = sol.value(states['E'])
        trajectories['dEdt'] = sol.value(states['dEdt'])

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
        E_data: Windkessel/Emax state for initialization.
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
    apply_dynamics_constraints(opti, N, dt_values, params, states, inor_values, config)

    # Build cost function (BP-only, no concentration term)
    print("  Building cost function on BP: obs VS model...")

    cost_fn = create_cost_function(config.cost_function_mode)
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


def resimulate_with_optimized_params(patient_id: int,
                                     trajectories,
                                    params_opt: Dict[str, float],
                                    injections_dict: Dict) -> Tuple:
    """Create new model with optimized parameters and simulate.

    Args:
        patient_id: Patient ID.
        params_opt: Dictionary of optimized parameters.
        injections_dict: Dictionary of injection protocols.

    Returns:
        Tuple of (t, Ad, Ac, Ap, E_emax, E_windkessel) from simulation.
    """
    pkpd_initial_conditions = {'Ad_0': trajectories['Ad'][0],
                               'Ac_0': trajectories['Ac'][0],
                               'Ap_0': trajectories['Ap'][0]}
    # Create model instance
    model = NorepinephrinePKPD(injections_dict, pkpd_initial_conditions)

    # Override parameters with optimized values
    model.C_endo = params_opt['C_endo']
    model.k_a = params_opt['k_a']
    model.V_c = params_opt['V_c']
    model.k_12 = params_opt['k_12']
    model.k_21 = params_opt['k_21']
    model.k_el = params_opt['k_el']
    model.E_0 = params_opt['E_0']
    model.E_max = params_opt['E_max']
    model.EC_50 = params_opt['EC_50']
    model.omega = params_opt['omega']
    model.zeta = params_opt['zeta']
    model.nu = params_opt['nu']

    # Simulate
    t, Ad, Ac, Ap, E_emax, E_windkessel = model.simulate(
        patient_id, t_end=2200, dt=0.5
    )

    return t, Ad, Ac, Ap, E_emax, E_windkessel


def get_initial_guess_from_pkpd(patient_id: int,
                                times: np.ndarray,
                                data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate initial guess for state trajectories from precomputed PKPD model.

    Loads trajectories from pkpd/ directory and interpolates to match observation times.

    Args:
        patient_id: Patient ID.
        times: Array of observation time points.
        data_dir: Base directory for patient data.

    Returns:
        Tuple of (Ad_data, Ac_data, Ap_data, E_data) arrays interpolated to observation times.
    """
    pkpd_path = f'{data_dir}/patient_{patient_id}/pkpd'

    # Load full trajectories from pkpd/ directory
    time_full = np.load(f'{pkpd_path}/time.npy')
    Ad_full = np.load(f'{pkpd_path}/Ad.npy')
    Ac_full = np.load(f'{pkpd_path}/Ac.npy')
    Ap_full = np.load(f'{pkpd_path}/Ap.npy')
    E_full = np.load(f'{pkpd_path}/bp_emax.npy')

    # Interpolate to match observation time points
    Ad_data = np.interp(times, time_full, Ad_full)
    Ac_data = np.interp(times, time_full, Ac_full)
    Ap_data = np.interp(times, time_full, Ap_full)
    E_data = np.interp(times, time_full, E_full)

    return Ad_data, Ac_data, Ap_data, E_data


if __name__ == "__main__":
    # NOTE: Set patient_ids=None to process ALL patients, or specify a list like [23, 45]

    # Configuration
    use_e0_constraint = False  # Toggle E_0 constraint mode
    output_subdir = 'opti-e0-constraint' if use_e0_constraint else 'opti'

    config = OptimizationConfig(
        patient_ids=[],  # None = all patients, or specify list like [23]
        max_data_points=1001,
        cost_function_mode='emax',
        use_e0_constraint=use_e0_constraint,  # E_0 constraint mode
        data_dir='codes/res',
        output_dir=output_subdir,
        ipopt_max_iter=5000,
        ipopt_tol=1e-6,
        ipopt_print_level=0  # 0=silent, 3=minimal, 5=verbose (default)
    )

    print("\n" + "="*70)
    print("CASADI PKPD PARAMETER OPTIMIZATION - PATIENT-BY-PATIENT")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Patients: {config.patient_ids}")
    print(f"  - Time sampling: Patient-specific (using actual observation times)")
    print(f"  - Max data points: {config.max_data_points} (subsample if exceeded)")
    print(f"  - Cost function mode: {config.cost_function_mode}")
    print(f"  - E_0 mode: {'Hard constraint to E0_indiv' if config.use_e0_constraint else 'Initial guess from E0_indiv'}")
    print(f"  - Output directory: {config.output_dir}/")
    print("="*70 + "\n")

    # Load data - pass patient_ids as-is (None or list)
    # load_observations and load_injections will handle None/[] by loading all patients
    injections_dict = load_injections(config.patient_ids, config.inj_csv_path)
    observations = load_observations(config.patient_ids, config.obs_csv_path)

    # Determine actual patient list after loading
    # Use intersection of patients with both observations and injections
    patients_with_obs = set(observations.keys())
    patients_with_inj = set(injections_dict.keys())
    available_patients = sorted(patients_with_obs & patients_with_inj)

    # Filter to only include requested patient IDs if specified
    if config.patient_ids is None or config.patient_ids == []:
        patient_ids = available_patients
    else:
        patient_ids = [pid for pid in config.patient_ids if pid in available_patients]

    print(f"Patient IDs after loading: {patient_ids}")

    if not patient_ids:
        print("ERROR: No patients found with both observations and injection data!")
        exit(1)

    print(f"Loaded data for {len(patient_ids)} patient(s): {patient_ids}")

    # Load patient-specific baseline E0 values
    print("Loading patient-specific E0_indiv values...")
    patient_e0_dict = load_patient_e0_indiv(patient_ids, config.obs_csv_path)
    print(f"  ✓ Loaded E0_indiv for {len(patient_e0_dict)} patients\n")

    # Get initial parameters for comparison
    physio = PhysiologicalConstants()
    params_initial = {
        'C_endo': physio.C_endo,
        'k_a': physio.k_a,
        'V_c': physio.V_c,
        'k_12': physio.k_12,
        'k_21': physio.k_21,
        'k_el': physio.k_el,
        'E_0': physio.E_0,
        'E_max': physio.E_max,
        'EC_50': physio.EC_50,
        'omega': physio.omega,
        'zeta': physio.zeta,
        'nu': physio.nu
    }

    for patient_id in patient_ids:
        print(f"\n{'#'*70}")
        print(f"# Processing Patient {patient_id}")
        print(f"{'#'*70}\n")

        print("Step 1: Extracting observation times and values...")
        # Get time points directly from blood pressure observations
        bp_obs_data = observations[patient_id]['blood_pressure']
        if not bp_obs_data:
            print(f"  ERROR: No blood pressure observations for patient {patient_id}")
            continue

        times = np.array([t for t, _ in bp_obs_data])
        BP_obs = np.array([v for _, v in bp_obs_data])
        n_original_observations = len(times)
        print(f"  ✓ Extracted {n_original_observations} observation time points")
        print(f"  ✓ Time range: [{times[0]:.1f}s, {times[-1]:.1f}s]")

        # Apply subsampling if needed
        if n_original_observations > config.max_data_points:
            print(f"  ⚠ Subsampling from {n_original_observations} to {config.max_data_points} points")
            indices = np.linspace(0, n_original_observations - 1, config.max_data_points, dtype=int)
            times = times[indices]
            BP_obs = BP_obs[indices]
            n_optimization_points = config.max_data_points
        else:
            n_optimization_points = n_original_observations

        print(f"  ✓ Using {n_optimization_points} points for optimization")

        print("\nStep 2: Loading initial guess from PKPD model trajectories...")
        Ad_data, Ac_data, Ap_data, E_data = get_initial_guess_from_pkpd(
            patient_id, times, config.data_dir
        )
        patient_e0 = patient_e0_dict[patient_id]
        print(f"  ✓ Loaded and interpolated PKPD trajectories for {len(times)} time points")
        print(f"  ✓ Patient-specific E0_indiv: {patient_e0:.2f} mmHg")

        print("\nStep 3: Precomputing injection rates...")
        inor_values = precompute_injection_rates(patient_id, times, injections_dict)
        print(f"  ✓ Precomputed INOR for {len(inor_values)} time points")

        print("\nCreating injection verification plot...")
        plot_injection_verification(
            patient_id, times, inor_values, injections_dict,
            config.data_dir, config.output_dir, n_optimization_points
        )

        print("\nStep 4: Running CasADi optimization...")
        result = optimize_patient_parameters(
            times, BP_obs, inor_values,
            Ad_data, Ac_data, Ap_data, E_data,
            config, patient_e0
        )
        print(f"  ✓ Final cost: {result.cost:.4f}")
        print(f"  ✓ Solve time: {result.solve_time:.2f}s")

        print("\nStep 5: Saving optimal parameters...")
        save_optimal_parameters(
            patient_id, result.params, result.cost,
            config.data_dir, config.output_dir,
            n_original_observations, n_optimization_points,
            config.cost_function_mode
        )

        print("\nStep 6: Printing optimization results...")
        print_optimization_results(patient_id, params_initial, result.params)

        print("Step 7: Re-simulating with optimized parameters...")
        resim_results = resimulate_with_optimized_params(
            patient_id, result.trajectories, result.params, injections_dict
        )
        print("  ✓ Re-simulation completed")

        # Compute equilibrium blood pressure
        E_equilibrium = compute_equilibrium_blood_pressure(
            times, inor_values, result.params
        )

        print("\nStep 8: Creating visualization plots...")
        plot_optimization_results(
            patient_id, observations, result.trajectories,
            result.params, resim_results, E_equilibrium,
            config.data_dir, config.output_dir,
            n_optimization_points, config.cost_function_mode
        )

        print("\nStep 9: Comparing CasADi vs PKPD trajectories...")
        plot_pkpd_vs_casadi_trajectories(
            patient_id, result.trajectories, resim_results,
            result.params, config.data_dir, config.output_dir,
            n_optimization_points, config.cost_function_mode
        )

    print(f"\n{'='*70}")
    print("ALL PATIENTS PROCESSED SUCCESSFULLY")
    print(f"{'='*70}\n")
