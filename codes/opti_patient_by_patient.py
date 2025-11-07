import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pkpd import NorepinephrinePKPD, load_injections, load_observations


PATIENT_IDS = [23]
N_DATA_POINTS = 100  # Number of points to subsample from trajectories
DATA_DIR = 'codes/res'
IS_LINEAR = True  # True for linear_no_lag, False for power_no_lag
COST_FUNCTION_MODE = 'emax'  # 'windkessel', 'emax', or 'both'
INIT_BP_MODEL = 'emax'  # 'windkessel' or 'emax'
OUTPUT_DIR = 'opti'  # Output subdirectory for results


def load_patient_data(patient_id, n_points, is_linear):
    """Load and subsample patient trajectory data from .npy files.

    Args:
        patient_id: Patient ID
        n_points: Number of points to subsample
        is_linear: True for linear_no_lag, False for power_no_lag

    Returns:
        times: np.array of shape (n_points,)
        Ad_data: np.array of shape (n_points,)
        Ac_data: np.array of shape (n_points,)
        Ap_data: np.array of shape (n_points,)
        E_data: np.array of shape (n_points,) - for windkessel initialization
        full_trajectories: dict with full resolution data
    """
    subdir = 'linear_no_lag' if is_linear else 'power_no_lag'
    base_path = f'{DATA_DIR}/patient_{patient_id}/{subdir}'

    # Load full trajectories
    time_full = np.load(f'{base_path}/time.npy')
    Ad_full = np.load(f'{base_path}/Ad.npy')
    Ac_full = np.load(f'{base_path}/Ac.npy')
    Ap_full = np.load(f'{base_path}/Ap.npy')
    bp_emax_full = np.load(f'{base_path}/bp_emax.npy')
    bp_windkessel_full = np.load(f'{base_path}/bp_windkessel.npy')

    # Subsample evenly
    indices = np.linspace(0, len(time_full)-1, n_points, dtype=int)
    times = time_full[indices]
    Ad_data = Ad_full[indices]
    Ac_data = Ac_full[indices]
    Ap_data = Ap_full[indices]
    E_data = bp_windkessel_full[indices] if INIT_BP_MODEL == 'windkessel' else bp_emax_full[indices]  # Use windkessel for E initialization

    full_trajectories = {
        'time': time_full,
        'Ad': Ad_full,
        'Ac': Ac_full,
        'Ap': Ap_full,
        'bp_emax': bp_emax_full,
        'bp_windkessel': bp_windkessel_full
    }

    return times, Ad_data, Ac_data, Ap_data, E_data, full_trajectories


def interpolate_observations(observations, patient_id, times):
    """Interpolate real observations to optimization time grid.

    Uses constant extrapolation outside observation range.

    Args:
        observations: Dict from load_observations()
        patient_id: Patient ID
        times: np.array of N+1 time points

    Returns:
        Cc_obs_interp: np.array of shape (N+1,)
        BP_obs_interp: np.array of shape (N+1,)
    """
    obs = observations[patient_id]

    # Extract concentration observations
    conc_obs = obs['concentration']  # list of (time, value) tuples
    if conc_obs:
        conc_times, conc_values = zip(*conc_obs)
        conc_times = np.array(conc_times)
        conc_values = np.array(conc_values)
    else:
        raise ValueError(f"No concentration observations for patient {patient_id}")

    # Extract blood pressure observations
    bp_obs = obs['blood_pressure']
    if bp_obs:
        bp_times, bp_values = zip(*bp_obs)
        bp_times = np.array(bp_times)
        bp_values = np.array(bp_values)
    else:
        raise ValueError(f"No blood pressure observations for patient {patient_id}")

    # Interpolate with constant extrapolation (left/right fill)
    Cc_obs_interp = np.interp(times, conc_times, conc_values)
    BP_obs_interp = np.interp(times, bp_times, bp_values)
    
    plt.plot(times, Cc_obs_interp)

    return Cc_obs_interp, BP_obs_interp


def precompute_injection_rates(patient_id, times, injections_dict):
    """Precompute INOR injection rates at each time point.

    Args:
        patient_id: Patient ID
        times: np.array of time points
        injections_dict: Dictionary of injection protocols

    Returns:
        inor_values: np.array of injection rates at each time point
    """
    model = NorepinephrinePKPD(injections_dict)
    inor_values = np.array([model.INOR(t, patient_id) for t in times])
    return inor_values


# ============================================================================
# CASADI OPTIMIZATION
# ============================================================================

def optimize_patient_parameters(times, Cc_obs, BP_obs, inor_values,
                                Ad_data, Ac_data, Ap_data, E_data):
    """Run CasADi optimization to estimate PKPD parameters.

    Args:
        times: np.array of time points
        Cc_obs: Interpolated concentration observations
        BP_obs: Interpolated blood pressure observations
        inor_values: Precomputed injection rates
        Ad_data, Ac_data, Ap_data: State data for initialization
        E_data: Windkessel state for initialization

    Returns:
        params_opt: dict of optimized parameters
        traj_opt: dict of optimized trajectories
        cost_value: final cost value
    """
    opti = ca.Opti()
    N = len(times) - 1  # Number of intervals
    dt_values = np.diff(times)  # Variable time steps

    print(f"  Setting up optimization problem: N={N} intervals, dt range=[{dt_values.min():.2f}, {dt_values.max():.2f}]s")

    # PK parameters (6 - always optimized)
    C_endo = opti.variable()
    k_a = opti.variable()
    V_c = opti.variable()
    k_12 = opti.variable()
    k_21 = opti.variable()
    k_el = opti.variable()

    # True model for fixed parameter values
    true_model = NorepinephrinePKPD()

    # PD Emax parameters (3 - conditional)
    if COST_FUNCTION_MODE in ['emax', 'both']:
        E_0 = opti.variable()
        E_max = opti.variable()
        EC_50 = opti.variable()
    else:
        # Fixed values when not optimizing Emax
        E_0 = true_model.E_0
        E_max = true_model.E_max
        EC_50 = true_model.EC_50

    # PD Windkessel parameters (3 - conditional)
    if COST_FUNCTION_MODE in ['windkessel', 'both']:
        omega = opti.variable()
        zeta = opti.variable()
        nu = opti.variable()
    else:
        # Fixed values when not optimizing Windkessel
        omega = true_model.omega
        zeta = true_model.zeta
        nu = true_model.nu

    # Parameter bounds (physical constraints)
    opti.subject_to(C_endo >= 0)
    opti.subject_to(k_a >= 0)
    opti.subject_to(V_c > 0)
    opti.subject_to(k_12 >= 0)
    opti.subject_to(k_21 >= 0)
    opti.subject_to(k_el >= 0)

    if COST_FUNCTION_MODE in ['emax', 'both']:
        opti.subject_to(E_0 > 0)
        opti.subject_to(E_max > E_0)
        opti.subject_to(EC_50 > 0)

    if COST_FUNCTION_MODE in ['windkessel', 'both']:
        opti.subject_to(omega > 0)
        opti.subject_to(zeta > 0)
        opti.subject_to(nu > 0)

    # PK states: Ad, Ac, Ap (always needed)
    Ad = opti.variable(N + 1)
    Ac = opti.variable(N + 1)
    Ap = opti.variable(N + 1)

    # PD Windkessel states: E, dE/dt (only if windkessel mode)
    if COST_FUNCTION_MODE in ['windkessel', 'both']:
        E = opti.variable(N + 1)
        dEdt = opti.variable(N + 1)

    opti.subject_to(Ad[0] == 0.0)
    opti.subject_to(Ac[0] == 0.0)
    opti.subject_to(Ap[0] == 0.0)

    if COST_FUNCTION_MODE in ['windkessel', 'both']:
        opti.subject_to(E[0] == 57.09)
        opti.subject_to(dEdt[0] == 0.0)

    print("  Building dynamics constraints...")

    for k in range(N):
        dt = dt_values[k]

        # PK equations (Euler implicit at k+1) - ALWAYS
        # dAd/dt = -k_a * Ad + INOR
        opti.subject_to(Ad[k+1] == Ad[k] + dt * (-k_a * Ad[k+1] + inor_values[k+1]))

        # dAc/dt = k_a * Ad - (k_12 + k_el) * Ac + k_21 * Ap
        opti.subject_to(Ac[k+1] == Ac[k] + dt * (
            k_a * Ad[k+1] - (k_12 + k_el) * Ac[k+1] + k_21 * Ap[k+1]
        ))

        # dAp/dt = k_12 * Ac - k_21 * Ap
        opti.subject_to(Ap[k+1] == Ap[k] + dt * (
            k_12 * Ac[k+1] - k_21 * Ap[k+1]
        ))

        # Windkessel equations (only if windkessel mode)
        if COST_FUNCTION_MODE in ['windkessel', 'both']:
            # Compute concentration at k+1 (needed for PD)
            Cc_kp1 = C_endo + Ac[k+1] / V_c

            # dE/dt = dEdt
            opti.subject_to(E[k+1] == E[k] + dt * dEdt[k+1])

            # d²E/dt² = nu * Cc - 2*zeta*omega*dEdt - omega²*E
            opti.subject_to(dEdt[k+1] == dEdt[k] + dt * (
                nu * Cc_kp1 - 2 * zeta * omega * dEdt[k+1] - omega**2 * E[k+1]
            ))

    print("  Building cost function...")

    # Compute concentration trajectory
    Cc = C_endo + Ac / V_c

    cost_conc = ca.sumsqr(Cc - Cc_obs)

    # Blood pressure cost term (flexible based on mode)
    cost_bp = 0

    if COST_FUNCTION_MODE in ['windkessel', 'both']:
        # Windkessel cost: compare E with BP observations
        cost_bp += ca.sumsqr(E - BP_obs)

    if COST_FUNCTION_MODE in ['emax', 'both']:
        # Emax cost: compute E_emax algebraically and compare with BP observations
        E_emax = E_0 + (E_max - E_0) * Cc / (Cc + EC_50)
        cost_bp += ca.sumsqr(E_emax - BP_obs)

    cost_total = cost_conc + cost_bp

    opti.minimize(cost_total)

    # Configure IPOPT
    opts = {
        'ipopt.max_iter': 5000,
        'ipopt.tol': 1e-6,
        'ipopt.acceptable_tol': 1e-4,
        'ipopt.acceptable_iter': 15,
        'ipopt.print_level': 5,
        'print_time': False
    }
    opti.solver('ipopt', opts)

    print("  Setting initial guesses...")

    # Initialize PK parameters with true values from pkpd model
    model_for_init = NorepinephrinePKPD()
    opti.set_initial(C_endo, model_for_init.C_endo)
    opti.set_initial(k_a, model_for_init.k_a)
    opti.set_initial(V_c, model_for_init.V_c)
    opti.set_initial(k_12, model_for_init.k_12)
    opti.set_initial(k_21, model_for_init.k_21)
    opti.set_initial(k_el, model_for_init.k_el)

    # Initialize PD parameters (only if they are variables)
    if COST_FUNCTION_MODE in ['emax', 'both']:
        opti.set_initial(E_0, model_for_init.E_0)
        opti.set_initial(E_max, model_for_init.E_max)
        opti.set_initial(EC_50, model_for_init.EC_50)

    if COST_FUNCTION_MODE in ['windkessel', 'both']:
        opti.set_initial(omega, model_for_init.omega)
        opti.set_initial(zeta, model_for_init.zeta)
        opti.set_initial(nu, model_for_init.nu)

    # Initialize state trajectories with .npy data
    opti.set_initial(Ad, Ad_data)
    opti.set_initial(Ac, Ac_data)
    opti.set_initial(Ap, Ap_data)

    if COST_FUNCTION_MODE in ['windkessel', 'both']:
        opti.set_initial(E, E_data)
        opti.set_initial(dEdt, np.zeros(N+1))  # Start with zero velocity

    print("  Solving optimization problem...")
    try:
        sol = opti.solve()
        print("  ✓ Optimization converged successfully!")
    except RuntimeError as e:
        print(f"  ⚠ Solver did not fully converge: {str(e).split('return_status is')[1].strip().strip(chr(39))}")
        print("  ⚠ Using best solution found (may still be reasonable)...")
        sol = opti.debug

    # Extract PK parameters (always optimized)
    params_opt = {
        'C_endo': sol.value(C_endo),
        'k_a': sol.value(k_a),
        'V_c': sol.value(V_c),
        'k_12': sol.value(k_12),
        'k_21': sol.value(k_21),
        'k_el': sol.value(k_el),
    }

    # Extract PD parameters (conditional)
    if COST_FUNCTION_MODE in ['emax', 'both']:
        params_opt['E_0'] = sol.value(E_0)
        params_opt['E_max'] = sol.value(E_max)
        params_opt['EC_50'] = sol.value(EC_50)
    else:
        # Use fixed values
        params_opt['E_0'] = float(E_0)
        params_opt['E_max'] = float(E_max)
        params_opt['EC_50'] = float(EC_50)

    if COST_FUNCTION_MODE in ['windkessel', 'both']:
        params_opt['omega'] = sol.value(omega)
        params_opt['zeta'] = sol.value(zeta)
        params_opt['nu'] = sol.value(nu)
    else:
        # Use fixed values
        params_opt['omega'] = float(omega)
        params_opt['zeta'] = float(zeta)
        params_opt['nu'] = float(nu)

    # Extract state trajectories
    traj_opt = {
        'times': times,
        'Ad': sol.value(Ad),
        'Ac': sol.value(Ac),
        'Ap': sol.value(Ap),
    }

    if COST_FUNCTION_MODE in ['windkessel', 'both']:
        traj_opt['E'] = sol.value(E)
        traj_opt['dEdt'] = sol.value(dEdt)

    # Extract cost value
    cost_value = sol.value(cost_total)

    return params_opt, traj_opt, cost_value


# ============================================================================
# RE-SIMULATION WITH OPTIMIZED PARAMETERS
# ============================================================================

def resimulate_with_optimized_params(patient_id, params_opt, injections_dict):
    """Create new model with optimized parameters and simulate.

    Args:
        patient_id: Patient ID
        params_opt: dict of optimized parameters
        injections_dict: Dictionary of injection protocols

    Returns:
        t, Ad, Ac, Ap, E_emax, E_windkessel: Simulation results
    """
    # Create model instance
    model = NorepinephrinePKPD(injections_dict, is_linear=IS_LINEAR)

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
    t, Ad, Ac, Ap, E_emax, E_windkessel = model.simulate(patient_id, t_end=2200, dt=0.5)

    return t, Ad, Ac, Ap, E_emax, E_windkessel


# ============================================================================
# SAVE OPTIMAL PARAMETERS
# ============================================================================

def save_optimal_parameters(patient_id, params_opt, cost_value=None):
    """Save optimized parameters to JSON file.

    Args:
        patient_id: Patient ID
        params_opt: Dictionary of optimized parameters
        cost_value: Optional final cost value
    """
    output_dir = f'{DATA_DIR}/patient_{patient_id}/{OUTPUT_DIR}/{N_DATA_POINTS}_points'
    os.makedirs(output_dir, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    params_dict = {k: float(v) for k, v in params_opt.items()}

    # Add metadata
    params_dict['patient_id'] = int(patient_id)
    params_dict['cost_function_mode'] = COST_FUNCTION_MODE
    params_dict['is_linear'] = IS_LINEAR
    params_dict['n_data_points'] = N_DATA_POINTS

    if cost_value is not None:
        params_dict['final_cost'] = float(cost_value)

    json_path = f'{output_dir}/params.json'
    with open(json_path, 'w') as f:
        json.dump(params_dict, f, indent=2)

    print(f"  ✓ Parameters saved to {json_path}")


# ============================================================================
# RESULTS PRINTING
# ============================================================================

def print_optimization_results(patient_id, params_initial, params_opt):
    """Print comparison table of true vs optimized parameters.

    Args:
        patient_id: Patient ID
        params_true: dict of true parameter values
        params_opt: dict of optimized parameter values
    """
    print(f"\n{'='*70}")
    print(f"PATIENT {patient_id} - OPTIMIZATION RESULTS")
    print(f"{'='*70}")
    print(f"{'Parameter':<15} {'True Value':>15} {'Optimized':>15}")
    print(f"{'-'*70}")

    for param_name in params_initial.keys():
        true_val = params_initial[param_name]
        opt_val = params_opt[param_name]
        print(f"{param_name:<15} {true_val:>15.6f} {opt_val:>15.6f}")

    print(f"{'='*70}\n")


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_optimization_results(patient_id, observations, traj_opt, params_opt, resim_results):
    """Create comparison plots for concentration and blood pressure.

    Args:
        patient_id: Patient ID
        observations: Dict from load_observations()
        traj_opt: Dict of optimized trajectories from CasADi
        params_opt: Optimized parameters
        resim_results: Re-simulation results tuple
    """
    output_dir = f'{DATA_DIR}/patient_{patient_id}/{OUTPUT_DIR}/{N_DATA_POINTS}_points'
    os.makedirs(output_dir, exist_ok=True)

    obs = observations[patient_id]

    # Extract REAL observations
    conc_obs = obs['concentration']
    conc_times, conc_values = zip(*conc_obs)
    conc_times = np.array(conc_times)
    conc_values = np.array(conc_values)

    bp_obs = obs['blood_pressure']
    bp_times, bp_values = zip(*bp_obs)
    bp_times = np.array(bp_times)
    bp_values = np.array(bp_values)

    # Get CasADi trajectories
    times_casadi = traj_opt['times']
    Ac_casadi = traj_opt['Ac']
    Cc_casadi = params_opt['C_endo'] + Ac_casadi / params_opt['V_c']

    # Get PKPD resimulated trajectories
    t_resim, _, Ac_resim, _, E_emax_resim, E_windkessel_resim = resim_results
    Cc_resim = params_opt['C_endo'] + Ac_resim / params_opt['V_c']

    # ========================================================================
    # PLOT 1: CONCENTRATION
    # ========================================================================

    fig1, ax1 = plt.subplots(figsize=(12, 6))

    ax1.scatter(conc_times, conc_values, c='red', s=80, zorder=5,
                label='Real Observations', alpha=0.8, edgecolors='darkred')
    ax1.plot(times_casadi, Cc_casadi, 'b-', linewidth=2, label='CasADi Optimized', alpha=0.7)
    ax1.plot(t_resim, Cc_resim, 'g--', linewidth=2, label='PKPD Resimulated', alpha=0.7)

    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('NOR Concentration (nmol/L)', fontsize=12)
    ax1.set_title(f'Patient {patient_id} - NOR Concentration (N={N_DATA_POINTS})',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cc_opt.png', dpi=150)
    plt.close()

    # ========================================================================
    # PLOT 2: BLOOD PRESSURE
    # ========================================================================

    fig2, ax2 = plt.subplots(figsize=(12, 6))

    ax2.scatter(bp_times, bp_values, c='red', s=80, zorder=5,
                label='Real Observations', alpha=0.8, edgecolors='darkred')

    if COST_FUNCTION_MODE == 'emax':
        E_casadi = params_opt['E_0'] + (params_opt['E_max'] - params_opt['E_0']) * Cc_casadi / (Cc_casadi + params_opt['EC_50'])
        ax2.plot(times_casadi, E_casadi, 'b-', linewidth=2, label='CasADi Optimized (Emax)', alpha=0.7)
        ax2.plot(t_resim, E_emax_resim, 'g--', linewidth=2, label='PKPD Resimulated (Emax)', alpha=0.7)
    elif COST_FUNCTION_MODE == 'windkessel':
        E_casadi = traj_opt['E']
        ax2.plot(times_casadi, E_casadi, 'b-', linewidth=2, label='CasADi Optimized (Windkessel)', alpha=0.7)
        ax2.plot(t_resim, E_windkessel_resim, 'g--', linewidth=2, label='PKPD Resimulated (Windkessel)', alpha=0.7)
    else:  # both
        E_casadi = traj_opt['E']
        ax2.plot(times_casadi, E_casadi, 'b-', linewidth=2, label='CasADi Optimized (Windkessel)', alpha=0.7)
        ax2.plot(t_resim, E_windkessel_resim, 'g--', linewidth=2, label='PKPD Resimulated (Windkessel)', alpha=0.7)

    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('MAP (mmHg)', fontsize=12)
    ax2.set_title(f'Patient {patient_id} - Blood Pressure (N={N_DATA_POINTS})',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bp_opt.png', dpi=150)
    plt.close()

    print(f"  ✓ Plots saved to {output_dir}/")


def plot_interpolation_verification(patient_id, observations, times_interp, Cc_interp, Bp_interp):
    """Create verification plots for BP/Cc interpolation across full time span.

    Args:
        patient_id: Patient ID
        observations: Dict from load_observations()
    """
    output_dir = f'{DATA_DIR}/patient_{patient_id}/data'
    os.makedirs(output_dir, exist_ok=True)

    obs = observations[patient_id]

    # Extract raw observations
    conc_obs = obs['concentration']
    conc_times, conc_values = zip(*conc_obs)
    conc_times = np.array(conc_times)
    conc_values = np.array(conc_values)

    bp_obs = obs['blood_pressure']
    bp_times, bp_values = zip(*bp_obs)
    bp_times = np.array(bp_times)
    bp_values = np.array(bp_values)
    
    # Plot Concentration
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    ax1.scatter(conc_times, conc_values, c='red', s=80, zorder=5,
                label='Real Observations', alpha=0.8, edgecolors='darkred')
    ax1.plot(times_interp, Cc_interp, 'b-', linewidth=2,
             label='Interpolation (constant extrapolation)', alpha=0.7)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Concentration (ng/mL)', fontsize=12)
    ax1.set_title(f'Patient {patient_id} - Concentration Interpolation Verification',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cc.png', dpi=150)
    plt.close()

    # Plot Blood Pressure
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.scatter(bp_times, bp_values, c='red', s=80, zorder=5,
                label='Real Observations', alpha=0.8, edgecolors='darkred')
    ax2.plot(times_interp, Bp_interp, 'b-', linewidth=2,
             label='Interpolation (constant extrapolation)', alpha=0.7)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('MAP (mmHg)', fontsize=12)
    ax2.set_title(f'Patient {patient_id} - Blood Pressure Interpolation Verification',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bp.png', dpi=150)
    plt.close()

    print(f"  ✓ Interpolation verification plots saved to {output_dir}/")


def plot_pkpd_vs_casadi_trajectories(patient_id, traj_opt, resim_results, params_opt):
    """Compare CasADi optimization trajectories vs PKPD model re-simulation.

    Args:
        patient_id: Patient ID
        traj_opt: Dict of optimized trajectories from CasADi (keys: Ad, Ac, Ap, E, dEdt)
        resim_results: Tuple from resimulate_with_params (t, Ad, Ac, Ap, E_emax, E_windkessel)
        params_opt: Dict of optimized parameters
    """
    output_dir = f'{DATA_DIR}/patient_{patient_id}/{OUTPUT_DIR}/{N_DATA_POINTS}_points'
    os.makedirs(output_dir, exist_ok=True)

    t_resim, Ad_resim, Ac_resim, Ap_resim, E_emax_resim, E_windkessel_resim = resim_results

    # Create comparison plot with subplots for each state
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Patient {patient_id} - CasADi vs PKPD Model Trajectories',
                 fontsize=16, fontweight='bold')

    # Get CasADi times (assuming uniform grid from traj_opt)
    times_casadi = traj_opt.get('times', None)
    if times_casadi is None:
        N = len(traj_opt['Ad']) - 1
        times_casadi = np.linspace(t_resim[0], t_resim[-1], N + 1)

    # Plot Ad
    ax = axes[0, 0]
    ax.plot(times_casadi, traj_opt['Ad'], 'b-', linewidth=2, label='CasADi', alpha=0.7)
    ax.plot(t_resim, Ad_resim, 'r--', linewidth=2, label='PKPD Model', alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Ad (ng)', fontsize=11)
    ax.set_title('Depot Compartment (Ad)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot Ac
    ax = axes[0, 1]
    ax.plot(times_casadi, traj_opt['Ac'], 'b-', linewidth=2, label='CasADi', alpha=0.7)
    ax.plot(t_resim, Ac_resim, 'r--', linewidth=2, label='PKPD Model', alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Ac (ng)', fontsize=11)
    ax.set_title('Central Compartment (Ac)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot Ap
    ax = axes[1, 0]
    ax.plot(times_casadi, traj_opt['Ap'], 'b-', linewidth=2, label='CasADi', alpha=0.7)
    ax.plot(t_resim, Ap_resim, 'r--', linewidth=2, label='PKPD Model', alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Ap (ng)', fontsize=11)
    ax.set_title('Peripheral Compartment (Ap)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot E (BP effect)
    ax = axes[1, 1]
    if COST_FUNCTION_MODE == 'emax':
        # CasADi doesn't have E trajectory in emax mode, compute it
        Cc_casadi = params_opt['C_endo'] + traj_opt['Ac'] / params_opt['V_c']
        E_casadi = params_opt['E_0'] + (params_opt['E_max'] - params_opt['E_0']) * Cc_casadi / (Cc_casadi + params_opt['EC_50'])
        ax.plot(times_casadi, E_casadi, 'b-', linewidth=2, label='CasADi (Emax)', alpha=0.7)
        ax.plot(t_resim, E_emax_resim, 'r--', linewidth=2, label='PKPD Model (Emax)', alpha=0.7)
    elif COST_FUNCTION_MODE == 'windkessel':
        ax.plot(times_casadi, traj_opt['E'], 'b-', linewidth=2, label='CasADi (Windkessel)', alpha=0.7)
        ax.plot(t_resim, E_windkessel_resim, 'r--', linewidth=2, label='PKPD Model (Windkessel)', alpha=0.7)
    else:  # both
        ax.plot(times_casadi, traj_opt['E'], 'b-', linewidth=2, label='CasADi (Windkessel)', alpha=0.7)
        ax.plot(t_resim, E_windkessel_resim, 'r--', linewidth=2, label='PKPD Model (Windkessel)', alpha=0.7)

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('MAP (mmHg)', fontsize=11)
    ax.set_title('Blood Pressure Effect (E)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/pkpd_vs_casadi_traj.png', dpi=150)
    plt.close()

    print(f"  ✓ CasADi vs PKPD comparison plot saved to {output_dir}/")


def main():
    print("\n" + "="*70)
    print("CASADI PKPD PARAMETER OPTIMIZATION - PATIENT-BY-PATIENT")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Patients: {PATIENT_IDS}")
    print(f"  - Data points: {N_DATA_POINTS}")
    print(f"  - Model: {'Linear' if IS_LINEAR else 'Non-linear'}")
    print(f"  - Cost function mode: {COST_FUNCTION_MODE}")
    print(f"  - Output directory: {OUTPUT_DIR}/")
    print("="*70 + "\n")

    injections = load_injections(PATIENT_IDS)
    observations = load_observations(PATIENT_IDS)

    # True parameters for comparison
    pkpd_model = NorepinephrinePKPD(injections, is_linear=IS_LINEAR)
    params_true = {
        'C_endo': pkpd_model.C_endo,
        'k_a': pkpd_model.k_a,
        'V_c': pkpd_model.V_c,
        'k_12': pkpd_model.k_12,
        'k_21': pkpd_model.k_21,
        'k_el': pkpd_model.k_el,
        'E_0': pkpd_model.E_0,
        'E_max': pkpd_model.E_max,
        'EC_50': pkpd_model.EC_50,
        'omega': pkpd_model.omega,
        'zeta': pkpd_model.zeta,
        'nu': pkpd_model.nu
    }

    for patient_id in PATIENT_IDS:
        print(f"\n{'#'*70}")
        print(f"# Processing Patient {patient_id}")
        print(f"{'#'*70}\n")

        print("Step 1: Loading .npy trajectories for initialization...")
        times, Ad_data, Ac_data, Ap_data, E_data, full_traj = load_patient_data(
            patient_id, N_DATA_POINTS, IS_LINEAR
        )
        print(f"  ✓ Loaded {len(times)} time points")

        print("\nStep 2: Interpolating real observations to time grid...")
        Cc_interp, BP_interp = interpolate_observations(observations, patient_id, times)
        print(f"  ✓ Interpolated {len(Cc_interp)} concentration observations")
        print(f"  ✓ Interpolated {len(BP_interp)} blood pressure observations")

        print("\nStep 3: Precomputing injection rates...")
        inor_values = precompute_injection_rates(patient_id, times, injections)
        print(f"  ✓ Precomputed INOR for {len(inor_values)} time points")

        print("\nStep 4: Running CasADi optimization...")
        params_opt, traj_opt, cost_value = optimize_patient_parameters(
            times, Cc_interp, BP_interp, inor_values,
            Ad_data, Ac_data, Ap_data, E_data
        )
        print(f"  ✓ Final cost: {cost_value:.4f}")

        print("\nStep 5: Saving optimal parameters...")
        save_optimal_parameters(patient_id, params_opt, cost_value)

        print("\nStep 6: Printing optimization results...")
        print_optimization_results(patient_id, params_true, params_opt)

        print("Step 7: Re-simulating with optimized parameters...")
        resim_results = resimulate_with_optimized_params(patient_id, params_opt, injections)
        print("  ✓ Re-simulation completed")

        print("\nStep 8: Creating visualization plots...")
        plot_optimization_results(patient_id, observations, traj_opt, params_opt, resim_results)

        print("\nStep 9: Comparing CasADi vs PKPD trajectories...")
        plot_pkpd_vs_casadi_trajectories(patient_id, traj_opt, resim_results, params_opt)

    print(f"\n{'='*70}")
    print("ALL PATIENTS PROCESSED SUCCESSFULLY")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
