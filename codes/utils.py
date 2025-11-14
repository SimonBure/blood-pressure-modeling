"""Utility functions for data loading, saving, and printing."""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, Tuple, Optional, List
from pkpd import NorepinephrinePKPD


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_observations(patient_ids: Optional[List[int]] = None,
                     csv_path: str = 'codes/data/joachim.csv') -> Dict[int, Dict]:
    """Load observation data for specified patients.

    Args:
        patient_ids: List of patient IDs to load, or None for all patients.
        csv_path: Path to CSV file with observations.

    Returns:
        Dictionary mapping patient_id to dict with 'concentration' and 'blood_pressure' lists.
        Each list contains (time, value) tuples.
    """
    df = pd.read_csv(csv_path)
    available_patients = sorted(df['id'].unique())

    # If no patients specified, load all
    if not patient_ids:
        patient_ids = available_patients
    else:
        # Validate that requested patients exist in the dataset
        invalid_patients = [pid for pid in patient_ids if pid not in available_patients]
        if invalid_patients:
            print(f"\n  ERROR: The following patient IDs do not exist in observations dataset:")
            print(f"    Requested: {invalid_patients}")
            print(f"    Available: {available_patients}")
            exit(1)

    df_obs = df[df['obs'] != '.'].copy()
    df_obs['obs'] = pd.to_numeric(df_obs['obs'])
    df_obs['obsid'] = pd.to_numeric(df_obs['obsid'])

    observations_dict = {}
    for pid in patient_ids:
        patient_obs = df_obs[df_obs['id'] == pid]
        conc_obs = patient_obs[patient_obs['obsid'] == 0]
        bp_obs = patient_obs[patient_obs['obsid'] == 1]
        observations_dict[pid] = {
            'concentration': list(zip(conc_obs['time(s)'].values, conc_obs['obs'].values)),
            'blood_pressure': list(zip(bp_obs['time(s)'].values, bp_obs['obs'].values))
        }
    return observations_dict


def load_injections(patient_ids: List[int],
                   csv_path: str = 'codes/data/injections.csv') -> Dict[int, Tuple]:
    """Load injection protocols for specified patients.

    Args:
        patient_ids: List of patient IDs to load injection data for.
        csv_path: Path to CSV file with injection data.

    Returns:
        Dictionary mapping patient_id to (times, amounts, durations) tuple.
    """
    inj_df = pd.read_csv(csv_path)
    injections_dict = {}

    for pid in patient_ids:
        p_inj = inj_df[inj_df['patient_id'] == pid].sort_values('injection_time_s')
        if len(p_inj) > 0:
            injections_dict[pid] = (
                p_inj['injection_time_s'].values,
                p_inj['amount_nmol'].values,
                p_inj['duration_s'].values
            )
        else:
            # Patient has no injection data, use empty arrays
            injections_dict[pid] = (
                np.array([]),
                np.array([]),
                np.array([])
            )

    return injections_dict


def load_patient_data(patient_id: int, n_points: int, is_linear: bool,
                     data_dir: str, init_bp_model: str) -> Tuple:
    """Load and subsample patient trajectory data from .npy files.

    Args:
        patient_id: Patient ID.
        n_points: Number of points to subsample.
        is_linear: True for linear_no_lag, False for power_no_lag.
        data_dir: Base directory for patient data.
        init_bp_model: 'windkessel' or 'emax' - which BP model to use for initialization.

    Returns:
        Tuple of (times, Ad_data, Ac_data, Ap_data, E_data, full_trajectories).
    """
    subdir = 'linear_no_lag' if is_linear else 'power_no_lag'
    base_path = f'{data_dir}/patient_{patient_id}/{subdir}'

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
    E_data = bp_windkessel_full[indices] if init_bp_model == 'windkessel' else bp_emax_full[indices]

    full_trajectories = {
        'time': time_full,
        'Ad': Ad_full,
        'Ac': Ac_full,
        'Ap': Ap_full,
        'bp_emax': bp_emax_full,
        'bp_windkessel': bp_windkessel_full
    }

    return times, Ad_data, Ac_data, Ap_data, E_data, full_trajectories


def interpolate_observations(observations: Dict[int, Dict],
                            patient_id: int,
                            times: np.ndarray) -> np.ndarray:
    """Interpolate blood pressure observations to optimization time grid.

    Uses constant extrapolation outside observation range.

    Args:
        observations: Dict from load_observations().
        patient_id: Patient ID.
        times: np.array of N+1 time points.

    Returns:
        BP_obs: np.array of shape (N+1,) with interpolated blood pressure.
    """
    obs = observations[patient_id]

    # Extract blood pressure observations
    bp_obs = obs['blood_pressure']
    if bp_obs:
        bp_times, bp_values = zip(*bp_obs)
        bp_times = np.array(bp_times)
        bp_values = np.array(bp_values)
    else:
        raise ValueError(f"No blood pressure observations for patient {patient_id}")

    # Interpolate with constant extrapolation (left/right fill)
    BP_obs_interp = np.interp(times, bp_times, bp_values)

    return BP_obs_interp


# ==============================================================================
# DATA SAVING
# ==============================================================================

def save_optimal_parameters(patient_id: int,
                           params_opt: Dict[str, float],
                           cost_value: float,
                           data_dir: str,
                           output_dir: str,
                           n_data_points: int,
                           cost_function_mode: str,
                           is_linear: bool) -> None:
    """Save optimized parameters to JSON file.

    Args:
        patient_id: Patient ID.
        params_opt: Dictionary of optimized parameters.
        cost_value: Final cost value.
        data_dir: Base data directory.
        output_dir: Output subdirectory name.
        n_data_points: Number of data points used.
        cost_function_mode: Cost function mode used.
        is_linear: Whether linear model was used.
    """
    output_path = f'{data_dir}/patient_{patient_id}/{output_dir}/{n_data_points}_points'
    os.makedirs(output_path, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    params_dict = {k: float(v) for k, v in params_opt.items()}

    # Add metadata
    params_dict['patient_id'] = int(patient_id)
    params_dict['cost_function_mode'] = cost_function_mode
    params_dict['is_linear'] = is_linear
    params_dict['n_data_points'] = n_data_points
    params_dict['final_cost'] = float(cost_value)

    json_path = f'{output_path}/params.json'
    with open(json_path, 'w') as f:
        json.dump(params_dict, f, indent=2)

    print(f"  ✓ Parameters saved to {json_path}")


# ==============================================================================
# RESULTS PRINTING
# ==============================================================================

def print_optimization_results(patient_id: int,
                              params_initial: Dict[str, float],
                              params_opt: Dict[str, float]) -> None:
    """Print comparison table of initial vs optimized parameters.

    Args:
        patient_id: Patient ID.
        params_initial: dict of initial parameter values.
        params_opt: dict of optimized parameter values.
    """
    print(f"\n{'='*70}")
    print(f"PATIENT {patient_id} - OPTIMIZATION RESULTS")
    print(f"{'='*70}")
    print(f"{'Parameter':<15} {'Initial Value':>15} {'Optimized':>15}")
    print(f"{'-'*70}")

    for param_name in params_initial.keys():
        init_val = params_initial[param_name]
        opt_val = params_opt[param_name]
        print(f"{param_name:<15} {init_val:>15.6f} {opt_val:>15.6f}")

    print(f"{'='*70}\n")
