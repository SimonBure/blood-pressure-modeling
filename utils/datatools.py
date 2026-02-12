"""Utility functions for data loading, saving, and printing."""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, Tuple, Optional, List


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_observations(patient_ids: Optional[List[int]] = None,
                     csv_path: str = 'data/joachim.csv') -> Dict[int, Dict]:
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


def load_injections(patient_ids: Optional[List[int]] = None,
                   csv_path: str = 'data/injections.csv') -> Dict[int, Tuple]:
    """Load injection protocols for specified patients.

    Args:
        patient_ids: List of patient IDs to load injection data for, or None for all patients.
        csv_path: Path to CSV file with injection data.

    Returns:
        Dictionary mapping patient_id to (times, amounts, durations) tuple.
    """
    inj_df = pd.read_csv(csv_path)

    # If no patients specified, load all available patients
    if not patient_ids:
        patient_ids = sorted(inj_df['patient_id'].unique())

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


def load_patient_e0_indiv(patient_ids: Optional[List[int]] = None,
                          csv_path: str = 'data/joachim.csv') -> Dict[int, float]:
    """Load individual baseline E0 (starting blood pressure) for each patient.

    Args:
        patient_ids: List of patient IDs to load, or None for all patients.
        csv_path: Path to CSV file with observations.

    Returns:
        Dictionary mapping patient_id to E0_indiv value (baseline BP in mmHg).
    """
    df = pd.read_csv(csv_path)
    available_patients = sorted(df['id'].unique())

    # If no patients specified, load all
    if not patient_ids:
        patient_ids = available_patients

    e0_dict = {}
    for pid in patient_ids:
        patient_data = df[df['id'] == pid]
        if not patient_data.empty:
            # E0_indiv is constant per patient (same value in all rows)
            e0_dict[pid] = float(patient_data['E0_indiv'].iloc[0])

    return e0_dict


def load_resimulated_trajectories(patient_id: int,
                                  res_dir: str,
                                  pkpd_dir: str) -> Tuple:
    """Load resimulated trajectories from disk.

    Args:
        patient_id: Patient ID.
        res_dir: Base results directory (e.g., 'results').
        pkpd_dir: PKPD output subdirectory (e.g., 'opti').

    Returns:
        Tuple of (t, Ad, Ac, Ap, E_emax) arrays.

    Raises:
        FileNotFoundError: If trajectory files don't exist.
    """
    traj_path = f'{res_dir}/patient_{patient_id}/pkpd/{pkpd_dir}'
    time_file = f'{traj_path}/time.npy'

    if not os.path.exists(time_file):
        raise FileNotFoundError(
            f"\nERROR: Cannot run plot-only mode for patient {patient_id}\n"
            f"  Missing: {traj_path}/*.npy\n"
            f"  → Run resimulation first with mode='resim_and_plot' or mode='full'"
        )

    try:
        t = np.load(f'{traj_path}/time.npy')
        Ad = np.load(f'{traj_path}/Ad.npy')
        Ac = np.load(f'{traj_path}/Ac.npy')
        Ap = np.load(f'{traj_path}/Ap.npy')
        E_emax = np.load(f'{traj_path}/bp_emax.npy')
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"\nERROR: Incomplete trajectory files for patient {patient_id}\n"
            f"  Missing: {traj_path}/*.npy\n"
            f"  → Run resimulation with mode='resim_and_plot' or mode='full'"
        ) from e

    return t, Ad, Ac, Ap, E_emax


def load_patient_covariates(
    patient_ids: List[int], csv_path: str = "data/joachim.csv"
) -> pd.DataFrame:
    """Load biological covariables for each patient from joachim.csv.

    Args:
        patient_ids: List of patient IDs to load covariables for.
        csv_path: Path to joachim.csv file.

    Returns:
        DataFrame with one row per patient containing covariable values.
        Columns: patient_id, hr, pi, age, weight, height, sex, DFG, E0_indiv, HTA, IECARA, TABAC
    """
    df = pd.read_csv(csv_path)

    # Filter to requested patients
    df_filtered = df[df["id"].isin(patient_ids)]

    # Covariables are constant per patient (except hr which is time-varying)
    # We'll take the first value for each patient
    covariable_cols = [
        "id",
        "hr",
        "pi",
        "age",
        "weight",
        "height",
        "sex",
        "DFG",
        "E0_indiv",
        "HTA",
        "IECARA",
        "TABAC",
    ]

    # Group by patient and take first row to get per-patient covariables
    covariables_df = (
        df_filtered.groupby("id")[covariable_cols].first().reset_index(drop=True)
    )
    covariables_df.rename(columns={"id": "patient_id"}, inplace=True)

    return covariables_df


# ==============================================================================
# DATA SAVING
# ==============================================================================

def save_optimal_parameters(patient_id: int,
                           params_opt: Dict[str, float],
                           cost_value: float,
                           data_dir: str,
                           output_dir: str,
                           n_original_observations: int,
                           n_optimization_points: int) -> None:
    """Save optimized parameters to JSON file.

    Args:
        patient_id: Patient ID.
        params_opt: Dictionary of optimized parameters.
        cost_value: Final cost value.
        data_dir: Base data directory.
        output_dir: Output subdirectory name.
        n_original_observations: Total number of available observations.
        n_optimization_points: Number of points actually used for optimization (after subsampling).
    """
    output_path = f'{data_dir}/patient_{patient_id}/{output_dir}'
    os.makedirs(output_path, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    params_dict = {k: float(v) for k, v in params_opt.items()}

    # Add metadata
    params_dict['patient_id'] = int(patient_id)
    params_dict['n_original_observations'] = n_original_observations
    params_dict['n_optimization_points'] = n_optimization_points
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
