"""
Data preparation functions for optimization pipeline.

This module handles data loading, preprocessing, and initial guess preparation
for the PKPD parameter optimization.
"""

import numpy as np
from typing import Dict, Tuple
from pkpd import NorepinephrinePKPD


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


def get_initial_guess_from_pkpd(patient_id: int,
                                times: np.ndarray,
                                data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate initial guess for state trajectories from precomputed PKPD model.

    Loads trajectories from pkpd/standalone/ directory and interpolates to match observation times.

    Args:
        patient_id: Patient ID.
        times: Array of observation time points.
        data_dir: Base directory for patient data.

    Returns:
        Tuple of (Ad_data, Ac_data, Ap_data, E_data) arrays interpolated to observation times.

    Raises:
        FileNotFoundError: If PKPD standalone trajectories don't exist.
    """
    pkpd_path = f'{data_dir}/patient_{patient_id}/pkpd/standalone'

    try:
        # Load full trajectories from pkpd/standalone/ directory
        time_full = np.load(f'{pkpd_path}/time.npy')
        Ad_full = np.load(f'{pkpd_path}/Ad.npy')
        Ac_full = np.load(f'{pkpd_path}/Ac.npy')
        Ap_full = np.load(f'{pkpd_path}/Ap.npy')
        E_full = np.load(f'{pkpd_path}/bp_emax.npy')
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"\nERROR: Cannot load PKPD standalone trajectories for patient {patient_id}\n"
            f"  Missing: {pkpd_path}/*.npy\n"
            f"  → Run standalone PKPD simulation first: python pkpd.py"
        ) from e

    # Interpolate to match observation time points
    Ad_data = np.interp(times, time_full, Ad_full)
    Ac_data = np.interp(times, time_full, Ac_full)
    Ap_data = np.interp(times, time_full, Ap_full)
    E_data = np.interp(times, time_full, E_full)

    return Ad_data, Ac_data, Ap_data, E_data


def extract_observation_data(patient_id: int,
                             observations: Dict,
                             max_data_points: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Extract and optionally subsample observation data for a patient.

    Args:
        patient_id: Patient ID.
        observations: Dictionary of observations from load_observations().
        max_data_points: Maximum number of data points to use (subsample if exceeded).

    Returns:
        Tuple of (times, BP_obs, n_original_observations, n_optimization_points).

    Raises:
        ValueError: If no blood pressure observations exist for patient.
    """
    # Get blood pressure observations
    bp_obs_data = observations[patient_id]['blood_pressure']
    if not bp_obs_data:
        raise ValueError(f"No blood pressure observations for patient {patient_id}")

    # Extract times and values
    times = np.array([t for t, _ in bp_obs_data])
    BP_obs = np.array([v for _, v in bp_obs_data])
    n_original_observations = len(times)

    # Apply subsampling if needed
    if n_original_observations > max_data_points:
        print(f"  ⚠ Subsampling from {n_original_observations} to {max_data_points} points")
        indices = np.linspace(0, n_original_observations - 1, max_data_points, dtype=int)
        times = times[indices]
        BP_obs = BP_obs[indices]
        n_optimization_points = max_data_points
    else:
        n_optimization_points = n_original_observations

    return times, BP_obs, n_original_observations, n_optimization_points


def prepare_optimization_inputs(patient_id: int,
                                config,
                                observations: Dict,
                                injections_dict: Dict,
                                patient_e0: float) -> Dict:
    """Prepare all inputs needed for optimization.

    Orchestrates data extraction, initial guess loading, and injection precomputation.

    Args:
        patient_id: Patient ID.
        config: OptimizationConfig object.
        observations: Dictionary of observations.
        injections_dict: Dictionary of injection protocols.
        patient_e0: Patient-specific baseline E0 value.

    Returns:
        Dictionary containing:
            - times: Array of time points
            - BP_obs: Array of blood pressure observations
            - inor_values: Array of injection rates
            - Ad_data, Ac_data, Ap_data, E_data: Initial guess trajectories
            - patient_e0: Baseline E0 value
            - n_original_observations: Original number of observations
            - n_optimization_points: Number of points used in optimization

    Raises:
        ValueError: If patient data is missing.
        FileNotFoundError: If PKPD standalone trajectories don't exist.
    """
    # Extract observation data
    times, BP_obs, n_original_obs, n_optim_points = extract_observation_data(
        patient_id, observations, config.max_data_points
    )

    # Load initial guess from PKPD model
    Ad_data, Ac_data, Ap_data, E_data = get_initial_guess_from_pkpd(
        patient_id, times, config.data_dir
    )

    # Precompute injection rates
    inor_values = precompute_injection_rates(patient_id, times, injections_dict)

    return {
        'times': times,
        'BP_obs': BP_obs,
        'inor_values': inor_values,
        'Ad_data': Ad_data,
        'Ac_data': Ac_data,
        'Ap_data': Ap_data,
        'E_data': E_data,
        'patient_e0': patient_e0,
        'n_original_observations': n_original_obs,
        'n_optimization_points': n_optim_points
    }
