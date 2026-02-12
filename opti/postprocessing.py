"""
Post-processing functions for optimization pipeline.

This module handles resimulation with optimized parameters, saving/loading
trajectories, and preparing data for visualization.
"""

import os
import json
import numpy as np
from typing import Dict, Tuple, Optional
from pkpd import NorepinephrinePKPD


def resimulate_with_optimized_params(patient_id: int,
                                     trajectories,
                                     params_opt: Dict[str, float],
                                     injections_dict: Dict,
                                     observation_times: Optional[np.ndarray] = None) -> Tuple:
    """Create new model with optimized parameters and simulate.

    Args:
        patient_id: Patient ID.
        trajectories: CasADi trajectories (contains initial conditions).
        params_opt: Dictionary of optimized parameters.
        injections_dict: Dictionary of injection protocols.
        observation_times: Optional array of specific time points to simulate at.
                          If provided, simulates on these exact time points.
                          If None, uses fixed grid (t_end=2200, dt=0.5).

    Returns:
        Tuple of (t, Ad, Ac, Ap, E_emax) from simulation.
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

    # Simulate with custom time points if provided
    if observation_times is not None:
        t, Ad, Ac, Ap, E_emax = model.simulate(
            patient_id, t_eval=observation_times
        )
    else:
        # Fallback to fixed grid for backward compatibility
        t, Ad, Ac, Ap, E_emax = model.simulate(
            patient_id, t_end=2200, dt=0.5
        )

    return t, Ad, Ac, Ap, E_emax


def save_resimulated_trajectories(patient_id: int,
                                  resim_results: Tuple,
                                  data_dir: str,
                                  output_dir: str) -> None:
    """Save resimulated PKPD trajectories to disk.

    Args:
        patient_id: Patient ID
        resim_results: Tuple of (t, Ad, Ac, Ap, E_emax)
        data_dir: Base data directory (e.g., 'results')
        output_dir: Output subdirectory (e.g., 'opti')
    """
    output_path = f'{data_dir}/patient_{patient_id}/pkpd/{output_dir}'
    os.makedirs(output_path, exist_ok=True)

    t, Ad, Ac, Ap, E_emax = resim_results

    np.save(f'{output_path}/time.npy', t)
    np.save(f'{output_path}/Ad.npy', Ad)
    np.save(f'{output_path}/Ac.npy', Ac)
    np.save(f'{output_path}/Ap.npy', Ap)
    np.save(f'{output_path}/bp_emax.npy', E_emax)

    print(f"  ✓ Resimulated trajectories saved to {output_path}/")


def load_optimized_parameters(patient_id: int,
                              data_dir: str,
                              output_dir: str) -> Dict[str, float]:
    """Load optimized parameters from disk.

    Args:
        patient_id: Patient ID.
        data_dir: Base data directory (e.g., 'results').
        output_dir: Output subdirectory (e.g., 'opti').

    Returns:
        Dictionary of optimized parameters.

    Raises:
        FileNotFoundError: If params.json doesn't exist.
    """
    params_path = f'{data_dir}/patient_{patient_id}/{output_dir}/params.json'

    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"\nERROR: Cannot run resimulation mode for patient {patient_id}\n"
            f"  Missing: {params_path}\n"
            f"  → Run optimization first with mode='full'"
        )

    with open(params_path, 'r') as f:
        params_data = json.load(f)

    return params_data


def run_resimulation(patient_id: int,
                    params_opt: Dict[str, float],
                    injections_dict: Dict,
                    observation_times: np.ndarray,
                    data_dir: str,
                    output_dir: str) -> Tuple:
    """Orchestrate resimulation and saving for a patient.

    Creates dummy trajectories dictionary with initial conditions from standalone PKPD,
    then resimulates and saves results.

    Args:
        patient_id: Patient ID.
        params_opt: Dictionary of optimized parameters.
        injections_dict: Dictionary of injection protocols.
        observation_times: Array of time points to simulate at.
        data_dir: Base data directory.
        output_dir: Output subdirectory.

    Returns:
        Tuple of (t, Ad, Ac, Ap, E_emax) from resimulation.
    """
    # Get initial conditions from standalone PKPD
    from opti.data_preparation import get_initial_guess_from_pkpd
    Ad_data, Ac_data, Ap_data, _ = get_initial_guess_from_pkpd(
        patient_id, observation_times[:1], data_dir  # Just need first time point
    )

    # Create dummy trajectories dict with initial conditions
    trajectories = {
        'Ad': Ad_data,
        'Ac': Ac_data,
        'Ap': Ap_data
    }

    # Resimulate with optimized parameters
    print("  Re-simulating with optimized parameters...")
    resim_results = resimulate_with_optimized_params(
        patient_id, trajectories, params_opt, injections_dict,
        observation_times=observation_times
    )
    print("  ✓ Re-simulation completed (on observation time points)")

    # Save trajectories
    print("  Saving resimulated trajectories...")
    save_resimulated_trajectories(patient_id, resim_results, data_dir, output_dir)

    return resim_results
