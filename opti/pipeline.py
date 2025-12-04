"""
PKPD Optimization Pipeline Orchestrator

This module provides a unified interface to run the optimization pipeline
in different modes:
- 'full': Complete pipeline (data prep -> optimization -> resimulation -> plotting)
- 'resim_and_plot': Load optimized params, resimulate, save, and plot
- 'plot_only': Load saved trajectories and create plots only

Usage:
    python -m opti.pipeline
"""

import numpy as np
from typing import Dict, Tuple

from opti.config import OptimizationConfig, PhysiologicalConstants
from opti.data_preparation import prepare_optimization_inputs
from opti.optimizer import optimize_patient_parameters
from opti.postprocessing import (
    resimulate_with_optimized_params,
    save_resimulated_trajectories,
    load_optimized_parameters,
    load_resimulated_trajectories,
    run_resimulation
)
from utils.datatools import (
    load_observations,
    load_injections,
    load_patient_e0_indiv,
    save_optimal_parameters,
    print_optimization_results
)
from utils.plots import (
    plot_optimization_results,
    plot_pkpd_vs_casadi_trajectories,
    plot_injection_verification
)
from opti.optim import compute_equilibrium_blood_pressure


def run_pipeline(config: OptimizationConfig, mode: str = 'full') -> None:
    """
    Run optimization pipeline with specified mode.

    Args:
        config: Optimization configuration
        mode: Pipeline mode - 'full' | 'resim_and_plot' | 'plot_only'

    Raises:
        ValueError: If mode is invalid
    """
    # Validate mode
    valid_modes = ['full', 'resim_and_plot', 'plot_only']
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of: {valid_modes}\n"
            f"  - 'full': Run complete pipeline (data prep -> optim -> resim -> plots)\n"
            f"  - 'resim_and_plot': Load params, resimulate, save, and plot\n"
            f"  - 'plot_only': Load trajectories and create plots only"
        )

    # Print configuration header
    print("\n" + "="*70)
    print("CASADI PKPD PARAMETER OPTIMIZATION - PATIENT-BY-PATIENT")
    print("="*70)
    print(f"Pipeline Mode: {mode.upper()}")
    print(f"Configuration:")
    print(f"  - Patients: {config.patient_ids}")
    if mode == 'full':
        print(f"  - Time sampling: Patient-specific (using actual observation times)")
        print(f"  - Max data points: {config.max_data_points} (subsample if exceeded)")
        print(f"  - Cost function mode: {config.cost_function_mode}")
        print(f"  - E_0 mode: {'Hard constraint to E0_indiv' if config.use_e0_constraint else 'Initial guess from E0_indiv'}")
    print(f"  - Output directory: {config.output_dir}/")
    print("="*70 + "\n")

    # Load CSV data (needed for all modes)
    print("Loading CSV data...")
    injections_dict = load_injections(config.patient_ids, config.inj_csv_path)
    observations = load_observations(config.patient_ids, config.obs_csv_path)

    # Determine actual patient list after loading
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
        return

    print(f"Loaded data for {len(patient_ids)} patient(s): {patient_ids}")

    # Load patient-specific baseline E0 values (needed for full and resim_and_plot modes)
    if mode in ['full', 'resim_and_plot']:
        print("Loading patient-specific E0_indiv values...")
        patient_e0_dict = load_patient_e0_indiv(patient_ids, config.obs_csv_path)
        print(f"  ✓ Loaded E0_indiv for {len(patient_e0_dict)} patients\n")

    # Get initial parameters for comparison (only needed in full mode)
    if mode == 'full':
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

    # Process each patient with mode-specific logic
    for patient_id in patient_ids:
        print(f"\n{'#'*70}")
        print(f"# Processing Patient {patient_id}")
        print(f"{'#'*70}\n")

        try:
            if mode == 'full':
                run_full_pipeline_patient(
                    patient_id, config, observations, injections_dict,
                    patient_e0_dict, params_initial
                )
            elif mode == 'resim_and_plot':
                run_resim_and_plot_patient(
                    patient_id, config, observations, injections_dict,
                    patient_e0_dict
                )
            elif mode == 'plot_only':
                run_plot_only_patient(
                    patient_id, config, observations, injections_dict
                )
        except FileNotFoundError as e:
            print(f"\n❌ ERROR for patient {patient_id}:")
            print(str(e))
            print("\nSkipping to next patient...\n")
            continue
        except Exception as e:
            print(f"\n❌ UNEXPECTED ERROR for patient {patient_id}: {e}")
            print("\nSkipping to next patient...\n")
            continue

    print(f"\n{'='*70}")
    print("ALL PATIENTS PROCESSED SUCCESSFULLY")
    print(f"{'='*70}\n")


def run_full_pipeline_patient(patient_id: int,
                               config: OptimizationConfig,
                               observations: Dict,
                               injections_dict: Dict,
                               patient_e0_dict: Dict[int, float],
                               params_initial: Dict[str, float]) -> None:
    """
    Execute complete optimization pipeline for one patient.

    Steps:
        1. Extract observation data
        2. Load initial guess from PKPD
        3. Precompute injection rates
        4. Run CasADi optimization
        5. Save optimal parameters
        6. Print optimization results
        7. Resimulate with optimized parameters
        7b. Save resimulated trajectories
        8. Create visualization plots
        9. Compare CasADi vs PKPD trajectories

    Args:
        patient_id: Patient ID
        config: Optimization configuration
        observations: Dictionary of patient observations
        injections_dict: Dictionary of injection protocols
        patient_e0_dict: Dictionary of patient-specific E0_indiv values
        params_initial: Initial parameters for comparison
    """
    # Step 1-3: Data preparation
    print("Steps 1-3: Preparing optimization inputs...")
    inputs = prepare_optimization_inputs(
        patient_id, config, observations, injections_dict,
        patient_e0_dict[patient_id]
    )

    times = inputs['times']
    BP_obs = inputs['BP_obs']
    inor_values = inputs['inor_values']
    Ad_data = inputs['Ad_data']
    Ac_data = inputs['Ac_data']
    Ap_data = inputs['Ap_data']
    E_data = inputs['E_data']
    patient_e0 = inputs['patient_e0']
    n_original_observations = inputs['n_original_observations']
    n_optimization_points = inputs['n_optimization_points']

    print("\nCreating injection verification plot...")
    plot_injection_verification(
        patient_id, times, inor_values, injections_dict,
        config.data_dir, config.output_dir, n_optimization_points
    )

    # Step 4: Optimization
    print("\nStep 4: Running CasADi optimization...")
    result = optimize_patient_parameters(
        times, BP_obs, inor_values,
        Ad_data, Ac_data, Ap_data, E_data,
        config, patient_e0
    )
    print(f"  ✓ Final cost: {result.cost:.4f}")
    print(f"  ✓ Solve time: {result.solve_time:.2f}s")

    # Step 5: Save parameters
    print("\nStep 5: Saving optimal parameters...")
    save_optimal_parameters(
        patient_id, result.params, result.cost,
        config.data_dir, config.output_dir,
        n_original_observations, n_optimization_points,
        config.cost_function_mode
    )

    # Step 6: Print results
    print("\nStep 6: Printing optimization results...")
    print_optimization_results(patient_id, params_initial, result.params)

    # Step 7: Resimulation
    print("\nStep 7: Re-simulating with optimized parameters...")
    resim_results = resimulate_with_optimized_params(
        patient_id, result.trajectories, result.params, injections_dict,
        observation_times=times
    )
    print("  ✓ Re-simulation completed (on observation time points)")

    # Step 7b: Save trajectories
    print("\nStep 7b: Saving resimulated trajectories...")
    save_resimulated_trajectories(
        patient_id, resim_results,
        config.data_dir, config.output_dir
    )

    # Compute equilibrium blood pressure
    E_equilibrium = compute_equilibrium_blood_pressure(
        times, inor_values, result.params
    )

    # Step 8: Main visualization
    print("\nStep 8: Creating visualization plots...")
    plot_optimization_results(
        patient_id, observations, result.trajectories,
        result.params, resim_results, E_equilibrium,
        config.data_dir, config.output_dir,
        n_optimization_points, config.cost_function_mode
    )

    # Step 9: Comparison plot
    print("\nStep 9: Comparing CasADi vs PKPD trajectories...")
    plot_pkpd_vs_casadi_trajectories(
        patient_id, result.trajectories, resim_results,
        result.params, config.data_dir, config.output_dir,
        n_optimization_points, config.cost_function_mode
    )


def run_resim_and_plot_patient(patient_id: int,
                                config: OptimizationConfig,
                                observations: Dict,
                                injections_dict: Dict,
                                patient_e0_dict: Dict[int, float]) -> None:
    """
    Load optimized parameters, resimulate, save trajectories, and create plots.

    Steps:
        1. Load optimized parameters from disk
        2. Extract observation times
        3. Resimulate with optimized parameters
        4. Save resimulated trajectories
        5. Create visualization plots
        6. Compare CasADi vs PKPD trajectories

    Args:
        patient_id: Patient ID
        config: Optimization configuration
        observations: Dictionary of patient observations
        injections_dict: Dictionary of injection protocols
        patient_e0_dict: Dictionary of patient-specific E0_indiv values

    Raises:
        FileNotFoundError: If optimized parameters not found
    """
    # Step 1: Load parameters
    print("Step 1: Loading optimized parameters from disk...")
    params_opt = load_optimized_parameters(
        patient_id, config.data_dir, config.output_dir
    )
    print(f"  ✓ Loaded params.json for patient {patient_id}")

    # Step 2: Extract observation times
    print("\nStep 2: Extracting observation times...")
    bp_obs_data = observations[patient_id]['blood_pressure']
    if not bp_obs_data:
        raise ValueError(f"No blood pressure observations for patient {patient_id}")

    times = np.array([t for t, _ in bp_obs_data])
    n_optimization_points = len(times)
    print(f"  ✓ Extracted {n_optimization_points} observation time points")
    print(f"  ✓ Time range: [{times[0]:.1f}s, {times[-1]:.1f}s]")

    # Step 3-4: Resimulate and save
    print("\nSteps 3-4: Re-simulating and saving trajectories...")
    resim_results = run_resimulation(
        patient_id, params_opt, injections_dict, times,
        config.data_dir, config.output_dir
    )

    # Compute equilibrium blood pressure
    from opti.data_preparation import precompute_injection_rates
    inor_values = precompute_injection_rates(patient_id, times, injections_dict)
    E_equilibrium = compute_equilibrium_blood_pressure(
        times, inor_values, params_opt
    )

    # Create dummy trajectories dict for plotting (from first point of resim)
    t_resim, Ad_resim, Ac_resim, Ap_resim, _, _ = resim_results
    trajectories = {
        'times': times,
        'Ad': Ad_resim,
        'Ac': Ac_resim,
        'Ap': Ap_resim
    }

    # Step 5: Main visualization
    print("\nStep 5: Creating visualization plots...")
    plot_optimization_results(
        patient_id, observations, trajectories,
        params_opt, resim_results, E_equilibrium,
        config.data_dir, config.output_dir,
        n_optimization_points, config.cost_function_mode
    )

    # Step 6: Comparison plot
    print("\nStep 6: Comparing CasADi vs PKPD trajectories...")
    plot_pkpd_vs_casadi_trajectories(
        patient_id, trajectories, resim_results,
        params_opt, config.data_dir, config.output_dir,
        n_optimization_points, config.cost_function_mode
    )


def run_plot_only_patient(patient_id: int,
                          config: OptimizationConfig,
                          observations: Dict,
                          injections_dict: Dict) -> None:
    """
    Load saved trajectories and parameters, then create plots only.

    Steps:
        1. Load optimized parameters from disk
        2. Load resimulated trajectories from disk
        3. Extract observation times
        4. Create visualization plots
        5. Compare CasADi vs PKPD trajectories

    Args:
        patient_id: Patient ID
        config: Optimization configuration
        observations: Dictionary of patient observations
        injections_dict: Dictionary of injection protocols

    Raises:
        FileNotFoundError: If parameters or trajectories not found
    """
    # Step 1: Load parameters
    print("Step 1: Loading optimized parameters from disk...")
    params_opt = load_optimized_parameters(
        patient_id, config.data_dir, config.output_dir
    )
    print(f"  ✓ Loaded params.json for patient {patient_id}")

    # Step 2: Load trajectories
    print("\nStep 2: Loading resimulated trajectories from disk...")
    resim_results = load_resimulated_trajectories(
        patient_id, config.data_dir, config.output_dir
    )
    print(f"  ✓ Loaded 6 trajectory files for patient {patient_id}")

    # Step 3: Extract observation times
    print("\nStep 3: Extracting observation times...")
    bp_obs_data = observations[patient_id]['blood_pressure']
    if not bp_obs_data:
        raise ValueError(f"No blood pressure observations for patient {patient_id}")

    times = np.array([t for t, _ in bp_obs_data])
    n_optimization_points = len(times)
    print(f"  ✓ Extracted {n_optimization_points} observation time points")

    # Compute equilibrium blood pressure
    from opti.data_preparation import precompute_injection_rates
    inor_values = precompute_injection_rates(patient_id, times, injections_dict)
    E_equilibrium = compute_equilibrium_blood_pressure(
        times, inor_values, params_opt
    )

    # Create trajectories dict from loaded data
    t_resim, Ad_resim, Ac_resim, Ap_resim, _, _ = resim_results
    trajectories = {
        'times': times,
        'Ad': Ad_resim,
        'Ac': Ac_resim,
        'Ap': Ap_resim
    }

    # Step 4: Main visualization
    print("\nStep 4: Creating visualization plots...")
    plot_optimization_results(
        patient_id, observations, trajectories,
        params_opt, resim_results, E_equilibrium,
        config.data_dir, config.output_dir,
        n_optimization_points, config.cost_function_mode
    )

    # Step 5: Comparison plot
    print("\nStep 5: Comparing CasADi vs PKPD trajectories...")
    plot_pkpd_vs_casadi_trajectories(
        patient_id, trajectories, resim_results,
        params_opt, config.data_dir, config.output_dir,
        n_optimization_points, config.cost_function_mode
    )


if __name__ == "__main__":
    """
    Example usage - run pipeline with different modes.
    """
    # Configuration
    use_e0_constraint = False  # Toggle E_0 constraint mode
    output_subdir = 'opti-e0-constraint' if use_e0_constraint else 'opti'

    config = OptimizationConfig(
        patient_ids=[],  # None = all patients, or specify list like [23]
        max_data_points=1001,
        cost_function_mode='emax',
        use_e0_constraint=use_e0_constraint,
        data_dir='results',
        output_dir=output_subdir,
        ipopt_max_iter=5000,
        ipopt_tol=1e-6,
        ipopt_print_level=0  # 0=silent, 3=minimal, 5=verbose
    )

    # Select pipeline mode
    # mode = 'full'             # Complete pipeline: data prep -> optim -> resim -> plots
    mode = 'resim_and_plot'   # Load params, resimulate, save, and plot
    # mode = 'plot_only'        # Load trajectories and create plots only

    # Run pipeline
    run_pipeline(config, mode=mode)
