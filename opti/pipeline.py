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

from typing import Dict, List
import numpy as np

from opti.config import OptimizationConfig, PhysiologicalConstants
from opti.data_preparation import (
    prepare_optimization_inputs,
    compute_equilibrium_blood_pressure,
    precompute_injection_rates
    )
from opti.optimizer import optimize_patient_parameters
from opti.postprocessing import (
    resimulate_with_optimized_params,
    save_resimulated_trajectories,
    load_optimized_parameters,
    run_resimulation
)
from utils.datatools import (
    load_all_patient_ids,
    load_observations,
    load_injections,
    load_patient_e0_indiv,
    load_resimulated_trajectories,
    save_optimal_parameters,
    print_optimization_results
)
from utils.plots import (
    plot_optimization_results,
    plot_pkpd_vs_casadi_trajectories,
    plot_injection_verification
)


def get_actual_patient_ids(config_patients_ids):
    if config_patients_ids == "all":
        return load_all_patient_ids()
    elif isinstance(config_patients_ids, int):
        return [config_patients_ids]
    elif isinstance(config_patients_ids, List):
        return config_patients_ids
    else:
        raise ValueError(f"Invalid patient_ids: {config_patients_ids}")


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
    print("Configuration:")
    print(f"  - Patients: {config.patient_ids}")
    if mode == 'full':
        print("  - Time sampling: Patient-specific (using actual observation times)")
        print(f"  - Max data points: {config.max_data_points} (subsample if exceeded)")
        print(f"  - E_0 mode: {'Hard constraint to E0_indiv' if config.use_e0_constraint else 'Initial guess from E0_indiv'}")
        print(f"  - Parameter bounds: {'Paper-based (mu +/- 3*sigma)' if config.use_paper_bounds else 'Default (non-negativity only)'}")
    print(f"  - Output directory: {config.output_dir}/")
    print("="*70 + "\n")
    
    if config.use_paper_bounds:
        print("Parameter bounds:")
        print(config.param_bounds)

    # Load CSV data (needed for all modes)
    print("Loading CSV data...")
    patients_ids = get_actual_patient_ids(config.patient_ids)
    
    injections_dict = load_injections(patients_ids, config.inj_csv_path)
    observations = load_observations(patients_ids, config.obs_csv_path)

    print(f"Patient IDs after loading: {patients_ids}")

    if not patients_ids:
        print("ERROR: No patients found with both observations and injection data!")
        return

    print(f"Loaded data for {len(patients_ids)} patient(s): {patients_ids}")

    # Load patient-specific baseline E0 values (needed for full and resim_and_plot modes)
    if mode in ['full', 'resim_and_plot']:
        print("Loading patient-specific E0_indiv values...")
        patient_e0_dict = load_patient_e0_indiv(patients_ids, config.obs_csv_path)
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
            'EC_50': physio.EC_50
        }

    # Process each patient with mode-specific logic
    for p_id in patients_ids:
        print(f"\n{'#'*70}")
        print(f"# Processing Patient {p_id}")
        print(f"{'#'*70}\n")

        try:
            if mode == 'full':
                run_full_pipeline_patient(
                    p_id, config, observations, injections_dict,
                    patient_e0_dict, params_initial
                )
            elif mode == 'resim_and_plot':
                run_resim_and_plot_patient(
                    p_id, config, observations, injections_dict,
                )
            elif mode == 'plot_only':
                run_plot_only_patient(
                    p_id, config, observations, injections_dict
                )
        except FileNotFoundError as e:
            print(f"\n❌ ERROR for patient {p_id}:")
            print(str(e))
            print("\nSkipping to next patient...\n")
            continue
        # except Exception as e:
        #     print(f"\n❌ UNEXPECTED ERROR for patient {patient_id}: {e}")
        #     print("\nSkipping to next patient...\n")
        #     continue

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
        Ad_data, Ac_data, Ap_data,
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
        inor_values, result.params
    )

    # Step 8: Main visualization
    print("\nStep 8: Creating visualization plots...")
    plot_optimization_results(
        patient_id, observations, result.trajectories,
        result.params, resim_results, E_equilibrium,
        config.data_dir, config.output_dir,
        n_optimization_points
    )

    # Step 9: Comparison plot
    print("\nStep 9: Comparing CasADi vs PKPD trajectories...")
    plot_pkpd_vs_casadi_trajectories(
        patient_id, result.trajectories, resim_results,
        result.params, config.data_dir, config.output_dir,
        n_optimization_points,
    )


def run_resim_and_plot_patient(patient_id: int,
                                config: OptimizationConfig,
                                observations: Dict,
                                injections_dict: Dict) -> None:
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
    inor_values = precompute_injection_rates(patient_id, times, injections_dict)
    E_equilibrium = compute_equilibrium_blood_pressure(
        inor_values, params_opt
    )

    # Create dummy trajectories dict for plotting (from first point of resim)
    _, a_d_resim, a_c_resim, a_p_resim, _ = resim_results
    trajectories = {
        'times': times,
        'Ad': a_d_resim,
        'Ac': a_c_resim,
        'Ap': a_p_resim
    }

    # Step 5: Main visualization
    print("\nStep 5: Creating visualization plots...")
    plot_optimization_results(
        patient_id, observations, trajectories,
        params_opt, resim_results, E_equilibrium,
        config.data_dir, config.output_dir,
        n_optimization_points,
    )

    # Step 6: Comparison plot
    print("\nStep 6: Comparing CasADi vs PKPD trajectories...")
    plot_pkpd_vs_casadi_trajectories(
        patient_id, trajectories, resim_results,
        params_opt, config.data_dir, config.output_dir,
        n_optimization_points
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
    inor_values = precompute_injection_rates(patient_id, times, injections_dict)
    E_equilibrium = compute_equilibrium_blood_pressure(
        inor_values, params_opt
    )

    # Create trajectories dict from loaded data
    _, a_d_resim, a_c_resim, a_p_resim, _ = resim_results
    trajectories = {
        'times': times,
        'Ad': a_d_resim,
        'Ac': a_c_resim,
        'Ap': a_p_resim
    }

    # Step 4: Main visualization
    print("\nStep 4: Creating visualization plots...")
    plot_optimization_results(
        patient_id, observations, trajectories,
        params_opt, resim_results, E_equilibrium,
        config.data_dir, config.output_dir,
        n_optimization_points,
    )

    # Step 5: Comparison plot
    print("\nStep 5: Comparing CasADi vs PKPD trajectories...")
    plot_pkpd_vs_casadi_trajectories(
        patient_id, trajectories, resim_results,
        params_opt, config.data_dir, config.output_dir,
        n_optimization_points,
    )


if __name__ == "__main__":
    USE_E0_CONSTRAINT = False
    USE_PAPER_BOUNDS = True    # Toggle biologically realistic bounds from paper (mu +/- 3*sigma)

    some_config = OptimizationConfig(
        patient_ids=[5, 6],  # int for 1 patient, list for multiple patients, 'all' for every patient
        max_data_points=5000,
        use_e0_constraint=USE_E0_CONSTRAINT,
        use_paper_bounds=USE_PAPER_BOUNDS,
        ipopt_max_iter=5000,
        ipopt_tol=1e-6,
        ipopt_print_level=0  # 0=silent, 3=minimal, 5=verbose
    )

    # Select pipeline mode
    PIPELINE_MODE = 'full'  # Complete pipeline: data prep -> optim -> resim -> plots
    # PIPELINE_MODE = 'resim_and_plot'   # Load params, resimulate, save, and plot
    # PIPELINE_MODE = 'plot_only'        # Load trajectories and create plots only

    run_pipeline(config=some_config, mode=PIPELINE_MODE)
