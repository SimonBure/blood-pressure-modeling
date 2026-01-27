"""
PKPD Model Quality Analysis

Analyzes the quality of PKPD model predictions by computing Mean Absolute Error (MAE)
between observed and resimulated blood pressure values for each patient, then
correlates these errors with biological covariables.

Covariables analyzed:
- Continuous: hr, pi, age, weight, height, DFG, E0_indiv
- Binary: sex, HTA, IECARA, TABAC
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from utils.datatools import load_observations, load_patient_covariates, load_resimulated_trajectories


def compute_patient_mean_err(patient_id: int, observations: Dict, resim_bp: np.ndarray
) -> float:
    bp_obs_data = observations[patient_id]["blood_pressure"]
    obs_bp = np.array([v for _, v in bp_obs_data])
        
    return np.mean(obs_bp - resim_bp)
    

def compute_patient_mean_sqr(patient_id: int, observations: Dict, resim_bp: np.ndarray
) -> float:
    bp_obs_data = observations[patient_id]["blood_pressure"]
    obs_bp = np.array([v for _, v in bp_obs_data])
    
    return np.mean((obs_bp - resim_bp) ** 2)


def compute_patient_mae(
    patient_id: int, observations: Dict, resim_bp: np.ndarray
) -> float:
    """Compute Mean Absolute Error between observed and resimulated BP.

    Args:
        patient_id: Patient ID.
        observations: Dictionary from load_observations().
        resim_time: Time array from resimulation.
        resim_bp: BP array from resimulation.

    Returns:
        MAE value (mmHg).
    """
    # Extract observed BP data
    bp_obs_data = observations[patient_id]["blood_pressure"]
    obs_bp = np.array([v for _, v in bp_obs_data])

    # Compute MAE
    return np.mean(np.abs(obs_bp - resim_bp))


def save_quality_analysis(results_df: pd.DataFrame, res_dir: str, pkpd_dir:str, chosen_metric: str) -> None:
    """Save quality analysis results to CSV.

    Args:
        results_df: DataFrame with MAE and covariables.
        res_dir: Base results directory.
    """
    output_path = f"{res_dir}/stats/pkpd-quality/{pkpd_dir}/"
    os.makedirs(output_path, exist_ok=True)

    csv_path = f"{output_path}/{chosen_metric}_vs_covariates.csv"
    results_df.to_csv(csv_path, index=False)

    print(f"\n  ✓ Quality analysis saved to {csv_path}")


def print_quality_summary(results_df: pd.DataFrame, chosen_metric: str) -> None:
    """Print summary statistics of model quality.

    Args:
        results_df: DataFrame with MAE and covariables.
    """
    print("\n" + "=" * 80)
    print("MODEL QUALITY SUMMARY")
    print("=" * 80)

    metric_values = results_df[chosen_metric].values
    print(f"\n{chosen_metric.upper()} Statistics (n={len(metric_values)} patients):")
    print(f"  Mean:   {np.mean(metric_values):.4f} mmHg")
    print(f"  Median: {np.median(metric_values):.4f} mmHg")
    print(f"  Std:    {np.std(metric_values):.4f} mmHg")
    print(f"  Min:    {np.min(metric_values):.4f} mmHg")
    print(f"  Max:    {np.max(metric_values):.4f} mmHg")

    # Identify best and worst performers
    best_idx = np.argmin(metric_values)
    worst_idx = np.argmax(metric_values)

    best_patient = results_df.iloc[best_idx]["patient_id"]
    worst_patient = results_df.iloc[worst_idx]["patient_id"]

    print(
        f"\n  Best fit:  Patient {best_patient} ({chosen_metric.upper()} = {metric_values[best_idx]:.4f} mmHg)"
    )
    print(
        f"  Worst fit: Patient {worst_patient} ({chosen_metric.upper()} = {metric_values[worst_idx]:.4f} mmHg)"
    )

    print("\n" + "=" * 80)


def main():
    """Main function to run PKPD quality analysis."""
    # Configuration
    results_dir = "results"
    bp_type = "emax"  # or 'windkessel'
    obs_dir = "data/joachim.csv"
    
    is_constrained = True
    
    pkpd_dir = "opti-constrained" if is_constrained else "opti"

    chosen_metric = 'mae'  #  'mae' 'mean-error' 'mean-squares'
    
    # Get list of all patient directories
    print("Scanning for patient directories...")
    patient_dirs = [
        d
        for d in os.listdir(results_dir)
        if d.startswith("patient_") and os.path.isdir(os.path.join(results_dir, d))
    ]
    patient_ids = sorted([int(d.split("_")[1]) for d in patient_dirs])
    # print(f"Found {len(patient_ids)} patient directories: {patient_ids}\n")

    if not patient_ids:
        print("ERROR: No patient directories found!")
        return
    
    observations = load_observations(patient_ids, obs_dir)
    covariates = load_patient_covariates(patient_ids, obs_dir)
    
    metrics = []
    
    for id in patient_ids:        
        t, Ad, Ac, Ap, E_max, E_windkessel = load_resimulated_trajectories(id, results_dir, pkpd_dir)
        
        modeled_bp = E_max if bp_type == "emax" else E_windkessel

        if chosen_metric == 'mae':
            metrics.append(compute_patient_mae(id, observations, modeled_bp))
        elif chosen_metric == 'mean-squares':
            metrics.append(compute_patient_mean_sqr(id, observations, modeled_bp))
        elif chosen_metric == 'mean-error':
            metrics.append(compute_patient_mean_err(id, observations, modeled_bp))
        else:
            pass
        
    # Create final DF
    metrics_df = pd.DataFrame({"patient_id": patient_ids, chosen_metric: metrics}, columns=["patient_id", chosen_metric])
    
    results_df = metrics_df.merge(covariates, on="patient_id", how="left")

    # Print summary
    print_quality_summary(results_df, chosen_metric)

    # Save results
    save_quality_analysis(results_df, results_dir, pkpd_dir, chosen_metric)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
