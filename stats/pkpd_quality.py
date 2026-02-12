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
import json

import numpy as np
import pandas as pd

from utils.datatools import load_observations, load_resimulated_trajectories


def compute_mean_err(observations: np.ndarray, resim_bp: np.ndarray):        
    return np.mean(observations - resim_bp)
    

def compute_mean_sqr_err(observations: np.ndarray, resim_bp: np.ndarray):
    return np.mean((observations - resim_bp) ** 2)


def compute_mae(observations: np.ndarray, ref_curve: np.ndarray):
    """Compute Mean Absolute Error between observed and resimulated BP.

    Args:
        observations: Dictionary from load_observations().
        ref_curve: BP array from resimulation.

    Returns:
        MAE value (mmHg).
    """
    # Compute MAE
    return np.mean(np.abs(observations - ref_curve))


def save_quality_analysis(results_df: pd.DataFrame, res_dir: str, pkpd_dir:str, chosen_metric: str) -> None:
    """Save quality analysis results to CSV.

    Args:
        results_df: DataFrame with MAE and covariables.
        res_dir: Base results directory.
    """
    output_path = f"{res_dir}/stats/pkpd-quality/{pkpd_dir}/"
    os.makedirs(output_path, exist_ok=True)

    csv_path = os.path.join(output_path, f"{chosen_metric}.csv")
    json_path = os.path.join(output_path, f"{chosen_metric}_summary.json")
    results_df.to_csv(csv_path, index=False)
    
    bp_col = f"{chosen_metric}-BP"
    ac_col = f"{chosen_metric}-Ac"
    
    summary_stats = {
        'BP': {
            'mean': np.mean(results_df[bp_col]),
            'std': np.std(results_df[bp_col]),
            'median': np.median(results_df[bp_col]),
            'min': np.min(results_df[bp_col]),
            'max': np.max(results_df[bp_col])
        },
        'Ac': {
            'mean': np.mean(results_df[ac_col]),
            'std': np.std(results_df[ac_col]),
            'median': np.median(results_df[ac_col]),
            'min': np.min(results_df[ac_col]),
            'max': np.max(results_df[ac_col])
        }
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=4, separators=(',', ': '))
        

    print(f"\n  ✓ Quality analysis saved to {csv_path}")


def print_quality_summary(metrics_df: pd.DataFrame) -> None:
    """Print summary statistics of model quality.

    Args:
        results_df: DataFrame with MAE and covariables.
    """
    print("\n" + "=" * 80)
    print("MODEL QUALITY SUMMARY")
    print("=" * 80)

    cols = metrics_df.columns[1:]
    
    print(f"Statistics (n={len(metrics_df[cols[0]])} patients):")

    for c_name in cols:
        print(f"\n{c_name.upper()} ")
        print(f"  Mean:   {np.mean(metrics_df[c_name]):.4f} mmHg")
        print(f"  Median: {np.median(metrics_df[c_name]):.4f} mmHg")
        print(f"  Std:    {np.std(metrics_df[c_name]):.4f} mmHg")
        print(f"  Min:    {np.min(metrics_df[c_name]):.4f} mmHg")
        print(f"  Max:    {np.max(metrics_df[c_name]):.4f} mmHg")

    print("\n" + "=" * 80)


def main():
    """Main function to run PKPD quality analysis."""
    # Configuration
    results_dir = "results"
    obs_dir = "data/joachim.csv"
    
    is_constrained = False
    
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
    
    bp_errors = np.zeros(len(patient_ids))
    ac_errors = np.zeros(len(patient_ids))
    
    for index, p_id in enumerate(patient_ids):
        patient_obs = observations[p_id]
        a_c_obs = np.array([ac for _, ac in patient_obs["concentration"]])
        t_ac_obs = np.array([t for t, _ in patient_obs["concentration"]])
        bp_obs = np.array([bp for _, bp in patient_obs["blood_pressure"]])
        
        # t, _, Ac, __, modeled_bp = load_resimulated_trajectories(id, results_dir, pkpd_dir)
        resim_res = load_resimulated_trajectories(p_id, results_dir, pkpd_dir)
        t, _, a_c, _, modeled_bp = resim_res
        
        a_c = np.interp(t_ac_obs, t, a_c)  # Ac reference points at time compatible with ac_obs
        
        if chosen_metric == 'mae':
            bp_err = compute_mae(bp_obs, modeled_bp)
            ac_err = compute_mae(a_c_obs, a_c)
        elif chosen_metric == 'mean-squares':
            bp_err = compute_mean_sqr_err(bp_obs, modeled_bp)
            ac_err = compute_mean_sqr_err(a_c_obs, a_c)
        elif chosen_metric == 'mean-error':
            bp_err = compute_mean_err(bp_obs, modeled_bp)
            ac_err = compute_mean_err(a_c_obs, a_c)
        else:
            raise ValueError("Wrong metric choice.")
            
        bp_errors[index] = bp_err
        ac_errors[index] = ac_err
        
    # Create final DF
    bp_err_colname = f"{chosen_metric}-BP"
    ac_err_colname = f"{chosen_metric}-Ac"
    metrics_df = pd.DataFrame({"patient_id": patient_ids, bp_err_colname: bp_errors, ac_err_colname: ac_errors},
                              columns=["patient_id", bp_err_colname, ac_err_colname])
    print(metrics_df.head())

    print_quality_summary(metrics_df)
    
    save_quality_analysis(metrics_df, results_dir, pkpd_dir, chosen_metric)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
