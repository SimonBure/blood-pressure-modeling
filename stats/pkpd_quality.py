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
from utils.datatools import load_observations


def load_patient_covariables(patient_ids: List[int],
                             csv_path: str = 'data/joachim.csv') -> pd.DataFrame:
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
    df_filtered = df[df['id'].isin(patient_ids)]

    # Covariables are constant per patient (except hr which is time-varying)
    # We'll take the first value for each patient
    covariable_cols = ['id', 'hr', 'pi', 'age', 'weight', 'height', 'sex',
                      'DFG', 'E0_indiv', 'HTA', 'IECARA', 'TABAC']

    # Group by patient and take first row to get per-patient covariables
    covariables_df = df_filtered.groupby('id')[covariable_cols].first().reset_index(drop=True)
    covariables_df.rename(columns={'id': 'patient_id'}, inplace=True)

    return covariables_df


def load_resimulated_bp(patient_id: int,
                        data_dir: str,
                        output_dir: str,
                        bp_type: str = 'windkessel') -> Tuple[np.ndarray, np.ndarray]:
    """Load resimulated blood pressure trajectories from pkpd/opti/ directory.

    Args:
        patient_id: Patient ID.
        data_dir: Base data directory (e.g., 'results').
        output_dir: Output subdirectory (e.g., 'opti').
        bp_type: Type of BP model ('emax' or 'windkessel').

    Returns:
        Tuple of (time array, BP array).

    Raises:
        FileNotFoundError: If trajectory files don't exist.
    """
    pkpd_path = f'{data_dir}/patient_{patient_id}/pkpd/{output_dir}'

    time_file = f'{pkpd_path}/time.npy'
    bp_file = f'{pkpd_path}/bp_{bp_type}.npy'

    if not os.path.exists(time_file) or not os.path.exists(bp_file):
        raise FileNotFoundError(
            f"Missing resimulated trajectories for patient {patient_id} in {pkpd_path}"
        )

    time = np.load(time_file)
    bp = np.load(bp_file)

    return time, bp


def compute_patient_mae(patient_id: int,
                        observations: Dict,
                        resim_time: np.ndarray,
                        resim_bp: np.ndarray) -> float:
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
    bp_obs_data = observations[patient_id]['blood_pressure']
    obs_times = np.array([t for t, _ in bp_obs_data])
    obs_bp = np.array([v for _, v in bp_obs_data])

    # Interpolate resimulated BP to observation times
    resim_bp_at_obs = np.interp(obs_times, resim_time, resim_bp)

    # Compute MAE
    mae = np.mean(np.abs(obs_bp - resim_bp_at_obs))

    return mae


def analyze_model_quality(patient_ids: List[int],
                          data_dir: str = 'results',
                          output_dir: str = 'opti',
                          bp_type: str = 'windkessel',
                          obs_csv_path: str = 'data/joachim.csv') -> pd.DataFrame:
    """Analyze PKPD model quality across all patients.

    For each patient:
    1. Load observed BP from joachim.csv
    2. Load resimulated BP from pkpd/opti/
    3. Compute MAE between observed and resimulated BP
    4. Load biological covariables
    5. Combine into results DataFrame

    Args:
        patient_ids: List of patient IDs to analyze.
        data_dir: Base data directory.
        output_dir: Output subdirectory (e.g., 'opti' or 'opti-e0-constraint').
        bp_type: BP model type ('emax' or 'windkessel').
        obs_csv_path: Path to observations CSV.

    Returns:
        DataFrame with columns: patient_id, mae, hr, pi, age, weight, height,
        sex, DFG, E0_indiv, HTA, IECARA, TABAC
    """
    print("\n" + "="*80)
    print("PKPD MODEL QUALITY ANALYSIS")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Output subdirectory: {output_dir}")
    print(f"BP type: {bp_type}")
    print(f"Number of patients: {len(patient_ids)}")
    print("="*80 + "\n")

    # Load observations
    print("Loading observations...")
    observations = load_observations(patient_ids, obs_csv_path)
    print(f"  ✓ Loaded observations for {len(observations)} patients")

    # Load covariables
    print("\nLoading patient covariables...")
    covariables_df = load_patient_covariables(patient_ids, obs_csv_path)
    print(f"  ✓ Loaded covariables for {len(covariables_df)} patients")

    # Compute MAE for each patient
    print("\nComputing MAE for each patient...")
    mae_results = []
    missing_patients = []

    for patient_id in patient_ids:
        try:
            # Load resimulated BP
            resim_time, resim_bp = load_resimulated_bp(
                patient_id, data_dir, output_dir, bp_type
            )

            # Compute MAE
            mae = compute_patient_mae(patient_id, observations, resim_time, resim_bp)

            mae_results.append({
                'patient_id': patient_id,
                'mae': mae
            })

            print(f"  Patient {patient_id}: MAE = {mae:.4f} mmHg")

        except FileNotFoundError as e:
            print(f"  ⚠ Patient {patient_id}: {e}")
            missing_patients.append(patient_id)

    if missing_patients:
        print(f"\n  ⚠ WARNING: Missing resimulated data for {len(missing_patients)} patient(s): {missing_patients}")

    # Convert to DataFrame
    mae_df = pd.DataFrame(mae_results)

    # Merge with covariables
    results_df = mae_df.merge(covariables_df, on='patient_id', how='left')

    print(f"\n  ✓ Successfully computed MAE for {len(mae_df)} patients")

    return results_df


def save_quality_analysis(results_df: pd.DataFrame,
                         data_dir: str) -> None:
    """Save quality analysis results to CSV.

    Args:
        results_df: DataFrame with MAE and covariables.
        data_dir: Base data directory.
    """
    output_path = f'{data_dir}/stats/pkpd-quality'
    os.makedirs(output_path, exist_ok=True)

    csv_path = f'{output_path}/quality_vs_covariables.csv'
    results_df.to_csv(csv_path, index=False)

    print(f"\n  ✓ Quality analysis saved to {csv_path}")


def print_quality_summary(results_df: pd.DataFrame) -> None:
    """Print summary statistics of model quality.

    Args:
        results_df: DataFrame with MAE and covariables.
    """
    print("\n" + "="*80)
    print("MODEL QUALITY SUMMARY")
    print("="*80)

    mae_values = results_df['mae'].values
    print(f"\nMAE Statistics (n={len(mae_values)} patients):")
    print(f"  Mean:   {np.mean(mae_values):.4f} mmHg")
    print(f"  Median: {np.median(mae_values):.4f} mmHg")
    print(f"  Std:    {np.std(mae_values):.4f} mmHg")
    print(f"  Min:    {np.min(mae_values):.4f} mmHg")
    print(f"  Max:    {np.max(mae_values):.4f} mmHg")

    # Identify best and worst performers
    best_idx = np.argmin(mae_values)
    worst_idx = np.argmax(mae_values)

    best_patient = results_df.iloc[best_idx]['patient_id']
    worst_patient = results_df.iloc[worst_idx]['patient_id']

    print(f"\n  Best fit:  Patient {best_patient} (MAE = {mae_values[best_idx]:.4f} mmHg)")
    print(f"  Worst fit: Patient {worst_patient} (MAE = {mae_values[worst_idx]:.4f} mmHg)")

    print("\n" + "="*80)


def main():
    """Main function to run PKPD quality analysis."""
    # Configuration
    data_dir = 'results'
    use_e0_constraint = False  # Toggle E_0 constraint mode
    bp_type = 'emax'  # or 'windkessel'

    output_dir = 'opti-e0-constraint' if use_e0_constraint else 'opti'

    # Get list of all patient directories
    print("Scanning for patient directories...")
    patient_dirs = [d for d in os.listdir(data_dir)
                   if d.startswith('patient_') and os.path.isdir(os.path.join(data_dir, d))]
    patient_ids = sorted([int(d.split('_')[1]) for d in patient_dirs])
    print(f"Found {len(patient_ids)} patient directories: {patient_ids}\n")

    if not patient_ids:
        print("ERROR: No patient directories found!")
        return

    # Run quality analysis
    results_df = analyze_model_quality(
        patient_ids,
        data_dir=data_dir,
        output_dir=output_dir,
        bp_type=bp_type
    )

    # Print summary
    print_quality_summary(results_df)

    # Save results
    save_quality_analysis(results_df, data_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
