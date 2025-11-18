"""
Population-level statistics analysis for optimized PKPD parameters.

This script analyzes optimized parameters across all patients and generates:
- Mean and standard deviation for each parameter
- Histograms showing parameter distributions
- Summary statistics saved to JSON
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from utils import load_patient_e0_indiv


def get_patient_directories(res_dir: str) -> List[int]:
    """Get list of patient IDs from result directories.

    Args:
        res_dir: Path to results directory.

    Returns:
        Sorted list of patient IDs.
    """
    patient_ids = []
    for dirname in os.listdir(res_dir):
        if dirname.startswith('patient_') and os.path.isdir(os.path.join(res_dir, dirname)):
            try:
                pid = int(dirname.split('_')[1])
                patient_ids.append(pid)
            except (IndexError, ValueError):
                continue
    return sorted(patient_ids)


def load_all_parameters(res_dir: str, patient_ids: List[int],
                       opti_subdir: str = 'opti') -> Dict[int, Dict]:
    """Load optimized parameters from all patients.

    Args:
        res_dir: Path to results directory.
        patient_ids: List of patient IDs to load.
        opti_subdir: Optimization subdirectory name ('opti' or 'opti-e0-constraint').

    Returns:
        Dictionary mapping patient_id to params dict.

    Raises:
        FileNotFoundError: If any patient is missing params.json in the specified subdirectory.
    """
    all_params = {}
    missing_patients = []

    for pid in patient_ids:
        params_path = os.path.join(res_dir, f'patient_{pid}', opti_subdir, 'params.json')
        if not os.path.exists(params_path):
            missing_patients.append(pid)
        else:
            with open(params_path, 'r') as f:
                all_params[pid] = json.load(f)

    if missing_patients:
        raise FileNotFoundError(
            f"Missing {opti_subdir}/params.json for patient(s): {missing_patients}\n"
            f"Please run optimization for these patients first."
        )

    return all_params


def compute_statistics(all_params: Dict[int, Dict]) -> Dict[str, Dict[str, float]]:
    """Compute statistics for each parameter across all patients.

    Args:
        all_params: Dictionary mapping patient_id to params dict.

    Returns:
        Dictionary mapping parameter name to stats dict (mean, std, min, max, median).
    """
    # Define which parameters to analyze (exclude metadata)
    metadata_keys = {'patient_id', 'cost_function_mode', 'n_observation_points',
                    'n_original_observations', 'n_optimization_points', 'final_cost'}

    # Get all numeric parameter names
    first_patient_params = next(iter(all_params.values()))
    param_names = [k for k in first_patient_params.keys() if k not in metadata_keys]

    # Also include final_cost and observation counts as numeric stats
    # Handle both old (n_observation_points) and new (n_original_observations, n_optimization_points) formats
    if 'n_optimization_points' in first_patient_params:
        param_names.extend(['final_cost', 'n_original_observations', 'n_optimization_points'])
    else:
        param_names.extend(['final_cost', 'n_observation_points'])

    stats = {}
    for param_name in param_names:
        values = np.array([params[param_name] for params in all_params.values()])
        stats[param_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }

    return stats


def print_statistics(stats: Dict[str, Dict[str, float]], n_patients: int) -> None:
    """Print statistics in a formatted table.

    Args:
        stats: Dictionary of statistics per parameter.
        n_patients: Number of patients analyzed.
    """
    print("\n" + "="*80)
    print(f"POPULATION STATISTICS - {n_patients} PATIENTS")
    print("="*80)

    # Group parameters by category
    pk_params = ['C_endo', 'k_a', 'V_c', 'k_12', 'k_21', 'k_el']
    pd_emax_params = ['E_0', 'E_max', 'EC_50']
    pd_windkessel_params = ['omega', 'zeta', 'nu']
    metadata_params = ['n_original_observations', 'n_optimization_points', 'n_observation_points', 'final_cost']

    def print_param_group(title: str, params: List[str]) -> None:
        print(f"\n{title}")
        print("-"*80)
        print(f"{'Parameter':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
        print("-"*80)
        for param in params:
            if param in stats:
                s = stats[param]
                print(f"{param:<20} {s['mean']:>12.6f} {s['std']:>12.6f} {s['min']:>12.6f} {s['max']:>12.6f}")

    print_param_group("PK PARAMETERS", pk_params)
    print_param_group("PD EMAX PARAMETERS", pd_emax_params)
    print_param_group("PD WINDKESSEL PARAMETERS", pd_windkessel_params)
    print_param_group("OPTIMIZATION METADATA", metadata_params)

    print("\n" + "="*80)


def save_statistics(stats: Dict[str, Dict[str, float]],
                   output_dir: str,
                   n_patients: int) -> None:
    """Save statistics to JSON file.

    Args:
        stats: Dictionary of statistics per parameter.
        output_dir: Directory to save results.
        n_patients: Number of patients analyzed.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Add metadata to stats
    output_data = {
        'n_patients': n_patients,
        'parameters': stats
    }

    json_path = os.path.join(output_dir, 'stats.json')
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nStatistics saved to {json_path}")


def plot_histograms(all_params: Dict[int, Dict], output_dir: str) -> None:
    """Generate and save histograms for each parameter.

    Args:
        all_params: Dictionary mapping patient_id to params dict.
        output_dir: Directory to save histogram plots.
    """
    hist_dir = os.path.join(output_dir, 'hists')
    os.makedirs(hist_dir, exist_ok=True)

    # Define which parameters to plot (exclude metadata)
    metadata_keys = {'patient_id', 'cost_function_mode'}
    first_patient_params = next(iter(all_params.values()))
    param_names = [k for k in first_patient_params.keys() if k not in metadata_keys]

    # Group parameters for better visualization
    param_groups = {
        'PK Parameters': ['C_endo', 'k_a', 'V_c', 'k_12', 'k_21', 'k_el'],
        'PD Emax Parameters': ['E_0', 'E_max', 'EC_50'],
        'PD Windkessel Parameters': ['omega', 'zeta', 'nu'],
        'Optimization Results': ['n_original_observations', 'n_optimization_points', 'n_observation_points', 'final_cost']
    }

    # Plot individual histograms
    print("\nGenerating histograms...")
    for param_name in param_names:
        values = np.array([params[param_name] for params in all_params.values()])

        plt.figure(figsize=(10, 6))
        plt.hist(values, bins='auto', edgecolor='black', alpha=0.7)
        plt.xlabel(param_name, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Distribution of {param_name} across {len(all_params)} patients', fontsize=14)

        # Add statistics to plot
        mean_val = np.mean(values)
        std_val = np.std(values)
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.4f}')
        plt.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5,
                   label=f'Mean - Std: {mean_val - std_val:.4f}')
        plt.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5,
                   label=f'Mean + Std: {mean_val + std_val:.4f}')

        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        hist_path = os.path.join(hist_dir, f'{param_name}_distribution.png')
        plt.savefig(hist_path, dpi=150)
        plt.close()

    # Create combined plots for each parameter group
    for group_name, group_params in param_groups.items():
        available_params = [p for p in group_params if p in param_names]
        if not available_params:
            continue

        n_params = len(available_params)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_params == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, param_name in enumerate(available_params):
            values = np.array([params[param_name] for params in all_params.values()])

            axes[idx].hist(values, bins='auto', edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel(param_name, fontsize=10)
            axes[idx].set_ylabel('Frequency', fontsize=10)
            axes[idx].set_title(param_name, fontsize=11, fontweight='bold')

            mean_val = np.mean(values)
            axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2)
            axes[idx].grid(True, alpha=0.3)

        # Hide empty subplots
        for idx in range(n_params, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f'{group_name} Distribution (n={len(all_params)} patients)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        group_filename = group_name.lower().replace(' ', '_') + '_combined.png'
        combined_path = os.path.join(hist_dir, group_filename)
        plt.savefig(combined_path, dpi=150)
        plt.close()

    print(f"Histograms saved to {hist_dir}/")


def compare_e0_optimized_vs_observed(all_params: Dict[int, Dict],
                                      output_dir: str) -> None:
    """Compare optimized E_0 values against observed values from joachim.csv.

    Args:
        all_params: Dictionary mapping patient_id to params dict.
        output_dir: Directory to save comparison results.
    """
    # Load observed E_0 from joachim.csv using utility function
    patient_ids = list(all_params.keys())
    observed_e0 = load_patient_e0_indiv(patient_ids)

    # Build comparison data
    comparison_data = []
    for pid in sorted(all_params.keys()):
        optimized_e0 = all_params[pid]['E_0']
        obs_e0 = observed_e0.get(pid, np.nan)
        diff = optimized_e0 - obs_e0 if not np.isnan(obs_e0) else np.nan
        pct_diff = (diff / obs_e0 * 100) if obs_e0 != 0 and not np.isnan(obs_e0) else np.nan

        comparison_data.append({
            'patient_id': pid,
            'E0_optimized': optimized_e0,
            'E0_observed': obs_e0,
            'difference': diff,
            'pct_difference': pct_diff
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Print table
    print("\n" + "="*80)
    print("E_0 COMPARISON: OPTIMIZED VS OBSERVED (joachim.csv)")
    print("="*80)
    print(f"{'Patient':<10} {'Optimized':>15} {'Observed':>15} {'Diff':>12} {'% Diff':>10}")
    print("-"*80)

    for _, row in comparison_df.iterrows():
        pid = int(row['patient_id'])
        opt = row['E0_optimized']
        obs = row['E0_observed']
        diff = row['difference']
        pct = row['pct_difference']

        if np.isnan(obs):
            print(f"{pid:<10} {opt:>15.6f} {'N/A':>15} {'N/A':>12} {'N/A':>10}")
        else:
            print(f"{pid:<10} {opt:>15.6f} {obs:>15.6f} {diff:>12.6f} {pct:>9.2f}%")

    # Summary statistics
    valid_diffs = comparison_df['difference'].dropna()
    if len(valid_diffs) > 0:
        print("-"*80)
        print(f"{'Mean diff':<10} {'':<15} {'':<15} {valid_diffs.mean():>12.6f}")
        print(f"{'Std diff':<10} {'':<15} {'':<15} {valid_diffs.std():>12.6f}")
        print(f"{'MAE':<10} {'':<15} {'':<15} {valid_diffs.abs().mean():>12.6f}")

    print("="*80)

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'e0_comparison.csv')
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nE_0 comparison saved to {csv_path}")


def main():
    """Main function to run population statistics analysis."""
    # Configuration
    res_dir = 'codes/res'
    use_e0_constraint = True  # Toggle E_0 constraint mode
    
    opti_subdir = 'opti-e0-constraint' if use_e0_constraint else 'opti'
    
    output_dir = '0_population-e0-constraint' if use_e0_constraint else '0_population' 
    output_dir = os.path.join(res_dir, output_dir)

    # Choose which optimization results to analyze:
    # 'opti' = E_0 as initial guess (optimized)
    # 'opti-e0-constraint' = E_0 as hard constraint (fixed to observed value)

    print("\n" + "="*80)
    print("PKPD PARAMETER POPULATION STATISTICS ANALYSIS")
    print("="*80)
    print(f"Analyzing results from: {opti_subdir}/")
    if opti_subdir == 'opti':
        print("  (E_0 optimized as free parameter)")
    elif opti_subdir == 'opti-e0-constraint':
        print("  (E_0 constrained to observed E0_indiv)")

    # Get all patient directories
    print("\nScanning for patient directories...")
    patient_ids = get_patient_directories(res_dir)
    print(f"Found {len(patient_ids)} patient directories: {patient_ids}")

    if not patient_ids:
        print("ERROR: No patient directories found!")
        return

    # Load all parameters
    print("\nLoading optimized parameters...")
    try:
        all_params = load_all_parameters(res_dir, patient_ids, opti_subdir)
        print(f"Successfully loaded parameters for {len(all_params)} patients")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return

    # Compare E_0 optimized vs observed
    compare_e0_optimized_vs_observed(all_params, res_dir)

    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(all_params)

    # Print statistics
    print_statistics(stats, len(all_params))

    # Save statistics
    save_statistics(stats, output_dir, len(all_params))

    # Generate histograms
    plot_histograms(all_params, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved in: {output_dir}/")
    print(f"  - stats.json: Summary statistics")
    print(f"  - hists/: Parameter distribution histograms")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
