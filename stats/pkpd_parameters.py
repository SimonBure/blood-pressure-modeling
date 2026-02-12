"""
Population-level statistics analysis for optimized PKPD parameters.

This script analyzes optimized parameters across all patients and generates:
- Mean, standard deviation, min, max, and median for each parameter
- Boxplot visualizations for parameter distributions
- Summary statistics saved to JSON
- Comparison between optimized E_0 and observed E0_indiv values
"""

import os
import json
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


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
            with open(params_path, 'r', encoding='utf-8') as f:
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
    metadata_keys = {'patient_id', 'n_observation_points',
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
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nStatistics saved to {json_path}")


def plot_boxplots(all_params: Dict[int, Dict], output_dir: str) -> None:
    """Generate and save boxplots for each parameter.

    Args:
        all_params: Dictionary mapping patient_id to params dict.
        output_dir: Directory to save boxplot figures.
    """
    boxplot_dir = os.path.join(output_dir, 'boxplots')
    os.makedirs(boxplot_dir, exist_ok=True)

    # Define which parameters to plot (exclude metadata)
    metadata_keys = {'patient_id'}
    first_patient_params = next(iter(all_params.values()))
    param_names = [k for k in first_patient_params.keys() if k not in metadata_keys]

    print("\nGenerating boxplots...")
    n_plots = 0

    # Generate individual boxplots for each parameter
    for param_name in param_names:
        values = np.array([params[param_name] for params in all_params.values()])

        # Create figure
        _, ax = plt.subplots(figsize=(8, 6))

        # Create boxplot
        ax.boxplot([values], patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2),
                   whiskerprops=dict(linewidth=1.5),
                   capprops=dict(linewidth=1.5))

        # Add statistics text
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)

        stats_text = (f"n = {len(values)} patients\n"
                     f"Mean: {mean_val:.6f}\n"
                     f"Median: {median_val:.6f}\n"
                     f"Std: {std_val:.6f}\n"
                     f"Min: {np.min(values):.6f}\n"
                     f"Max: {np.max(values):.6f}")

        ax.text(0.98, 0.97, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9,
               family='monospace')

        # Formatting
        ax.set_ylabel(param_name, fontsize=12, fontweight='bold')
        ax.set_title(f'Distribution of {param_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save figure
        safe_param_name = param_name.replace('_', '-').lower()
        boxplot_path = os.path.join(boxplot_dir, f'{safe_param_name}.png')
        plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
        plt.close()

        n_plots += 1

    print(f"  ✓ Generated {n_plots} boxplots in {boxplot_dir}/")


def main():
    """Main function to run population statistics analysis."""
    # Configuration
    res_dir = 'results'
    is_constrained = True  # Toggle E_0 constraint mode

    opti_subdir = 'opti-constrained' if is_constrained else 'opti'

    # Output to results/stats/pkpd-parameters/ (simplified path)
    output_dir = os.path.join(res_dir, 'stats', 'pkpd-parameters', opti_subdir)

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


    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(all_params)

    # Print statistics
    print_statistics(stats, len(all_params))

    # Save statistics
    save_statistics(stats, output_dir, len(all_params))

    # Generate boxplots
    plot_boxplots(all_params, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved in: {output_dir}/")
    print("  - stats.json: Summary statistics")
    print("  - e0_comparison.csv: E_0 comparison table")
    print("  - boxplots/: Parameter distribution boxplots")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
