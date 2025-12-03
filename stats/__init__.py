"""Statistics analysis package for PKPD optimization results.

This package contains modules for analyzing PKPD parameter optimization results:
- pkpd_parameters: Population-level parameter statistics
- pkpd_quality: Model quality analysis (MAE vs covariables)
"""

from stats.pkpd_parameters import (
    get_patient_directories,
    load_all_parameters,
    compute_statistics,
    print_statistics,
    save_statistics,
    plot_boxplots,
    compare_e0_optimized_vs_observed
)

from stats.pkpd_quality import (
    load_patient_covariables,
    load_resimulated_bp,
    compute_patient_mae,
    analyze_model_quality,
    save_quality_analysis,
    print_quality_summary
)

__all__ = [
    # pkpd_parameters
    'get_patient_directories',
    'load_all_parameters',
    'compute_statistics',
    'print_statistics',
    'save_statistics',
    'plot_boxplots',
    'compare_e0_optimized_vs_observed',
    # pkpd_quality
    'load_patient_covariables',
    'load_resimulated_bp',
    'compute_patient_mae',
    'analyze_model_quality',
    'save_quality_analysis',
    'print_quality_summary',
]
