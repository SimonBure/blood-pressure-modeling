#!/usr/bin/env python3
"""Test script to verify all imports work correctly."""

print("Testing imports...")

try:
    import casadi as ca

    print("✓ casadi imported successfully (version {})".format(ca.__version__))
except ImportError as e:
    print("✗ Failed to import casadi:", e)
    exit(1)

try:
    from opti.config import OptimizationConfig, PhysiologicalConstants

    print("✓ opti.config imported successfully")
except ImportError as e:
    print("✗ Failed to import opti.config:", e)
    exit(1)

try:
    from opti.optim import optimize_patient_parameters

    print("✓ opti.optim.optimize_patient_parameters imported successfully")
except ImportError as e:
    print("✗ Failed to import optimize_patient_parameters:", e)
    exit(1)

try:
    from utils.datatools import load_observations, load_injections

    print("✓ utils.datatools imported successfully")
except ImportError as e:
    print("✗ Failed to import utils.datatools:", e)
    exit(1)

try:
    from utils.plots import plot_optimization_results

    print("✓ utils.plots imported successfully")
except ImportError as e:
    print("✗ Failed to import utils.plots:", e)
    exit(1)

try:
    from pkpd import NorepinephrinePKPD

    print("✓ pkpd imported successfully")
except ImportError as e:
    print("✗ Failed to import pkpd:", e)
    exit(1)

try:
    from stats.pkpd_parameters import (
        get_patient_directories,
        load_all_parameters,
        compute_statistics,
    )

    print("✓ stats.pkpd_parameters imported successfully")
except ImportError as e:
    print("✗ Failed to import stats.pkpd_parameters:", e)
    exit(1)

try:
    from stats.pkpd_quality import load_patient_covariates, analyze_model_quality

    print("✓ stats.pkpd_quality imported successfully")
except ImportError as e:
    print("✗ Failed to import stats.pkpd_quality:", e)
    exit(1)

try:
    from stats import get_patient_directories, analyze_model_quality

    print("✓ stats package-level imports successful")
except ImportError as e:
    print("✗ Failed to import from stats package:", e)
    exit(1)

print("\n" + "=" * 60)
print("SUCCESS! All imports working correctly!")
print("=" * 60)
