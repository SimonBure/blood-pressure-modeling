# Blood Pressure Modeling (bp-modeling)

A pharmacokinetic-pharmacodynamic (PK/PD) modeling framework for analyzing blood pressure response to norepinephrine in ICU patients.

## Project Overview

This project implements a comprehensive PK/PD model for norepinephrine-induced blood pressure regulation. It combines:
- Multi-compartment pharmacokinetic modeling
- Emax dose-response relationships
- Patient-specific parameter optimization using CasADi

## File Structure

```
modeling-bp/
├── data/                           # Patient data
│   ├── injections.csv             # Norepinephrine injection protocols
│   ├── joachim.csv                # Patient blood pressure observations
│   └── joachim_meta.txt           # Metadata
│
├── pkpd.py                         # Core PK/PD model (NorepinephrinePKPD class)
│
├── opti/                           # Optimization framework
│   ├── config.py                  # Configuration dataclasses and constants
│   ├── cost_functions.py          # Objective functions for optimization
│   ├── optim.py                   # Main optimization routines
│   ├── results.py                 # Results dataclass and handling
│   ├── parameters_stats.py        # Statistical analysis of parameters
│   └── res/                       # Output directory (gitignored)
│
└── utils/                          # Utility modules
    ├── datatools.py               # Data loading and processing
    ├── injections.py              # Injection protocol utilities
    └── plots.py                   # Visualization functions
```

## Requirements

### Core Dependencies
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- SciPy
- CasADi (for optimization)

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Code Modules

### PK/PD Model (`pkpd.py`)
Implements the physiological model with the `NorepinephrinePKPD` class:
- **PK Component**: Three-compartment model (depot, central, peripheral) with endogenous production
- **PD Component**: Emax model for dose-response for hemodynamics
- Supports custom initial conditions for trajectory matching

### Optimization (`opti/`)
Patient-specific parameter estimation:
- **config.py**: Defines optimization settings, bounds, and physiological constants
- **optim.py**: CasADi-based nonlinear optimization with multiple shooting
- **cost_functions.py**: Weighted least squares and regularization terms
- **results.py**: Structured result storage and export
- **parameters_stats.py**: Cross-patient statistical analysis

### Utilities (`utils/`)
Supporting functionality:
- **datatools.py**: CSV parsing, data validation, observation loading
- **injections.py**: Injection protocol processing
- **plots.py**: Matplotlib-based visualization (trajectories, comparisons, residuals)

## Usage

### Running Optimization

```python
from opti.optim import optimize_patient
from opti.config import OptimizationConfig

# Configure optimization
config = OptimizationConfig(
    n_control_intervals=50,
    enable_e0_constraint=False
)

# Run optimization for a patient
result = optimize_patient(patient_id=1, config=config)
```

### Statistical Analysis

```python
from opti.parameters_stats import analyze_parameters

# Analyze optimized parameters across patients
analyze_parameters(results_directory="res/0_population")
```

## Research Context

This work is part of a PhD research project at MINES Paris - PSL focusing on personalized hemodynamic management in intensive care. The model aims to:
- Quantify inter-patient variability in drug response
- Enable predictive control strategies
- Support clinical decision-making for vasopressor titration

## License

Academic research project - contact for collaboration inquiries.
