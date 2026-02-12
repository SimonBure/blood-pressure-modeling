"""Configuration management for PKPD parameter optimization."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from utils.physiological_constants import PhysiologicalConstants

@dataclass
class OptimizationConfig:
    """Configuration for PKPD parameter optimization.

    Attributes:
        patient_ids: List of patient IDs to optimize. None = all patients from observations.
        max_data_points: Maximum number of optimization time points. If patient has more
            observations, uniform subsampling is applied.
        use_e0_constraint: If True, E_0 is constrained to patient's E0_indiv (hard constraint).
            If False, E0_indiv is used only as initial guess. Results saved to different directories.
        use_paper_bounds: If True, applies biologically realistic bounds from paper (mu +/- 3*sigma).
            Results are saved to 'opti-constrained' instead of 'opti'.
        data_dir: Base directory for patient data.
        output_dir: Output subdirectory name within patient directories.
        obs_csv_path: Path to observations CSV file.
        inj_csv_path: Path to injections CSV file.
        ipopt_max_iter: Maximum IPOPT iterations.
        ipopt_tol: IPOPT convergence tolerance.
        ipopt_acceptable_tol: IPOPT acceptable convergence tolerance.
        ipopt_acceptable_iter: Number of acceptable iterations before termination.
        ipopt_print_level: IPOPT verbosity (0-12).
        param_bounds: Parameter bounds as dict of (lower, upper) tuples.
    """

    # Patient selection
    patient_ids: Optional[List[int]] = None

    # Subsampling configuration
    max_data_points: int = 400  # Maximum optimization points (subsample if exceeded)

    # Model configuration
    use_e0_constraint: bool = False  # If True, E_0 is constrained to E0_indiv; if False, used as initial guess only
    use_paper_bounds: bool = False  # If True, uses biologically realistic bounds from paper (mu +/- 3*sigma)

    # Paths
    data_dir: str = 'results'
    output_dir: str = 'opti'
    obs_csv_path: str = 'data/joachim.csv'
    inj_csv_path: str = 'data/injections.csv'

    # Optimization settings
    ipopt_max_iter: int = 5000
    ipopt_tol: float = 1e-6
    ipopt_acceptable_tol: float = 1e-4
    ipopt_acceptable_iter: int = 15
    ipopt_print_level: int = 5

    # Parameter bounds (lower, upper) - None means unbounded
    param_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default parameter bounds if not provided."""
        # Auto-set output directory based on paper bounds flag
        if self.output_dir == 'opti' and self.use_paper_bounds:
            self.output_dir = 'opti-constrained'

        if not self.param_bounds:
            if self.use_paper_bounds:
                # Use biologically realistic bounds from paper (mu +/- 3*sigma)
                self.param_bounds = PhysiologicalConstants.PAPER_PARAM_BOUNDS.copy()
            else:
                # Default bounds (non-negativity only)
                self.param_bounds = {
                    # PK parameters - all non-negative
                    'C_endo': (0, None),
                    'k_a': (0, None),
                    'V_c': (1e-6, None),  # Strictly positive (avoid division by zero)
                    'k_12': (0, None),
                    'k_21': (0, None),
                    'k_el': (0, None),
                    # PD Emax parameters - all positive
                    'E_0': (1e-6, None),
                    'E_max': (1e-6, None),  # E_max > E_0 enforced separately
                    'EC_50': (1e-6, None)
                }

    def get_ipopt_options(self) -> dict:
        """Get IPOPT solver options dictionary."""
        return {
            'ipopt.max_iter': self.ipopt_max_iter,
            'ipopt.tol': self.ipopt_tol,
            'ipopt.acceptable_tol': self.ipopt_acceptable_tol,
            'ipopt.acceptable_iter': self.ipopt_acceptable_iter,
            'ipopt.print_level': self.ipopt_print_level,
            'print_time': False
        }
