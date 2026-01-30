"""Configuration management for PKPD parameter optimization."""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


class PhysiologicalConstants:
    """Default physiological parameter values and simulation settings."""

    # Initial conditions
    INITIAL_AD = 0.0
    INITIAL_AC = 0.0
    INITIAL_AP = 0.0
    INITIAL_E_WINDKESSEL = 57.09  # mmHg
    INITIAL_DEDT = 0.0

    # Default PK parameters
    C_ENDO_DEFAULT = 0.81  # nmol/L
    K_A_DEFAULT = 0.02  # 1/h
    V_C_DEFAULT = 0.49  # L
    K_12_DEFAULT = 0.06  # 1/h
    K_21_DEFAULT = 0.04  # 1/h
    K_EL_DEFAULT = 0.05  # 1/h

    # Default PD Emax parameters
    E_0_DEFAULT = 57.09  # mmHg
    E_MAX_DEFAULT = 113.52  # mmHg
    EC_50_DEFAULT = 15.7  # nmol/L

    # Default PD Windkessel parameters
    OMEGA_DEFAULT = 1.01  # rad/s
    ZETA_DEFAULT = 19.44  # dimensionless
    NU_DEFAULT = 2.12  # mmHg/(nmol/L)

    # Standard deviations from paper Table 2 (log-space, for log-normal distribution)
    # These are the ω (omega) parameters of the inter-individual variability
    SIGMA_C_ENDO = 0.51
    SIGMA_K_A = 0.65
    SIGMA_V_C = 0.36
    SIGMA_K_12 = 0.28
    SIGMA_K_21 = 0.58
    SIGMA_K_EL = 0.31
    SIGMA_E_0 = 0.22
    SIGMA_E_MAX = 0.51
    SIGMA_EC_50 = 0.59
    SIGMA_OMEGA = 0.2
    SIGMA_ZETA = 0.66
    SIGMA_NU = 0.4

    # Simulation defaults
    T_END_DEFAULT = 2200  # seconds
    DT_DEFAULT = 0.5  # seconds

    # Plot settings
    FIGSIZE_LARGE = (14, 6)
    FIGSIZE_MEDIUM = (12, 6)
    FIGSIZE_COMPARISON = (16, 10)
    DPI = 150

    # Paper-based parameter bounds (computed from Table 2 values)
    # Will be populated by get_paper_param_bounds() after class definition
    PAPER_PARAM_BOUNDS: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    @staticmethod
    def compute_lognormal_bounds(theta_pop: float, omega: float, n_sigma: float = 3.0) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute bounds for a log-normal distributed parameter.

        For a log-normal distribution, individual parameters are modeled as:
            θ_i = θ_pop × exp(η_i)  where η_i ~ N(0, ω²)

        The bounds at ±n_sigma standard deviations in log-space are:
            θ_lower = θ_pop × exp(-n_sigma × ω)
            θ_upper = θ_pop × exp(+n_sigma × ω)

        Args:
            theta_pop: Population parameter value (Value column from paper Table 2)
            omega: Inter-individual variability in log-space (Standard deviation from Table 2)
            n_sigma: Number of standard deviations for bounds (default: 3)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        lower = theta_pop * math.exp(-n_sigma * omega)
        upper = theta_pop * math.exp(+n_sigma * omega)

        return (lower, upper)

    @classmethod
    def get_paper_param_bounds(cls) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """
        Compute biologically realistic parameter bounds from paper values.

        Uses the log-normal formula: θ_lower = θ_pop × exp(-3ω), θ_upper = θ_pop × exp(+3ω)

        Returns:
            Dictionary mapping parameter names to (lower, upper) bound tuples
        """
        return {
            # PK parameters
            'C_endo': cls.compute_lognormal_bounds(cls.C_ENDO_DEFAULT, cls.SIGMA_C_ENDO),
            'k_a': cls.compute_lognormal_bounds(cls.K_A_DEFAULT, cls.SIGMA_K_A),
            'V_c': cls.compute_lognormal_bounds(cls.V_C_DEFAULT, cls.SIGMA_V_C),
            'k_12': cls.compute_lognormal_bounds(cls.K_12_DEFAULT, cls.SIGMA_K_12),
            'k_21': cls.compute_lognormal_bounds(cls.K_21_DEFAULT, cls.SIGMA_K_21),
            'k_el': cls.compute_lognormal_bounds(cls.K_EL_DEFAULT, cls.SIGMA_K_EL),
            # PD Emax parameters
            'E_0': cls.compute_lognormal_bounds(cls.E_0_DEFAULT, cls.SIGMA_E_0),
            'E_max': cls.compute_lognormal_bounds(cls.E_MAX_DEFAULT, cls.SIGMA_E_MAX),
            'EC_50': cls.compute_lognormal_bounds(cls.EC_50_DEFAULT, cls.SIGMA_EC_50),
            # PD Windkessel parameters
            'omega': cls.compute_lognormal_bounds(cls.OMEGA_DEFAULT, cls.SIGMA_OMEGA),
            'zeta': cls.compute_lognormal_bounds(cls.ZETA_DEFAULT, cls.SIGMA_ZETA),
            'nu': cls.compute_lognormal_bounds(cls.NU_DEFAULT, cls.SIGMA_NU),
        }

    # Convenience properties for cleaner access
    @property
    def Ad_0(self) -> float:
        return self.INITIAL_AD

    @property
    def Ac_0(self) -> float:
        return self.INITIAL_AC

    @property
    def Ap_0(self) -> float:
        return self.INITIAL_AP

    @property
    def E_0_init(self) -> float:
        return self.INITIAL_E_WINDKESSEL

    @property
    def dEdt_0(self) -> float:
        return self.INITIAL_DEDT

    @property
    def C_endo(self) -> float:
        return self.C_ENDO_DEFAULT

    @property
    def k_a(self) -> float:
        return self.K_A_DEFAULT

    @property
    def V_c(self) -> float:
        return self.V_C_DEFAULT

    @property
    def k_12(self) -> float:
        return self.K_12_DEFAULT

    @property
    def k_21(self) -> float:
        return self.K_21_DEFAULT

    @property
    def k_el(self) -> float:
        return self.K_EL_DEFAULT

    @property
    def E_0(self) -> float:
        return self.E_0_DEFAULT

    @property
    def E_max(self) -> float:
        return self.E_MAX_DEFAULT

    @property
    def EC_50(self) -> float:
        return self.EC_50_DEFAULT

    @property
    def omega(self) -> float:
        return self.OMEGA_DEFAULT

    @property
    def zeta(self) -> float:
        return self.ZETA_DEFAULT

    @property
    def nu(self) -> float:
        return self.NU_DEFAULT


# Pre-compute paper bounds at class definition time
PhysiologicalConstants.PAPER_PARAM_BOUNDS = PhysiologicalConstants.get_paper_param_bounds()


@dataclass
class OptimizationConfig:
    """Configuration for PKPD parameter optimization.

    Attributes:
        patient_ids: List of patient IDs to optimize. None = all patients from observations.
        max_data_points: Maximum number of optimization time points. If patient has more
            observations, uniform subsampling is applied.
        cost_function_mode: Cost function type - 'emax', 'windkessel', or 'both'.
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
    cost_function_mode: str = 'emax'
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
                    'EC_50': (1e-6, None),
                    # PD Windkessel parameters - all positive
                    'omega': (1e-6, None),
                    'zeta': (1e-6, None),
                    'nu': (1e-6, None),
                }

        # Validate cost function mode
        valid_modes = ['emax', 'windkessel', 'both']
        if self.cost_function_mode not in valid_modes:
            raise ValueError(
                f"cost_function_mode must be one of {valid_modes}, "
                f"got '{self.cost_function_mode}'"
            )

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
