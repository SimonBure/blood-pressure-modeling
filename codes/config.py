"""Configuration management for PKPD parameter optimization."""

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

    # Simulation defaults
    T_END_DEFAULT = 2200  # seconds
    DT_DEFAULT = 0.5  # seconds

    # Plot settings
    FIGSIZE_LARGE = (14, 6)
    FIGSIZE_MEDIUM = (12, 6)
    FIGSIZE_COMPARISON = (16, 10)
    DPI = 150

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


@dataclass
class OptimizationConfig:
    """Configuration for PKPD parameter optimization.

    Attributes:
        patient_ids: List of patient IDs to optimize. None = all patients from observations.
        max_data_points: Maximum number of optimization time points. If patient has more
            observations, uniform subsampling is applied.
        cost_function_mode: Cost function type - 'emax', 'windkessel', or 'both'.
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

    # Paths
    data_dir: str = 'codes/res'
    output_dir: str = 'opti'
    obs_csv_path: str = 'codes/data/joachim.csv'
    inj_csv_path: str = 'codes/data/injections.csv'

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
        if not self.param_bounds:
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
