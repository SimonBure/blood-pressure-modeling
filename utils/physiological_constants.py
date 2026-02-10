"""Physiological constants and simulation settings for the PKPD model."""
from typing import Dict, Tuple, Optional
import math

class PhysiologicalConstants:
    """Default physiological parameter values and simulation settings."""

    # Initial conditions
    INITIAL_AD = 0.0
    INITIAL_AC = 0.0
    INITIAL_AP = 0.0
    INITIAL_E_WINDKESSEL = 57.09  # mmHg
    INITIAL_DEDT = 0.0

    # Population PK parameters
    C_ENDO_POP = 0.81  # nmol/L
    K_A_POP = 0.02  # 1/h
    V_C_POP = 0.49  # L
    K_12_POP = 0.06  # 1/h
    K_21_POP = 0.04  # 1/h
    K_EL_POP = 0.05  # 1/h

    # Population PD Emax parameters
    E_0_POP = 57.09  # mmHg
    E_MAX_POP = 113.52  # mmHg
    EC_50_POP = 15.7  # nmol/L

    # Population PD Windkessel parameters
    OMEGA_POP = 1.01  # rad/s
    ZETA_POP = 19.44  # dimensionless
    NU_POP = 2.12  # mmHg/(nmol/L)

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
            'C_endo': cls.compute_lognormal_bounds(cls.C_ENDO_POP, cls.SIGMA_C_ENDO),
            'k_a': cls.compute_lognormal_bounds(cls.K_A_POP, cls.SIGMA_K_A),
            'V_c': cls.compute_lognormal_bounds(cls.V_C_POP, cls.SIGMA_V_C),
            'k_12': cls.compute_lognormal_bounds(cls.K_12_POP, cls.SIGMA_K_12),
            'k_21': cls.compute_lognormal_bounds(cls.K_21_POP, cls.SIGMA_K_21),
            'k_el': cls.compute_lognormal_bounds(cls.K_EL_POP, cls.SIGMA_K_EL),
            # PD Emax parameters
            'E_0': cls.compute_lognormal_bounds(cls.E_0_POP, cls.SIGMA_E_0),
            'E_max': cls.compute_lognormal_bounds(cls.E_MAX_POP, cls.SIGMA_E_MAX),
            'EC_50': cls.compute_lognormal_bounds(cls.EC_50_POP, cls.SIGMA_EC_50),
            # PD Windkessel parameters
            'omega': cls.compute_lognormal_bounds(cls.OMEGA_POP, cls.SIGMA_OMEGA),
            'zeta': cls.compute_lognormal_bounds(cls.ZETA_POP, cls.SIGMA_ZETA),
            'nu': cls.compute_lognormal_bounds(cls.NU_POP, cls.SIGMA_NU),
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
        return self.C_ENDO_POP

    @property
    def k_a(self) -> float:
        return self.K_A_POP

    @property
    def V_c(self) -> float:
        return self.V_C_POP

    @property
    def k_12(self) -> float:
        return self.K_12_POP

    @property
    def k_21(self) -> float:
        return self.K_21_POP

    @property
    def k_el(self) -> float:
        return self.K_EL_POP

    @property
    def E_0(self) -> float:
        return self.E_0_POP

    @property
    def E_max(self) -> float:
        return self.E_MAX_POP

    @property
    def EC_50(self) -> float:
        return self.EC_50_POP

    @property
    def omega(self) -> float:
        return self.OMEGA_POP

    @property
    def zeta(self) -> float:
        return self.ZETA_POP

    @property
    def nu(self) -> float:
        return self.NU_POP
    
    def get_constants_dict(self) -> Dict[str, float]:
        """
        Return all PKPD parameters in a dictionary.

        Returns:
            Dictionary containing all PKPD parameters with their names as keys.
        """
        return {
            # PK parameters
            'C_endo': self.C_ENDO_POP,
            'k_a': self.K_A_POP,
            'V_c': self.V_C_POP,
            'k_12': self.K_12_POP,
            'k_21': self.K_21_POP,
            'k_el': self.K_EL_POP,

            # PD Emax parameters
            'E_0': self.E_0_POP,
            'E_max': self.E_MAX_POP,
            'EC_50': self.EC_50_POP,

            # PD Windkessel parameters
            'omega': self.OMEGA_POP,
            'zeta': self.ZETA_POP,
            'nu': self.NU_POP,
        }


# Pre-compute paper bounds at class definition time
PhysiologicalConstants.PAPER_PARAM_BOUNDS = PhysiologicalConstants.get_paper_param_bounds()