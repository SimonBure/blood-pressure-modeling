"""Cost function abstractions for PKPD parameter optimization."""

import casadi as ca
from typing import Dict


class CostFunction:
    """Base class for cost functions."""

    def compute(self, opti_vars: Dict, observations: Dict) -> ca.MX:
        """Compute cost function value.

        Args:
            opti_vars: Dictionary of optimization variables (parameters and states).
            observations: Dictionary of observation data.

        Returns:
            CasADi expression for cost value.
        """
        raise NotImplementedError("Subclasses must implement compute()")


class EmaxBPCost(CostFunction):
    """Blood pressure cost function using Emax pharmacodynamic model.

    Computes squared error between Emax-predicted BP and observed BP.
    """

    def compute(self, opti_vars: Dict, observations: Dict) -> ca.MX:
        """Compute Emax BP cost.

        Args:
            opti_vars: Must contain 'C_endo', 'Ac', 'V_c', 'E_0', 'E_max', 'EC_50'.
            observations: Must contain 'BP_obs' (observed blood pressure array).

        Returns:
            Sum of squared errors: sum((E_emax - BP_obs)^2)
        """
        # Compute concentration
        Cc = opti_vars['C_endo'] + opti_vars['Ac'] / opti_vars['V_c']

        # Compute Emax effect
        E_emax = (opti_vars['E_0'] +
                  (opti_vars['E_max'] - opti_vars['E_0']) * Cc /
                  (Cc + opti_vars['EC_50']))

        # Squared error cost
        return ca.sumsqr(E_emax - observations['BP_obs'])


class WindkesselBPCost(CostFunction):
    """Blood pressure cost function using Windkessel dynamic model.

    Computes squared error between Windkessel-predicted BP and observed BP.
    """

    def compute(self, opti_vars: Dict, observations: Dict) -> ca.MX:
        """Compute Windkessel BP cost.

        Args:
            opti_vars: Must contain 'E' (Windkessel state variable).
            observations: Must contain 'BP_obs' (observed blood pressure array).

        Returns:
            Sum of squared errors: sum((E - BP_obs)^2)
        """
        return ca.sumsqr(opti_vars['E'] - observations['BP_obs'])


class CombinedBPCost(CostFunction):
    """Combined cost function using both Emax and Windkessel models.

    Computes sum of Emax and Windkessel costs.
    """

    def __init__(self):
        self.emax_cost = EmaxBPCost()
        self.windkessel_cost = WindkesselBPCost()

    def compute(self, opti_vars: Dict, observations: Dict) -> ca.MX:
        """Compute combined cost.

        Args:
            opti_vars: Must contain variables for both Emax and Windkessel.
            observations: Must contain 'BP_obs'.

        Returns:
            Sum of Emax and Windkessel costs.
        """
        cost_emax = self.emax_cost.compute(opti_vars, observations)
        cost_windkessel = self.windkessel_cost.compute(opti_vars, observations)
        return cost_emax + cost_windkessel


def create_cost_function(mode: str) -> CostFunction:
    """Factory function to create cost function based on mode.

    Args:
        mode: Cost function mode - 'emax', 'windkessel', or 'both'.

    Returns:
        CostFunction instance.

    Raises:
        ValueError: If mode is not recognized.
    """
    if mode == 'emax':
        return EmaxBPCost()
    elif mode == 'windkessel':
        return WindkesselBPCost()
    elif mode == 'both':
        return CombinedBPCost()
    else:
        raise ValueError(
            f"Unknown cost function mode: '{mode}'. "
            f"Must be 'emax', 'windkessel', or 'both'."
        )
