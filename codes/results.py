"""Result dataclass for optimization outputs."""

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class OptimizationResult:
    """Container for PKPD parameter optimization results.

    Attributes:
        params: Dictionary of optimized parameter values (name -> value).
        trajectories: Dictionary of optimized state trajectories (name -> array).
        cost: Final cost function value.
        converged: Whether optimization converged successfully.
        iterations: Number of solver iterations performed.
        solve_time: Time spent solving optimization (seconds).
    """

    params: Dict[str, float]
    trajectories: Dict[str, np.ndarray]
    cost: float
    converged: bool
    iterations: int
    solve_time: float

    def to_dict(self) -> dict:
        """Convert result to dictionary (excluding large trajectory arrays).

        Returns:
            Dictionary with params, cost, converged, iterations, solve_time.
        """
        return {
            'params': self.params,
            'cost': self.cost,
            'converged': self.converged,
            'iterations': self.iterations,
            'solve_time': self.solve_time,
        }

    @classmethod
    def from_dict(cls, data: dict, trajectories: Dict[str, np.ndarray]):
        """Reconstruct OptimizationResult from dictionary and trajectories.

        Args:
            data: Dictionary from to_dict() method.
            trajectories: Dictionary of trajectory arrays.

        Returns:
            OptimizationResult instance.
        """
        return cls(
            params=data['params'],
            trajectories=trajectories,
            cost=data['cost'],
            converged=data['converged'],
            iterations=data['iterations'],
            solve_time=data['solve_time'],
        )
