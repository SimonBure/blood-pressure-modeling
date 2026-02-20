"""CasADi symbolic PKPD integrator with variational equations for sensitivity analysis."""

from __future__ import annotations

import casadi as ca
import numpy as np

from sensitivity.sensitivity_history import SensitivityHistory


class PKPDIntegrator:
    """Symbolic 3-compartment PK + Emax PD integrator with forward sensitivity.

    Augments the PK ODE with variational equations to propagate S = ∂x/∂p
    alongside the states, then computes ∂E/∂p algebraically at each step.

    States  x = [Ad, Ac, Ap]        (N_STATES = 3)
    Params  p = [C_endo, k_a, V_c, k_12, k_21, k_el, E_0, E_max, EC_50]  (N_PARAMS = 9)
    INOR(t) passed as a numerical scalar per step — not differentiated.
    """

    PARAM_NAMES: list[str] = [
        'C_endo', 'k_a', 'V_c', 'k_12', 'k_21', 'k_el', 'E_0', 'E_max', 'EC_50',
    ]
    N_STATES: int = 3
    N_PARAMS: int = 9
    N_S_FLAT: int = 27  # N_STATES × N_PARAMS

    def __init__(self) -> None:
        self._F_rk4 = self._build_rk4_step()
        self._F_pd = self._build_pd_sensitivity()

    # -------------------------------------------------------------------------
    # Symbolic builders (called once at construction)
    # -------------------------------------------------------------------------

    def _build_rk4_step(self) -> ca.Function:
        """RK4 step for the augmented system [x(3); vec(S)(27)].

        Returns
        -------
        ca.Function
            Signature: (z[30], p[9], inor, dt) -> z_next[30]
        """
        x = ca.SX.sym('x', self.N_STATES)
        p = ca.SX.sym('p', self.N_PARAMS)
        S_flat = ca.SX.sym('s', self.N_S_FLAT)
        inor = ca.SX.sym('inor')
        dt = ca.SX.sym('dt')

        z = ca.vertcat(x, S_flat)

        # Parameter unpacking (canonical order)
        _C_endo, k_a, _V_c, k_12, k_21, k_el, _E_0, _E_max, _EC_50 = [p[i] for i in range(9)]

        # PK ODE  f: R^3 × R^9 × R → R^3
        f = ca.vertcat(
            -k_a * x[0] + inor,
            k_a * x[0] - (k_12 + k_el) * x[1] + k_21 * x[2],
            k_12 * x[1] - k_21 * x[2],
        )

        # Symbolic Jacobians of f
        dfdx = ca.jacobian(f, x)    # (3, 3)
        dfdp = ca.jacobian(f, p)    # (3, 9)  — last 3 cols (E_0, E_max, EC_50) are zero

        # Variational equation:  dS/dt = (∂f/∂x) S + (∂f/∂p)
        S = ca.reshape(S_flat, self.N_STATES, self.N_PARAMS)  # column-major
        dSdt = dfdx @ S + dfdp                                # (3, 9)

        # Augmented RHS (shape 30)
        rhs = ca.vertcat(f, ca.vec(dSdt))

        # RK4 via symbolic substitution (p and inor stay fixed across all stages)
        k1 = rhs
        k2 = ca.substitute(rhs, z, z + dt / 2 * k1)
        k3 = ca.substitute(rhs, z, z + dt / 2 * k2)
        k4 = ca.substitute(rhs, z, z + dt * k3)
        z_next = z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return ca.Function(
            'rk4',
            [z, p, inor, dt], [z_next],
            ['z', 'p', 'inor', 'dt'], ['z_next'],
        )

    def _build_pd_sensitivity(self) -> ca.Function:
        """Algebraic PD output E and its parameter sensitivity ∂E/∂p.

        Uses the chain rule:
            ∂E/∂p = (∂E/∂x) @ S + (∂E/∂p)_direct

        Returns
        -------
        ca.Function
            Signature: (x[3], p[9], S_flat[27]) -> (E[1,1], dEdp[1,9])
        """
        x = ca.SX.sym('x', self.N_STATES)
        p = ca.SX.sym('p', self.N_PARAMS)
        S_flat = ca.SX.sym('s', self.N_S_FLAT)

        C_endo, _k_a, V_c, _k_12, _k_21, _k_el, E_0, E_max, EC_50 = [p[i] for i in range(9)]

        Cc = C_endo + x[1] / V_c
        E = E_0 + (E_max - E_0) * Cc / (Cc + EC_50)

        # Direct Jacobians of E
        dEdx = ca.jacobian(E, x)          # (1, 3) — non-zero only for Ac column
        dEdp_direct = ca.jacobian(E, p)   # (1, 9) — non-zero for C_endo, V_c, E_0, E_max, EC_50

        # Chain rule
        S = ca.reshape(S_flat, self.N_STATES, self.N_PARAMS)
        dEdp = dEdx @ S + dEdp_direct     # (1, 9)

        return ca.Function(
            'pd_sens',
            [x, p, S_flat], [E, dEdp],
            ['x', 'p', 's'], ['E', 'dEdp'],
        )

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    def simulate(
        self,
        x0: np.ndarray,
        p_val: np.ndarray,
        inor_traj: np.ndarray,
        t: np.ndarray,
    ) -> SensitivityHistory:
        """Simulate augmented system [x; vec(S)] and compute PD sensitivity at each step.

        Parameters
        ----------
        x0 : np.ndarray, shape (3,)
            Initial state [Ad_0, Ac_0, Ap_0].
        p_val : np.ndarray, shape (9,)
            Parameter values in canonical PARAM_NAMES order.
        inor_traj : np.ndarray, shape (N+1,)
            Precomputed INOR(t_k) for each time point. Value at index k is used
            for the step from t[k] to t[k+1].
        t : np.ndarray, shape (N+1,)
            Time grid (may be non-uniform).

        Returns
        -------
        SensitivityHistory
        """
        N = len(t) - 1
        states = np.empty((N + 1, self.N_STATES))
        s_flat = np.empty((N + 1, self.N_S_FLAT))
        E = np.empty(N + 1)
        dEdp = np.empty((N + 1, self.N_PARAMS))

        # Initial conditions: x = x0, S = 0
        z_cur = np.concatenate([x0, np.zeros(self.N_S_FLAT)])
        states[0] = x0
        s_flat[0] = np.zeros(self.N_S_FLAT)

        res_E0, res_dEdp0 = self._F_pd(x0, p_val, np.zeros(self.N_S_FLAT))
        E[0] = float(res_E0)
        dEdp[0] = np.array(res_dEdp0).flatten()

        for k in range(N):
            dt_k = t[k + 1] - t[k]
            z_cur = self._F_rk4(z_cur, p_val, inor_traj[k], dt_k).full().flatten()
            states[k + 1] = z_cur[:self.N_STATES]
            s_flat[k + 1] = z_cur[self.N_STATES:]

            res_E, res_dEdp = self._F_pd(z_cur[:self.N_STATES], p_val, z_cur[self.N_STATES:])
            E[k + 1] = float(res_E)
            dEdp[k + 1] = np.array(res_dEdp).flatten()

        # ca.vec() uses column-major (Fortran) order: v[3*j + i] = S[i, j].
        # reshape(N+1, N_PARAMS, N_STATES) with C-order gives arr[t, col, row] = s_flat[t, 3*col+row],
        # then transpose(0,2,1) yields S_states[t, i, j] = s_flat[t, 3*j + i] = S[i, j].
        S_states = s_flat.reshape(N + 1, self.N_PARAMS, self.N_STATES).transpose(0, 2, 1)

        return SensitivityHistory(
            t=t,
            states=states,
            S_states=S_states,
            E=E,
            dEdp=dEdp,
        )
