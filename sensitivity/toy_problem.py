import os
from typing import Dict

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results/sensitivity"


def analytical_sol(t: np.ndarray, x_0: float, v_0: float, theta: Dict[str, float]):
    omega = theta['omega']
    zeta = theta['zeta']
    omega_d = omega * np.sqrt(1 - zeta**2)
    return np.exp(-zeta * omega * t) * x_0 * (
        np.cos(omega_d * t) + np.sin(omega_d * t) * (v_0 + zeta * omega * x_0) / omega_d
    )



def build_rk4_integrator() -> ca.Function:
    """
    Build a single RK4 step for the augmented system [x(2); vec(S)(4)].

    The augmented state z = [pos, vel, dpos/domega, dvel/domega, dpos/dzeta, dvel/dzeta].
    Parameters p = [omega, zeta].

    Returns
    -------
    ca.Function
        Signature: (z[6], p[2], dt) -> z_next[6]
    """
    # --- Symbols ---
    x = ca.SX.sym('x', 2)       # [pos, vel]
    p = ca.SX.sym('p', 2)       # [omega, zeta]
    S_flat = ca.SX.sym('s', 4)  # vec(S), column-major
    dt = ca.SX.sym('dt')

    z = ca.vertcat(x, S_flat)   # augmented state (6,)

    omega = p[0]
    zeta = p[1]

    # --- ODE right-hand side ---
    f = ca.vertcat(
        x[1],
        -omega**2 * x[0] - 2 * zeta * omega * x[1]
    )

    # --- Symbolic Jacobians (the core CasADi symbolic differentiation) ---
    dfdx = ca.jacobian(f, x)   # shape (2, 2)
    dfdp = ca.jacobian(f, p)   # shape (2, 2)

    # --- Variational equation: dS/dt = (df/dx) S + (df/dp) ---
    # CasADi reshape is column-major: S[:,0] = d(x)/d(omega), S[:,1] = d(x)/d(zeta)
    S = ca.reshape(S_flat, 2, 2)
    dSdt = dfdx @ S + dfdp

    # --- Augmented RHS ---
    rhs = ca.vertcat(f, ca.vec(dSdt))   # shape (6,)

    # --- RK4 via symbolic substitution ---
    k1 = rhs
    k2 = ca.substitute(rhs, z, z + dt / 2 * k1)
    k3 = ca.substitute(rhs, z, z + dt / 2 * k2)
    k4 = ca.substitute(rhs, z, z + dt * k3)
    z_next = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return ca.Function(
        'rk4_step',
        [z, p, dt], [z_next],
        ['z', 'p', 'dt'], ['z_next']
    )


def simulate(
    F_rk4: ca.Function,
    x0: np.ndarray,
    p_val: np.ndarray,
    N: int,
    dt: float,
) -> np.ndarray:
    """
    Simulate the augmented system (state + sensitivity) over N steps.

    Parameters
    ----------
    F_rk4 : ca.Function
        One-step RK4 integrator from build_rk4_integrator().
    x0 : np.ndarray, shape (2,)
        Initial state [pos_0, vel_0].
    p_val : np.ndarray, shape (2,)
        Parameter values [omega, zeta].
    N : int
        Number of time steps.
    dt : float
        Time step size.

    Returns
    -------
    np.ndarray, shape (N+1, 6)
        Columns 0-1: state [pos, vel]
        Columns 2-5: vec(S) = [dpos/domega, dvel/domega, dpos/dzeta, dvel/dzeta]
    """
    z_cur = np.concatenate([x0, np.zeros(4)])  # S(0) = 0
    history = np.empty((N + 1, 6))
    history[0] = z_cur
    for k in range(N):
        z_cur = F_rk4(z_cur, p_val, dt).full().flatten()
        history[k + 1] = z_cur
    return history



def _rk4_numpy_step(x: np.ndarray, omega: float, zeta: float, dt: float) -> np.ndarray:
    """Single RK4 step for the plain (non-augmented) harmonic oscillator."""
    def f(state):
        return np.array([
            state[1],
            -omega**2 * state[0] - 2 * zeta * omega * state[1]
        ])
    k1 = f(x)
    k2 = f(x + dt / 2 * k1)
    k3 = f(x + dt / 2 * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def finite_difference_jacobian(
    p_val: np.ndarray,
    x0: np.ndarray,
    N: int,
    dt: float,
    eps: float = 1e-5,
) -> dict:
    """
    Compute the Jacobian dx/dp at each time step via forward finite differences.

    Returns
    -------
    dict with keys: dpos_domega, dvel_domega, dpos_dzeta, dvel_dzeta
        Each value is an np.ndarray of shape (N+1,).
    """
    omega, zeta = p_val

    def simulate_np(om, ze):
        x = x0.copy()
        hist = [x.copy()]
        for _ in range(N):
            x = _rk4_numpy_step(x, om, ze, dt)
            hist.append(x.copy())
        return np.array(hist)  # shape (N+1, 2)

    h_nom  = simulate_np(omega,       zeta)
    h_p_om = simulate_np(omega + eps, zeta)
    h_p_ze = simulate_np(omega,       zeta + eps)

    return {
        'dpos_domega': (h_p_om[:, 0] - h_nom[:, 0]) / eps,
        'dvel_domega': (h_p_om[:, 1] - h_nom[:, 1]) / eps,
        'dpos_dzeta':  (h_p_ze[:, 0] - h_nom[:, 0]) / eps,
        'dvel_dzeta':  (h_p_ze[:, 1] - h_nom[:, 1]) / eps,
    }


def validate_sensitivities(
    history: np.ndarray,
    fd_jac: dict,
    t: np.ndarray,
) -> None:
    """Print a comparison table at t=0, T/2, T."""
    N = len(t) - 1
    check_indices = [0, N // 2, N]
    elements = [
        ('dpos/domega', history[:, 2], fd_jac['dpos_domega']),
        ('dvel/domega', history[:, 3], fd_jac['dvel_domega']),
        ('dpos/dzeta',  history[:, 4], fd_jac['dpos_dzeta']),
        ('dvel/dzeta',  history[:, 5], fd_jac['dvel_dzeta']),
    ]

    print("\n=== SENSITIVITY VALIDATION ===")
    print(f"{'Time':>6}  {'Element':>14}  {'Variational':>14}  {'FD (eps=1e-5)':>14}  {'Rel Error':>10}")
    print("-" * 68)

    for idx in check_indices:
        for name, var_vals, fd_vals in elements:
            v, f_val = var_vals[idx], fd_vals[idx]
            if abs(f_val) > 1e-12:
                rel_err = abs(v - f_val) / abs(f_val) * 100
                rel_str = f"{rel_err:.4f}%"
                if rel_err > 0.1:
                    rel_str += " !"
            else:
                rel_str = "---"
            print(f"{t[idx]:>6.2f}  {name:>14}  {v:>14.6f}  {f_val:>14.6f}  {rel_str:>10}")
        print()


# =============================================================================
# SECTION 4: PLOTTING
# =============================================================================

def plot_states(
    t: np.ndarray,
    history: np.ndarray,
    x0: np.ndarray,
    params: Dict[str, float],
) -> None:
    """Plot state trajectories pos(t) and vel(t), with analytical solution overlay."""
    pos_analytical = analytical_sol(t, x0[0], x0[1], params)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, history[:, 0], color='steelblue', linewidth=1.5, label='RK4 (CasADi)')
    axes[0].plot(t, pos_analytical, color='steelblue', linewidth=1.0,
                 linestyle='--', alpha=0.6, label='Analytical')
    axes[0].set_ylabel("Position")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='k', alpha=0.2, linewidth=0.8)
    axes[0].legend(fontsize=8)

    axes[1].plot(t, history[:, 1], color='darkorange', linewidth=1.5)
    axes[1].set_ylabel("Velocity")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color='k', alpha=0.2, linewidth=0.8)

    omega = params['omega']
    zeta = params['zeta']
    fig.suptitle(f"Damped Harmonic Oscillator  (ω={omega}, ζ={zeta})", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "toy_problem.png"))
    plt.close()


def plot_jacobians(
    t: np.ndarray,
    history: np.ndarray,
    fd_jac: dict,
) -> None:
    """
    Plot all 4 Jacobian elements dx/dp over time.

    Column-major indexing of history:
      history[:,2] = dpos/domega   history[:,4] = dpos/dzeta
      history[:,3] = dvel/domega   history[:,5] = dvel/dzeta
    """
    elements = [
        (0, 0, 'dpos/dω', history[:, 2], fd_jac['dpos_domega'], 'steelblue'),
        (1, 0, 'dvel/dω', history[:, 3], fd_jac['dvel_domega'], 'firebrick'),
        (0, 1, 'dpos/dζ', history[:, 4], fd_jac['dpos_dzeta'],  'seagreen'),
        (1, 1, 'dvel/dζ', history[:, 5], fd_jac['dvel_dzeta'],  'darkorchid'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for row, col, label, var_vals, fd_vals, color in elements:
        ax = axes[row][col]
        ax.plot(t, var_vals, color=color, linewidth=1.5, label='Variational')
        ax.plot(t, fd_vals,  color=color, linewidth=1.0, linestyle='--',
                alpha=0.6, label='FD check')
        ax.axhline(0, color='k', alpha=0.2, linewidth=0.8)
        ax.set_ylabel(label)
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Jacobian  ∂x/∂p  over Time  (Variational Equations)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "jacobian.png"))
    plt.close()


def plot_normalized_sensitivity(
    t: np.ndarray,
    history: np.ndarray,
    p_val: np.ndarray,
) -> None:
    """
    Plot dimensionless (normalized) sensitivity: S_rel_ij = (p_j / scale_i) * (dx_i/dp_j).

    scale_i = max|x_i(t)| so that the result is interpretable as:
    "fraction of peak state amplitude changed per fractional parameter change".
    """
    pos_scale = np.max(np.abs(history[:, 0]))
    vel_scale = np.max(np.abs(history[:, 1]))

    omega, zeta = p_val

    elements = [
        (0, 0, 'dpos/dω  (norm.)', omega / pos_scale * history[:, 2], 'steelblue'),
        (1, 0, 'dvel/dω  (norm.)', omega / vel_scale * history[:, 3], 'firebrick'),
        (0, 1, 'dpos/dζ  (norm.)', zeta  / pos_scale * history[:, 4], 'seagreen'),
        (1, 1, 'dvel/dζ  (norm.)', zeta  / vel_scale * history[:, 5], 'darkorchid'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for row, col, label, vals, color in elements:
        ax = axes[row][col]
        ax.plot(t, vals, color=color, linewidth=1.5)
        ax.axhline(0, color='k', alpha=0.2, linewidth=0.8)
        ax.set_ylabel(label)
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Normalized Sensitivity  (p/scale) · ∂x/∂p  —  dimensionless",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "normalized_sensitivity.png"))
    plt.close()


def plot_integrated_sensitivity(
    t: np.ndarray,
    history: np.ndarray,
    p_val: np.ndarray,
) -> None:
    """
    Bar chart of L2-integrated sensitivity: L2_ij = sqrt( integral (dx_i/dp_j)^2 dt ).

    Groups bars by parameter (omega, zeta), with one bar per state (pos, vel).
    A third 'combined' bar shows L2_j = sqrt(L2_pos_j^2 + L2_vel_j^2).
    """
    omega, zeta = p_val

    # L2 norm of each Jacobian element over time (trapezoidal integration)
    l2_pos_omega = np.sqrt(np.trapezoid(history[:, 2] ** 2, t))
    l2_vel_omega = np.sqrt(np.trapezoid(history[:, 3] ** 2, t))
    l2_pos_zeta  = np.sqrt(np.trapezoid(history[:, 4] ** 2, t))
    l2_vel_zeta  = np.sqrt(np.trapezoid(history[:, 5] ** 2, t))

    # Per-parameter aggregate (L2 over all states)
    l2_omega = np.sqrt(l2_pos_omega**2 + l2_vel_omega**2)
    l2_zeta  = np.sqrt(l2_pos_zeta**2  + l2_vel_zeta**2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: per (state, param) breakdown ---
    ax = axes[0]
    x_pos = np.array([0.0, 1.0])  # group positions for omega, zeta
    bar_w = 0.3
    bars_pos = ax.bar(x_pos - bar_w / 2, [l2_pos_omega, l2_pos_zeta],
                      width=bar_w, color='steelblue', label='pos')
    bars_vel = ax.bar(x_pos + bar_w / 2, [l2_vel_omega, l2_vel_zeta],
                      width=bar_w, color='darkorange', label='vel')
    ax.bar_label(bars_pos, fmt='%.3f', padding=2, fontsize=8)
    ax.bar_label(bars_vel, fmt='%.3f', padding=2, fontsize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'ω={omega}', f'ζ={zeta}'])
    ax.set_ylabel(r'$\sqrt{\int_0^T (\partial x_i / \partial p_j)^2\, dt}$')
    ax.set_title("L2 Sensitivity by State")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # --- Right: aggregate per parameter ---
    ax = axes[1]
    colors = ['steelblue', 'seagreen']
    bars = ax.bar([f'ω={omega}', f'ζ={zeta}'], [l2_omega, l2_zeta],
                  color=colors, width=0.4)
    ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=9)
    ax.set_ylabel(r'$\sqrt{\sum_i \int_0^T (\partial x_i / \partial p_j)^2\, dt}$')
    ax.set_title("L2 Sensitivity — Combined (all states)")
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle("L2-Integrated Parameter Sensitivity", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "integrated_sensitivity.png"))
    plt.close()


def main():
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    omega_nom = 1.5          # natural frequency (rad/s)
    zeta_nom  = 0.3          # damping ratio (-)
    T         = 10.0         # simulation duration (s)
    N         = 200          # number of time steps
    x0        = np.array([1.0, 0.0])   # [pos_0, vel_0]

    dt    = T / N
    t     = np.linspace(0.0, T, N + 1)
    p_val = np.array([omega_nom, zeta_nom])
    params = {'omega': omega_nom, 'zeta': zeta_nom}

    print("=" * 60)
    print("DAMPED HARMONIC OSCILLATOR — SENSITIVITY ANALYSIS")
    print("=" * 60)
    print(f"  omega = {omega_nom},  zeta = {zeta_nom}")
    print(f"  T = {T}s,  N = {N} steps,  dt = {dt:.4f}s")
    print(f"  Initial condition: {x0}")

    # -------------------------------------------------------------------------
    # Build integrator
    # -------------------------------------------------------------------------
    print("\nBuilding RK4 integrator (variational equations)...")
    F_rk4 = build_rk4_integrator()
    print(f"  {F_rk4}")

    # -------------------------------------------------------------------------
    # Simulate augmented system
    # -------------------------------------------------------------------------
    print("\nSimulating augmented system [state | sensitivity]...")
    history = simulate(F_rk4, x0, p_val, N, dt)
    print(f"  Done. history shape: {history.shape}")

    # -------------------------------------------------------------------------
    # Validate against finite differences
    # -------------------------------------------------------------------------
    print("\nComputing finite-difference Jacobian for validation...")
    fd_jac = finite_difference_jacobian(p_val, x0, N, dt)
    validate_sensitivities(history, fd_jac, t)

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Generating plots...")
    plot_states(t, history, x0, params)
    plot_jacobians(t, history, fd_jac)
    plot_normalized_sensitivity(t, history, p_val)
    plot_integrated_sensitivity(t, history, p_val)

    print("\nDone.")


if __name__ == "__main__":
    main()
