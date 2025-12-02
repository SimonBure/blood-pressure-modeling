"""Plotting functions for PKPD optimization visualization."""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple
from ..pkpd import NorepinephrinePKPD


def plot_optimization_results(patient_id: int,
                              observations: Dict,
                              traj_opt: Dict,
                              params_opt: Dict[str, float],
                              resim_results: Tuple,
                              E_equilibrium: np.ndarray,
                              data_dir: str,
                              output_dir: str,
                              n_data_points: int,
                              cost_function_mode: str) -> None:
    """Create comparison plots for concentration and blood pressure.

    Args:
        patient_id: Patient ID.
        observations: Dict from load_observations().
        traj_opt: Dict of optimized trajectories from CasADi.
        params_opt: Optimized parameters dict.
        resim_results: Re-simulation results tuple.
        E_equilibrium: Array of equilibrium blood pressure values over time.
        data_dir: Base data directory.
        output_dir: Output subdirectory name.
        n_data_points: Number of data points used.
        cost_function_mode: Cost function mode.
    """
    output_path = f'{data_dir}/patient_{patient_id}/{output_dir}/{n_data_points}_points'
    os.makedirs(output_path, exist_ok=True)

    obs = observations[patient_id]

    # Extract REAL observations
    conc_obs = obs['concentration']
    conc_times, conc_values = zip(*conc_obs)
    conc_times = np.array(conc_times)
    conc_values = np.array(conc_values)

    bp_obs = obs['blood_pressure']
    bp_times, bp_values = zip(*bp_obs)
    bp_times = np.array(bp_times)
    bp_values = np.array(bp_values)

    # Get CasADi trajectories
    times_casadi = traj_opt['times']
    Ac_casadi = traj_opt['Ac']
    Cc_casadi = params_opt['C_endo'] + Ac_casadi / params_opt['V_c']

    # Get PKPD resimulated trajectories
    t_resim, _, Ac_resim, _, E_emax_resim, E_windkessel_resim = resim_results
    Cc_resim = params_opt['C_endo'] + Ac_resim / params_opt['V_c']

    # ========================================================================
    # PLOT 1: CONCENTRATION
    # ========================================================================

    fig1, ax1 = plt.subplots(figsize=(12, 6))

    ax1.scatter(conc_times, conc_values, c='red', s=80, zorder=5,
                label='Real Observations', alpha=0.8, edgecolors='darkred')
    ax1.plot(times_casadi, Cc_casadi, 'b-', linewidth=2, label='CasADi Optimized', alpha=0.7)
    ax1.plot(t_resim, Cc_resim, 'g--', linewidth=2, label='PKPD Resimulated', alpha=0.7)

    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('NOR Concentration (nmol/L)', fontsize=12)
    ax1.set_title(f'Patient {patient_id} - NOR Concentration (N={n_data_points})',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_path}/cc_opt.png', dpi=150)
    plt.close()

    # ========================================================================
    # PLOT 2: BLOOD PRESSURE
    # ========================================================================

    fig2, ax2 = plt.subplots(figsize=(12, 6))

    ax2.scatter(bp_times, bp_values, c='red', s=80, zorder=5,
                label='Real Observations', alpha=0.8, edgecolors='darkred')

    if cost_function_mode == 'emax':
        E_casadi = (params_opt['E_0'] +
                   (params_opt['E_max'] - params_opt['E_0']) * Cc_casadi /
                   (Cc_casadi + params_opt['EC_50']))
        ax2.plot(times_casadi, E_casadi, 'b-', linewidth=2,
                label='CasADi Optimized (Emax)', alpha=0.7)
        ax2.plot(t_resim, E_emax_resim, 'g--', linewidth=2,
                label='PKPD Resimulated (Emax)', alpha=0.7)
        # Add equilibrium blood pressure line
        ax2.plot(times_casadi, E_equilibrium, 'k--', linewidth=2,
                label='MAP équilibre', alpha=0.7)
    elif cost_function_mode == 'windkessel':
        E_casadi = traj_opt['E']
        ax2.plot(times_casadi, E_casadi, 'b-', linewidth=2,
                label='CasADi Optimized (Windkessel)', alpha=0.7)
        ax2.plot(t_resim, E_windkessel_resim, 'g--', linewidth=2,
                label='PKPD Resimulated (Windkessel)', alpha=0.7)
    else:  # both
        E_casadi = traj_opt['E']
        ax2.plot(times_casadi, E_casadi, 'b-', linewidth=2,
                label='CasADi Optimized (Windkessel)', alpha=0.7)
        ax2.plot(t_resim, E_windkessel_resim, 'g--', linewidth=2,
                label='PKPD Resimulated (Windkessel)', alpha=0.7)
        # Add equilibrium blood pressure line (using Emax parameters from 'both' mode)
        ax2.plot(times_casadi, E_equilibrium, 'k--', linewidth=2,
                label='MAP équilibre', alpha=0.7)

    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('MAP (mmHg)', fontsize=12)
    ax2.set_title(f'Patient {patient_id} - Blood Pressure (N={n_data_points})',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Set y-axis limit based on actual BP dynamics (excluding equilibrium peaks)
    max_bp = np.max(bp_values)
    ax2.set_ylim(top=max_bp + 5)

    plt.tight_layout()
    plt.savefig(f'{output_path}/bp_opt.png', dpi=150)
    plt.close()

    print(f"  ✓ Plots saved to {output_path}/")


def plot_pkpd_vs_casadi_trajectories(patient_id: int,
                                     traj_opt: Dict,
                                     resim_results: Tuple,
                                     params_opt: Dict[str, float],
                                     data_dir: str,
                                     output_dir: str,
                                     n_data_points: int,
                                     cost_function_mode: str) -> None:
    """Compare CasADi optimization trajectories vs PKPD model re-simulation.

    Args:
        patient_id: Patient ID.
        traj_opt: Dict of optimized trajectories from CasADi.
        resim_results: Tuple from resimulate_with_params.
        params_opt: Dict of optimized parameters.
        data_dir: Base data directory.
        output_dir: Output subdirectory name.
        n_data_points: Number of data points used.
        cost_function_mode: Cost function mode.
    """
    output_path = f'{data_dir}/patient_{patient_id}/{output_dir}/{n_data_points}_points'
    os.makedirs(output_path, exist_ok=True)

    t_resim, Ad_resim, Ac_resim, Ap_resim, E_emax_resim, E_windkessel_resim = resim_results

    # Create comparison plot with subplots for each state
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Patient {patient_id} - CasADi vs PKPD Model Trajectories',
                 fontsize=16, fontweight='bold')

    # Get CasADi times
    times_casadi = traj_opt.get('times', None)
    if times_casadi is None:
        N = len(traj_opt['Ad']) - 1
        times_casadi = np.linspace(t_resim[0], t_resim[-1], N + 1)

    # Plot Ad
    ax = axes[0, 0]
    ax.plot(times_casadi, traj_opt['Ad'], 'b-', linewidth=2, label='CasADi', alpha=0.7)
    ax.plot(t_resim, Ad_resim, 'r--', linewidth=2, label='PKPD Model', alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Ad (ng)', fontsize=11)
    ax.set_title('Depot Compartment (Ad)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot Ac
    ax = axes[0, 1]
    ax.plot(times_casadi, traj_opt['Ac'], 'b-', linewidth=2, label='CasADi', alpha=0.7)
    ax.plot(t_resim, Ac_resim, 'r--', linewidth=2, label='PKPD Model', alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Ac (ng)', fontsize=11)
    ax.set_title('Central Compartment (Ac)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot Ap
    ax = axes[1, 0]
    ax.plot(times_casadi, traj_opt['Ap'], 'b-', linewidth=2, label='CasADi', alpha=0.7)
    ax.plot(t_resim, Ap_resim, 'r--', linewidth=2, label='PKPD Model', alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Ap (ng)', fontsize=11)
    ax.set_title('Peripheral Compartment (Ap)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot E (BP effect)
    ax = axes[1, 1]
    if cost_function_mode == 'emax':
        Cc_casadi = params_opt['C_endo'] + traj_opt['Ac'] / params_opt['V_c']
        E_casadi = (params_opt['E_0'] +
                   (params_opt['E_max'] - params_opt['E_0']) * Cc_casadi /
                   (Cc_casadi + params_opt['EC_50']))
        ax.plot(times_casadi, E_casadi, 'b-', linewidth=2, label='CasADi (Emax)', alpha=0.7)
        ax.plot(t_resim, E_emax_resim, 'r--', linewidth=2, label='PKPD Model (Emax)', alpha=0.7)
    elif cost_function_mode == 'windkessel':
        ax.plot(times_casadi, traj_opt['E'], 'b-', linewidth=2,
               label='CasADi (Windkessel)', alpha=0.7)
        ax.plot(t_resim, E_windkessel_resim, 'r--', linewidth=2,
               label='PKPD Model (Windkessel)', alpha=0.7)
    else:  # both
        ax.plot(times_casadi, traj_opt['E'], 'b-', linewidth=2,
               label='CasADi (Windkessel)', alpha=0.7)
        ax.plot(t_resim, E_windkessel_resim, 'r--', linewidth=2,
               label='PKPD Model (Windkessel)', alpha=0.7)

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('MAP (mmHg)', fontsize=11)
    ax.set_title('Blood Pressure Effect (E)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_path}/pkpd_vs_casadi_traj.png', dpi=150)
    plt.close()

    print(f"  ✓ CasADi vs PKPD comparison plot saved to {output_path}/")


def plot_injection_verification(patient_id: int,
                                times: np.ndarray,
                                inor_values: np.ndarray,
                                injections_dict: Dict,
                                data_dir: str,
                                output_dir: str,
                                n_data_points: int) -> None:
    """Create verification plot for injection rate sampling.

    Plots computed INOR(t) at optimization times vs actual injection events
    to verify that time discretization captures all injections properly.

    Args:
        patient_id: Patient ID.
        times: Array of optimization time points.
        inor_values: Array of computed injection rates at each time point.
        injections_dict: Dictionary mapping patient_id to (times, amounts, durations).
        data_dir: Base data directory.
        output_dir: Output subdirectory name.
        n_data_points: Number of data points used.
    """
    output_path = f'{data_dir}/patient_{patient_id}/{output_dir}/{n_data_points}_points'
    os.makedirs(output_path, exist_ok=True)

    # Extract actual injection events for this patient
    injection_times, injection_amounts, injection_durations = injections_dict[patient_id]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot computed INOR(t) as continuous line
    ax.plot(times, inor_values, 'b-', linewidth=2, label='Computed INOR(t)', alpha=0.8)

    # Plot actual injection events as vertical red lines
    if len(injection_times) > 0:
        # Compute injection rates (amount / duration) in µg/min
        # Note: amounts are in nmol, need to convert appropriately
        injection_rates = injection_amounts / injection_durations  # nmol/s
        injection_rates = injection_rates * 60  # convert to nmol/min

        # Plot vertical lines at injection start times
        for i, (t_inj, rate) in enumerate(zip(injection_times, injection_rates)):
            label = 'Injection Events' if i == 0 else None
            ax.vlines(t_inj, 0, np.max(inor_values) * 1.1, colors='red',
                     linestyles='--', linewidth=1.5, alpha=0.7, label=label)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Injection Rate (µg/min)', fontsize=12)
    ax.set_title(f'Patient {patient_id} - Injection Rate Verification (N={n_data_points-1})',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(f'{output_path}/injections_verif.png', dpi=150)
    plt.close()

    print(f"  ✓ Injection verification plot saved to {output_path}/")
