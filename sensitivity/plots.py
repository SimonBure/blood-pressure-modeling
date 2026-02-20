"""Plotting functions for PKPD sensitivity analysis."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from sensitivity.sensitivity_history import SensitivityHistory

RESULTS_DIR = "results/sensitivity"
OUTPUT_NAMES = ['Ad', 'Ac', 'Ap', 'E']
OUTPUT_COLORS = ['steelblue', 'darkorange', 'seagreen', 'firebrick']


def plot_normalized_sensitivity(
    history: SensitivityHistory,
    norm_sens: np.ndarray,
    param_names: list[str],
    output_names: list[str] = OUTPUT_NAMES,
    injection_times: np.ndarray | None = None,
    save_dir: str = RESULTS_DIR,
) -> None:
    """Plot dimensionless normalized sensitivity for all outputs over time.

    One subplot per output (4 rows). Each subplot shows one line per parameter (9 lines).
    Dashed vertical lines mark injection start times when provided.

    Parameters
    ----------
    history : SensitivityHistory
    norm_sens : np.ndarray, shape (N+1, 4, 9)
        From compute_normalized_sensitivity().
    param_names : list[str]
        Parameter names in canonical order (length 9).
    output_names : list[str]
        Output names for subplot labels.
    injection_times : np.ndarray, optional
        Start times (seconds) of each injection — drawn as dashed vertical lines.
    save_dir : str
        Directory to save the figure.
    """
    n_outputs = len(output_names)
    n_params = len(param_names)
    t = history.t

    colors = plt.cm.tab10(np.linspace(0, 0.9, n_params))

    fig, axes = plt.subplots(n_outputs, 1, figsize=(12, 3 * n_outputs), sharex=True)

    for i, (ax, out_name) in enumerate(zip(axes, output_names)):
        for j in range(n_params):
            vals = norm_sens[:, i, j]
            if not np.all(np.isnan(vals)):
                ax.plot(t, vals, color=colors[j], linewidth=1.2, label=param_names[j])
        ax.axhline(0, color='k', alpha=0.2, linewidth=0.8)
        if injection_times is not None:
            for t_inj in injection_times:
                ax.axvline(t_inj, color='red', linestyle='--', linewidth=1.2, alpha=0.6)
        ax.set_ylabel(f'(p/scale)·∂{out_name}/∂p', fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, ncols=3, loc='upper right')

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Normalized Sensitivity  (p / max|output|) · ∂output/∂p", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pkpd_normalized_sensitivity.png"), dpi=150)
    plt.close()


def plot_l2_norms(
    l2: np.ndarray,
    param_names: list[str],
    output_names: list[str] = OUTPUT_NAMES,
    save_dir: str = RESULTS_DIR,
) -> None:
    """Bar chart of L2-integrated sensitivity norms.

    Left panel:  grouped bars — x-axis = parameters, 4 bars per group (one per output).
    Right panel: aggregate per parameter = sqrt(Σ_i L2[i,j]²).

    Parameters
    ----------
    l2 : np.ndarray, shape (4, 9)
        From compute_l2_norms().
    param_names : list[str]
    output_names : list[str]
    save_dir : str
    """
    n_outputs = len(output_names)
    n_params = len(param_names)

    # Aggregate L2 per parameter: sqrt(sum_i L2[i,j]^2)
    l2_agg = np.sqrt(np.sum(l2 ** 2, axis=0))  # shape (9,)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: per (output, param) breakdown ---
    ax = axes[0]
    bar_w = 0.18
    x_centers = np.arange(n_params, dtype=float)
    offsets = np.linspace(-(n_outputs - 1) / 2, (n_outputs - 1) / 2, n_outputs) * bar_w

    for i, (out_name, color, offset) in enumerate(zip(output_names, OUTPUT_COLORS, offsets)):
        bars = ax.bar(
            x_centers + offset, l2[i],
            width=bar_w, color=color, label=out_name, alpha=0.85,
        )
        ax.bar_label(bars, fmt='%.2f', padding=1, fontsize=6, rotation=90)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(param_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(r'$\sqrt{\int_0^T (\partial \mathrm{output} / \partial p_j)^2\, dt}$')
    ax.set_title("L2 Sensitivity by Output")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Right: aggregate per parameter ---
    ax = axes[1]
    param_colors = plt.cm.tab10(np.linspace(0, 0.9, n_params))
    bars = ax.bar(param_names, l2_agg, color=param_colors, width=0.6)
    ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=8)
    ax.set_xticklabels(param_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(r'$\sqrt{\sum_i L2[i,j]^2}$')
    ax.set_title("Aggregate L2 Sensitivity (all outputs)")
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle("L2-Integrated Parameter Sensitivity", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pkpd_l2_sensitivity.png"), dpi=150)
    plt.close()
