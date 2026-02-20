"""Runner script for PKPD sensitivity analysis on a specific patient."""

import os

import numpy as np

from pkpd import NorepinephrinePKPD
from opti.postprocessing import load_optimized_parameters
from utils.datatools import load_injections
from sensitivity.pkpd_integrator import PKPDIntegrator
from sensitivity.sensitivity import compute_normalized_sensitivity, compute_l2_norms
from sensitivity.plots import RESULTS_DIR, plot_normalized_sensitivity, plot_l2_norms

# =============================================================================
# CONFIGURATION
# =============================================================================

PATIENT_ID = 1
RES_DIR = "results"
OPTI_OUTPUT_DIR = "opti-constrained"   # subdirectory under results/patient_{id}/ containing params.json

T_END = 2200.0   # simulation end time (seconds)
DT = 0.5         # time step (seconds)


# =============================================================================
# HELPERS
# =============================================================================

def build_inor_trajectory(pkpd_model: NorepinephrinePKPD, patient_id: int, t: np.ndarray) -> np.ndarray:
    """Evaluate INOR at every point in the time grid.

    Parameters
    ----------
    pkpd_model : NorepinephrinePKPD
        Model instance with injection protocol loaded.
    patient_id : int
    t : np.ndarray, shape (N+1,)

    Returns
    -------
    np.ndarray, shape (N+1,)
    """
    return np.array([pkpd_model.INOR(t_k, patient_id) for t_k in t])


def params_to_array(params: dict[str, float]) -> np.ndarray:
    """Convert parameter dict to array in canonical PARAM_NAMES order."""
    return np.array([params[name] for name in PKPDIntegrator.PARAM_NAMES])


def print_l2_ranking(l2: np.ndarray, param_names: list[str], output_names: list[str]) -> None:
    """Print L2 norm summary sorted by aggregate impact."""
    l2_agg = np.sqrt(np.sum(l2 ** 2, axis=0))
    ranking = np.argsort(l2_agg)[::-1]

    print("\n=== L2 SENSITIVITY RANKING (aggregate, all outputs) ===")
    print(f"  {'Rank':>4}  {'Parameter':>10}  {'Aggregate L2':>14}")
    print("  " + "-" * 34)
    for rank, idx in enumerate(ranking, start=1):
        print(f"  {rank:>4}  {param_names[idx]:>10}  {l2_agg[idx]:>14.4f}")

    print("\n=== L2 NORMS PER (OUTPUT, PARAM) ===")
    header = f"  {'Output':>6} | " + " | ".join(f"{p:>8}" for p in param_names)
    print(header)
    print("  " + "-" * len(header))
    for i, out in enumerate(output_names):
        row = f"  {out:>6} | " + " | ".join(f"{l2[i, j]:>8.4f}" for j in range(len(param_names)))
        print(row)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    patient_id = PATIENT_ID

    print("=" * 60)
    print("PKPD PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)
    print(f"  Patient ID   : {patient_id}")
    print(f"  Params from  : {RES_DIR}/patient_{patient_id}/{OPTI_OUTPUT_DIR}/params.json")

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\nLoading optimized parameters...")
    params_opt = load_optimized_parameters(patient_id, RES_DIR, OPTI_OUTPUT_DIR)
    print(f"  {params_opt}")

    print("Loading injection protocol...")
    injections_dict = load_injections([patient_id])

    # -------------------------------------------------------------------------
    # Build time grid and INOR trajectory
    # -------------------------------------------------------------------------
    print("Building time grid and INOR trajectory...")
    pkpd_model = NorepinephrinePKPD(injections_dict)
    pkpd_model.set_parameters(params_opt)

    t = np.arange(0.0, T_END + DT, DT)
    inor_traj = build_inor_trajectory(pkpd_model, patient_id, t)
    print(f"  N={len(t)-1} steps, dt={DT}s, T={T_END}s")
    print(f"  INOR non-zero steps: {np.sum(inor_traj > 0)}")

    p_val = params_to_array(params_opt)
    x0 = np.array([0.0, 0.0, 0.0])

    # -------------------------------------------------------------------------
    # Build integrator and simulate
    # -------------------------------------------------------------------------
    print("\nBuilding CasADi RK4 integrator (variational equations)...")
    integrator = PKPDIntegrator()
    print(f"  rk4:     {integrator._F_rk4}")
    print(f"  pd_sens: {integrator._F_pd}")

    print("Simulating augmented system [states | sensitivity]...")
    history = integrator.simulate(x0, p_val, inor_traj, t)
    print(f"  Done — states: {history.states.shape}, S_states: {history.S_states.shape}")

    # -------------------------------------------------------------------------
    # Compute metrics
    # -------------------------------------------------------------------------
    print("\nComputing sensitivity metrics...")
    norm_sens = compute_normalized_sensitivity(history, p_val)
    l2 = compute_l2_norms(history, p_val)
    print(f"  norm_sens shape: {norm_sens.shape}")
    print(f"  l2 shape:        {l2.shape}")

    print_l2_ranking(l2, PKPDIntegrator.PARAM_NAMES, ['Ad', 'Ac', 'Ap', 'E'])

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    patient_save_dir = os.path.join(RESULTS_DIR, f"patient_{patient_id}")
    os.makedirs(patient_save_dir, exist_ok=True)
    print(f"\nSaving plots to {patient_save_dir}/")

    injection_times = injections_dict[patient_id][0] if patient_id in injections_dict else None
    plot_normalized_sensitivity(
        history, norm_sens, PKPDIntegrator.PARAM_NAMES,
        injection_times=injection_times,
        save_dir=patient_save_dir,
    )
    print("  ✓ pkpd_normalized_sensitivity.png")

    plot_l2_norms(l2, PKPDIntegrator.PARAM_NAMES, save_dir=patient_save_dir)
    print("  ✓ pkpd_l2_sensitivity.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
