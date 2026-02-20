"""
Docstring for pkpd
"""
import os
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from utils.physiological_constants import PhysiologicalConstants
from utils.datatools import get_patient_ids, load_observations, load_injections


# ============================================================================
# PKPD MODEL CLASS
# ============================================================================

class NorepinephrinePKPD:
    # PK parameters
    C_endo: float
    k_a: float
    V_c: float
    k_12: float
    k_21: float
    k_el: float
    # PD Emax parameters
    E_0: float
    E_max: float
    EC_50: float
    
    # Initial conditions
    Ad_0: float
    Ac_0: float
    Ap_0: float
    dEdt_0: float

    def __init__(self, injections_dict=None, initial_conditions=None):
        self.injections_dict = injections_dict if injections_dict is not None else {}

        # Set default parameters using the paper physiological constants
        self.set_parameters(PhysiologicalConstants().get_constants_dict())

        # Initial conditions
        self.Ad_0 = 0.0
        self.Ac_0 = 0.0
        self.Ap_0 = 0.0
        self.dEdt_0 = 0.0
        
        if initial_conditions is not None:
            self.set_initial_conditions(initial_conditions)
        
    def set_parameters(self, params: Dict):
        for key, value in params.items():
            setattr(self, key, value)
            
    def get_parameters(self):
        return {key: value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(value)}
            
    def set_initial_conditions(self, initial_conditions: Dict):
        for key, value in initial_conditions.items():
            setattr(self, key, value)

    def INOR(self, t, patient_id):
        if patient_id not in self.injections_dict:
            return 0.0

        times, amounts, durations = self.injections_dict[patient_id]
                
        for i, t_start in enumerate(times):
            t_end = t_start + durations[i]
            if t_start <= t < t_end:
                return amounts[i] / durations[i]
        return 0.0

    def pk_rhs(self, t, y, patient_id):
        Ad, Ac, Ap = y
        dAd_dt = -self.k_a * Ad + self.INOR(t, patient_id)
        dAc_dt = self.k_a * Ad - (self.k_12 + self.k_el) * Ac + self.k_21 * Ap
        dAp_dt = self.k_12 * Ac - self.k_21 * Ap
        return np.array([dAd_dt, dAc_dt, dAp_dt])

    def euler_implicit_pk(self, y_n, t_n, dt, patient_id) -> np.ndarray:
        t_np1 = t_n + dt
        def residual(y_np1):
            return y_np1 - y_n - dt * self.pk_rhs(t_np1, y_np1, patient_id)
        y_np1 = fsolve(residual, y_n)
        return np.array(y_np1)

    def compute_concentration(self, Ac):
        return self.C_endo + Ac / self.V_c

    def pd_emax(self, Cc):
        return self.E_0 + (self.E_max - self.E_0) * Cc / (Cc + self.EC_50)
    
    def bp_obs_to_x_variables(self, blood_pressure_obs):
        return (self.E_0 - self.E_max) * self.V_c * self.EC_50 / (blood_pressure_obs - self.E_max)
    
    def x4(self):
        return self.V_c * (self.EC_50 + self.C_endo)

    def simulate(self, patient_id, t_end=2200, dt=0.5, t_eval=None):
        """Simulate PKPD model using Euler implicit method.

        Args:
            patient_id: Patient ID for injection protocol.
            t_end: End time in seconds (used only if t_eval is None).
            dt: Time step in seconds (used only if t_eval is None).
            t_eval: Optional array of specific time points to evaluate at.
                   If provided, simulation uses these exact points with variable dt.
                   If None, uses uniform grid with fixed dt.

        Returns:
            Tuple of (t, Ad, Ac, Ap, E_emax) arrays.
        """
        # Use custom time points if provided, otherwise use uniform grid
        if t_eval is not None:
            t = np.asarray(t_eval)
        else:
            t = np.arange(0, t_end + dt, dt)

        n_steps = len(t)

        a_d = np.zeros(n_steps)
        a_c = np.zeros(n_steps)
        a_p = np.zeros(n_steps)
        c_c = np.zeros(n_steps)
        e_emax = np.zeros(n_steps)

        a_d[0] = self.Ad_0
        a_c[0] = self.Ac_0
        a_p[0] = self.Ap_0
        c_c[0] = self.compute_concentration(a_c[0])

        e_emax[0] = self.pd_emax(c_c[0])

        for i in range(n_steps - 1):
            # Use variable time step when using custom time points
            dt_i = t[i+1] - t[i]

            y_pk = np.array([a_d[i], a_c[i], a_p[i]])
            y_pk_new = self.euler_implicit_pk(y_pk, t[i], dt_i, patient_id)

            a_d[i+1] = y_pk_new[0]
            a_c[i+1] = y_pk_new[1]
            a_p[i+1] = y_pk_new[2]

            c_c[i+1] = self.compute_concentration(a_c[i+1])
            e_emax[i+1] = self.pd_emax(c_c[i+1])

        return t, a_d, a_c, a_p, e_emax


def save_patient_results(patient_id, results, model, observations_dict=None,
                        save_graph=True, save_res=True, output_subdir='linear_no_lag'):
    """Save simulation results (graphs and numpy arrays) for a patient.

    Args:
        patient_id: Patient ID
        results: Tuple of (time, Ad, Ac, Ap, bp_emax)
        model: NorepinephrinePKPD instance
        observations_dict: Dictionary of observations for plotting
        save_graph: Whether to save plots
        save_res: Whether to save numpy arrays
        output_subdir: Subdirectory name (e.g., 'linear_no_lag', 'power_no_lag')
    """
    if not save_graph and not save_res:
        return

    output_dir = f'results/patient_{patient_id}/pkpd/{output_subdir}'
    os.makedirs(output_dir, exist_ok=True)
    time, a_d, a_c, a_p, bp_emax = results

    if save_res:
        np.save(f'{output_dir}/time.npy', time)
        np.save(f'{output_dir}/Ad.npy', a_d)
        np.save(f'{output_dir}/Ac.npy', a_c)
        np.save(f'{output_dir}/Ap.npy', a_p)
        np.save(f'{output_dir}/bp_emax.npy', bp_emax)

    if save_graph:
        plt.figure(figsize=(10, 6))
        plt.plot(time, bp_emax, 'b-', label='Emax (Simulated)')
        
        if observations_dict and patient_id in observations_dict:
            bp_obs = observations_dict[patient_id]['blood_pressure']
            if bp_obs:
                obs_times, obs_values = zip(*bp_obs)
                plt.scatter(obs_times, obs_values, c='red', s=30, label='Measured Data Points', zorder=5)
        plt.xlabel('Time (s)')
        plt.ylabel('MAP (mmHg)')
        plt.title(f'Patient {patient_id} - Blood Pressure')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_dir}/blood_pressure_evol.png')
        plt.close()

        c_c = a_c / model.V_c
        plt.figure(figsize=(10, 6))
        plt.plot(time, c_c, 'b-', label='Simulated NOR Concentration')
        if observations_dict and patient_id in observations_dict:
            conc_obs = observations_dict[patient_id]['concentration']
            if conc_obs:
                obs_times, obs_values = zip(*conc_obs)
                plt.scatter(obs_times, obs_values, c='red', s=30, label='Measured Data Points', zorder=5)
        plt.xlabel('Time (s)')
        plt.ylabel('NOR Concentration (nmol/L)')
        plt.title(f'Patient {patient_id} - NOR Plasma Concentration')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_dir}/nor_conc_evol.png')
        plt.close()


if __name__ == "__main__":
    # Patients to simulate (empty list = all patients from observation dataset)
    PATIENTS = 'all'  # int for one patient, list for multiple patients or 'all' for every patients

    # Output control
    SAVE_GRAPHS = True
    SAVE_NUMPY_RESULTS = True
    OUTPUT_SUBDIR = 'standalone'

    print("\n" + "="*70)
    print("PKPD SIMULATION - NOREPINEPHRINE MODEL")
    print("="*70)

    print("\nLoading observation data...")
    patients_ids = get_patient_ids(PATIENTS)
    observations = load_observations(patients_ids)
    print(f"  ✓ Loaded observations for {len(observations)} patients")

    print("\nLoading injection protocols...")
    injections = load_injections(patients_ids)

    pkpd_model = NorepinephrinePKPD(injections)

    print("\n" + "-"*70)
    print("SIMULATION META-PARAMETERS")
    print("-"*70)
    print(f"  Save graphs: {SAVE_GRAPHS}")
    print(f"  Save numpy arrays: {SAVE_NUMPY_RESULTS}")
    print(f"  Output directory: results/patient_<id>/pkpd/{OUTPUT_SUBDIR}/")
    print("\n  PKPD Model Parameters:")
    print(f"    PK: C_endo={pkpd_model.C_endo}, k_a={pkpd_model.k_a}, V_c={pkpd_model.V_c}, k_12={pkpd_model.k_12}, k_21={pkpd_model.k_21}, k_el={pkpd_model.k_el}")
    print(f"    PD Emax: E_0={pkpd_model.E_0}, E_max={pkpd_model.E_max}, EC_50={pkpd_model.EC_50}")
    print("-"*70)

    print("\n" + "="*70)
    print(f"STARTING PKPD SIMULATION FOR {len(patients_ids)}")
    print("="*70 + "\n")

    # Simulate each patient
    for index, p_id in enumerate(patients_ids):
        res = pkpd_model.simulate(p_id, t_end=2200, dt=0.5)

        # Save results
        save_patient_results(
            p_id,
            res,
            pkpd_model,
            observations,
            save_graph=SAVE_GRAPHS,
            save_res=SAVE_NUMPY_RESULTS,
            output_subdir=OUTPUT_SUBDIR
        )

        print(f"Patient {p_id} processed !")

    print("\n" + "="*70)
    print("SIMULATION COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"  Results saved in: results/patient_<id>/pkpd/{OUTPUT_SUBDIR}/")
    if SAVE_GRAPHS:
        print("    - Graphs: blood_pressure_evol.png, nor_conc_evol.png")
    if SAVE_NUMPY_RESULTS:
        print("    - Arrays: time.npy, Ad.npy, Ac.npy, Ap.npy, bp_emax.npy")
    print("="*70 + "\n")
