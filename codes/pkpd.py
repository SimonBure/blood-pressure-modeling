import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd
import os


# ============================================================================
# PKPD MODEL CLASS
# ============================================================================

class NorepinephrinePKPD:

    def __init__(self, injections_dict=None, is_linear=True):
        self.injections_dict = injections_dict if injections_dict is not None else {}

        # PK model parameters
        self.C_endo = 0.81
        self.k_a = 0.02
        self.V_c = 0.49
        self.k_12 = 0.06
        self.k_21 = 0.04
        self.k_el = 0.05
        self.gamma = 1.0 if is_linear else 1.46
        self.beta = 1.0 if is_linear else 1.21

        # PD model -- Emax equation
        self.E_0 = 57.09
        self.E_max = 113.52
        self.EC_50 = 15.7

        # PD model -- Windkessel equation
        self.omega = 1.01
        self.zeta = 19.44
        self.nu = 2.12

        # Initial conditions
        self.Ad_0 = 0.0
        self.Ac_0 = 0.0
        self.Ap_0 = 0.0
        self.E_0_init = 57.09
        self.dEdt_0 = 0.0

    def INOR(self, t, patient_id):
        if patient_id not in self.injections_dict:
            return 0.0

        times, amounts, durations = self.injections_dict[patient_id]

        for i in range(len(times)):
            t_start = times[i]
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

    def euler_implicit_pk(self, y_n, t_n, dt, patient_id):
        t_np1 = t_n + dt
        def residual(y_np1):
            return y_np1 - y_n - dt * self.pk_rhs(t_np1, y_np1, patient_id)
        y_np1 = fsolve(residual, y_n)
        return y_np1

    def compute_concentration(self, Ac):
        return self.C_endo + Ac / self.V_c

    def pd_emax(self, Cc):
        return self.E_0 + (self.E_max - self.E_0) * Cc / (Cc + self.EC_50)

    def pd_windkessel_rhs(self, t, y_pd, Cc):
        E, V = y_pd
        dE_dt = V
        dV_dt = self.nu * Cc - 2 * self.zeta * self.omega * V - self.omega**2 * E
        return np.array([dE_dt, dV_dt])

    def euler_implicit_windkessel(self, y_pd_n, t_n, dt, Cc_np1):
        t_np1 = t_n + dt
        def residual(y_pd_np1):
            return y_pd_np1 - y_pd_n - dt * self.pd_windkessel_rhs(t_np1, y_pd_np1, Cc_np1)
        y_pd_np1 = fsolve(residual, y_pd_n)
        return y_pd_np1

    def simulate(self, patient_id, t_end=2200, dt=0.5):
        t = np.arange(0, t_end + dt, dt)
        n_steps = len(t)

        Ad = np.zeros(n_steps)
        Ac = np.zeros(n_steps)
        Ap = np.zeros(n_steps)
        Cc = np.zeros(n_steps)
        E_emax = np.zeros(n_steps)
        E_windkessel = np.zeros(n_steps)

        Ad[0] = self.Ad_0
        Ac[0] = self.Ac_0
        Ap[0] = self.Ap_0
        Cc[0] = self.compute_concentration(Ac[0])

        E_emax[0] = self.pd_emax(Cc[0])
        y_pd = np.array([self.E_0_init, self.dEdt_0])
        E_windkessel[0] = y_pd[0]

        for i in range(n_steps - 1):
            y_pk = np.array([Ad[i], Ac[i], Ap[i]])
            y_pk_new = self.euler_implicit_pk(y_pk, t[i], dt, patient_id)

            Ad[i+1] = y_pk_new[0]
            Ac[i+1] = y_pk_new[1]
            Ap[i+1] = y_pk_new[2]

            Cc[i+1] = self.compute_concentration(Ac[i+1])
            E_emax[i+1] = self.pd_emax(Cc[i+1])

            y_pd = self.euler_implicit_windkessel(y_pd, t[i], dt, Cc[i+1])
            E_windkessel[i+1] = y_pd[0]

        return t, Ad, Ac, Ap, E_emax, E_windkessel


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_observations(patient_ids=None, csv_path='./codes/data/joachim.csv'):
    """Load observation data for specified patients.

    Args:
        patient_ids: List of patient IDs to load, or None/empty list for all patients
        csv_path: Path to CSV file with observations

    Returns:
        Dictionary with patient observations (concentration and blood pressure)
    """
    df = pd.read_csv(csv_path)
    available_patients = sorted(df['id'].unique())

    # If no patients specified, load all
    if not patient_ids:
        patient_ids = available_patients
    else:
        # Validate that requested patients exist in the dataset
        invalid_patients = [pid for pid in patient_ids if pid not in available_patients]
        if invalid_patients:
            print(f"\n  ERROR: The following patient IDs do not exist in observations dataset:")
            print(f"    Requested: {invalid_patients}")
            print(f"    Available: {available_patients}")
            exit(1)

    df_obs = df[df['obs'] != '.'].copy()
    df_obs['obs'] = pd.to_numeric(df_obs['obs'])
    df_obs['obsid'] = pd.to_numeric(df_obs['obsid'])

    observations_dict = {}
    for pid in patient_ids:
        patient_obs = df_obs[df_obs['id'] == pid]
        conc_obs = patient_obs[patient_obs['obsid'] == 0]
        bp_obs = patient_obs[patient_obs['obsid'] == 1]
        observations_dict[pid] = {
            'concentration': list(zip(conc_obs['time(s)'].values, conc_obs['obs'].values)),
            'blood_pressure': list(zip(bp_obs['time(s)'].values, bp_obs['obs'].values))
        }
    return observations_dict


def load_injections(patient_ids, csv_path='./codes/data/injections.csv'):
    """Load injection protocols for specified patients.

    Args:
        patient_ids: List of patient IDs to load injection data for
        csv_path: Path to CSV file with injection data

    Returns:
        Dictionary mapping patient_id to (times, amounts, durations)
    """
    inj_df = pd.read_csv(csv_path)
    injections_dict = {}

    for pid in patient_ids:
        p_inj = inj_df[inj_df['patient_id'] == pid].sort_values('injection_time_s')
        if len(p_inj) > 0:
            injections_dict[pid] = (
                p_inj['injection_time_s'].values,
                p_inj['amount_nmol'].values,
                p_inj['duration_s'].values
            )
        else:
            # Patient has no injection data, use empty arrays
            injections_dict[pid] = (
                np.array([]),
                np.array([]),
                np.array([])
            )
    return injections_dict


def save_patient_results(patient_id, results, model, observations_dict=None,
                        save_graph=True, save_res=True, output_subdir='linear_no_lag'):
    """Save simulation results (graphs and numpy arrays) for a patient.

    Args:
        patient_id: Patient ID
        results: Tuple of (time, Ad, Ac, Ap, bp_emax, bp_windkessel)
        model: NorepinephrinePKPD instance
        observations_dict: Dictionary of observations for plotting
        save_graph: Whether to save plots
        save_res: Whether to save numpy arrays
        output_subdir: Subdirectory name (e.g., 'linear_no_lag', 'power_no_lag')
    """
    if not save_graph and not save_res:
        return

    output_dir = f'codes/res/patient_{patient_id}/{output_subdir}'
    os.makedirs(output_dir, exist_ok=True)
    time, Ad, Ac, Ap, bp_emax, bp_windkessel = results

    if save_res:
        np.save(f'{output_dir}/time.npy', time)
        np.save(f'{output_dir}/Ad.npy', Ad)
        np.save(f'{output_dir}/Ac.npy', Ac)
        np.save(f'{output_dir}/Ap.npy', Ap)
        np.save(f'{output_dir}/bp_emax.npy', bp_emax)
        np.save(f'{output_dir}/bp_windkessel.npy', bp_windkessel)

    if save_graph:
        plt.figure(figsize=(10, 6))
        plt.plot(time, bp_emax, 'b-', label='Emax (Simulated)')
        plt.plot(time, bp_windkessel, color='orange', linestyle='--', label='Windkessel (Simulated)')
        if observations_dict and patient_id in observations_dict:
            bp_obs = observations_dict[patient_id]['blood_pressure']
            if bp_obs:
                obs_times, obs_values = zip(*bp_obs)
                plt.scatter(obs_times, obs_values, c='red', marker='o', s=30, label='Measured Data Points', zorder=5)
        plt.xlabel('Time (s)')
        plt.ylabel('MAP (mmHg)')
        plt.title(f'Patient {patient_id} - Blood Pressure')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_dir}/blood_pressure_evol.png')
        plt.close()

        Cc = Ac / model.V_c
        plt.figure(figsize=(10, 6))
        plt.plot(time, Cc, 'b-', label='Simulated NOR Concentration')
        if observations_dict and patient_id in observations_dict:
            conc_obs = observations_dict[patient_id]['concentration']
            if conc_obs:
                obs_times, obs_values = zip(*conc_obs)
                plt.scatter(obs_times, obs_values, c='red', marker='o', s=30, label='Measured Data Points', zorder=5)
        plt.xlabel('Time (s)')
        plt.ylabel('NOR Concentration (nmol/L)')
        plt.title(f'Patient {patient_id} - NOR Plasma Concentration')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_dir}/nor_conc_evol.png')
        plt.close()


def decide_simulation_token(i):
    """Return rotating progress indicator."""
    if i % 4 == 0:
        return '|'
    elif i % 4 == 1:
        return '/'
    elif i % 4 == 2:
        return '-'
    else:
        return '\\'


# ============================================================================
# META-PARAMETERS DEFINITION
# ============================================================================

# Patients to simulate (empty list = all patients from observation dataset)
patients = []  # Example: [23, 20, 15] or [] for all

# Model type: linear (exponents = 1.0) or non-linear (exponents != 1.0)
is_linear = False

# Output control
save_graphs = True
save_numpy_results = True
output_subdirectory = 'linear_no_lag' if is_linear else 'power_no_lag'

# ============================================================================
# PREPROCESSING & LOADINGS
# ============================================================================

print("\n" + "="*70)
print("PKPD SIMULATION - NOREPINEPHRINE MODEL")
print("="*70)

# Load observation data (primary data source)
print("\nLoading observation data...")
if not patients:
    print(f"  → No patients specified, loading all patients from dataset")
else:
    print(f"  → Loading {len(patients)} specified patient(s): {patients}")

observations_dict = load_observations(patients)
patients = [int(p) for p in sorted(observations_dict.keys())]  # Update patients list with loaded IDs (convert to int)
print(f"  ✓ Loaded observations for {len(observations_dict)} patients")

# Load injection data for the patients with observations
print("\nLoading injection protocols...")
injections_dict = load_injections(patients)
patients_with_injections = [pid for pid in patients if len(injections_dict[pid][0]) > 0]
patients_without_injections = [pid for pid in patients if len(injections_dict[pid][0]) == 0]
print(f"  ✓ Loaded injection data for {len(patients_with_injections)} patients")
if patients_without_injections:
    print(f"  ⚠ Warning: {len(patients_without_injections)} patient(s) have no injection data: {patients_without_injections}")

pkpd_model = NorepinephrinePKPD(injections_dict, is_linear=is_linear)

np.set_printoptions(precision=1)

print("\n" + "-"*70)
print("SIMULATION META-PARAMETERS")
print("-"*70)
print(f"  Model type: {'Linear (gamma=1.0, beta=1.0)' if is_linear else f'Non-linear (gamma={pkpd_model.gamma}, beta={pkpd_model.beta})'}")
print(f"  Number of patients: {len(patients)}")
print(f"  Patient IDs: {patients}")
print(f"  Save graphs: {save_graphs}")
print(f"  Save numpy arrays: {save_numpy_results}")
print(f"  Output directory: codes/res/patient_<id>/{output_subdirectory}/")
print("\n  PKPD Model Parameters:")
print(f"    PK: C_endo={pkpd_model.C_endo}, k_a={pkpd_model.k_a}, V_c={pkpd_model.V_c}, k_12={pkpd_model.k_12}, k_21={pkpd_model.k_21}, k_el={pkpd_model.k_el}")
print(f"    PK exponents: gamma={pkpd_model.gamma}, beta={pkpd_model.beta}")
print(f"    PD Emax: E_0={pkpd_model.E_0}, E_max={pkpd_model.E_max}, EC_50={pkpd_model.EC_50}")
print(f"    PD Windkessel: omega={pkpd_model.omega}, zeta={pkpd_model.zeta}, nu={pkpd_model.nu}")
print("-"*70)


# ============================================================================
# SIMULATION
# ============================================================================

print("\n" + "="*70)
print(f"STARTING SIMULATION FOR {len(patients)} PATIENT(S)")
print("="*70 + "\n")

# Simulate each patient
for i, patient_id in enumerate(patients):
    # Progress indicator
    progress_token = decide_simulation_token(i)
    print(f"{progress_token} Patient {patient_id:3d} ", end='', flush=True)

    # Run simulation
    results = pkpd_model.simulate(patient_id, t_end=2200, dt=0.5)

    # Save results
    save_patient_results(
        patient_id,
        results,
        pkpd_model,
        observations_dict,
        save_graph=save_graphs,
        save_res=save_numpy_results,
        output_subdir=output_subdirectory
    )

    print("✓")


# ============================================================================
# COMPLETION
# ============================================================================

print("\n" + "="*70)
print("SIMULATION COMPLETED SUCCESSFULLY")
print("="*70)
print(f"  Results saved in: codes/res/patient_<id>/{output_subdirectory}/")
if save_graphs:
    print(f"    - Graphs: blood_pressure_evol.png, nor_conc_evol.png")
if save_numpy_results:
    print(f"    - Arrays: time.npy, Ad.npy, Ac.npy, Ap.npy, bp_emax.npy, bp_windkessel.npy")
print("="*70 + "\n")
