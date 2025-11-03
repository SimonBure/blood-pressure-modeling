import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd
import os
import sys


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


def load_observations(patient_ids, csv_path='./codes/data/joachim.csv'):
    df = pd.read_csv(csv_path)
    print(f"Total patients in joachim.csv = {len(df['id'].unique())}")
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


def save_patient_results(patient_id, results, model, observations_dict=None, save_graph=True, save_res=True):
    if not save_graph and not save_res:
        return

    output_dir = f'res/patient_{patient_id}'
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


def load_injections() -> dict:
    inj_df = pd.read_csv('./codes/data/injections.csv')
    injections_dict = {}

    for pid in inj_df['patient_id'].unique():
        p_inj = inj_df[inj_df['patient_id'] == pid].sort_values('injection_time_s')
        injections_dict[pid] = (
            p_inj['injection_time_s'].values,
            p_inj['amount_nmol'].values,
            p_inj['duration_s'].values
        )
    return injections_dict

def check_patient_id(patient_ids: list, injections_dict: dict):
    for pid in patient_ids:
        if pid not in injections_dict:
            print(f"Erreur: Patient ID {pid} n'existe pas dans injections_dict")
            print(f"IDs disponibles: {sorted(injections_dict.keys())}")
            sys.exit(1)
            
def decide_simulation_token(i: int):
    if i % 4 == 0:
        simulation_token = '|'
    elif i % 4 == 1:
        simulation_token = '/'
    elif i % 4 == 2:
        simulation_token = '-'
    else:
        simulation_token = '\\'
    return simulation_token
    
def main():
    import argparse

    parser = argparse.ArgumentParser(description='Simulation PK/PD Norepinephrine')
    parser.add_argument('patient_ids', nargs='*', type=int, help='Liste des IDs patients (vide = tous les patients)')
    parser.add_argument('--no-graph', action='store_true', help='Ne pas sauvegarder les graphiques')
    parser.add_argument('--no-res', action='store_true', help='Ne pas sauvegarder les résultats numpy')
    args = parser.parse_args()

    injections_dict = load_injections()

    # If no ID provided by CLI --> loop over all existing patients
    if args.patient_ids:
        patient_list = args.patient_ids
        check_patient_id(patient_list, injections_dict)
        
    else:
        patient_list = sorted(injections_dict.keys())

    observations_dict = load_observations(patient_list)  # only needed obs
    linear_model = NorepinephrinePKPD(injections_dict, is_linear=True)
    

    print(f"Simulation de {len(patient_list)} patient(s)...")
    for i, patient_id in enumerate(patient_list):
        simulation_token = decide_simulation_token(i)
        
        print(f"{simulation_token} Patient {patient_id}")
        
        results = linear_model.simulate(patient_id)
        save_patient_results(patient_id, results, linear_model, observations_dict, not args.no_graph, not args.no_res)
    print("\nSimulations terminées!")


if __name__ == "__main__":
    main()
