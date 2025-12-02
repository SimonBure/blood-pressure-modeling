import pandas as pd

df = pd.read_csv('./codes/data/joachim.csv')

df_injections = df[df['amount'].notna() & (df['amount'] != '.')].copy()

df_injections['injection_time_s'] = df_injections['time(s)']
df_injections['amount_nmol'] = pd.to_numeric(df_injections['amount'])
df_injections['duration_s'] = pd.to_numeric(df_injections['tinf'])

output_df = df_injections[['id', 'injection_time_s', 'amount_nmol', 'duration_s']].rename(columns={'id': 'patient_id'})

unique_patients = output_df['patient_id'].nunique()
print(f'Total unique patients: {unique_patients}')

output_df.to_csv('injections.csv', index=False)
print(f'Saved injections.csv with {len(output_df)} injection events')
