'''
Code to attain the pre-ML AM1-DFT MAE metrics for every dataset.

Usage:

    python pre_ml.py

Saves the tabulated pre-ml metrics for every dataset in one CSV file.

INFORMATION:
Below is tracking for what each distortion energy equals for each dataset.
                Diels-Alder     [3+2]           Malonate        Nitroamine
distortion 1    Diene           Dipole          Nucleophile     Nucleophile
distortion 2    Dienophile      Dipolarophile   MA              MA 

'''

# Imports
import os
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define what the distortion energies are so that it can be stored in one dataframe.
dist_d = {
    'da_endo':{'dist_1_am1':'distortion_energy_di_am1', 'dist_2_am1':'distortion_energy_dp_am1', 'dist_1_dft':'distortion_energy_di_dft', 'dist_2_dft':'distortion_energy_dp_dft'},
    'da_exo':{'dist_1_am1':'distortion_energy_di_am1', 'dist_2_am1':'distortion_energy_dp_am1', 'dist_1_dft':'distortion_energy_di_dft', 'dist_2_dft':'distortion_energy_dp_dft'},
    'cyalk_da':{'dist_1_am1':'distortion_energy_di_am1', 'dist_2_am1':'distortion_energy_dp_am1', 'dist_1_dft':'distortion_energy_di_dft', 'dist_2_dft':'distortion_energy_dp_dft'},
    'cypro_da':{'dist_1_am1':'distortion_energy_di_am1', 'dist_2_am1':'distortion_energy_dp_am1', 'dist_1_dft':'distortion_energy_di_dft', 'dist_2_dft':'distortion_energy_dp_dft'},
    'ma':{'dist_1_am1':'distortion_energy_0_am1', 'dist_2_am1':'distortion_energy_1_am1', 'dist_1_dft':'distortion_energy_0_dft', 'dist_2_dft':'distortion_energy_1_dft'},
    'mal':{'dist_1_am1':'distortion_energy_0_am1', 'dist_2_am1':'distortion_energy_1_am1', 'dist_1_dft':'distortion_energy_0_dft', 'dist_2_dft':'distortion_energy_1_dft'},
    'tt_solvent':{'dist_1_am1':'distortion_energy_1_am1', 'dist_2_am1':'distortion_energy_2_am1', 'dist_1_dft':'distortion_energy_1_dft', 'dist_2_dft':'distortion_energy_2_dft'},
}
# Load in data from files
datasets = [x[0] for x in os.walk('_f_selection')][1:]
files = {}
for d in datasets:
    for file in os.listdir(d):
        if file.startswith('manual') and 'removed' not in file:
            if 'la_da' in file or 'da_exo' in file:
                pass
            else:
                files[file.split('manual_')[-1].split('.pkl')[0]] = (f'{d}\{file}')

metrics = {'q_barrier':{}, 'e_barrier':{}, 'interaction_energy':{}, 'distortion_energy_1':{}, 'distortion_energy_2':{}}
# Calculate MAEs
for d in files.keys():
    x = pd.read_pickle(files[d])
    dist_cols = [col for col in x.columns if col.startswith('distortion')]
    metrics['distortion_energy_1'][d] = np.round(mean_absolute_error(x[dist_d[d]['dist_1_dft']], x[dist_d[d]['dist_1_am1']]), 2)
    metrics['distortion_energy_2'][d] = np.round(mean_absolute_error(x[dist_d[d]['dist_2_dft']], x[dist_d[d]['dist_2_am1']]), 2)
    metrics['q_barrier'][d] = np.round(mean_absolute_error(x['q_barrier_dft'], x['q_barrier_am1']), 2)
    metrics['e_barrier'][d] = np.round(mean_absolute_error(x['e_barrier_dft'], x['e_barrier_am1']), 2)
    metrics['interaction_energy'][d] = np.round(mean_absolute_error(x['interaction_energies_dft'], x['interaction_energies_am1']), 2)

# Save MAE results to a .csv file.
df = pd.DataFrame(metrics)
print(df)
df.to_csv('pre_ml_mae.csv')

# Calculate RMSEs
for d in files.keys():
    x = pd.read_pickle(files[d])
    dist_cols = [col for col in x.columns if col.startswith('distortion')]
    metrics['distortion_energy_1'][d] = np.round(math.sqrt(mean_squared_error(x[dist_d[d]['dist_1_dft']], x[dist_d[d]['dist_1_am1']])), 2)
    metrics['distortion_energy_2'][d] = np.round(math.sqrt(mean_squared_error(x[dist_d[d]['dist_2_dft']], x[dist_d[d]['dist_2_am1']])), 2)
    metrics['q_barrier'][d] = np.round(math.sqrt(mean_squared_error(x['q_barrier_dft'], x['q_barrier_am1'])), 2)
    metrics['e_barrier'][d] = np.round(math.sqrt(mean_squared_error(x['e_barrier_dft'], x['e_barrier_am1'])), 2)
    metrics['interaction_energy'][d] = np.round(math.sqrt(mean_squared_error(x['interaction_energies_dft'], x['interaction_energies_am1'])), 2)

# Save MAE results to a .csv file.
df = pd.DataFrame(metrics)
print(df)
df.to_csv('pre_ml_rmse.csv')