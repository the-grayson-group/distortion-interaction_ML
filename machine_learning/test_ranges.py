'''
Code to get the test set ranges from a ml_results.pkl file.

'''

import numpy as np
import pandas as pd

def _get_range(l):
    
    try:
        maxs = sum([max(l[0]), max(l[1]), max(l[2]), max(l[3]), max(l[4])])/len(l)
        mins = sum([min(l[0]), min(l[1]), min(l[2]), min(l[3]), min(l[4])])/len(l)
        return f'{np.round(mins, 2)} - {np.round(maxs, 2)}'
    except:
        return '0 - 0'


df = pd.read_pickle('ml_results.pkl')[['model_target', 'y_test_true']]
df['ranges'] = df['y_test_true'].apply(_get_range)

print(df)