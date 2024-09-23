"""
#########################################################################################################################
#                                                   ml_tabulate.py                                                      #
#                                                                                                                       #
#                        This code takes the ML results for a given dataset and will tabulate the                       #
#                        results and save them in a .csv file for ease of reading.                                      #
#                                                                                                                       #
#                                                                                                                       #
#            Usage:                                                                                                     #
#                                                                                                                       #
#                 python ml_tabulate.py path/to/ml_results.pkl                                                          #
#                                                                                                                       #
#                                                                                                                       #
#########################################################################################################################
#                                                      Authors                                                          #
#                                                                                                                       #
#------------------------------------------------  Samuel G. Espley ----------------------------------------------------#
#                                                                                                                       #
#########################################################################################################################
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
pd.options.mode.chained_assignment = None  # default='warn'

def combine(l1, l2):
    '''
    Function to be used with .apply for dataframes. Combines two values into a string.
    '''
    value = f'{l1} ± {l2}'
    return value

def _ave_map(l):
    '''
    Function to be used with .apply for dataframes. Creates the average from a list.
    '''
    try:        
        value = '{:.2f}'.format(np.round(sum(l)/len(l), 2))
        return value
    except:
        return '---'
    
def _model_target(l):    
    '''
    Function to be used with .apply for dataframes. Gets the model target and model type.
    '''
    model = model_d[l.split('_')[0]]
    target = l.split('_', 1)[-1].replace('st_nn_', '')
    return model, target

# Check valid ml_results.pkl file has been passed.
if len(sys.argv) == 2:
    if 'ml_results.pkl' not in sys.argv[-1]:
        print('Pass a valid file (ml_results.pkl).')
        exit()
    else:
        df = pd.read_pickle(sys.argv[-1])
else:
    print('Pass a valid file (ml_results.pkl).')
    exit()

# Remove model architecture.
df = df.drop(columns=['model'])

# Define model name dictionary.
model_d = {
    'ridge':'Ridge',
    'krr':'KRR',
    'svr':'SVR',
    '2':'2 Layer NN',
    '4':'4 Layer NN',
    'rf':'Random Forest'
}
# Get the path to save files to later.
path = sys.argv[-1].split('ml_results.pkl')[0]
# Attain 100% training data metrics.
standard_metrics = df[['model_target', 'y_val_pred', 'std_error_val', 'y_test_pred', 'std_error_test', 'y_train_pred', 'std_error_train', 'y_train_CV']]
metrics = {}
# Iterate through rows and save calues to metrics dictionary.
for index, row in standard_metrics.iterrows():
    model = model_d[row['model_target'].split('_')[0]]
    target = row['model_target'].split('_', 1)[-1].replace('st_nn_', '')
    val_ave = '{:.2f}'.format(np.round(sum(row['y_val_pred'])/len(row['y_val_pred']), 2))
    std_val_ave = '{:.2f}'.format(np.round(sum(row['std_error_val'])/len(row['std_error_val']), 2))
    test_ave = '{:.2f}'.format(np.round(sum(row['y_test_pred'])/len(row['y_test_pred']), 2))
    std_test_ave = '{:.2f}'.format(np.round(sum(row['std_error_test'])/len(row['std_error_test']), 2))
    try:
        train_ave = '{:.2f}'.format(np.round(sum(row['y_train_pred'])/len(row['y_train_pred']), 2))
        std_train_ave = '{:.2f}'.format(np.round(sum(row['std_error_train'])/len(row['std_error_train']), 2))
    except:
        train_ave = '{:.2f}'.format(np.round(sum(row['y_train_CV'])/len(row['y_train_CV']), 2))
        std_train_ave = '---'
    metrics[row['model_target']] = {
        'model':model,
        'target':target,
        'val':f'{val_ave} ± {std_val_ave}',
        'test':f'{test_ave} ± {std_test_ave}',
        'train':f'{train_ave} ± {std_train_ave}'
    }
# Build DataFrame and save.
metric_df = pd.DataFrame(metrics).transpose()
metric_df = metric_df.replace({'nan ± nan': '-'}, regex=True)
metric_df = metric_df.replace({' ± ---': ''}, regex=True)
# print(metric_df)
metric_df.to_csv(f'{path}cleaned_results.csv', index=False)

# Learning curve metrics.
splits = {'test':['model_target'], 'train':['model_target'], 'val':['model_target']}
for col in df.columns:
    if 'values' in col or 'true' in col or 'scores' in col:
        pass
    else:
        try:
            float(col.split('_')[0])
            # splits.append(col)
            for sets in splits.keys():
                if sets in col:
                    splits[sets].append(col)
                else:
                    pass
        except:
            pass

# Get each individual dataset.
test_df = df[splits['test']]
train_df = df[splits['train']]
val_df = df[splits['val']]

# Get test set metrics.
for column in test_df.columns:
        if column == 'model_target':
            pass
        else: 
            test_df[f'{column}_ave'] = test_df[column].apply(_ave_map)
            test_df = test_df.drop(columns=[column])
# Loop through each split.
for split in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    mae = f'{split}_y_test_pred_ave'
    std = f'{split}_std_error_test_ave'
    test_df[f'{split}_test'] = test_df.apply(lambda x: combine(x[mae], x[std]), axis=1)
    test_df = test_df.drop(columns=[mae, std])
# Get targets and models.
test_df['model'], test_df['target'] = zip(*test_df['model_target'].apply(_model_target))
test_df = test_df.reindex(sorted(test_df.columns), axis=1)

# Get val set metrics.
for column in val_df.columns:
        if column == 'model_target':
            pass
        else: 
            val_df[f'{column}_ave'] = val_df[column].apply(_ave_map)
            val_df = val_df.drop(columns=[column])

# Loop through each split.
for split in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    mae = f'{split}_y_val_pred_ave'
    std = f'{split}_std_error_val_ave'
    val_df[f'{split}_val'] = val_df.apply(lambda x: combine(x[mae], x[std]), axis=1)
    val_df = val_df.drop(columns=[mae, std])
# Get targets and models.
val_df['model'], val_df['target'] = zip(*val_df['model_target'].apply(_model_target))
val_df = val_df.reindex(sorted(val_df.columns), axis=1)

# Get train set metrics.
for column in train_df.columns:
        if column == 'model_target':
            pass
        else: 
            train_df[f'{column}_ave'] = train_df[column].apply(_ave_map)
            train_df = train_df.drop(columns=[column])

# Loop through each split.
for split in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    mae = f'{split}_y_train_CV_ave'
    train_df = train_df.rename(columns={mae:f'{split}_train'})
# Get targets and models.
train_df['model'], train_df['target'] = zip(*train_df['model_target'].apply(_model_target))
train_df = train_df.reindex(sorted(train_df.columns), axis=1)

# Combine training splits. 
lc_df = pd.merge(train_df, val_df, on=['model', 'model_target', 'target'])
lc_df = pd.merge(lc_df, test_df, on=['model', 'model_target', 'target'])
lc_df = lc_df.replace({'--- ± ---': '---'}, regex=True)

# Save results.
lc_df.to_csv(f'{path}lc_results.csv', index=False)

