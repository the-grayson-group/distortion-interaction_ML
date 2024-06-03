"""
#####################################################################################################################################
#                                                        ml_rebuild.py                                                              #
#                                                                                                                                   #
#                        This code takes the dataset and the tuned hyperparameters and will run ML                                  #
#                        analysis on one random state. This code also takes in an external test set                                 #
#                        that will be evaluated if the features can be matched correctly.                                           #
#                                                                                                                                   #
#   Usage:                                                                                                                          #
#                                                                                                                                   #
#     python ml_rebuild.py 23 path/to/first/dataset.pkl path/to/second/dataset.pkl path/to/hps.pkl                                  #
#                                                                                                                                   #                    
#     for %x in (22 23 14 1 2) do python ml_rebuild.py %x path/to/first/dataset.pkl path/to/second/dataset.pkl path/to/hps.pkl      #  
#                                                                                                                                   #                                                                                                                     #
#                        The first dataset is what the model is trained upon, with the second dataset                               #
#                        being the external test set that you want to evaluate.                                                     #
#                                                                                                                                   #
#####################################################################################################################################
#                                                           Authors                                                                 #
#                                                                                                                                   #
#-----------------------------------------------------  Samuel G. Espley -----------------------------------------------------------#
#                                                                                                                                   #
#####################################################################################################################################
"""

# Standard imports
import os
import sys
import time
import shutil
import pickle
import inspect
import logging
import inquirer
import warnings
import datetime
import itertools
import numpy as np
import pandas as pd
from numpy.random import seed
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Remove build information associated with tensorflow
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
# TensorFlow imports
import tensorflow as tf
from tensorflow import random
from tensorflow import keras
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping
from keras.activations import elu, relu, selu
from keras import layers, Sequential, models, backend as K
# sklearn imports
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

# Set random seed.
try:
    seed(int(sys.argv[-4]))
    random.set_seed(int(sys.argv[-4]))
    tf.keras.utils.set_random_seed(int(sys.argv[-4]))
except:
    print('Invalid seed passed - argument formats as:\npython ml_analysis.py   random_seed    path/to/dataset.pkl    path/to/hps.pkl')
    exit()

def timeit(func):
    '''
    Function to time a given function utilising wrapper capabilities.
    '''
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f'Function {func.__name__} took {total_time:.4f} seconds.\n---')
        #print(f'Function \033[1m{func.__name__}\033[0m took \033[92m{total_time:.4f}\033[0m seconds')
        return result
    return timeit_wrapper

class General():

    def __init__(self):
        
        # Pull out path to dataset.
        if len(sys.argv) <= 4:
            logger.info('Please provide path to datasets and hyperparameter file.')
            print('Error - check log.')
            exit()
        elif len(sys.argv) == 5:
            path = sys.argv[-3]
            test_path = sys.argv[-2]
            hps_path = sys.argv[-1]
            if not path.endswith('pkl'):
                logger.info('Please provide a valid path to a dataset.')
                print('Error - check log.')
                exit()
            elif not hps_path.endswith('pkl'):
                logger.info('Please provide a valid path to a hyperparameter file.')
                print('Error - check log.')
                exit()

        # Store random seed.
        self.randomseed = int(sys.argv[-4])

        # Load in hps file.
        with open(hps_path, 'rb') as f:
            hps = pickle.load(f)

        # Load in file from path above - 
        df = pd.read_pickle(path)
        d = path.split('/')[-1].split('_')[0:2]
        if d[1].endswith('.pkl') == True:
            d[1] = d[1].split('.')[0]
        logger.info(f'Dataset: {d[1]} \nFeature Selection Method: {d[0]}\n---')

        # Load in external test set and store dataframes.
        if test_path == 'None':
            self.test_df == None
        else:
            self.test_df = pd.read_pickle(test_path)  
            self.test_df = self.test_df[df.columns] 
        rxn_number = None
        self.df = df
        self.rxn_number = rxn_number
        self.hps = hps

    def _perform_train_test_split(X, y):
        '''
        Function to run the train test split. 
        80:10:10 train:validation:test

        Arguments:
        X (DataFrame): A DataFrame containing the X values.
        y (DataFrame): A DataFrame containing the y values.

        Returns:
        split_dict (Dictionary): A dictionary containing DataFrames of all the different train:validation:test splits.
        '''
        # Perform the train test splits and store in split_dict.
        X = X[X['reaction_number'] != 365]
        X = X[X['reaction_number'] != 366]
        X = X[X['reaction_number'] != 368]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=int(sys.argv[-4]))
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=int(sys.argv[-4]))
        split_dict = {'X_train': X_train, 'X_val':X_val, 'X_test':X_test,
                      'y_train': y_train, 'y_val':y_val, 'y_test':y_test}
        # Log shape information about the different splits.
        logger.info('Train/Test/Validation Split Shapes:')
        logger.info('X Train: '+str(X_train.shape)+
                    '\nX Test: '+str(X_test.shape)+
                    '\nX Validation: '+str(X_val.shape))
        logger.info('y Train: '+str(y_train.shape)+
                    '\ny Test: '+str(y_test.shape)+
                    '\ny Validation: '+str(y_val.shape)+
                    '\n---')
        return split_dict

    def _get_dft_features(df):
        '''
        Function to get any feature that is DFT derived and store column names in list.
        
        Arguments:
        df (DataFrame): A DataFrame of features.

        Returns:
        X (DataFrame): A DataFrame containing the X data (features).
        y (DataFrame): A DataFrame containing the y data (target).
        '''
        remove_targets = []
        for col in df:
            if '_dft' in col:
                remove_targets.append(col)
        y = df[remove_targets]
        X = df.drop(columns=remove_targets)
        logger.info('X Shape: '+str(X.shape)+'\ny Shape: '+str(y.shape)+'\n---')
        return X, y
    
    def _standardise_data(split_dict):
        '''
        Function to standardise the data - performed for both instances with only X and, X and y.
        Using standard scaler in sklearn to perform this.

        Arguments:
        split_dict (Dictionary): A dictionary containing DataFrames of all the different train:validation:test splits.

        Returns:
        data_dict (Dictionary): A dictionary containing the standardised data and scalers all stored in one place.
        '''
        ### Standardise only X
        sc_X = StandardScaler()
        s_train = pd.DataFrame(sc_X.fit_transform(split_dict['X_train']),
                                columns=split_dict['X_train'].columns,
                                  index=split_dict['X_train'].index)
        # Perform same transformation to the X test and Validation Sets
        s_test = pd.DataFrame(sc_X.transform(split_dict['X_test']),
                                columns=split_dict['X_test'].columns,
                                  index=split_dict['X_test'].index)
        s_val = pd.DataFrame(sc_X.transform(split_dict['X_val']),
                                columns=split_dict['X_val'].columns,
                                  index=split_dict['X_val'].index)
        only_X = {'X_train': s_train, 'X_val':s_val, 'X_test':s_test,
                  'y_train': split_dict['y_train'], 'y_val':split_dict['y_val'], 'y_test':split_dict['y_test']}
        
        ### Standardise  y
        scaler_dict = {}
        scaled_ys = {}
        for target in split_dict['y_train'].keys():
            sc_y = StandardScaler()
            ys_train = pd.DataFrame(sc_y.fit_transform(pd.DataFrame(split_dict['y_train'][target])),
                                    columns=pd.DataFrame(split_dict['y_train'][target]).columns,
                                    index=pd.DataFrame(split_dict['y_train'][target]).index)
            
            ys_test = pd.DataFrame(sc_y.transform(pd.DataFrame(split_dict['y_test'][target])),
                                    columns=pd.DataFrame(split_dict['y_test'][target]).columns,
                                    index=pd.DataFrame(split_dict['y_test'][target]).index)
            
            ys_val = pd.DataFrame(sc_y.transform(pd.DataFrame(split_dict['y_val'][target])),
                                    columns=pd.DataFrame(split_dict['y_val'][target]).columns,
                                    index=pd.DataFrame(split_dict['y_val'][target]).index)
            scaled_ys[target] = {'train':ys_train, 'test':ys_test, 'val':ys_val}
            scaler_dict[target] = sc_y

        # Build DataFrames and concatenate the data - store in data_dict.
        all_ys_train, all_ys_test, all_ys_val = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()     
        for target in scaled_ys.keys():
            all_ys_train = pd.concat([all_ys_train, scaled_ys[target]['train']], axis=1)
            all_ys_test = pd.concat([all_ys_test, scaled_ys[target]['test']], axis=1)
            all_ys_val = pd.concat([all_ys_val, scaled_ys[target]['val']], axis=1)
        both_X_y = {'X_train':s_train, 'X_val':s_val, 'X_test':s_test,
                  'y_train':all_ys_train, 'y_val':all_ys_val, 'y_test':all_ys_test}
        data_dict = {'just_X': only_X, 'both_X_y':both_X_y, 'X_scaler':sc_X, 'y_scaler':scaler_dict}
        return data_dict
    
    def _save_results(results_d):
        '''
        Function to update the machine learning results dataframe.

        Arguments:
        results_d (Dictionary): A dictionary containing the ML results for the given model tested.

        Outputs:
        Saves results to a file called ml_results.pkl.
        '''
        # Check if the ml_results.pkl file exists.
        if os.path.isfile('ml_results.pkl') == True:
            ml_results = pd.read_pickle('ml_results.pkl')
            # Check if model already been tested.
            if results_d['model_target'] in ml_results['model_target'].tolist():   
                rs_tested = ml_results['random_state'][ml_results['model_target'] == results_d['model_target']].iloc[0]
                # Check if more than four random states have been checked and if this specific random state has been checked.
                if len(rs_tested) > 4 or int(results_d['random_state']) in rs_tested:
                    logger.info(f'{results_d["model_target"]} already tested 5 times or already tested with this random state - {results_d["random_state"]}')
                    pass
                else:
                    for metric in results_d.keys():
                        if metric == 'model_target':
                            pass
                        else:
                            ml_results[metric][ml_results['model_target'] == results_d['model_target']].iloc[0].append(results_d[metric])
            # Deal with if model isn't in dataframe.
            else:
                tdf = pd.DataFrame()
                for metric in results_d.keys():
                    if metric == 'model_target':
                        tdf[metric] =[results_d[metric]]
                    else:
                        tdf[metric] = [[results_d[metric]]]
                ml_results = pd.concat([ml_results, tdf], axis=0)
        # Deal with if ml_results.pkl does not exist.
        else:
            ml_results, tdf = pd.DataFrame(), pd.DataFrame()
            for metric in results_d.keys():
                if metric == 'model_target':
                    tdf[metric] = [results_d[metric]]
                else:
                    tdf[metric] = [[results_d[metric]]]
            ml_results = pd.concat([ml_results, tdf], axis=0)
        # Save results to a pickle file.
        ml_results.to_pickle('ml_results.pkl')
        # Toggle print below to visualise results.
        #print(ml_results)
                             
    def _runner(self):
        '''
        Function to run the ML analysis.

        This function utilises other fuctions to perform the splits of the data and partitions it before 
        then deciding which models to run and running those. This currently only works for single task models.
        '''
        ml = General()            
        # Load data and extract targets.
        self.df = self.df[self.df['reaction_number'] != 365]
        self.df = self.df[self.df['reaction_number'] != 366]
        self.df = self.df[self.df['reaction_number'] != 368]
        self.X, self.y = General._get_dft_features(self.df)

        # Perform Train/Test/Validation splits and standardise.
        self.splits_dict = General._perform_train_test_split(self.X, self.y)
        self.data_dict = General._standardise_data(self.splits_dict)
        # Pull out the targets for the external test set.
        if isinstance(self.test_df, pd.DataFrame):
            self.test_X, self.test_y = General._get_dft_features(self.test_df)
            for target in self.test_y.columns:
                logger.info(f"Pre ML ({target})\n{mean_absolute_error(self.test_y[target], self.test_X[target.replace('dft', 'am1')])}")
            # Standardise the external test sets based on the training data.        
            self.test_X = pd.DataFrame(self.data_dict['X_scaler'].transform(self.test_X),columns=self.test_X.columns,index=self.test_X.index)
            self.test_y_dict = {}
            for target in self.test_y.columns:
                # self.test_y_dict[target] = pd.DataFrame(self.data_dict['y_scaler'][target].transform(pd.DataFrame(self.test_y[target])), columns=pd.DataFrame(self.test_y[target]).columns, index=pd.DataFrame(self.test_y[target]).index)
                self.test_y_dict[target] = pd.DataFrame(pd.DataFrame(self.test_y[target]), columns=pd.DataFrame(self.test_y[target]).columns, index=pd.DataFrame(self.test_y[target]).index)
            self.test_d = {'X':self.test_X, 'y':self.test_y_dict}
        else:
            self.test_d = {}
        # Pull out the models that are to be evaluated. 
        self.run_models = [i for i in list(self.hps.keys()) if 'nn' not in i]
        # If ml_results is empty thus, no models have been run.
        metrics = SingleTask()._runner(self.run_models, self.data_dict, self.test_d, self.hps)

class SingleTask(General):

    def __init__(self):
        super().__init__()
    
        self.model_dict = {
                'ridge':Ridge(), 
                'krr':KernelRidge(kernel='rbf'),
                'svr':SVR(kernel='rbf')
            }

    def _runner(self, run_models, data_dict, test_d, hps):
        '''
        Runner for performing hyperparameter tuning for all single task models.

        Arguments:
        run_models (List): A list of the models to be evaluated.
        data_dict (Dictionary): A dictionary containing the standardised data and scalers all stored in one place.
        hps (Dictionary): A dictionary containing all the hyperparameters for each model.
        '''
        # Get targets.
        for mod in run_models:
            if 'nn' not in mod:
                model, target = mod.split('_', 1)[0], mod.split('_', 1)[1]
                m = self.model_dict[model].set_params(**hps[mod]['b_hps'])
                # Run sklearn model.
                SingleTask()._model_runner(data_dict, mod, m, test_d, target)
            else:
                model, target = mod.split('_')[0], mod.split('_', 3)[-1]
                hp = hps[mod]
                # Run TensorFlow NN.
                SingleTask()._nn_(data_dict, int(mod.split('_')[0]), hp, target)
        exit()

    def _model_runner(self, data_dict, mod, model, test_d, target):
        '''
        Function to run sklearn models. Performs percentage splits and stores all data (including predicted values) in a 
        DataFrame.

        Arguments:
        data_dict (Dictionary): A dictionary containing the standardised data and scalers all stored in one place.
        mod (String): The name of the model and target combined (e.g., krr_q_dft_barrier)
        model (String): The model to evaluate with fitted hyperparameters already set.
        target (String): The chosen target - dependent on dataset.

        '''        
        # Percentage training split metrics.
        split_d = {}
        for split in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
            f_model = model.fit(
                data_dict['just_X']['X_train'].sample(frac=split, random_state=int(sys.argv[-4])),
                data_dict['just_X']['y_train'][target].sample(frac=split, random_state=int(sys.argv[-4]))
            )
        #     # Predict using this fit model.
            metrics = SingleTask._sklearn_predict(data_dict, f_model, test_d, target)
            split_d[split] = metrics
        # Now fit model on 100% of training data and predict.
        f_model = model.fit(data_dict['just_X']['X_train'], data_dict['just_X']['y_train'][target])
        metrics = SingleTask._sklearn_predict(data_dict, f_model, test_d, target)
        # Log the metrics.
        logger.info(f"###\n{model} - {target}\n---\nExternal Test: {metrics['y_ext_pred']} kcal/mol\n###")

        # Add metrics to dictionary.
        results_d = {
                'model_target':str(mod),
                'model':model,
                'random_state':int(sys.argv[-4]),
                'y_ext_pred_values':metrics['y_ext_pred_values'],
                'y_ext_true':metrics['y_ext_true'][target].to_list(),
        }
        
        # Collate Metrics.
        for metric in ['y_ext_pred', 'std_error_ext']:
            results_d[metric] = metrics[metric]
        for dataframe in split_d.keys():
            results_d[f'{dataframe}_y_ext_pred_values'] = split_d[dataframe]['y_ext_pred_values']
            results_d[f'{dataframe}_y_ext_true'] = split_d[dataframe]['y_ext_true'][target].to_list()
            for metric in ['y_ext_pred', 'std_error_ext']:
                results_d[f'{dataframe}_{metric}'] = split_d[dataframe][metric]

        General._save_results(results_d)

    def _sklearn_predict(data_dict, f_model, test_d, target):
        '''
        Function to predict using a pre-fitted sklearn model.
        
        Arguments:
        data_dict (Dictionary): A dictionary containing the standardised data and scalers all stored in one place.
        f_model (sklearn model object): The fitted model to use for predictions.
        target (String): The target for the machine learning predictions.

        Returns:
        metrics (Dictionary): A dictionary of the metrics for the fitted model.
        '''
        metrics = {}
        # Make predictions on external test set.
        y_ext_pred = f_model.predict(test_d['X'])
        metrics['y_ext_pred_values'] = y_ext_pred
        
        # Generate metrics.
        metrics['y_ext_pred'] = np.round(mean_absolute_error(test_d['y'][target], y_ext_pred), 3)

        # Store values for plotting later.
        metrics['y_ext_true'] = test_d['y'][target]
        
        # Determine Standard Errors.
        lst = []
        for te, pr in zip(list(y_ext_pred), list(test_d['y'][target][target])):
            lst.append(np.abs(te-pr))
        standard_error = np.std(lst) / np.sqrt(len(lst))
        metrics['std_error_ext'] = standard_error
        return metrics

def main():
    # Set up logger
    global logger, start_formatter, formatter, fh, ch, target
    logger = logging.getLogger('ml_rebuild')
    logger.setLevel('INFO')
    fh = logging.FileHandler('ml_rebuild.log', mode='w')
    fh.setLevel('INFO')
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)

    # Change format
    start_formatter = logging.Formatter('%(message)s')
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    fh.setFormatter(start_formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Code to analysis machine learning models.
    try:    
        # Load data and extract targets.
        General()._runner()
        exit()
    except KeyboardInterrupt:
        logger.info(f'Keyboard interupt')

if __name__ == '__main__':
    main()
    
