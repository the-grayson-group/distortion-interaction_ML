"""
#########################################################################################################################
#                                                   ml_analysis.py                                                      #
#                                                                                                                       #
#                        This code takes the dataset and the tuned hyperparameters and will run ML                      #
#                        analysis over a given random state. This code can be looped to perform                         #
#                        the machine learning over multiple random states. It will max out after                        #
#                        five random states. Information required for plotting is all stored in a                       #
#                        pickle file (ml_results.pkl).                                                                  #
#                                                                                                                       #
#            Usage:                                                                                                     #
#                                                                                                                       #
#                 python ml_analysis.py 23 path/to/dataset.pkl path/to/hps.pkl                                          #
#                                                                                                                       #
#               OR                                                                                                      #
#                                                                                                                       #
#                 for %x in (22 23 14 1 2) do python ml_analysis.py %x path/to/dataset.pkl path/to/hps.pkl              #
#                                                                                                                       #
#                                                                                                                       #
#                                                                                                                       #
#########################################################################################################################
#                                                      Authors                                                          #
#                                                                                                                       #
#------------------------------------------------  Samuel G. Espley ----------------------------------------------------#
#                                                                                                                       #
#########################################################################################################################
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
    seed(int(sys.argv[-3]))
    random.set_seed(int(sys.argv[-3]))
    tf.keras.utils.set_random_seed(int(sys.argv[-3]))
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
        
        # Check if ml_results.pkl exists.
        if os.path.exists('ml_results.pkl') == True:
            self.ml_results = pd.read_pickle('ml_results.pkl')
        else:
            self.ml_results = pd.DataFrame(columns=['model_target'])

        # Check which models are tested in full.
        self.done = []
        for model in self.ml_results['model_target']:
            if len(self.ml_results['random_state'][self.ml_results['model_target'] == model].iloc[0]) > 4:
                self.done.append(model)
    
        # Pull out path to dataset.
        if len(sys.argv) == 1 or len(sys.argv) == 2 or len(sys.argv) == 3:
            logger.info('Please provide path to dataset and hyperparameter file.')
            print('Error - check log.')
            exit()
        elif len(sys.argv) == 4:
            path = sys.argv[-2]
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
        self.randomseed = int(sys.argv[-3])

        # Load in hps file.
        with open(hps_path, 'rb') as f:
            hps = pickle.load(f)

        # Load in file from path above - 
        df = pd.read_pickle(path)
        d = path.split('/')[-1].split('_')[0:2]
        if d[1].endswith('.pkl') == True:
            d[1] = d[1].split('.')[0]
        logger.info(f'Dataset: {d[1]} \nFeature Selection Method: {d[0]}\n---')
        
        # Deal with PCA datasets.
        if 'principal' not in df.columns[0]:
            rxn_number = df['reaction_number']
            remove = ['contributions', 'reactant', 'structure', 'path', 'reaction_number', 'sum_distortion_energies']
            to_remove = []
            for column in df.columns:
                for tag in remove:
                    if tag in column:
                        to_remove.append(column)
            df = df.drop(columns=to_remove)
            self.df = df
            self.rxn_number = rxn_number
            self.hps = hps
        else:
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=int(sys.argv[-3]))
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=int(sys.argv[-3]))
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
        self.X, self.y = General._get_dft_features(self.df)
        # Perform Train/Test/Validation splits and standardise.
        self.splits_dict = General._perform_train_test_split(self.X, self.y)
        self.data_dict = General._standardise_data(self.splits_dict)
        # Pull out the models that are to be evaluated.
        nn_models = [i for i in list(self.hps.keys()) if 'nn' in i]
        self.run_models = [i for i in list(self.hps.keys()) if 'nn' not in i]
        self.run_models = [i for i in self.run_models if i not in self.done]
        # print(self.run_models)
        # self.run_models.remove('sum_distortion_energies_dft')
        # print(self.run_models)
        # exit()
        self.remove = []
        # Loop through any NN models and only get one to run.
        for i in range(0, len(nn_models)):
            if nn_models[i] in self.done:
                self.remove.append(nn_models[i])
        nn_models = [i for i in nn_models if i not in self.remove]
        # Added code for temporarily removing the sums of distortion energy predictions.
        nn_models = [i for i in nn_models if 'sum_distortion_energies' not in i]
        self.run_models.append(nn_models[0])
        # Added code for temporarily removing the sums of distortion energy predictions.
        sum_remove = []
        for mod in self.run_models:
            if 'sum_distortion_energies' in mod:
                sum_remove.append(mod)
        self.run_models = [i for i in self.run_models if i not in sum_remove]
        # If ml_results is empty thus, no models have been run.
        metrics = SingleTask()._runner(self.run_models, self.data_dict, self.hps)

class SingleTask(General):

    def __init__(self):
        super().__init__()
    
        self.model_dict = {
                'ridge':Ridge(), 
                'krr':KernelRidge(kernel='rbf'),
                'svr':SVR(kernel='rbf')
            }

    def _runner(self, run_models, data_dict, hps):
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
                SingleTask()._model_runner(data_dict, mod, m, target)
            else:
                model, target = mod.split('_')[0], mod.split('_', 3)[-1]
                hp = hps[mod]
                # Run TensorFlow NN.
                SingleTask()._nn_(data_dict, int(mod.split('_')[0]), hp, target)

    def _model_runner(self, data_dict, mod, model, target):
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
                data_dict['just_X']['X_train'].sample(frac=split, random_state=int(sys.argv[-3])),
                data_dict['just_X']['y_train'][target].sample(frac=split, random_state=int(sys.argv[-3]))
            )
            # Predict using this fit model.
            metrics = SingleTask._sklearn_predict(data_dict, f_model, target, split)
            split_d[split] = metrics

        # Now fit model on 100% of training data and predict.
        f_model = model.fit(data_dict['just_X']['X_train'], data_dict['just_X']['y_train'][target])
        metrics = SingleTask._sklearn_predict(data_dict, f_model, target)

        # Log the metrics.
        logger.info(f'###\n{model} - {target}\n---\nTrain: {metrics["y_train_CV"]} kcal/mol \nTest: {metrics["y_test_pred"]} kcal/mol\n###')

        # Add metrics to dictionary.
        results_d = {
                'model_target':str(mod),
                'model':model,
                'random_state':int(sys.argv[-3]),
                'y_val_pred_values':metrics['y_val_pred_values'],
                'y_test_pred_values':metrics['y_test_pred_values'],
                'y_test_true':metrics['y_test_true'].to_list(),
                'y_val_true':metrics['y_val_true'].to_list(),
                'y_train_CV_scores':metrics['y_train_CV_scores'],
        }
        
        # Collate Metrics.
        for metric in ['y_train_CV', 'y_val_pred', 'y_test_pred', 'std_error_val', 'std_error_test']:
            results_d[metric] = metrics[metric]
        for dataframe in split_d.keys():
            results_d[f'{dataframe}_y_val_pred_values'] = split_d[dataframe]['y_val_pred_values']
            results_d[f'{dataframe}_y_test_pred_values'] = split_d[dataframe]['y_test_pred_values']
            results_d[f'{dataframe}_y_test_true'] = split_d[dataframe]['y_test_true'].to_list()
            results_d[f'{dataframe}_y_val_true'] = split_d[dataframe]['y_val_true'].to_list()
            results_d[f'{dataframe}_y_train_CV_scores'] = split_d[dataframe]['y_train_CV_scores']
            for metric in ['y_train_CV', 'y_val_pred', 'y_test_pred', 'std_error_val', 'std_error_test']:
                results_d[f'{dataframe}_{metric}'] = split_d[dataframe][metric]
        
        General._save_results(results_d)

    def _sklearn_predict(data_dict, f_model, target, split=1):
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
        # Get Train metrics.
        cv_scores = np.abs(cross_val_score(
            f_model, 
            data_dict['just_X']['X_train'].sample(frac=split, random_state=int(sys.argv[-3])),
            data_dict['just_X']['y_train'][target].sample(frac=split, random_state=int(sys.argv[-3])),
            cv=5,
            scoring='neg_mean_absolute_error'
        ))
        # Make predictions on test and validation.
        y_val_pred = f_model.predict(data_dict['just_X']['X_val'])
        y_test_pred = f_model.predict(data_dict['just_X']['X_test'])
        metrics['y_val_pred_values'] = y_val_pred
        metrics['y_test_pred_values'] = y_test_pred
        
        # Generate metrics.
        metrics['y_train_CV'] = np.round(sum(cv_scores)/len(cv_scores), 3)
        metrics['y_train_CV_scores'] = cv_scores
        metrics['y_val_pred'] = np.round(mean_absolute_error(data_dict['just_X']['y_val'][target], y_val_pred), 3)
        metrics['y_test_pred'] = np.round(mean_absolute_error(data_dict['just_X']['y_test'][target], y_test_pred), 3)

        # Store values for plotting later.
        metrics['y_test_true'] = data_dict['just_X']['y_test'][target]
        metrics['y_val_true'] = data_dict['just_X']['y_val'][target]

        # Determine Standard Errors.
        for pair in [
            ['std_error_val', y_val_pred, data_dict['just_X']['y_val'][target]], 
            ['std_error_test', y_test_pred, data_dict['just_X']['y_test'][target]]
        ]:
            lst = []
            for te, pr in zip(pair[1], pair[2]):
                lst.append(np.abs(te-pr))
            standard_error = np.std(lst) / np.sqrt(len(lst))
            metrics[pair[0]] = standard_error
        
        return metrics

    def _nn_(self, data_dict, model_size, hps, target):
        '''
        Single-task NN model builder using the Hyperband algorithm to search for a hyperparameter set.
        Will generate various networks of differing sizes that is defined in the _single_task_nn function.

        Arguments:
        data_dict (Dictionary): A dictionary containing the standardised data and scalers all stored in one place.
        model_size (String): An integer in the type of a string that denotes the number of hidden layers in the network.
        target (String): The target for the machine learning predictions.
        hps (Dictionary): A dictionary containing all the hyperparameters for the chosen model.

        Output:
        Saves the results to the ml_results.pkl.
        '''
        # Get the model target.
        model_target = f'{model_size}_st_nn_{target}'
        td = SingleTask._tensorflow_runner(data_dict, model_size, target, model_target, hps)

    def _tensorflow_runner(data_dict, model_size, target, model_target, hps):
        '''
        Function to perform the running of a TensorFlow model analysis.

        Arguments:
        data_dict (Dictionary): A dictionary containing the standardised data and scalers all stored in one place.
        model_size (String): An integer in the type of a string that denotes the number of hidden layers in the network.
        target (String): The target for the machine learning predictions.
        model_target (String): A combined string of the model and the target for the model.
        hps (Dictionary): A dictionary containing all the hyperparameters for the chosen model.

        Outputs:
        Saves the results to the ml_results.pkl.
        '''
        # Build network and store layers in a dictionary to avoid any issues with layers having similar names.
        layer_d = {}
        layer_d['inp_lay'] = keras.Input(shape=(data_dict['both_X_y']['X_train'].shape[1],), name='input')
        # Create network.
        for hidden_layers in range(1, model_size+1):
            if hidden_layers == 1:
                layer_d['dense_1'] = layers.Dense(units=hps['neurons_1'], name='dense_1', activation='elu', kernel_regularizer=l2(hps[f'reg_val_{hidden_layers}']))(layer_d['inp_lay'])
                layer_d['drop_1'] = layers.Dropout(rate=hps[f'dropout_rate_{hidden_layers}'], name='dropout_1')(layer_d['dense_1'])
            else:
                layer_d[f'dense_{hidden_layers}'] = layers.Dense(units=hps[f'neurons_{hidden_layers}'], name=f'dense_{hidden_layers}', activation='elu',kernel_regularizer=l2(hps[f'reg_val_{hidden_layers}']))(layer_d[f'drop_{hidden_layers-1}'])
                if hidden_layers == model_size:
                    layer_d['out_lay'] = layers.Dense(units=1, name=target)(layer_d[f'dense_{hidden_layers}'])
                else:
                    layer_d[f'drop_{hidden_layers}'] = layers.Dropout(rate=hps[f'dropout_rate_{hidden_layers}'], name=f'dropout_{hidden_layers}')(layer_d[f'dense_{hidden_layers}'])

        # Build Model
        model = models.Model(inputs=layer_d['inp_lay'], outputs=layer_d['out_lay'])
        # Toggle the below line for model picture generation.
        #keras.utils.plot_model(model, to_file=f'model_{hidden_layers}.png',rankdir='LR')
        model.compile(optimizer=Adam(learning_rate=hps['learning_rate']), loss='mean_absolute_error', metrics=['mean_absolute_error'])   

        # Fit the model.
        history = model.fit(
            x=data_dict['both_X_y']['X_train'],
            #x=X,
            y=data_dict['both_X_y']['y_train'][target],
            validation_data=(data_dict['both_X_y']['X_val'], data_dict['both_X_y']['y_val'][target]),
            batch_size=32,
            epochs=500,
            verbose=0,
            shuffle=False)
        
        # Get train metric.
        PD_y_train = model.predict(data_dict['both_X_y']['X_train'])
        PD_y_train = data_dict['y_scaler'][target].inverse_transform(np.array(PD_y_train).reshape(-1,1))
        y_train = data_dict['y_scaler'][target].inverse_transform(np.array(data_dict['both_X_y']['y_train'][target]).reshape(-1,1))
        train_mae = np.round(np.abs(mean_absolute_error(PD_y_train, y_train)), 3)
        
        # Get validation metric.
        PD_y_val = model.predict(data_dict['both_X_y']['X_val'])
        PD_y_val = data_dict['y_scaler'][target].inverse_transform(np.array(PD_y_val).reshape(-1,1))
        y_val = data_dict['y_scaler'][target].inverse_transform(np.array(data_dict['both_X_y']['y_val'][target]).reshape(-1,1))
        val_mae = np.round(np.abs(mean_absolute_error(PD_y_val, y_val)), 3)
        
        # Get test metric.
        PD_y_test = model.predict(data_dict['both_X_y']['X_test'])
        PD_y_test = data_dict['y_scaler'][target].inverse_transform(np.array(PD_y_test).reshape(-1,1))
        y_test = data_dict['y_scaler'][target].inverse_transform(np.array(data_dict['both_X_y']['y_test'][target]).reshape(-1,1))
        test_mae = np.round(np.abs(mean_absolute_error(PD_y_test, y_test)), 3)
        # Determine Standard Errors.
        std_errors = {}
        for pair in [['std_error_train', PD_y_train, y_train], ['std_error_val', PD_y_val, y_val], ['std_error_test', PD_y_test, y_test]]:
            lst = []
            for te, pr in zip(pair[1], pair[2]):
                lst.append(np.abs(te-pr))
            standard_error = np.std(lst) / np.sqrt(len(lst))
            std_errors[pair[0]] = standard_error

        # Collate Metrics.
        results_d = {
            'model_target':model_target,
            'random_state':int(sys.argv[-3]),
            'y_val_pred_values': [[i[0] for i in PD_y_val]],
            'y_test_pred_values':[ [i[0] for i in PD_y_test]],
            'y_train_pred_values': [[i[0] for i in PD_y_train]],
            'y_test_true': [[i[0] for i in y_test]],
            'y_val_true': [[i[0] for i in y_val]],
            'y_train_true': [[i[0] for i in y_train]],
            'y_val_pred': val_mae,
            'y_test_pred': test_mae,
            'y_train_pred': train_mae, 
            'std_error_val': std_errors['std_error_val'],
            'std_error_test': std_errors['std_error_test'],
            'std_error_train': std_errors['std_error_train'],
            'history_loss': [history.history['loss']],
            'history_val_loss': [history.history['val_loss']]
        }
        
        logger.info(
                f'NN with {model_size} hidden layers on {target}.\n \
                Train MAE: {train_mae}\n \
                Test MAE: {test_mae}\n \
                Validation MAE {val_mae}\n \
                ---'
            )    
        General._save_results(results_d)

def main():
    # Set up logger
    global logger, start_formatter, formatter, fh, ch, target
    logger = logging.getLogger('ml_analysis')
    logger.setLevel('INFO')
    fh = logging.FileHandler('ml_analysis.log', mode='w')
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
    
