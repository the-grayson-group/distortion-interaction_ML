"""
#####################################################################################
#                                  hyp_tuning.py                                    #
#                                                                                   #
#      This code performs hyperparameter tuning for multiple different              #
#      models with multiple different targets.                                      #
#      The models include sklearn standard models and NN's built with               #
#      TensorFlow.                                                                  #
#                                                                                   #
#      Usage:                                                                       #
#                                                                                   #
#             python hyp_tuning.py path/to/dataset.pkl                              #
#                                                                                   #
#####################################################################################
#                                       Authors                                     #
#                                                                                   #
#--------------------------------- Samuel G. Espley --------------------------------#
#                                                                                   #
#####################################################################################
"""

# Imports
import os
import sys
import time
import shutil
import pickle
import logging
import warnings
import datetime
import numpy as np
import pandas as pd
from numpy.random import seed
import matplotlib.pyplot as plt
# sklearn imports
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
# TensorFlow imports
 # Remove build information associated with tensorflow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from tensorflow import random
from keras.activations import elu
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping
from keras import layers, Sequential, models, backend as K

# Final depreciation warning removal - when debugging, hash this line.
tf.get_logger().setLevel('ERROR')

# Set seed for hyperparameter tuning.
seed(23)
random.set_seed(23)
tf.keras.utils.set_random_seed(23)

class General():

    def __init__():
        pass

    def _load_data():
        '''
        Function to load the data for hyperparameter tuning

        Returns:
        df (DataFrame): A DataFrame of the features and target for a given dataset.
        '''
        global d
        # Create figure directory if not already present.
        Checkpointing._check_fig_storage()
        # Pull out path to dataset.
        if len(sys.argv) == 1:
            logger.info('Please provide path to dataset.')
            print('Error - check log.')
            exit()
        else:
            path = sys.argv[1]
            if not path.endswith('pkl'):
                logger.info('Please provide a valid path to a dataset.')
                print('Error - check log.')
                exit()

        # Load in file from path above - 
        df = pd.read_pickle(path)
        d = path.split('/')[-1].split('_')[0:2]
        logger.info(f'Dataset: {d[1]} \nFeature Selection Method: {d[0]}\n---')
        
        if 'principal' not in df.columns[0]:
            rxn_number = df['reaction_number']
            remove = ['contributions', 'reactant', 'structure', 'path', 'reaction_number']
            to_remove = []
            for column in df.columns:
                for tag in remove:
                    if tag in column:
                        to_remove.append(column)
            df = df.drop(columns=to_remove)
            return df, rxn_number
        else:
            rxn_number = None

        return df, rxn_number

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=23)
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
        all_ys_train, all_ys_test, all_ys_val = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()     
        for target in scaled_ys.keys():
            all_ys_train = pd.concat([all_ys_train, scaled_ys[target]['train']], axis=1)
            all_ys_test = pd.concat([all_ys_test, scaled_ys[target]['test']], axis=1)
            all_ys_val = pd.concat([all_ys_val, scaled_ys[target]['val']], axis=1)
        both_X_y = {'X_train':s_train, 'X_val':s_val, 'X_test':s_test,
                  'y_train':all_ys_train, 'y_val':all_ys_val, 'y_test':all_ys_test}
        data_dict = {'just_X': only_X, 'both_X_y':both_X_y, 'X_scaler':sc_X, 'y_scaler':scaler_dict}
        return data_dict

    def _save(lst):
        '''
        Function to save the tuned hyperparameters to a file.
        
        Arguments:
        lst (List): A list of dictionaries containing tuned hyperparameters.
        '''
        # Check if checkpoint file exists - if True then load pre-run tuning.
        if os.path.isfile('checkpoint.pkl') == True:
            with open('checkpoint.pkl', 'rb') as f:
                tuned_dict = pickle.load(f)
        else:
            tuned_dict = {}
        # Loop through each dictionary and merge with tuned_dict.
        for dictionary in lst:
            if dictionary != {}:
                tuned_dict = {**tuned_dict, **dictionary}
            else:
                pass
        # Dump all hyperparams into dictionary.
        with open('hps.pkl', 'wb') as f:
                pickle.dump(tuned_dict, f)

class Checkpointing():

    def __init__():
        pass

    def _checkpoint(dictionary):
        '''
        Function to load in previous checkpoint files.

        Arguments:
        dictionary (Dictionary): A dictionary containing hyperparameters.
        '''
        # Determine if checkpointing file exists.
        if os.path.isfile('checkpoint.pkl') == False:
            with open('checkpoint.pkl', 'wb') as f:
                pickle.dump(dictionary, f)
            f.close()
        elif os.path.isfile('checkpoint.pkl') == True:
            # Load in existing data
            with open('checkpoint.pkl', 'rb') as f:
                prev = pickle.load(f)
            f.close()
            # Combine existing data and new data.
            data = {**prev, **dictionary}
            with open('checkpoint.pkl', 'wb') as f:
                pickle.dump(data, f)
            f.close()

    def _read_checkpoint():
        '''
        Function to read checkpoint and determine if model has already been run.
        Skips previously run models.

        Returns:
        models (List): A list of previously ran models.
        '''
        if os.path.isfile('checkpoint.pkl') == False:
            models = []
        else:
            with open('checkpoint.pkl', 'rb') as f:
                models = list(pickle.load(f).keys())
        return models

    def _check_fig_storage():
        '''
        Function to check if direcory 'figures' exists. If not, creates it.
        '''
        if os.path.isdir('figures') == True:
            pass
        else:
            os.mkdir('figures')
        
class SingleTask():

    def __init__():
        pass

    def _models():
        '''
        Storage of all model types and associated tuning spaces

        Returns:
        models (Dictionary): A dictionary containing all the models for single task learning.
        hp_values (Dictionary): A dictionary contraining all the hyperparameters for each of the models.
        '''
        models = {
                  'ridge':{'model':Ridge(), 'hp':['alpha', 'tol']},
                  'krr':{'model':KernelRidge(kernel='poly'), 'hp':['alpha', 'gamma']},
                  'svr':{'model':SVR(kernel='rbf'), 'hp':['gamma', 'epsilon', 'C', 'coef0', 'degree']},
                  'rf':{'model':RandomForestRegressor(), 'hp':['max_depth', 'n_estimators', 'max_features', 'min_samples_leaf']}
                  }
        hp_values = {'alpha':[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                     'l1_ratio':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                     'tol':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                     'gamma':{'krr':[None, 0.1, 0.5, 0.9], 'svr':['auto', 'scale']},
                     'epsilon':[0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1],
                     'C':[1, 30, 50],
                     'coef0': [0, 1], 
                     'degree': [1, 2, 3],
                     'max_depth':[3, 5, 7],
                     'n_estimators':[10, 50, 100],
                     'max_features':[10, 20, 30],
                     'min_samples_leaf':[1, 2, 3]}
        return models, hp_values
    
    def _runner(data_dict):
        '''
        Runner for performing hyperparameter tuning for all single task models.

        Arguments:
        data_dict (Dictionary): A dictionary containing the standardised data and scalers all stored in one place.

        Returns:
        tuned_dict (Dictionary): A dictionary containing the best hyperparameters for every possible single task target.
        '''
        tuned_dict = {}
        # Load in the models and hyperparameters.
        models, hp_values = SingleTask._models()
        # Loop through evert model.
        for model in models.keys():
            logger.info(f'Running {model}')
            
            hp_dict = {}
            for hp in models[model]['hp']:
                # Deal with differences in gamma hyperparameter.
                if hp == 'gamma':
                    hp_dict[hp] = hp_values[hp][model]
                else:
                    hp_dict[hp] = hp_values[hp]
            # Loop through possible targets and perform the grid search for best hyperparameters.
            for target in data_dict['just_X']['y_train'].keys():
                model_name = str(model+'_'+target)
                if model_name in Checkpointing._read_checkpoint():
                    logger.info(f'Model already tuned - skipping {model_name}')
                else:
                    t0 = time.time()
                    grid = GridSearchCV(models[model]['model'], hp_dict, cv=5, scoring='neg_mean_absolute_error')
                    grid.fit(data_dict['just_X']['X_train'], data_dict['just_X']['y_train'][target])
                    t1 = time.time()
                    total_t = datetime.timedelta(seconds=int(t1 - t0))
                    # Log information.
                    logger.info(
                        f'{model} on {target} finished running {total_t}.\n \
                        Train MAE: {np.round(np.abs(grid.best_score_), 3)}'
                    )
                    tuned_dict[model_name] = {'b_hps': grid.best_params_, 'train_mae':np.abs(grid.best_score_)}
                # Checkpoint results.
                Checkpointing._checkpoint(tuned_dict)
        return tuned_dict

    def _nn_model_builder(hp):
        '''
        Single-task NN model builder using the Hyperband algorithm to search for a hyperparameter set.
        Will generate various networks of differing sizes that is defined in the _single_task_nn function.

        Arguments:
        hp (Keras Tuner Object): This is the hyperparameter choice object that is fed into the model.

        Returns:
        model (TensorFlow model): A Tensorflow model with the hyperparameter choices loaded in.
        '''
        hp_storage = {
            'dropout_rate':[0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            'neurons':[32, 64, 128, 256, 512], 
            'reg_val':[2e-1, 1e-1, 1e-2, 1e-3],
            'learning_rate':[1e-3, 1e-4, 1e-5]
        }
        layer_d = {}
        layer_d['inp_lay'] = keras.Input(shape=(shape,), name='input')
        # layer_d['inp_lay'] = keras.Input()
        # Create network.
        for hidden_layers in range(1, network_size+1):
            specific_hp_storage = {}
            for key in hp_storage:
                if key == 'learning_rate':
                    specific_hp_storage[key] = hp.Choice(key, values=hp_storage[key])
                else:
                    specific_hp_storage[f'{key}_{hidden_layers}'] = hp.Choice(f'{key}_{hidden_layers}', values=hp_storage[key])
            if hidden_layers == 1:
                layer_d['dense_1'] = layers.Dense(units=specific_hp_storage['neurons_1'], name='dense_1', activation='elu', kernel_regularizer=l2(specific_hp_storage[f'reg_val_{hidden_layers}']))(layer_d['inp_lay'])
                layer_d['drop_1'] = layers.Dropout(rate=specific_hp_storage[f'dropout_rate_{hidden_layers}'], name='dropout_1')(layer_d['dense_1'])
            else:
                layer_d[f'dense_{hidden_layers}'] = layers.Dense(units=specific_hp_storage[f'neurons_{hidden_layers}'], name=f'dense_{hidden_layers}', activation='elu',kernel_regularizer=l2(specific_hp_storage[f'reg_val_{hidden_layers}']))(layer_d[f'drop_{hidden_layers-1}'])
                if hidden_layers == network_size:
                    layer_d['out_lay'] = layers.Dense(units=1, name=target)(layer_d[f'dense_{hidden_layers}'])
                else:
                    layer_d[f'drop_{hidden_layers}'] = layers.Dropout(rate=specific_hp_storage[f'dropout_rate_{hidden_layers}'], name=f'dropout_{hidden_layers}')(layer_d[f'dense_{hidden_layers}'])
        # Build Model
        model = models.Model(inputs=layer_d['inp_lay'], outputs=layer_d['out_lay'])
        model.compile(optimizer=Adam(learning_rate=specific_hp_storage['learning_rate']), loss='mean_absolute_error', metrics=['mean_absolute_error'])
        return model

    def _single_task_nn(data_dict):
        '''
        Function to run single task NN hyperparameter tuning.
        
        Arguments:
        data_dict (Dictionary): A dictionary containing the standardised data and scalers all stored in one place.

        Returns:
        tuned_dict (Dictionary): A dictionary containing the tuned hyperparameters for each model tuned.
        '''
        global shape, network_size, target
        tuned_dict = {}
        logger.info('Starting Single Task NN Tuning.')
        # shape = len(data_dict['both_X_y']['X_train'].columns)
        shape = data_dict['both_X_y']['X_train'].shape[1]
        


        # Pull out each target and check checkpoint file.
        for target in data_dict['both_X_y']['y_train'].keys():
            for network_size in [2, 4]:
                model_name = f'{network_size}_st_nn_{target}'
                if model_name in Checkpointing._read_checkpoint():
                    logger.info(f'Model already tuned - skipping {model_name}')
                else:
                    # Build the tuner.
                    tuner = kt.Hyperband(
                        SingleTask._nn_model_builder,
                        project_name=target,
                        objective=['mean_absolute_error', 'val_mean_absolute_error'],
                        max_epochs=100,
                        factor=3,
                        seed=23
                    )  
                    # Tuner search.
                    t0 = time.time()
                    tuner.search(
                        x=data_dict['both_X_y']['X_train'], 
                        y=data_dict['both_X_y']['y_train'][target],
                        validation_data=(data_dict['both_X_y']['X_val'], data_dict['both_X_y']['y_val'][target]),
                        epochs=100,
                        verbose=0
                    )

                    # Pull out the best hyperparameters.
                    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
                    tuned_dict[model_name] = best_hps.values

                    # Predict on train set.
                    model = tuner.hypermodel.build(best_hps)
                    history = model.fit(
                        x=data_dict['both_X_y']['X_train'], 
                        y=data_dict['both_X_y']['y_train'][target], 
                        validation_data=(data_dict['both_X_y']['X_val'], data_dict['both_X_y']['y_val'][target]),
                        batch_size=32,
                        # epochs=500,
                        epochs=10,
                        verbose=0
                    )

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

                    # Deal with times.
                    t1 = time.time()
                    total_t = datetime.timedelta(seconds=int(t1 - t0))

                    # Clear session and log information.
                    tf.keras.backend.clear_session()
                    logger.info(f'NN with {network_size} hidden layers on {target} finished running {total_t}.\n \
                                Train MAE: {train_mae}\n \
                                Test MAE: {test_mae}\n \
                                Validation MAE {val_mae}\n \
                                ---'
                            )       
                    # Remove directory containing trials.
                    shutil.rmtree(target, ignore_errors=False, onerror=None)
                    # Checkpoint results.
                    Checkpointing._checkpoint(tuned_dict)

        return tuned_dict

def main():
    # Set up logger
    global logger, start_formatter, formatter, fh, ch, target
    logger = logging.getLogger('hyperparameter_tuning')
    logger.setLevel('INFO')
    fh = logging.FileHandler('hyp_tuning.log', mode='w')
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

    # Code to hyperparameter tune.
    try:    
        # Load data and extract targets.
        df, rxn_number = General._load_data()
        X, y = General._get_dft_features(df)
        # Perform Train/Test/Validation splits and standardise.
        splits_dict = General._perform_train_test_split(X, y)
        data_dict = General._standardise_data(splits_dict)
        # Run hyperparameter tuning.
        ## Single Task
        current_task = 'single_task'
        logger.info('Single Task Models\n---')
        st_tuned_dict = SingleTask._runner(data_dict)
        st_nn_tuned_dict = SingleTask._single_task_nn(data_dict)
        # Save results
        General._save([st_tuned_dict, st_nn_tuned_dict])
        logger.info('Tuned hyperparameters saved to hps.pkl.')

    except KeyboardInterrupt:
        logger.info(f'Keyboard interupt - checkpointing current task ({current_task})')

if __name__ == '__main__':
    main()
    