"""
#####################################################################################################################################
#                                                ml_feature_importances.py                                                          #
#                                                                                                                                   #
#                        This code takes the dataset and the tuned hyperparameters and will run ML                                  #
#                        feature importances on one random state.                                                                   #
#                                                                                                                                   #
#   Usage:                                                                                                                          #
#                                                                                                                                   #
#     python ml_feature_importances.py 23 path/to/first/dataset.pkl path/to/hps.pkl                                                 #
#                                                                                                                                   #   
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
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from numpy.random import seed
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Remove build information associated with tensorflow
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
# TensorFlow imports
import tensorflow as tf
from tensorflow import random

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
    print('Invalid seed passed - argument formats as:\npython ml_feature_importances.py  random_seed  path/to/dataset.pkl  path/to/hps.pkl')
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
        if len(sys.argv) <= 3:
            logger.info('Please provide path to datasets and hyperparameter file.')
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
        self.run_models = [i for i in list(self.hps.keys()) if 'svr' in i]
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
        if target == 'sum_distortion_energies_dft':
            pass
        else:
            split_d = {}
            for split in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
                f_model = model.fit(
                    data_dict['just_X']['X_train'].sample(frac=split, random_state=int(sys.argv[-3])),
                    data_dict['just_X']['y_train'][target].sample(frac=split, random_state=int(sys.argv[-3]))
                )
            # Now fit model on 100% of training data and predict.
            f_model = model.fit(data_dict['just_X']['X_train'], data_dict['just_X']['y_train'][target])
            SingleTask._feature_importances_shuf(data_dict, mod, target, f_model)

    def _feature_importances_shuf(d, mod, model_name, f_model):
        '''
        Function to get the feature importances based on the test set - This function performs the feature importances based on shuffling one feature before testing. 
        This is typically more informative for Neural Networks.

        Arguments:
        d (Dictionary): A dictionary of training/test/validation data ready for machine learning.
        mod (String): A string of the model and target together.
        model_name (String): The name of the model to evaluate.
        f_model (TensorFlow Model): The model you want to test feature importances for.

        Output:
        Saves a figure with best and worst features based on the shuffle approach.
        '''
        # Shuffle each feature iteratively.
        test_x_dict = {}
        test_x_dict['Test MAE'] = d['just_X']['X_test']
        for col in d['just_X']['X_test'].columns:
            tdf = d['just_X']['X_test'].copy()
            tdf[col] = d['just_X']['X_test'][col].sample(frac=1, random_state=23).values
            test_x_dict[col] = tdf


        # Create a dictionary of the importance metrics.
        imp_d = {}
        for feature in test_x_dict.keys():
            # Test metrics.
            y_test_pred = f_model.predict(test_x_dict[feature])
            test_mae =  np.round(mean_absolute_error(d['just_X']['y_test'][model_name], y_test_pred), 3)
            imp_d[feature] = test_mae
        
        # Create the list of features.
        maxi = sorted(imp_d, key=imp_d.get, reverse=True)[:10]
        mini = sorted(imp_d, key=imp_d.get, reverse=False)[:10]
        maxi = list(reversed(maxi))
        mini = list(reversed(mini))
        maxi.append('Test MAE')
        mini.append('Test MAE')
        maxi_tt_imp_d = {}
        mini_tt_imp_d = {}
        for feature in maxi:
            maxi_tt_imp_d[feature] = imp_d[feature]
        for feature in mini:
            mini_tt_imp_d[feature] = imp_d[feature]
        
        # Create the colours for these.
        maxi_cs = []
        mini_cs = []
        for feature in maxi_tt_imp_d.keys():
            if feature == 'Test MAE':
                maxi_cs.append('#214cce')
            else:
                maxi_cs.append('#6d8ce8')
        for feature in mini_tt_imp_d.keys():
            if feature == 'Test MAE':
                mini_cs.append('#214cce')
            else:
                mini_cs.append('#6d8ce8')
        
        # # Plot the results.
        fig, ax = plt.subplots(2,1)
        ax[0].barh(list(maxi_tt_imp_d.keys()), list(maxi_tt_imp_d.values()), align='center', color=maxi_cs, edgecolor='navy')
        ax[0].axvline(x=1, linestyle='--', color='black')
        ax[0].set_xlabel('MAE / kcal mol$^{-1}$')
        ax[1].barh(list(mini_tt_imp_d.keys()), list(mini_tt_imp_d.values()), align='center', color=mini_cs, edgecolor='navy')
        ax[1].axvline(x=1, linestyle='--', color='black')
        ax[1].set_xlabel('MAE / kcal mol$^{-1}$')
        fig.supylabel('Feature Randomly Scrambled')
        ax[0].set_title(f'{mod}')
        fig.tight_layout()
        plt.show()

def main():
    # Set up logger
    global logger, start_formatter, formatter, fh, ch, target
    logger = logging.getLogger('ml_rebuild')
    logger.setLevel('INFO')
    fh = logging.FileHandler('ml_feature_importances.log', mode='w')
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
    
