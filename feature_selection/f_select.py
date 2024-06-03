"""
#####################################################################################
#                                    f_select.py                                    #
#                                                                                   #
#      This code performs feature selection based upon user inputs and the          #
#      various methods available to perform the selection. Implemented currently    #
#      is SFS, RFECV and PCA.                                                       #
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
import re
import time
import pickle
import logging
import inquirer
import numpy as np
import pandas as pd
from functools import wraps
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV, SequentialFeatureSelector


def timeit(func):
    '''
    Function to time a given function utilising wrapper capabilities
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
    
    def __init__():
        pass

    def _load_data():
        '''
        Function to load in given dataframe using pandas.

        Returns:
        df (DataFrame): A DataFrame of all the features.
        '''
        files = []
        pth = os.getcwd()
        for file in os.listdir():
            if file.endswith('.pkl'):
                files.append(file)
        questions = [inquirer.List('dataset_select',
                              message='Which dataset?',
                              choices=files)]
        answer = inquirer.prompt(questions)
        filename = pth+'\\'+answer['dataset_select']
        df = pd.read_pickle(filename)

        if os.path.isdir(pth+'\\_f_selection') == True:
            os.chdir(pth+'\\_f_selection')
        else:
            os.mkdir('_f_selection')
            os.chdir(pth+'\\_f_selection')

        return df

    def _clean_dist_contr(df):
        '''
        Function to clean the distorted contribution columns to create two separate columns from one.
        
        Arguments:
        df (DataFrame): A DataFrame containing all the features after feature selection.
        
        '''
        contributions_df = pd.DataFrame()
        for column in df:
            if 'contributions' in column:
                method = column.split('contributions_')[-1]
                for dictionary in df[column]:
                    for key in list(dictionary.keys()):
                        try:
                            k = int(key.split('_')[0])
                        except:
                            k = key.split('_')[0]
                        if isinstance(k, int):
                            dictionary[key.split('_')[-1]] = dictionary.pop(key)
                            rxn_num = int(key.split('_')[0])
                        else:
                            dictionary[key.split('_')[0]] = dictionary.pop(key)
                            rxn_num = int(key.split('_')[-1])
                    dictionary['reaction_number'] = rxn_num
                temp_df = df[column].apply(pd.Series)
                for new_column in temp_df:
                    if new_column == 'reaction_number':
                        pass
                    else:
                        temp_df = temp_df.rename(columns={new_column:'distortion_energy_'+new_column+'_'+method})
                if contributions_df.empty == True:
                    contributions_df = temp_df
                else:
                    contributions_df = contributions_df.merge(temp_df, on='reaction_number')
            else:
                pass
        df = df.merge(contributions_df, on='reaction_number')
        return df

    def _save_data(df, removed_features=None, exp_var_ratio=None):
        '''
        Function to save the dataset with the chosen feature selection method implemented.

        Arguments:
        df (DataFrame): A DataFrame containing all the features after feature selection.
        '''
        # Deal with filename input.
        f_name = input('Name of pickle file to save as: ')
        if 'pkl' in f_name:
            pass
        else:
            f_name = f_name+'.pkl'
        # Save data
        if 'principal' in list(df.columns)[0]:
            df.to_pickle(f_name)
            evr_name = f_name.split('.')[0]+'_exp_var_ratio.npy'
            with open(evr_name, 'wb') as f:
                np.save(f, exp_var_ratio)
            logger.info(f'PCA Transformed X and targets saved to {f_name}. Explained variance ratio saved to exp_var_ratio.npy')
        else:
            df.to_pickle(f_name)
            feat_name = f_name.split('.')[0]+'_removed_features.pkl'
            with open(feat_name, 'wb') as f:
                pickle.dump(removed_features, f)
            logger.info(f'Features selected saved to {f_name}. Removed features saved to {feat_name}')

    def _get_removed_features(df_initial, features):
        '''
        Function to find out all the removed features from the dataset.
        
        Arguments:
        df_initial (DataFrame): A DataFrame containing all the features prior to feature selection.
        features (np.array): A array of the selected features.

        Returns:
        removed_features (List): A list of removed features.
        '''
        removed_features = []
        for feature in list(features):
            if feature not in df_initial.columns:
                removed_features.append(feature)
        return removed_features
    
    def _get_dft_features(df):
        '''
        Function to get any feature that is DFT derived and store column names in list.
        
        Arguments:
        df (DataFrame): A DataFrame of features.

        Returns:
        remove_targets (List): A list of DFT derived features (based on column names).
        '''
        remove_targets = []
        for col in df:
            if '_dft' in col:
                remove_targets.append(col)
        return remove_targets
    
    def _add_targets(df, df_initial, targets):
        '''
        Function to readd the targets back to the dataset after feature selection has been completed.

        Arguments:
        df (DataFrame): A DataFrame containing all the features after feature selection.
        df_initial (DataFrame): A DataFrame containing all the features priot to feature selection.
        targets (List): A list of targets for ML analysis.

        Returns:
        new_df (DataFrame): A DataFrame that contains the feature selected features and targets.
        '''
        targets.append('reaction_number')
        target_df = df_initial[targets]
        new_df = df.merge(target_df, on='reaction_number')
        
        return new_df

class Manual():

    def __init__():
        pass
    
    def _manual_runner(df):
        
        feat_d = {
            'sterimol':[],
            'sasa':[],
            'buried_volume':[],
            'free_volume':[],
            'distance':[],
            'apt_charge':[],
            'apt_sum':[],
            'mulliken_charge':[],
            'mulliken_sum':[],
            'other':[]
        }

        features = list(df.columns)
        for f_type in feat_d.keys():
            [feat_d[f_type].append(col) for col in df.columns if f_type in col]
            [features.remove(col) for col in df.columns if f_type in col]
        cp_features = features
        feat_d['keep'] = [feat for feat in features if 'am1' in feat] + [feat for feat in features if 'dft' in feat]
        feat_d['keep'].append('reaction_number')

        for feature in feat_d['keep']:
            features.remove(feature)

        [feat_d['other'].append(col) for col in features]

        for f_type in ['sterimol', 'sasa', 'buried_volume', 'free_volume', 'apt_sum', 'mulliken_sum', 'other']:
            df = df.drop(columns=feat_d[f_type])
        # df = General._clean_dist_contr(df)

        return df._get_numeric_data()

class CorrCheck():

    def __init__():
        pass

    def _get_target(targets):
        '''
        Function to pull the user specified target from a list of targets obtained earlier.
        
        Arguments:
        targets (List): A list of targets for ML analysis.

        Returns:
        target['target_select'] (String): The feature the user has specified as the target.
        '''
        questions = [inquirer.List('target_select',
                              message='Which target do you want to feature select for?',
                              choices=targets)]
        target = inquirer.prompt(questions)
        return target['target_select']
    
    @timeit
    def _check_corr(df, removed_features = None):
        '''
        Function to do standard pandas implementation of correlation between targets and features.
        Avoiding the features that have non integer values.

        Arguments:
        df (DataFrame): A DataFrame containing all the features.

        Returns:
        total_remove (List): A list of removed features.
        numeric_df (DataFrame): A DataFrame containing only numeric features.
        target_variables (List): A list of the target features based on hard coded string.
        '''

        def _high_corr(numeric_df, _method='pearson'):
            '''
            Function to extract high correlating features in the dataset. Made into a function incase wanted to run multiple times with 
            different methods.
            
            Arguments:
            numeric_df (DataFrame): A DataFrame of all the numeric values in the dataset only.
            _method (String): The method chosen for the correlation calculation (defaults to 'pearson', 'kendall' and 'spearman' also available.)
            
            Returns:
            high_corr (Dictionary): A dictionary of each feature and which features it is highly correlated with.
            '''
            # Find the correlation of each feature against the rest of the features and append perfectly collinear features in a dictionary.
            high_corr = {}
            for col in numeric_df.columns:
                cols = []
                # For correlation, pearson is used here but could be other method - 'kendall' or 'spearman'.
                corr = pd.DataFrame(numeric_df.corr(method=_method)[col][:]).reset_index(drop=False).rename(columns={'index':'feature'})
                for value in corr[col]:
                    if value == 1 or value == -1: # Removing same feature pairs.
                        pass
                    else:
                        if value > 0.99: # Cutoff value for highly correlated features.
                            cols.append(corr['feature'][corr[col] == value].iloc[0])
                if cols == []:
                    pass
                else:
                    if 'barrier' in col:
                        pass
                    else:
                        high_corr[col] = cols
            return high_corr
        
        # Convert data to numeric and highlight issue columns.
        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        if removed_features == None:
            removed_features = []
        else:
            removed_features = removed_features
        # Remove any columns that are non numeric
        for col in numeric_df.columns:
            if numeric_df[col].isnull().values.any() == True:
                removed_features.append(col)
                del numeric_df[col]

        # Isolate target variables 
        target_variables = []
        for col in numeric_df.columns:
            if 'barrier' in col or 'distortion' in col or 'interaction' in col:
                if 'dft' in col:
                    target_variables.append(col)
   
        high_corr = _high_corr(numeric_df, _method='pearson')
        logger.info((str(len(high_corr))+' features with near perfect collinearity.'))
        logger.info('Removing near perfect collinear features and checking again.')
        # Use while loop to check that no near perfectly collinear features slipped through.
        total_remove = []
        while len(high_corr) != 0:
            # Extract list of all features to be removed.
            keep = []
            remove = []
            for feature_key, feature_value in high_corr.items():
                if feature_key in remove:
                    pass
                else:
                    keep.append(feature_key)
                    remove = remove + feature_value
            # Remove these features.
            numeric_df = numeric_df.drop(columns=remove)
            total_remove = total_remove + remove
            high_corr = _high_corr(numeric_df, _method='pearson')
            logger.info((str(len(high_corr))+' features with near perfect collinearity.'))
        else:
            logger.info('Near perfectly collinear features removed.')
        total_remove = total_remove + removed_features

        return numeric_df, total_remove, target_variables

    @timeit
    def _remove_low_variance(df, targets, removed_features = None):
        '''
        Function to remove any features that have a low variance.

        Arguments:
        df (DataFrame): A DataFrame containing all the features.

        Returns:
        cleaned_df (DataFrame): A DataFrame containing the features that do not have a low variance.
        '''
        # Assess variance in features
        logger.info('Finding features with low variance.')
        #target = CorrCheck._get_target(targets)
        X = df.drop(columns=[target])
        all_features = X.columns.tolist()
        selector = VarianceThreshold(threshold=0.05) 
        selector.fit_transform(X)
        features = selector.get_feature_names_out()
        
        # Count and obtain the removed features.
        remove = []
        for feature in all_features:
            if feature not in features:
                remove.append(feature)
        logger.info((str(len(remove))+' features with low variance - removing.'))
        if removed_features == None:
            total_remove = remove
        else:
            total_remove = list(set(removed_features + remove))
        # Remove features and return 
        cleaned_df = df.drop(columns=remove)
        logger.info('Low variance features removed.\n---')

        return cleaned_df, total_remove
    
    def _runner(df_initial):
        '''
        Function to run the correlation checks - pulled out from main so that manual feature
        selection can be performed.
        
        Arguments:
        df_initial (DataFrame): A dataframe containing all the features.
        
        Returns:
        df (DataFrame): A dataframe containing the dataset with removed features.
        total_remove (List): A list of all removed features.
        '''
        df, total_remove, target_variables = CorrCheck._check_corr(df_initial)
        target = CorrCheck._get_target(target_variables)
        df, total_remove = CorrCheck._remove_low_variance(df, target_variables, total_remove)
        return df, total_remove, target

class PCA_Class():

    def __init__():
        pass

    def _PCA(df):
        '''
        Function to perform PCA analysis from sklearn implemenetation. Saves 10 components and returns the transformed X features in a 
        DataFrame with the target.
        
        Arguments:
        df (DataFrame): A DataFrame containing all the features before feature selection.
        
        Returns:
        comb (DataFrame): A DataFrame containing the principal components and the target.
        '''
        logger.info('PCA - Selecting 10 components.')
        # Remove reaction number from features.
        y = df[target] 
        df = df.drop(columns=General._get_dft_features(df))
        X = preprocessing.normalize(df.drop(columns=['reaction_number']), 'l2')
        components = 10
        component_list = []

        # Create component names.
        for value in range(0, components):
            component_list.append('principal_component_'+str(value+1))
        # Perform the PCA.
        pca = PCA(n_components=components)
        transformed_X = pd.DataFrame(pca.fit_transform(X), columns=component_list)
        exp_var_ratio = np.array(pca.explained_variance_ratio_*100).astype(float)
        comb = pd.concat([transformed_X, y], axis=1)
        # Plot results from PCA.
        fig, ax = plt.subplots(1, 2, figsize=(6,6))
        plt.subplots_adjust(left=0.2, wspace=0.3)
        ax[0].barh(list(reversed(component_list)), np.flip(exp_var_ratio), color='royalblue')
        ax[1].scatter(comb['principal_component_1'], comb['principal_component_2'], c=comb[target], cmap='plasma')
        ax[1].set_xlabel('pc1')
        ax[1].set_ylabel('pc2')
        plt.show()
        return comb, exp_var_ratio

    def _clean_runner(df, df_initial):
        '''
        Function to add key features back to the DataFrame of all features, namely the distortion contributions.

        Arguments:
        df (DataFrame): A DataFrame containing all the features before feature selection.
        df_initial (DataFrame): A DataFrame containing the combination of all the features from the start.

        Returns:
        new_df (DataFrame): A DataFrame containing the features with the distortion contributions added as a separate column.
        '''
        features = list(df.columns)
        for column in df_initial.columns:
            if 'contribution' in column:
                if column not in df.columns:
                    features.append(column)
        new_df = df_initial[features]
        new_df = General._clean_dist_contr(new_df).drop(columns=['distortion_contributions_am1', 'distortion_contributions_dft'])
        return new_df
        
class FS():

    def __init__():
        pass
 
    @timeit
    def _SFS(df, targets):
        '''
        Function to perform Sequential Feature Selection (SFS) on the dataset based upon a user specified target. These are model specific feature selection however, this should give a good initial guess and further feature selection can be done at a later date should 
        it be required.
        Code is written for forward SFS but can be modified to reverse. Hard coded to look for 100 features using Ridge regression as the model.

        Arguments:
        df (DataFrame): A DataFrame containing all the features.
        targets (List): A list of targets for ML analysis.

        Returns:
        features (np.array): A array of the selected features.
        '''
        logger.info('SFS using Ridge Regression - Selecting 100 features.')
        y = df[target] 
        df = df.drop(columns=General._get_dft_features(df))
        X = df.drop(columns=['reaction_number'])
        mod = SVR(kernel='rbf') # 1 hour 5 minutes
        #mod = Ridge() # 5 minutes
        sfs = SequentialFeatureSelector(mod, n_features_to_select=100)
        X_new = sfs.fit_transform(X, y)
        features = sfs.get_feature_names_out()
        logger.info('Best features extracted.')
        return features
        
    @timeit
    def _RFECV(df, targets):
        '''
        Function to perform Recursive Feature Elimination Cross Validation (RFECV) on the dataset based upon a user specified target. These are model specific feature selection however, this should give a good initial guess and further feature selection can be done at a later date should it be required.
        Hard coded to look for a minimum 100 features using Ridge regression as the model.

        Arguments:
        df (DataFrame): A DataFrame containing all the features.
        targets (List): A list of targets for ML analysis.

        Returns:
        features (np.array): A array of the selected features.
        '''
        logger.info('RFECV using Ridge Regression - Selecting 100 features.')
        y = df[target] 
        df = df.drop(columns=General._get_dft_features(df))
        X = df.drop(columns=['reaction_number'])
        mod = Ridge() # less than 30 seconds.
        rfecv = RFECV(mod, step=1, cv=5, min_features_to_select=100)
        X_new = rfecv.fit_transform(X, y)
        features = rfecv.get_feature_names_out()
        logger.info('Best features extracted')
        return features

    def _clean_runner(df_initial, total_remove, features):
        '''
        Function to run a clean on the data.
        
        Arguments:
        df (DataFrame): A DataFrame containing all the features after feature selection.

        Returns:

        '''
        # Add useful columns that got removed by being non numeric back into the DataFrame.
        useful = ['structure', 'reactant', 'path', 'distortion']
        re_add = ['reaction_number']
        for feature in useful:
            for removed_feature in total_remove:
                collect = re.findall(feature, removed_feature)
                if collect != []:
                    re_add.append(removed_feature)
                else:
                    pass
        features = list(set(list(features) + re_add))
        df = df_initial[features]
        return df

def main():

    # Initial load of data
    df_initial = General._load_data()
    #df_initial = df_initial.head(30) # Un-hash for testing

    # Set up logger
    global logger, start_formatter, formatter, fh, ch, target
    logger = logging.getLogger('feature_extraction')
    logger.setLevel('INFO')
    fh = logging.FileHandler('feat_select.log', mode='w')
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

    # Initial checks and remove near perfectly collinear features.
    logger.info(('Dataset shape before - '+str(df_initial.shape)+'\n---'))

    # Determine which feature selection method to choose.
    questions = [inquirer.List('feat_select',
                              message='Which feature selection method do you want?',
                              choices=['Manual', 'SFS', 'RFECV', 'PCA'])]
    answer = inquirer.prompt(questions)
    if answer['feat_select'] == 'Manual':
        df = Manual._manual_runner(df_initial)
        logger.info(('Dataset shape out - '+str(df.shape)+'\n---'))
        General._save_data(df, removed_features=None, exp_var_ratio=None)
        exit()
    elif answer['feat_select'] == 'SFS':
        df, total_remove, target_variables = CorrCheck._runner(df_initial)
        features = FS._SFS(df, target_variables)
        df = FS._clean_runner(df_initial, total_remove, features)
        df = General._clean_dist_contr(df)
    elif answer['feat_select'] == 'RFECV':
        df, total_remove, target_variables = CorrCheck._runner(df_initial)
        features = FS._RFECV(df, target_variables)
        df = FS._clean_runner(df_initial, total_remove, features)
        df = General._clean_dist_contr(df)
    elif answer['feat_select'] == 'PCA':
        df, total_remove, target_variables = CorrCheck._runner(df_initial)
        df = PCA_Class._clean_runner(df, df_initial)
        X, exp_var_ratio = PCA_Class._PCA(df)
        logger.info(('Dataset shape out - '+str(X.shape)+'\n---'))
        General._save_data(X, removed_features=None, exp_var_ratio=exp_var_ratio)
        return 

    # Clean distortion contributions features into individual columns.
    logger.info(('Dataset shape out - '+str(df.shape)+'\n---'))
    # Return the targets back to the DataFrame
    df = General._add_targets(df, df_initial, target_variables)
    # Save the data and removed features
    removed_features = General._get_removed_features
    General._save_data(df, removed_features=removed_features, exp_var_ratio=None)

if __name__ == '__main__':
    main()