"""
#####################################################################################
#                                  f_extract.py                                     #
#                                                                                   #
#      This code performs feature extraction on all the passed data based           #
#      upon a yaml file. Features extracted are primarily charges, distances        #
#      and other physical organic chemistry based features.                         #
#                                                                                   #
#      Usage:                                                                       #
#                                                                                   #
#      Ensure that there is a config.yaml file present in the CWD. The layout       #
#      should be as follows:                                                        #
#                                                                                   #
#           # Malonate                                                              #
#           dataset : mal                                                           #
#           type : ts                                                               #
#           path: path/to/files                                                     #
#           spe : True                                                              #
#           common_atoms :                                                          #
#           gs : {ma : [1,2,3,4,5], nu : [1,2,3,4,5,6,7,8,9,10]}                    #
#           dist_gs : {ma : [1,2,3,4,5], nu : [1,2,3,4,5,6,7,8,9,10]}               #
#           ts : [1,2,3,4,5,6,7,8,9,10]                                             #
#           ca_from_file : False                                                    #
#           verbose : 0                                                             #
#                                                                                   #
#      python f_extract.py                                                          #
#                                                                                   #
#      This will generate a .pkl file of features. Once all feature have been       #
#      generated, the code can be used to collate the features into a single        #
#      dataframe as follows:                                                        #
#                                                                                   #
#      python f_extract.py -c                                                       #  
#                                                                                   #
#      This relies on having the following in the config.yaml file.                 #
#                                                                                   #
#           # Malonate                                                              #
#           gs_path : path/to/data/mal_gs.pkl                                       #
#           ts_path : path/to/data/mal_ts.pkl                                       #
#           dist_gs_path : path/to/data/mal_dist_gs.pkl                             #
#           am1_barriers_path : path/to/data/am1_ma_barriers.pkl                    #
#           dft_barriers_path :path/to/data/dft_ma_barriers.pkl                     #
#           filename : mal_features.pkl                                             #
#                                                                                   #                                                                               #
#####################################################################################
#                                       Authors                                     #
#                                                                                   #
#--------------------------------- Samuel G. Espley --------------------------------#
#                                                                                   #
#####################################################################################
"""

# Imports
import os
import time
import yaml
import cclib
import pickle
import logging
import argparse
import itertools
import numpy as np
import pandas as pd
from functools import wraps
from threading import Thread
import matplotlib.pyplot as plt
from morfeus import BuriedVolume, SASA, Sterimol, read_geometry

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
        logger.info(f'Function {func.__name__} took {total_time:.4f} seconds')
        if setup['verbose'] == 0:
            pass
        elif setup['verbose'] == 1:
            print(f'Function \033[1m{func.__name__}\033[0m took \033[92m{total_time:.4f}\033[0m seconds')
        return result
    return timeit_wrapper

class General():

    def __init__():
        pass

    def _load_config():
        '''
        Function to load in the configuration file for energy extraction.

        Returns
        setup (Dictionary): A dictionary containing setup information for extracting features.
        '''
        global setup
        with open('config.yaml', 'r') as file:
            setup = yaml.safe_load(file)
        return setup
    
    def _logger_finish(message):
        '''
        Small function to create the output for the logger after a process has complete.
        '''
        text = message+'\n---'
        logger.info(text)

    def _get_combinations(lst, take, reverse):
        '''
        Function to get every combination of items from a list.

        Arguments:
        lst (List): A list of the different values you want to calculate the combinations for.
        take (Integer): The value of the combinations you want e.g., take=2 means it gets paired combinations.
        reverse (Boolean): True or False to either get the reverse combinations in the list or not.

        Returns:
        combinations (List): A list of all combinations in a list format.
        '''
        combinations = list(itertools.combinations(lst, take))
        combinations = [list(t) for t in combinations]
        if reverse == False:
            return combinations
        else:
            combinations_reversed = []
            for pair in combinations:
                combinations_reversed.append(list(reversed(pair)))
            combinations = combinations + combinations_reversed

            return combinations

    def _setup(setup):
        '''
        Function to take the config file and prepare the information for extraction.

        Arguments:
        setup (Dictionary): A dictionary containing setup information for extracting features.
        
        Returns:
        dataset (String): A string of the dataset type.
        job_type (String): A string of the job type asked for.
        common_atoms (Dictionary): A dictionary of the common atoms to be extracted.
        '''
        # Extract type of job
        dataset = setup['dataset']
        job_type = setup['type']
        spe = setup['spe']

        if 'mol_types' in list(setup.keys()):
            if setup['mol_types'] != None:
                m_types = open(setup['mol_types'], 'rb')
                mol_types = pickle.load(m_types)
        else:
            mol_types = None

        if setup['ca_from_file'] == True:

            if setup['ca_file'] != None:
                try:
                    ca_file = open(setup['ca_file'], 'rb')
                    map_file = open(setup['mapping_file'], 'rb')
                    mapping = pickle.load(map_file)
                    common_atoms = pickle.load(ca_file)
                except:
                    print('Common atoms file not found.')
                    common_atoms = None
                    mapping = None
            else:
                print('Specify the common atoms file.')

        elif setup['ca_from_file'] == False:
            common_atoms = setup['common_atoms']
            mapping = None

        if common_atoms == None and mapping == None:
            print('Fail in loading common atoms.')
            exit()
        else:    
            return dataset, job_type, common_atoms, mapping, mol_types, spe

    def _load_files(job_type):
        '''
        Function to load the files and pull out the reaction number from the all the files in the current working directory. 
        Done as function to avoid running every instance.

        Arguments:
        job_type (String): The type of job pulled from the setup dictionary.

        Returns:
        data_dict (Dictionary): A dictionary containing 'meta data' and cclib extracted features such as charges and coordinates.
        '''
        la_dict = {
                'alcl3_1_3':0,
                'alcl3_1_4':1,
                'bf3_1_3':2, 
                'bf3_1_4':3,
                'i2_1_3':4,
                'i2_1_4':5, 
                'zncl2_1_3':6,
                'zncl2_1_4':7
            }
        files = []
        # Temporary line to check code works via a specific path - marked with # at the end.
        origin = os.getcwd()
        os.chdir(setup['path'])
        path = os.getcwd()
        for filename in os.listdir():
            if filename.endswith(".out") or filename.endswith(".log"):
                if 'SPE' in filename:
                    pass
                elif 'nuc' in filename:
                    pass
                elif setup['dataset'] == 'ma' and setup['type'] == 'dist_gs' and 'reactant_2' in filename:
                    pass
                elif setup['dataset'] == 'la_da' and setup['type'] == 'dist_gs' or setup['dataset'] == 'la_da' and setup['type'] == 'gs':
                    if 'dien' not in filename:
                        pass
                    else:
                        files.append(filename)
                else:
                    files.append(filename)
            else:
                continue
        # Alternate way of doing it where the structure is {file:rxn_number}
        rxn_file_dict = {}
        for file in files:
            if setup['dataset'] == 'ma':
                rxn_number = file.split('-')[1]
            elif setup['dataset'] == 'la_da':
                t_d = {'ts':3, 'gs':2, 'dist_gs':4}
                f = file.split('.out')[0]
                nme = f"{f.split('_')[0]}_{f.split('_', t_d[setup['type']])[-1]}"
                rxn_number = la_dict[nme]
            else:
                rxn_number = file.split('_')[1].replace('.out', '')
            rxn_file_dict[file] = rxn_number
        if setup['dataset'] == 'tt':
            data_dict = TT._load(files, job_type, path)
        elif setup['dataset'] == 'da':
            data_dict = DA._load(files, job_type, path) 
        elif setup['dataset'] == 'la_da':
            data_dict = LA_DA._load(files, job_type, path) 
        elif setup['dataset'] == 'ma':
            data_dict = MA._load(files, job_type, path)
        elif setup['dataset'] == 'mal':
            data_dict = MAL._load(files, job_type, path)
        os.chdir(origin) 
        return data_dict

    @timeit
    def _combine_features(feature_list, rxn_data):
        '''
        Function to combine all extracted features into a dataframe.

        Arguments:
        feature_list (List): A list of dictionaries that contain all the features. A specific order with data_dict first.
        rxn_data (Dictionary): A dictionary containing information about the dataset.

        Returns:
        combined_df (DataFrame): A dataframe of all the features combined for the entire dataset.
        '''
        information = ['reaction_number']
        completed_rxns = {}
        seen = []
        df_combined = pd.DataFrame()
        # Loop through all structures.
        for structure in feature_list[0].keys():
            rxn_number = feature_list[0][structure]['reaction_number']
            f_d = {}
            # Combine the features.
            combined_features = {**feature_list[0][structure], **feature_list[1][structure], **feature_list[2][structure], **feature_list[3][structure]}
            f_d[structure] = combined_features
            df = pd.DataFrame.from_dict(f_d, orient='index')
            if setup['type'] == 'gs' or setup['type'] == 'dist_gs':
                # Add check for ma dataset as requires different process to get suffix.
                if setup['dataset'] == 'ma':
                    suf = '_ma'
                else:
                    suf = '_'+rxn_data[rxn_number][feature_list[0][structure]['reactant']]['type']
            elif setup['type'] == 'ts':
                suf = '_ts'
            changes = {}
            for column in df.columns:
                if column in information:
                    pass
                else:
                    changes[column] = column+suf
            # Manipulate the dataframe by changing column names and index.
            df = df.rename(columns=changes)
            df = df.reset_index(drop=False).rename(columns={'index':'structure'})
            df = df.rename(columns={'structure':'structure'+suf})
            # Combine features for ground state reactants if they have the same reaction number.
            if setup['type'] == 'gs' or setup['type'] == 'dist_gs':
                if setup['dataset'] == 'ma':
                    df_combined = pd.concat([df, df_combined])
                elif setup['dataset'] == 'mal':
                    df_combined = pd.concat([df, df_combined])
                else:
                    if df['reaction_number'].loc[0] in seen:
                        df_rxn = df.merge(completed_rxns[df['reaction_number'].loc[0]], on='reaction_number')
                        df_combined = pd.concat([df_combined, df_rxn])
                    else:
                        completed_rxns[df['reaction_number'].loc[0]] =  df
                        seen.append(df['reaction_number'].loc[0])
            elif setup['type'] == 'ts':
                df_combined = pd.concat([df_combined, df])
        # Fix the pseudo label used for the Michael Addition dataset and remove unwanted ts features.
        if setup['dataset'] == 'ma' :
            df_combined.columns = df_combined.columns.str.replace('di', 'ma')
            df_combined.columns = df_combined.columns.str.replace('mas', 'dis')
            if setup['type'] == 'ts':
                df_combined = df_combined[df_combined.columns.drop(list(df_combined.filter(regex='dp')))]
        else:
            pass
        
        # Clean the features for the malonate data.
        if setup['dataset'] == 'mal' and setup['type'] == 'dist_gs':
            cleaned_df_combined = pd.DataFrame()
            rxn_nums = list(set(df_combined['reaction_number'].tolist()))
            for rn in rxn_nums:
                temp_df = df_combined[df_combined['reaction_number'] == rn]
                comb = pd.DataFrame(temp_df.iloc[0]).transpose().dropna(axis='columns').merge(pd.DataFrame(temp_df.iloc[-1]).transpose().dropna(axis='columns'), on='reaction_number').reset_index()
                cleaned_df_combined = pd.concat([cleaned_df_combined, comb], axis=0)
            df_combined = cleaned_df_combined

        return df_combined

    def _collate_features():
        '''
        Function to combine the features generated from previous runs.
        '''
        
        setup = General._load_config()

        filename = setup['filename']
        paths = {'gs_path':setup['gs_path'], 'ts_path':setup['ts_path'], 'dist_gs_path':setup['dist_gs_path'],
            'am1_barriers_path':setup['am1_barriers_path'], 'dft_barriers_path':setup['dft_barriers_path']}
        
        df_d = {}
        for name, path in paths.items():
            file_type = name.split('_path')[0]
            df_d[file_type] = pd.read_pickle(path)
               
        df = pd.DataFrame()
        for feature_type in df_d.keys():
            if 'barriers' in feature_type:
                suf = '_am1' if 'am1' in feature_type else '_dft' if 'dft' in feature_type else 0
                temp_df = df_d[feature_type]
                temp_df = temp_df.add_suffix(suf)
                temp_df = temp_df.rename(columns={str('id'+suf):'reaction_number'})
                temp_df = temp_df.sort_values('reaction_number')
                temp_df['reaction_number'] = temp_df['reaction_number'].astype('int')
                df = temp_df if df.empty else df.merge(temp_df, on='reaction_number')
            else:
                suf = '_'+feature_type
                temp_df = df_d[feature_type]
                temp_df = temp_df.add_suffix(suf)
                temp_df = temp_df.rename(columns={str('reaction_number'+suf):'reaction_number'})
                temp_df['reaction_number'] = temp_df['reaction_number'].astype('int')
                temp_df = temp_df.sort_values('reaction_number')
                df = temp_df if df.empty else df.merge(temp_df, on='reaction_number')
        
        df.to_pickle(filename)
        print(f'Features and barriers combined and saved to {filename}')

    def _save(df):
        '''
        Function to save the extracted features.

        Arguments:
        df (DataFrame): A DataFrame containing all the extracted features.
        '''
        # Create filename based on setup information
        filename = setup['dataset']+'_'+setup['type']+'.pkl'
        df.to_pickle(filename)

        return filename

    def _runner():

        if options.combine:
            General._collate_features()
        else:   
            datasets = {'tt': '[3+2] Cycloaddition', 'da':'Diels-Alder', 'la_da':'Lewis-Acid Diels-Alder', 'ma':'Michael Addition', 'mal':'Malonate Michael Addition'}
            # Load and create dictionaries.
            setup = General._load_config()
            dataset, job_type, common_atoms, mapping, file_names, spe = General._setup(setup)
            # Save information to logger and reformat
            logger.info(datasets[dataset])
            fh.setFormatter(formatter)
            # Load files
            logger.info('Loading data')
            data_dict = General._load_files(job_type)

            # Dealing with [3+2] dataset.
            if dataset == 'tt':
                # Perform clean on mapping, common_atoms and file_names.
                new_map = TT._clean_dictionary(mapping)
                new_ca = TT._clean_dictionary(common_atoms)
                m_types = TT._tidy_mol_types(file_names)
                new_map = TT._check_common_atoms(new_ca, new_map, m_types)
                # Get rxn_data dictionary.
                if job_type == 'gs' or job_type == 'dist_gs':
                    rxn_data = TT._get_gs_common_atoms(new_ca, new_map, m_types, file_names)
                elif job_type == 'ts':
                    rxn_data = TT._get_ts_common_atoms(new_ca)
                else:
                    print('Error in job type - check yaml file.')
            # Dealing with Diels-Alder dataset.
            elif dataset == 'da':
                # Get rxn_data dictionary.
                print('Diels-Alder\n---')
                rxn_data = DA._clean_diels_alder(data_dict)
            # Dealing with the Lewis-Acid Diels-Alder dataset.
            elif dataset == 'la_da':
                # Get rxn_data dictionary.
                print('Lewis-Acid Diels-Alder\n---')
                rxn_data = LA_DA._clean_la_diels_alder(data_dict)
            # Dealing with Michael Addition dataset.
            elif dataset == 'ma':
                # Get rxn_data dictionary.
                print('Michael Addition\n---') 
                rxn_data = MA._clean_michael_addition(data_dict)
            elif dataset == 'mal':
                print('Michael Addition Malonate\n---') 
                rxn_data = MAL._clean_malonate(data_dict)
            General._logger_finish('Loading data complete!\n---')
            
            # Extract morfeus features.
            logger.info('Starting feature extraction.\n---')
            logger.info('Extract morfeus features.')
            morf_buried_vol = Extract._morfeus_features(data_dict, rxn_data)
            General._logger_finish('Morfues features finished.')

            # Extract distances features.
            logger.info('Extract distances.')
            distances = Extract._distances(data_dict, rxn_data)
            General._logger_finish('Extracting distances finished.')
            
            # Extract charges features.
            logger.info('Extract charges.')
            charges = Extract._charges(common_atoms, data_dict, rxn_data)
            General._logger_finish('Extracting charges finished.')   
            
            # Combine all features.
            logger.info('Combine features.')
            df = General._combine_features([data_dict, morf_buried_vol, distances, charges], rxn_data)
            General._logger_finish('Combining features finished.')

            # Save features.
            logger.info('Saving features.')      
            filename = General._save(df)  
            General._logger_finish(f'Saving features finished - saved to {filename}.')
            General._logger_finish('Feature extraction finished.')

class TT():

    def __init__():
        pass

    def _load(files, job_type, path):
        '''
        Function to load files and get rxn_data dictionary.

        Arguments:
        files (List): A list of files to load in.
        job_type (String): What job type you intend to run.
        path (String): The main section of the path without the filename added.

        Returns:
        data_dict (Dictionary): A dictionary containing 'meta data' and cclib extracted features such as charges and coordinates.
        '''
        data_dict = {}
        d = {}
        for file in files:
            if job_type == 'gs' or job_type == 'dist_gs':
                filename = 'reactant_'+file.partition('reactant_')[2][0:1]
                number = file.split('_')[1]
            else:
                filename = file
                number = file.split('_')[1].replace('.out', '')
            d[file] = cclib.io.ccread(file)
            # Create the data_dict with cclib and 'meta data'
            data_dict[file] = {}
            data_dict[file]['reaction_number'] = number
            data_dict[file]['reactant'] = filename
            data_dict[file]['path'] = path+'\\'+file                
            data_dict[file]['coords'] = d[file].atomcoords[-1]      
            data_dict[file]['mulliken_charge'] = d[file].atomcharges['mulliken']
            data_dict[file]['mulliken_sum'] = d[file].atomcharges['mulliken_sum']
            data_dict[file]['apt_charge'] = d[file].atomcharges['apt']
            data_dict[file]['apt_sum'] = d[file].atomcharges['apt_sum']

        return data_dict
    
    def _check_common_atoms(new_ca, new_map, m_types):
        '''
        Function to check that the common atoms are present in the mapping for that specific structure. 
        If they are not, it will check the other reactant and fix and mistakes from the common atoms generation.
        
        Arguments:
        new_ca (Dictionary): A dictionary of all the tt common atoms for the TS structure.
        new_map (Dictionary): A dictionary of all the tt mapped atoms from TS structure to reactant 1 and reactant 2.
        m_types (Dictionary): A dictionary explaining the type of species of each reactant. Either di or dp for dipolar or dipolarophile respectively.

        Returns:
        adjusted_map (Dictionary): A dictionary of the adjusted and corrected mapping.
        '''

        # Determine the incorrect mappings.
        not_correct = []
        for rxn in m_types.keys():
            for molecule_type in m_types[rxn].keys():
                atom_number = new_ca[rxn][m_types[rxn][molecule_type]][0]
                if atom_number not in list(new_map[rxn][molecule_type].keys()): 
                    not_correct.append(rxn)
        not_correct = list(set(not_correct))  

        # Create the corrected mapping dictionary.
        corrected_d = {}
        for rxn in not_correct:
            corrected_d[rxn] = {'reactant_1':new_map[rxn]['reactant_2'], 'reactant_2':new_map[rxn]['reactant_1']}
            
        # Merge the two dictionaries.
        adjusted_map = {**new_map, **corrected_d}

        # Determine if any incorrect structures remain.
        not_correct = []
        for rxn in m_types.keys():
            for molecule_type in m_types[rxn].keys():
                atom_number = new_ca[rxn][m_types[rxn][molecule_type]][0]
                if atom_number not in list(adjusted_map[rxn][molecule_type].keys()): 
                    not_correct.append(rxn)
        not_correct = list(set(not_correct))  
    
        return adjusted_map
    
    def _clean_dictionary(d):
        '''
        Function to remove _reoptX's from file names.

        Arguments:
        d (Dictionary): A dictionary containing the mapping or common atoms information.

        Returns:
        adjusted_d (Dictionary): An adjusted dictionary with cleaned reaction names.
        '''
        new_d = {}
        for structure in d:
            if 'reopt' in structure:
                name = structure.replace('_reopt3.out', '.out')
                name  = name.replace('_reopt2.out', '.out')
                name  = name.replace('_reopt.out', '.out')
            else:
                name = structure
            if '.out' in structure:
                new_filename = name.split('_')[1].replace('.out','')
            else:
                new_filename = structure
            new_d[new_filename] = d[structure]

        return new_d

    def _tidy_mol_types(d):
        '''
        Function to tidy the mol_types dictionary for the tt dataset.

        Arguments:
        d (Dictionary): A dictionary loaded in that contains the molecule types for the tt dataset.

        Returns:
        new_d (Dictionary): A dictionary witht he adjusted dictionary to simplify later work.
        '''
        new_d = {}
        for rxn in d.keys():
            new_rxn_d = {}
            for reactant in d[rxn].keys():
                if 'reactant_1' in reactant:
                    new_rxn_d['reactant_1'] = d[rxn][reactant]
                elif 'reactant_2' in reactant:
                    new_rxn_d['reactant_2'] = d[rxn][reactant]
                else:
                    print('error')                   
            new_d[rxn] = new_rxn_d

        return new_d

    def _tidy_ca(d):
        '''
        Function to tidy the common atoms dictionary and set the rxn number as the primary key.

        Arguments:
        d (Dictionary): A dictionary of common atoms with file name as the key.

        Returns:
        new_d (Dictionary): A dictionary of common atoms with the rxn number as the key.
        '''
        new_d = {}
        for filename in d.keys():
            rxn_d = {}
            for key in d[filename]:
                if key in ['di', 'dp', 'reacting']:
                    rxn_d[key] = list(map(lambda i : i + 1, d[filename][key]))
                else:
                    rxn_d[key] = d[filename][key]
            new_d[filename] = rxn_d
        return new_d
    
    def _get_gs_common_atoms(new_ca, new_map, m_types, file_names):
        '''
        Function to get the common atoms for the ground state reactant and distorted structures.
        Utilises the mapping and common atoms for the TS structures.

        Arguments:
        new_ca (Dictionary): A dictionary of all the tt common atoms for the TS structure.
        new_map (Dictionary): A dictionary of all the tt mapped atoms from TS structure to reactant 1 and reactant 2.
        m_types (Dictionary): A dictionary explaining the type of species of each reactant. Either di or dp for dipolar or dipolarophile respectively.
        file_names (Dictionary): A dictionary that contains the file names of all the respective structures in the common atoms dictioanry.

        Returns:
        rxn_dict (Dictionary): A dictionary of the common atoms with associated filenames.
        '''      
        rxn_dict = {}
        for structure in new_ca.keys():
            if structure not in list(m_types.keys()):
                logger.warning(str(structure+' - not in m_types'))
                pass
            else:
                species_d = {}
                for species in ['di', 'dp']:
                    reactant_type = list(m_types[structure].keys())[list(m_types[structure].values()).index(species)]
                    f_name = list(file_names[structure].keys())[list(file_names[structure].values()).index(species)]
                    if 'reactant_1' in f_name:
                        name = 'reactant_1'
                    elif 'reactant_2' in f_name:
                        name = 'reactant_2'
                    else:
                        print(f_name, ' has incorrect format - check structure.')
                    common_atoms = []
                    for atom in new_ca[structure][species]:
                        mapped_atom = new_map[structure][reactant_type][atom]
                        common_atoms.append(mapped_atom)
                    species_d[name] = {'common_atoms':common_atoms, 'type':species}
            
                rxn_dict[structure] = species_d
        return rxn_dict

    def _get_ts_common_atoms(new_ca):
        '''
        Function to clean up the common atoms dictionary to get the same format as is for the ground state reactant dictionaries.

        Arguments:
        new_ca (Dictionary): A dictionary of all the tt common atoms for the TS structure.

        Returns:
        rxn_dict (Dictionary): A dictionary of the common atoms with associated filenames.
        '''
        rxn_data = new_ca.copy()
        for structure in rxn_data:
            rxn_data[structure]['file'] = rxn_data[structure].pop('name') 
        return rxn_data

class LA_DA():

    def __init__():
        pass

    def _clean_la_diels_alder(data_dict):
        '''
        Function to clean the Lewis-Acid Diels-Alder data to generate rxn_data

        Arguments:
        data_dict (Dictionary): A dictionary containing 'meta data' and cclib extracted features such as charges and coordinates.

        Returns:
        rxn_data (Dictionary): A dictionary containing information about the dataset.        
        '''
        la_dict = {
                'alcl3_1_3':0,
                'alcl3_1_4':1,
                'bf3_1_3':2, 
                'bf3_1_4':3,
                'i2_1_3':4,
                'i2_1_4':5, 
                'zncl2_1_3':6,
                'zncl2_1_4':7
            }
        reactant_d = {'reactant_1':'di', 'reactant_2':'dp'}
        rxn_data = {}
        # Loop through structures in data_dict.
        for structure in data_dict.keys():
            # Extract specific depending on gs, dist_gs or ts and create dictionary.
            if setup['type'] == 'dist_gs':
                rxn_number = data_dict[structure]['reaction_number']
                reactant_name = reactant_d[data_dict[structure]['reactant']]
                reactant_type = data_dict[structure]['reactant']
                d = {reactant_type:{'common_atoms':setup['common_atoms'][setup['type']][reactant_name], 'type':reactant_name}}
            elif setup['type'] == 'gs':
                rxn_number = data_dict[structure]['reaction_number']
                reactant_name = reactant_d[data_dict[structure]['reactant']]
                reactant_type = data_dict[structure]['reactant']
                d = {reactant_type:{'common_atoms':setup['common_atoms'][setup['type']][reactant_name], 'type':reactant_name}}
            elif setup['type'] == 'ts':
                rxn_number = data_dict[structure]['reaction_number']
                reacting_atoms = [setup['common_atoms']['gs']['di'][0], setup['common_atoms']['gs']['dp'][0], 
                                  setup['common_atoms']['gs']['di'][-1], setup['common_atoms']['gs']['dp'][-1]]
                rxn_data[rxn_number] = {'di':setup['common_atoms']['gs']['di'], 'dp':setup['common_atoms']['gs']['dp'], 'reacting':reacting_atoms, 'file':structure}
            if setup['type'] == 'gs' or setup['type'] == 'dist_gs':
                # Check if rxn_number is new or not and create combined dictionary.
                if rxn_number not in list(rxn_data.keys()):
                    rxn_data[rxn_number] = d
                else:
                    combined = {**rxn_data[rxn_number], **d}
                    rxn_data[rxn_number] = combined
                
        return rxn_data

    def _load(files, job_type, path):
        '''
        Function to load files and get rxn_data dictionary.

        Arguments:
        files (List): A list of files to load in.
        job_type (String): What job type you intend to run.
        path (String): The main section of the path without the filename added.

        Returns:
        data_dict (Dictionary): A dictionary containing 'meta data' and cclib extracted features such as charges and coordinates.
        '''
        la_dict = {
                'alcl3_1_3':0,
                'alcl3_1_4':1,
                'bf3_1_3':2, 
                'bf3_1_4':3,
                'i2_1_3':4,
                'i2_1_4':5, 
                'zncl2_1_3':6,
                'zncl2_1_4':7
            }
        t_d = {'ts':3, 'gs':2, 'dist_gs':4}
        

        data_dict = {}
        d = {}
        for file in files:
            f = file.split('.out')[0]
            nme = f"{f.split('_')[0]}_{f.split('_', t_d[setup['type']])[-1]}"
            number = la_dict[nme]
            if job_type == 'gs':
                if 'diene' in file:
                    filename = 'reactant_1'
                elif 'dienophile' in file:
                    filename = 'reactant_2' 
            elif job_type == 'dist_gs':
                if 'diene' in file:
                    filename = 'reactant_1'
                elif 'dienophile' in file:
                    filename = 'reactant_2' 
            else:
                filename = file
            d[file] = cclib.io.ccread(file)
            data_dict[file] = {}
            data_dict[file]['reaction_number'] = number
            data_dict[file]['reactant'] = filename
            data_dict[file]['path'] = path+'\\'+file                
            data_dict[file]['coords'] = d[file].atomcoords[-1]      
            data_dict[file]['mulliken_charge'] = d[file].atomcharges['mulliken']
            data_dict[file]['mulliken_sum'] = d[file].atomcharges['mulliken_sum']
            data_dict[file]['apt_charge'] = d[file].atomcharges['apt']
            data_dict[file]['apt_sum'] = d[file].atomcharges['apt_sum']

        return data_dict

class DA():

    def __init__():
        pass

    def _clean_diels_alder(data_dict):
        '''
        Function to clean the Diels-Alder data to generate rxn_data

        Arguments:
        data_dict (Dictionary): A dictionary containing 'meta data' and cclib extracted features such as charges and coordinates.

        Returns:
        rxn_data (Dictionary): A dictionary containing information about the dataset.        
        '''
        rxn_data = {}
        # Loop through structures in data_dict.
        for structure in data_dict.keys():
            # Extract specific depending on gs, dist_gs or ts and create dictionary.
            if setup['type'] == 'dist_gs':
                rxn_number = structure.split('_reactant')[0].split('_')[-1]
                if 'reactant_1' in structure:
                    reactant_name = 'di'
                    reactant_type = 'reactant_1'
                elif 'reactant_2' in structure:
                    reactant_name = 'dp'
                    reactant_type = 'reactant_2'
                d = {reactant_type:{'common_atoms':setup['common_atoms'][setup['type']][reactant_name], 'type':reactant_name}}
            elif setup['type'] == 'gs':
                rxn_number = structure.split('.out')[0].split('_')[-1]
                if 'di' in structure:
                    reactant_name = 'di'
                    reactant_type = 'reactant_1'
                elif 'dp' in structure:
                    reactant_name = 'dp'
                    reactant_type = 'reactant_2'
                d = {reactant_type:{'common_atoms':setup['common_atoms'][setup['type']][reactant_name], 'type':reactant_name}}
            elif setup['type'] == 'ts':
                rxn_number = structure.split('.out')[0].split('_')[-1]
                reacting_atoms = [setup['common_atoms']['gs']['di'][0], setup['common_atoms']['gs']['dp'][0], 
                                  setup['common_atoms']['gs']['di'][-1], setup['common_atoms']['gs']['dp'][-1]]
                rxn_data[rxn_number] = {'di':setup['common_atoms']['gs']['di'], 'dp':setup['common_atoms']['gs']['dp'], 'reacting':reacting_atoms, 'file':structure}
            if setup['type'] == 'gs' or setup['type'] == 'dist_gs':
                # Check if rxn_number is new or not and create combined dictionary.
                if rxn_number not in list(rxn_data.keys()):
                    rxn_data[rxn_number] = d
                else:
                    combined = {**rxn_data[rxn_number], **d}
                    rxn_data[rxn_number] = combined
                
        return rxn_data

    def _load(files, job_type, path):
        '''
        Function to load files and get rxn_data dictionary.

        Arguments:
        files (List): A list of files to load in.
        job_type (String): What job type you intend to run.
        path (String): The main section of the path without the filename added.

        Returns:
        data_dict (Dictionary): A dictionary containing 'meta data' and cclib extracted features such as charges and coordinates.
        '''
        data_dict = {}
        d = {}
        for file in files:
            if job_type == 'gs':
                if 'di' in file:
                    filename = 'reactant_1'
                elif 'dp' in file:
                    filename = 'reactant_2'
                number = file.split('_')[2].split('.out')[0]    
            elif job_type == 'dist_gs':
                filename = 'reactant_'+file.partition('reactant_')[2][0:1]
                number = file.split('_')[2]
            else:
                filename = file
                number = file.split('_')[2].replace('.out', '')
            d[file] = cclib.io.ccread(file)
            data_dict[file] = {}
            data_dict[file]['reaction_number'] = number
            data_dict[file]['reactant'] = filename
            data_dict[file]['path'] = path+'\\'+file                
            data_dict[file]['coords'] = d[file].atomcoords[-1]      
            data_dict[file]['mulliken_charge'] = d[file].atomcharges['mulliken']
            data_dict[file]['mulliken_sum'] = d[file].atomcharges['mulliken_sum']
            data_dict[file]['apt_charge'] = d[file].atomcharges['apt']
            data_dict[file]['apt_sum'] = d[file].atomcharges['apt_sum']

        return data_dict

class MA():

    def __init__():
        pass

    def _clean_michael_addition(data_dict):
        '''
        Function to clean the Michael Addition data to generate rxn_data

        Arguments:
        data_dict (Dictionary): A dictionary containing 'meta data' and cclib extracted features such as charges and coordinates.

        Returns:
        rxn_data (Dictionary): A dictionary containing information about the dataset.        
        '''
        rxn_data = {}
        for structure in data_dict.keys():
            rxn_number = structure.split('-')[1]
            if setup['type'] == 'dist_gs' or setup['type'] == 'gs':
                reactant_name = 'ma'
                reactant_type = 'reactant_1'
                d = {reactant_type:{'common_atoms':setup['common_atoms'][setup['type']][reactant_name], 'type':reactant_name}}                
                if rxn_number not in list(rxn_data.keys()):
                    rxn_data[rxn_number] = d
                else:
                    combined = {**rxn_data[rxn_number], **d}
                    rxn_data[rxn_number] = combined
            if setup['type'] == 'ts':
                reacting_atoms = [setup['common_atoms']['ts'][0], setup['common_atoms']['ts'][4]]
                rxn_data[rxn_number] = {'di':setup['common_atoms']['gs']['ma'], 'dp':setup['common_atoms']['gs']['nu'], 'reacting':reacting_atoms, 'file':structure}

        return rxn_data
           
    def _load(files, job_type, path):
        '''
        Function to load files and get rxn_data dictionary.

        Arguments:
        files (List): A list of files to load in.
        job_type (String): What job type you intend to run.
        path (String): The main section of the path without the filename added.

        Returns:
        data_dict (Dictionary): A dictionary containing 'meta data' and cclib extracted features such as charges and coordinates.
        '''
        data_dict = {}
        d = {}
        for file in files:
            if job_type == 'gs' or job_type == 'dist_gs':
                filename = 'reactant_1'
                number = file.split('-')[1]   
            else:
                filename = file
                number = file.split('-')[1]   
            d[file] = cclib.io.ccread(file)
            data_dict[file] = {}
            data_dict[file]['reaction_number'] = number
            data_dict[file]['reactant'] = filename
            data_dict[file]['path'] = path+'\\'+file                
            data_dict[file]['coords'] = d[file].atomcoords[-1]      
            data_dict[file]['mulliken_charge'] = d[file].atomcharges['mulliken']
            data_dict[file]['mulliken_sum'] = d[file].atomcharges['mulliken_sum']
            data_dict[file]['apt_charge'] = d[file].atomcharges['apt']
            data_dict[file]['apt_sum'] = d[file].atomcharges['apt_sum']

        return data_dict

class MAL():

    def __init__():
        pass

    def _clean_malonate(data_dict):
        '''
        Function to clean the Michael Addition data to generate rxn_data

        Arguments:
        data_dict (Dictionary): A dictionary containing 'meta data' and cclib extracted features such as charges and coordinates.

        Returns:
        rxn_data (Dictionary): A dictionary containing information about the dataset.        
        '''
        rxn_data = {}
        for structure in data_dict.keys():
            rxn_number = structure.split('_')[1]
            if setup['type'] == 'dist_gs' or setup['type'] == 'gs':
                if 'reactant_1' in structure:
                    reactant_name = 'ma'
                elif 'reactant_2' in structure:
                    reactant_name = 'nu'
                else:
                    reactant_name = 'ma'
                reactant_type = data_dict[structure]['reactant']
                d = {reactant_type:{'common_atoms':setup['common_atoms'][setup['type']][reactant_name], 'type':reactant_name}}                
                if rxn_number not in list(rxn_data.keys()):
                    rxn_data[rxn_number] = d
                else:
                    combined = {**rxn_data[rxn_number], **d}
                    rxn_data[rxn_number] = combined
            if setup['type'] == 'ts':
                reacting_atoms = [setup['common_atoms']['ts'][0], setup['common_atoms']['ts'][4]]
                rxn_data[rxn_number] = {'di':setup['common_atoms']['gs']['ma'], 'dp':setup['common_atoms']['gs']['nu'], 'reacting':reacting_atoms, 'file':structure}
        return rxn_data    

    def _load(files, job_type, path):
        '''
        Function to load files and get rxn_data dictionary.

        Arguments:
        files (List): A list of files to load in.
        job_type (String): What job type you intend to run.
        path (String): The main section of the path without the filename added.

        Returns:
        data_dict (Dictionary): A dictionary containing 'meta data' and cclib extracted features such as charges and coordinates.
        '''
        data_dict = {}
        d = {}
        for file in files:
            if job_type == 'gs' or job_type == 'dist_gs':
                filename = f"reactant_{file.split('_')[-1].replace('.log', '')}"
                number = file.split('_')[1]
            else:
                filename = file
                number = file.split('_')[1]
            d[file] = cclib.io.ccread(file)
            data_dict[file] = {}
            data_dict[file]['reaction_number'] = number
            data_dict[file]['reactant'] = filename
            data_dict[file]['path'] = path+'\\'+file                
            data_dict[file]['coords'] = d[file].atomcoords[-1]      
            data_dict[file]['mulliken_charge'] = d[file].atomcharges['mulliken']
            data_dict[file]['mulliken_sum'] = d[file].atomcharges['mulliken_sum']
            data_dict[file]['apt_charge'] = d[file].atomcharges['apt']
            data_dict[file]['apt_sum'] = d[file].atomcharges['apt_sum']

        return data_dict
    
class Extract():

    def __init__():
        pass

    def _buried_volume(common_atoms, elements, coords, structure, r_val):
        '''
        Function to extract the buried volumes for a given structure.

        Arguments:
        common_atoms (List): A list of common atoms to calculate buried volume features for.
        elements (List): Elements as atomic symbols or numbers.
        coords (Array): Coordinates of the structure.
        structure (String): The structure to calculate the values for.
        r_val: The return value for threading.

        Returns:
        f_d (Dictionary): A dictionary of morfeus bured volume features.
        '''
        f_d = {}
        morf_feats = {}
        for common_atom in common_atoms:
            position = common_atoms.index(common_atom)
            bv = BuriedVolume(elements, coords, common_atom).buried_volume
            fraction_bv = BuriedVolume(elements, coords, common_atom).fraction_buried_volume
            fv = BuriedVolume(elements, coords, common_atom).free_volume
            morf_feats['buried_volume_'+str(position)] = bv
            morf_feats['fraction_buried_volume_'+str(position)] = fraction_bv
            morf_feats['free_volume_'+str(position)] = fv
        f_d[structure] = morf_feats
        
        r_val [0] = f_d
        #return f_d

    def _sasa(common_atoms, elements, coords, structure, r_val):
        '''
        Function to extract the sasa features for a given structure.

        Arguments:
        common_atoms (List): A list of common atoms to calculate buried volume features for.
        elements (List): Elements as atomic symbols or numbers.
        coords (Array): Coordinates of the structure.
        structure (String): The structure to calculate the values for.
        r_val: The return value for threading.

        Returns:
        f_d (Dictionary): A dictionary of morfeus sasa features.
        '''
        f_d = {}
        sasa_feats= {}
        sasa = SASA(elements, coords)
        for common_atom in common_atoms:
            position = common_atoms.index(common_atom)
            sasa_atom = sasa.atom_areas[common_atom]
            sasa_feats['sasa_volume'] = sasa.volume
            sasa_feats['sasa_area'] = sasa.area
            sasa_feats['sasa_'+str(position)] = sasa_atom
        f_d[structure] = sasa_feats

        r_val [0] = f_d
        #return f_d

    def _sterimol(common_atoms, elements, coords, structure, r_val):
        '''
        Function to extract the sterimol features for a given structure.

        Arguments:
        common_atoms (List): A list of common atoms to calculate buried volume features for.
        elements (List): Elements as atomic symbols or numbers.
        coords (Array): Coordinates of the structure.
        structure (String): The structure to calculate the values for.
        r_val: The return value for threading.

        Returns:
        f_d (Dictionary): A dictionary of morfeus sterimol features.
        '''
        f_d = {}
        sterimol_feats = {}
        combinations = General._get_combinations(common_atoms, 2, reverse=True)
        for pair in combinations:
            sterimol = Sterimol(elements, coords, pair[0], pair[1])
            sterimol_feats['sterimol_L_value_'+str(common_atoms.index(pair[0]))+'_'+str(common_atoms.index(pair[1]))] = sterimol.L_value
            sterimol_feats['sterimol_B_1_value_'+str(common_atoms.index(pair[0]))+'_'+str(common_atoms.index(pair[1]))] = sterimol.B_1_value
            sterimol_feats['sterimol_B_5_value_'+str(common_atoms.index(pair[0]))+'_'+str(common_atoms.index(pair[1]))] = sterimol.B_5_value
        f_d[structure] = sterimol_feats

        r_val [0] = f_d
        #return f_d

    def _common_atom_parser(data_dict, structure, rxn_data):
        '''
        Function to generate the common atoms in a more readable and accessible format.

        Arguments:
        data_dict (Dictionary): A dictionary containing 'meta data' and cclib extracted features such as charges and coordinates.
        structure (String): The structure to calculate the common atoms for.
        rxn_data (Dictionary): A dictionary containing information about the dataset.   

        Returns:
        common_atoms (List): A list of common atoms for a given structure.
        path (String): The path to the given file.
        '''
        number = data_dict[structure]['reaction_number']
        path = data_dict[structure]['path']
        reactant = data_dict[structure]['reactant']

        common_atoms = []
        if setup['type'] == 'ts':
            # Get the common atoms if not ma dataset
            if setup['dataset'] != 'ma':
                if number not in list(rxn_data.keys()):
                    logger.warning(f'{number} not in rxn_data')
                    pass
                else:
                    for species in rxn_data[number]:
                        if isinstance(rxn_data[number][species], list):
                            for value in rxn_data[number][species]:
                                common_atoms.append(value)
                    common_atoms = list(set(common_atoms))
            # Get the common atoms for the ma dataset.
            elif setup['dataset'] == 'ma':
                common_atoms = setup['common_atoms']['ts']
            elif setup['dataset'] == 'mal':
                common_atoms = setup['common_atoms']['ts']

        elif setup['type'] == 'gs' or setup['type'] =='dist_gs':
            common_atoms = rxn_data[number][reactant]['common_atoms']
        return common_atoms, path
    
    @timeit
    def _morfeus_features(data_dict, rxn_data):
        '''
        Function to extract morfeus features for all common atoms for the given structures. Relies on the used functions and utilised threading.
        
        Arguments:
        data_dict (Dictionary): A dictionary containing 'meta data' and cclib extracted features such as charges and coordinates.
        rxn_data (Dictionary): A dictionary containing information about the dataset.   

        Returns:
        feat_dict (Dictionary): A dictionary containing the features from morfeus for the entire dataset.
        '''
        feat_dict = {}
        # Read in each file iteratively and create a temporary file.
        for structure in data_dict.keys():
            common_atoms, path = Extract._common_atom_parser(data_dict, structure, rxn_data)
            f = cclib.io.ccread(path)
            cclib.io.ccwrite(f, outputtype='xyz', outputdest='temp.xyz')
            elements, coords = read_geometry('temp.xyz')

            # Set the return variables.
            bv= [None]*1
            sasa= [None]*1
            sterimol= [None]*1

            # Set the target functions.
            t1 = Thread(target=Extract._buried_volume, args=(common_atoms, elements, coords, structure, bv))
            t2 = Thread(target=Extract._sasa, args=(common_atoms, elements, coords, structure, sasa))
            t3 = Thread(target=Extract._sterimol(common_atoms, elements, coords, structure, sterimol))
            
            # Start and join the threads.
            t1.start(), t2.start(), t3.start()
            t1.join(), t2.join(), t3.join()

            # Combine the feature sets.
            feat_dict[structure] = {**bv[0][structure], **sasa[0][structure], **sterimol[0][structure]}

            # Remove the temporary file.
            os.remove('temp.xyz')
        return feat_dict

    @timeit
    def _distances(data_dict, rxn_data):
        '''
        Function to get the distances between all common atoms - depends on job type as well as the dataset.
        
        Arguments:
        data_dict (Dictionary): A dictionary containing 'meta data' and cclib extracted features such as charges and coordinates.
        rxn_data (Dictionary): A dictionary containing information about the dataset.   

        Returns:
        feat_dict (Dictionary): A dictionary containing the distance features for the entire dataset.
        '''

        def _ts_dist(rxn_data, dataset):
            '''
            Function to extract the bond forming distances for the given dataset (one reaction centre or two reaction centres).

            Arguments:
            rxn_data (Dictionary): A dictionary containing information about the dataset. 
            dataset (String): A string denoting which dataset dat extraction is being performed upon.

            Returns:
            distance_feats (Dictionary): A dictionary containing the transition state reaction distances.
            '''
            
            distance_feats = {}
            if dataset == 'tt' or dataset == 'da' or dataset == 'la_da':
                if data_dict[structure]['reaction_number'] not in list(rxn_data.keys()):
                    logger.warning(str(str(data_dict[structure]['reaction_number'])+'not in rxn_data'))
                    pass
                else:
                    atoms = rxn_data[data_dict[structure]['reaction_number']]['reacting']
                    reacting_atom_zero = data_dict[structure]['coords'][atoms[0]-1] # -1 because coords start at 0 di
                    reacting_atom_one = data_dict[structure]['coords'][atoms[1]-1] # -1 because coords start at 0 dp
                    reacting_atom_two = data_dict[structure]['coords'][atoms[2]-1] # -1 because coords start at 0 di
                    reacting_atom_three = data_dict[structure]['coords'][atoms[3]-1] # -1 because coords start at 0 dp

                    reacting_distance_zero = np.linalg.norm(np.array(reacting_atom_zero)-np.array(reacting_atom_one))
                    reacting_distance_one = np.linalg.norm(np.array(reacting_atom_two)-np.array(reacting_atom_three))

                    distance_feats['reacting_distance_0'] = reacting_distance_zero
                    distance_feats['reacting_distance_1'] = reacting_distance_one
                    distance_feats['reacting_distance_diff'] = np.absolute(reacting_distance_zero - reacting_distance_one)

            elif dataset == 'ma':
                atoms = rxn_data[data_dict[structure]['reaction_number']]['reacting']
                reacting_atom_zero = data_dict[structure]['coords'][atoms[0]-1]
                reacting_atom_one = data_dict[structure]['coords'][atoms[1]-1] 

                reacting_distance_zero = np.linalg.norm(np.array(reacting_atom_zero)-np.array(reacting_atom_one))
                distance_feats['reacting_distance_0'] = reacting_distance_zero
            
            elif dataset == 'mal':
                reacting_atom_zero = data_dict[structure]['coords'][0]
                reacting_atom_one = data_dict[structure]['coords'][4] 

                reacting_distance_zero = np.linalg.norm(np.array(reacting_atom_zero)-np.array(reacting_atom_one))
                distance_feats['reacting_distance_0'] = reacting_distance_zero

            return distance_feats
                    
        f_d = {}
        for structure in data_dict.keys():
            common_atoms, path = Extract._common_atom_parser(data_dict, structure, rxn_data)
            # Get ground state reactant distances
            if setup['type'] == 'gs' or setup['type'] == 'dist_gs':
                combinations = General._get_combinations(common_atoms, 2, reverse=False)
                distance_feats = {}
                for pair in combinations:
                    atom_zero = data_dict[structure]['coords'][pair[0]-1] # -1 because coords start at 0
                    atom_one = data_dict[structure]['coords'][pair[1]-1] # -1 because coords start at 0
                    distance = np.linalg.norm(np.array(atom_zero)-np.array(atom_one))
                    distance_feats['distance_'+str(common_atoms.index(pair[0]))+'_'+str(common_atoms.index(pair[1]))] = distance
                f_d[structure] = distance_feats
            elif setup['type'] == 'ts':
                distance_feats = _ts_dist(rxn_data, setup['dataset'])
                f_d[structure] = distance_feats
        return f_d

    @timeit
    def _charges(common_atoms, data_dict, rxn_data):
        '''
        Function to get the correct charges from the cclib output (only for the specified common atoms).

        Arguments:
        common_atoms (List): A list of common atoms for the given system (Created in this function for gs and dist_gs structures).
        data_dict (Dictionary): A dictionary containing 'meta data' and cclib extracted features such as charges and coordinates.
        rxn_data (Dictionary): A dictionary containing information about the dataset.

        Returns:
        c_d (Dictionary): A dictionary containing the charges for the given dataset and common atoms.
        ''' 
        c_d = {}
        for structure in data_dict.keys():
            strucuture_c_d = {}
            if setup['type'] == 'gs' or setup['type'] == 'dist_gs':
                common_atoms, path = Extract._common_atom_parser(data_dict, structure, rxn_data)
                species = rxn_data[data_dict[structure]['reaction_number']][data_dict[structure]['reactant']]['type']
                charge_dict = {}
                for atom in common_atoms:
                    position = '_'+str(common_atoms.index(atom))
                    mul_charge = data_dict[structure]['mulliken_charge'][atom-1]
                    mul_sum = data_dict[structure]['mulliken_sum'][atom-1]
                    apt_charge = data_dict[structure]['apt_charge'][atom-1]
                    apt_sum = data_dict[structure]['apt_sum'][atom-1]
                    if bool(charge_dict) == False:
                        charge_dict = {'mulliken_charge'+position:mul_charge, 'mulliken_sum'+position:mul_sum,
                                           'apt_charge'+position:apt_charge, 'apt_sum'+position:apt_sum}
                    else:
                        new_charge_dict = {'mulliken_charge'+position:mul_charge, 'mulliken_sum'+position:mul_sum,
                                           'apt_charge'+position:apt_charge, 'apt_sum'+position:apt_sum}
                        charge_dict = {**charge_dict, **new_charge_dict}
                c_d[structure] = charge_dict

            elif setup['type'] == 'ts':
                rxn_number = data_dict[structure]['reaction_number']
                if rxn_number not in list(rxn_data.keys()):
                        logger.warning(f'{rxn_number} not in rxn_data')
                        pass
                else:
                    for species in rxn_data[rxn_number].keys():
                        charge_dict={}
                        if species == 'reacting' or species == 'file':
                            pass
                        else:
                            for atom in rxn_data[rxn_number][species]:
                                position = '_'+species+'_'+str(rxn_data[rxn_number][species].index(atom))
                                mul_charge = data_dict[structure]['mulliken_charge'][atom-1]
                                mul_sum = data_dict[structure]['mulliken_sum'][atom-1]
                                apt_charge = data_dict[structure]['apt_charge'][atom-1]
                                apt_sum = data_dict[structure]['apt_sum'][atom-1]
                                if bool(charge_dict) == False:
                                    charge_dict = {'mulliken_charge'+position:mul_charge, 'mulliken_sum'+position:mul_sum,
                                            'apt_charge'+position:apt_charge, 'apt_sum'+position:apt_sum}
                                else:
                                    new_charge_dict = {'mulliken_charge'+position:mul_charge, 'mulliken_sum'+position:mul_sum,
                                            'apt_charge'+position:apt_charge, 'apt_sum'+position:apt_sum}
                                    charge_dict = {**charge_dict, **new_charge_dict}
                            if rxn_number not in list(strucuture_c_d.keys()):
                                strucuture_c_d[rxn_number] = charge_dict
                            else:
                                combined_charges = {**strucuture_c_d[rxn_number], **charge_dict}
                c_d[structure] = combined_charges

        return c_d
           
def main():

    # Set up logger
    global logger, start_formatter, formatter, fh, ch
    logger = logging.getLogger('feature_extraction')
    logger.setLevel('INFO')
    fh = logging.FileHandler('feat_extr.log', mode='w')
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

    # Run feature extraction
    General._runner()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='f_extract',
        description='Python script for extracting features from the [3+2], Diels-Alder and Michael addition datasets.')
    parser.add_argument('-c', dest='combine', action='store_true', required=False, help='Argument to combine the different feature files and barriers.')
    (options, args) = parser.parse_known_args()
    main()