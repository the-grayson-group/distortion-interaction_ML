'''
Python script to match the DFT GSs to their respective distorted structures.

Usage:

python match_gss.py -m extract
python match_gss.py -m match
'''

import os
import cclib
import shutil
import pickle
import argparse

def _get_distorted_masses():
    '''
    Function to extract the masses from the distorted structures and save to a pickle.
    
    Returns:
    mass_dct (Dictionary): A dictionary of masses.
    '''

    # Get the path to all files
    files = []
    for file in os.listdir():
        if file.endswith('SPE.out'):
            files.append(file)

    # Extract the reaction number and data associated with file
    data_dict = {}
    rxns = []
    for file in files:
        fname = '_'+str(file.split('_')[1])+'_'
        rxns.append(fname)
        data_dict[file] = cclib.io.ccread(file)

    # Get the files that correspond to the same reactions
    rxn_dct = {}
    for rxn in set(rxns):
        grouped_rxns = []
        for file in data_dict.keys():
            if rxn in file:
                grouped_rxns.append(file)
        rxn_dct[rxn] = grouped_rxns

    # Extract the masses for each reaction and assign it in a DoD
    mass_dct = {}
    for rxn in rxn_dct.keys():
        rxn_mass = {}
        for file in rxn_dct[rxn]:
            rxn_mass[file] = sum(list(data_dict[file].atomnos))
        mass_dct[rxn] = rxn_mass

    # Save the dictionary as a pickle file.
    with open('rxn_rct_masses.pkl', 'wb') as f:
        pickle.dump(mass_dct, f)
        print('Dictionary of masses saved to rxn_rct_masses.pkl')

    return mass_dct

def _get_matches():
    '''
    Function to match the reactions from the dictionary to the originals.

    '''

    # Load the dicitonary
    with open('rxn_rct_masses.pkl', 'rb') as f:
        mass_dct = pickle.load(f)

    # Loop through each directory/reaction
    mapped_files = {}
    for rxn in mass_dct.keys():
        rxn_clean = rxn.replace('_','')
        pth = rxn_clean+'/single_point_logs'
        os.chdir(pth)
        gss = os.listdir()
        gss = [x for x in gss if x.startswith('r')]
        gss = [x for x in gss if not 'alt' in x]
        assert len(gss) == 2, f'Too many files in {rxn} directory'
        # Loop through all files in that directory to get their masses
        temp_d = {}
        for file in gss:
            mass = sum(list(cclib.io.ccread(file).atomnos))
            matched_file = list(mass_dct[rxn].keys())[list(mass_dct[rxn].values()).index(mass)]
            temp_d[file] = matched_file
        mapped_files[rxn] = temp_d
        os.chdir('../../')
    
    # Loop through the mapped dict and copies files back
    for rxn in mapped_files.keys():
        rxn_clean = rxn.replace('_','')
        pth = rxn_clean+'/single_point_logs'
        os.chdir(pth)
        for key, value in mapped_files[rxn].items():
            dest = '../../'+value
            dest = dest.replace('ts', 'gs')
            shutil.copy(key, dest)
        os.chdir('../../')

def main():

    if options.extractormatch == 'extract':
        _get_distorted_masses()
    elif options.extractormatch == 'match':
        _get_matches()
    else:
        print('Not a valid keyword - use either "extract" or "match".')

if __name__ == '__main__':

    # Argument parse
    parser = argparse.ArgumentParser(
        prog='match_gss',
        description='Python script for either extracting masses or matching them.')
    parser.add_argument('-m', dest='extractormatch', default=None, required=True, help='String of either "extract" or "match" depending on user function.')
    (options, args) = parser.parse_known_args()

    main()

