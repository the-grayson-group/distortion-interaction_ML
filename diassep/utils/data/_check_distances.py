
# Imports
import os
import cclib
import pickle
import argparse
import numpy as np


class check_tss():

    def __init__(self):

        data_dict = check_tss._get_freqs()
        check = check_tss._get_distances(data_dict)
        if len(check) == 0:
            print('All structures have two bond forming distances within 0.6 Angstroms of eachother.')
        else:
            print('Check these structures: ')
            for structure in check:
                print(structure)

    def _get_data():
        '''
        Function to take all data in current working directory and parse
        the appropriate data using cclib.

        Returns:

        data_dict (dictionary): Dictionary with filename (key) and parsed data (value) pairs.
        
        '''
        data_dict = {}
        files = []
        for filename in os.listdir():
            if filename.endswith(".out") or filename.endswith(".log"):
                files.append(filename)
            else:
                continue
            
        for file in files:
            data_dict[file] = cclib.io.ccread(file)
            
        return data_dict 
      
    def _get_freqs():
        '''
        Function to take output from _get_data and specify the values needed to calculate the
        separate species.

        Returns:

        data_dict (dictionary of dictionaries): DoD with filename (key) and coords1, coords2 and 
        vectors (nested keys and values).
        '''
        data_dict = {}
        dct = check_tss._get_data() # Use _get_data to parse files
        atomic_masses = {1:'H', 6:'C', 7:'N', 8:'O', 9:'F', 17:'Cl', 35:'Br'}

        for file in dct:
            f_check = []
            for vib in dct[file].vibfreqs:
                if vib < 0:
                    f_check.append(file)
                    position = dct[file].vibfreqs.tolist().index(vib)
            try: # Check for multiple negative frequencies
                assert len(f_check) == 1
            except:
                print(file, " has multiple negative frequencies - check structure.")

            # Get just the filename without reopt extensions
            
            if '_reopt' in file:
                filename = file.split('_reopt')[0]
            else:
                filename = file.split('.out')[0]    
            
            data_dict[file] = {} # Create nested dictionary
            data_dict[file]['coords1'] = dct[file].atomcoords[-1]
            data_dict[file]['coords2'] = dct[file].atomcoords[-1] - dct[file].vibdisps[position]
            data_dict[file]['disp_vect'] = dct[file].vibdisps[position]
            data_dict[file]['atomnos'] = dct[file].atomnos
            

            atom_symbs = []
            for atom in data_dict[file]['atomnos']:
                atom_symbs.append(atomic_masses[atom])
            
            data_dict[file]['atomsymb'] = np.array(atom_symbs)
            data_dict[file]['name'] = filename

        return data_dict
    
    def _clean_common_atoms(d):
        
        '''
        Function that deals with common atoms. Essentially, the file names across different solvation systems change depending upon how many reoptimisations 
        had to be performed. To navigate this, this function strips the extensions and takes the pure file name from the 'name' column and makes it the key.

        Arguments:
        
        d (Dictionary): A dictionary that has the structure of filename (with extensions) as key and a sub-dictionary with a 'name' key (no extensions.)

        Returns:

        new_d (Dictionary): A dictionary with the updated, no extension keys.
        '''

        d_new = {}
        for structure in d.keys():
            fix = d[structure]
            name = d[structure]['name']
            d_new[name] = fix

        return d_new

    def _get_distances(data_dict):
        '''
        Function to check the distances of the reacting atoms from a TS optimisation.

        Arguments:
        data_dict (dictionary): Dictionary with filename (key) and parsed data (value) pairs.

        Returns:

        greater (List): A list of distances that are above 0.6 Angstrom difference (Diels-Alder/Cycloaddition)
        '''
        c = str(options.commonatoms)
        with open(c, 'rb') as common_atoms:
            load_dict = pickle.load(common_atoms)
        common_atoms_dict = check_tss._clean_common_atoms(load_dict)

        diff_dict = {}
        for structure in data_dict.keys():
            # Get the name
            name = data_dict[structure]['name']
            # Get the coordinates to check
            atom_1 = data_dict[structure]['coords1'][common_atoms_dict[name]['reacting'][0]]
            atom_2 = data_dict[structure]['coords1'][common_atoms_dict[name]['reacting'][1]]
            atom_3 = data_dict[structure]['coords1'][common_atoms_dict[name]['reacting'][2]]
            atom_4 = data_dict[structure]['coords1'][common_atoms_dict[name]['reacting'][3]]
            # Check the distances and add to dictionary if over 0.6 A
            distance_1 = np.linalg.norm(np.array(atom_1)-np.array(atom_2))
            distance_2 = np.linalg.norm(np.array(atom_3)-np.array(atom_4))
            if np.abs(distance_1 - distance_2) > 0.6:
                diff_dict[structure] = np.abs(distance_1 - distance_2)
            else:
                pass
        return diff_dict

def main():
    check_tss()

if __name__ == '__main__':

    # Argument parse
    parser = argparse.ArgumentParser(
        prog='check_distances',
        description='Python script to check ts distances for cycloadditions based on a common_atoms.pkl.')
    parser.add_argument('-c', dest='commonatoms', default=None, required=True, help='Utilises a dictionary of common atoms to separate structures.')
    
    (options, args) = parser.parse_known_args()
    main()