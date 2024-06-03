"""
#####################################################################################
#                                    get_di_dp.py                                   #
#                                                                                   #
#   This code takes a frequency calculation from Gaussian and determines the types  #
#   of reactant that is present. This code is structured to calculate whether the   #
#   reactants are dipoles or dipolarophiles - primarily for the [3+2] dataset.      #
#   Usage requires a .yaml file to run which directs the code to the correct        #
#   directory.                                                                      #
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
import yaml
import time
import cclib
import pickle
import numpy as np
from math import sqrt
from functools import wraps
from molml.utils import get_connections

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function \033[1m{func.__name__}\033[0m took \033[92m{total_time:.4f}\033[0m seconds')
        return result
    return timeit_wrapper

class general():
    
    def __init__():
        pass
    
    @timeit
    def _load_config():
        '''
        Function to load in the configuration file for energy extraction.

        Returns
        setup (Dictionary): A dictionary containing setup information for extracting features.
        '''
        with open('config.yaml', 'r') as file:
            setup = yaml.safe_load(file)
        return setup

    def _get_data():
        
        data_dict = {}
        files = []
        for filename in os.listdir():
            if filename.endswith(".out"):
                    files.append(filename)
            else:
                    continue
        for file in files:
            data_dict[file] = cclib.io.ccread(file)
        
        return data_dict
    
    def _clean(lol):
        '''
        Function to clean a list of lists into just a list.

        Arguments:
        lol (List of Lists): A list of lists to be converted into a list.

        Returns:
        new_lst (List): A list created from the list of lists.
        '''

        new_lst = []
        for lst in lol:
            lst = [str(i) for i in lst]
            new_lst.append('_'.join(lst))
        return new_lst
    
    @timeit
    def _get_freqs():
        
        data_dict = {}
        dct = general._get_data() # Use _get_data to parse files
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

            filename = file.split('.out')[0]   

            data_dict[file] = {} # Create nested dictionary
            data_dict[file]['coords1'] = dct[file].atomcoords[-1]

            data_dict[file]['coords2'] = dct[file].atomcoords[-1] - dct[file].vibdisps[position]
            data_dict[file]['coords3'] = dct[file].atomcoords[-1] + dct[file].vibdisps[position]
            data_dict[file]['disp_vect'] = dct[file].vibdisps[position]
            data_dict[file]['atomnos'] = dct[file].atomnos

            atom_symbs = []
            for atom in data_dict[file]['atomnos']:
                atom_symbs.append(atomic_masses[atom])
            
            data_dict[file]['atomsymb'] = np.array(atom_symbs)
            data_dict[file]['name'] = filename
           
        return data_dict
    
    def _get_distances(data_dict, column):
        '''
        Function to get distance pairs of all atoms in the system 

        Arguments:
        data_dict (dictionary): Dictionary with filename (key) and parsed data (value) pairs.
        column (String): A string equally either 'coords1' 'coords2' or 'coords3'

        Returns:

        dist_dict (Dictionary): A dictionary of all the possible atom-atom distance differences (under 3 A).
        '''
        dist_dict = {}
        for structure in data_dict.keys():
            dst = {}
            for xyz1 in data_dict[structure][column]:
                position1 = data_dict[structure][column].tolist().index(list(xyz1))
                for xyz2 in data_dict[structure][column]:
                    position2 = data_dict[structure][column].tolist().index(list(xyz2))
                    position = str(position1)+'_'+str(position2)
                    distance = np.linalg.norm(np.array(xyz2)-np.array(xyz1))
                    if distance < 5:
                        dst[position] = distance
            dist_dict[structure] = dst
        return dist_dict

    @timeit
    def _get_differences(data_dict):
        '''
        Function to get the differences in distances of before and after the vibrational displacements.
        Also obtains the vector pair directions for each of the biggest changing atom distances.
        
        Arguments:
        data_dict (dictionary): Dictionary with filename (key) and parsed data (value) pairs.

        Returns:

        diff_dict (Dictionary): A dictionary of the largests atom-atom distance differences 
        vect_dict (Dictionary): A dictionary of the paired atoms and their displacement vector directions (cos(theta))
        '''
        before = general._get_distances(data_dict, 'coords1')
        after = general._get_distances(data_dict, 'coords2')

        diff_dict = {} 
        vect_dict = {} 
        # Get distance differences for each pair and eliminate by a threshold
        for structure in before.keys():
            diff_dst = {}
            for pair in before[structure]:
                try:
                    aft = after[structure][pair]
                    bef = before[structure][pair]
                    diff = abs(bef-aft)
                    if diff > 0.3: # This is an arbitrary distance currently to reduce computation
                        diff_dst[pair] = diff
                except:
                    pass

                 # Remove any duplicate pairs (e.g., 1_0 and 0_1)
                temp = []
                clean = {}
                for key, val in diff_dst.items():
                    if val not in temp:
                        temp.append(val)
                        clean[key] = val      
                diff_dict[structure] = clean

        # Using distance pairs in diff dict, calculate the vector direction (looking for negative values).
        for name in diff_dict.keys():
            vectors = {}
            for pair in diff_dict[name]:
                first, last = pair.split('_')
                # Get the first atom's displacement vector, square root every value and make absolute
                first_vect = data_dict[name]['disp_vect'].tolist()[int(first)]
                abs_first_vect = [sqrt(abs(val)) for val in first_vect]

                # Get the last atom's displacement vector, square root every value and make absolute
                last_vect = data_dict[name]['disp_vect'].tolist()[int(last)]
                abs_last_vect = [sqrt(abs(val)) for val in last_vect]

                # Calculate the value of cos(theta)
                # The more negative the value the more anti-parallel the vectors are
                value = np.dot(first_vect, last_vect)/(np.dot(abs_first_vect, abs_last_vect)+1e-7)
                vectors[pair] = value
            vect_dict[name] = vectors
                
        #  Add a check here to return the TS pair of atoms.
        temp_diff_dict, temp_vect_dict = diff_dict.copy(), vect_dict.copy()
        react_dict = {}
        for name in diff_dict.keys():
            max_diff = max(temp_diff_dict[name], key=temp_diff_dict[name].get)
            min_vect = min(temp_vect_dict[name], key=temp_vect_dict[name].get)
            if max_diff == min_vect:
                reacting_atoms = max_diff
            else:
                # Remove max and min from resepective lists
                del temp_diff_dict[name][max(diff_dict[name], key=diff_dict[name].get)]
                del temp_vect_dict[name][min(vect_dict[name], key=vect_dict[name].get)]
                second_max_diff = max(temp_diff_dict[name], key=temp_diff_dict[name].get)
                second_min_vect = min(temp_vect_dict[name], key=temp_vect_dict[name].get)
                if max_diff == second_min_vect and min_vect == second_max_diff:
                    reacting_atoms = max_diff
                elif max_diff != second_min_vect or min_vect != second_max_diff:
                    if max_diff in temp_vect_dict[name].keys():
                        reacting_atoms = max_diff
                else:
                    raise Exception('Error with structure - ', name)
            react_dict[name] = reacting_atoms

        return diff_dict, vect_dict, react_dict
    
    @timeit
    def _get_connected_atoms(data_dict):
        '''
        Function to get the full connectivity of the whole molecule from the perspective of the reacting atom.

        Arguments:
        data_dict (dictionary): Dictionary with filename (key) and parsed data (value) pairs.

        Returns:

        connect_dict (Dictionary): A dictionary containing the list of atoms that are connected to the reacting atom.
        '''
        # Create the distance/vector dictionaries and the reacting atoms for every structure.
        dist_diff, vect_diff, react_dict = general._get_differences(data_dict)

        connect_dict = {}

        # Loop through every structure
        for structure, atoms in react_dict.items():
            connected_atoms = []
            already_checked = []
            # Get the first reacting atom for the current structure and calculate its connectivity storing it in connected_atoms/already_checked
            atom_one = atoms.split('_')[0] 
            connectivity = get_connections(data_dict[structure]['atomsymb'].tolist(), data_dict[structure]['coords1'].tolist())
            for connected_atom in connectivity[int(atom_one)].keys():
                connected_atoms.append(connected_atom)
            connected_atoms.append(int(atom_one))
            already_checked.append(int(atom_one))
            # Now loop through the connected atoms to get the next set of connected atoms checking if its in already_checked to avoid extra computation
            for new_atom in connected_atoms:
                if new_atom not in already_checked:
                    for new_a in connectivity[int(new_atom)].keys():
                        connected_atoms.append(new_a)
                    already_checked.append(new_atom)
            connect_dict[structure] = [*set(connected_atoms)] # Removes duplicate atoms in the list
        return connect_dict

    @timeit
    def _get_edges(data_dict, coords='coords1'):
        '''
        Function to get a dictionary of all edges in the system. Default to use coords1.

        Arguments:

        data_dict (Dictionary): A dictionary of all the information for all the reactions.
        coords (String): A string choosing the coordinates to use - defaults as coords1 (optimised coordinates).

        Returns:
        edge_dct (Dictionary): A dictionary of all the edges for all the reactions.
        '''
        edge_dct = {}
        for structure in data_dict:
            dct = get_connections(data_dict[structure]['atomsymb'].tolist(),
                                data_dict[structure][coords].tolist())
        # Use dct to generate a list of list of edges in the system
            edges = []
            for first_atom in dct.keys():
                for second_atom in dct[first_atom].keys():
                    edges.append(list([first_atom,second_atom]))
            h_positions = [] 
            for position, atom in zip(range(0, len(list(data_dict[structure]['atomsymb']))),list(data_dict[structure]['atomsymb'])):
                if atom == 'H':
                    h_positions.append(position)
            for edge in edges:
                for h_pos in h_positions:
                    if h_pos in edge:
                        edges.remove(edge)

            edge_dct[structure] = edges
        return edge_dct

    def _get_shortest_path(edges, starting_node, goal):
        '''
        Function to find the shortest path between two given points and a list of edges (connections).

        Arguments:
        edges (List): A list of lists of all possible connections in the molecule.
        starting_node (Int): The starting point for the path.
        goal (Int): The finishing point for the path.

        Returns:
        A list containing the path from starting_node to goal. 
        '''

        visited = []
        queue = [[starting_node]]
    
        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node not in visited:
                neighbours = []
                for edge in edges:
                    if edge[0] == node:
                        neighbours.append(edge[1])
                    elif edge[1] == node:
                        neighbours.append(edge[0])
                for neighbour in neighbours:
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)
                    
                    if neighbour == goal:
                        return new_path
                
                visited.append(node)
                
        return []

    @timeit
    def _get_molecule_type(edge_dict, reacting_atoms):
        
        redo = {}
        for structure in reacting_atoms:
            if len(reacting_atoms[structure]) != 2:
                redo[structure.split('_')[1].split('.out')[0]] = structure
                print('Error in structure: - ', structure)
            else:
                mol_type_d = {}
                for structure in edge_dict:
                    reacting = reacting_atoms[structure]
                    react_atoms = []
                    for pair in reacting:
                        react_atoms.append(pair.split('_')[0])
                        react_atoms.append(pair.split('_')[-1])
                    d = {}
                    for atom1 in react_atoms:
                        for atom2 in react_atoms:
                            if atom1 == atom2:
                                pass
                            else:
                                path = general._get_shortest_path(edge_dict[structure], int(atom1), int(atom2))
                                if len(path) == 3:
                                    d['di'] = [atom1, atom2]
                                elif len(path) == 2:
                                    d['dp'] = [atom1, atom2]
                    mol_type_d[structure] = d


        return mol_type_d, redo

    def _remove_dupes(reacting_atoms):
        '''
        Function to remove duplicate atom pairs from reacting_atoms for all structures.

        Function:
        reacting_atoms (Dictionary): A dictionary of all structures and their equivalent reacting atoms.

        Returns:
        reacting_atoms (Dictionary): A dictionary of all structures and their equivalent reacting atoms now without duplicates.
        '''
        for structure in reacting_atoms.keys():
            for value in reacting_atoms[structure]:
                new_value = str(value.split('_')[-1])+'_'+str(value.split('_')[0])
                try:
                    reacting_atoms[structure].remove(new_value)
                except:
                    pass
        return reacting_atoms

    @timeit
    def _get_reacting_atoms(data_dict):
        '''
        Function to get the common atoms involved and isolating them as either diene/dipolar or dienophile/dipolarophile.

        Returns:
        reacting_centres (Dictionary) = A dictionary of the reacting centres for all reactions.
        check (List): A list of structures to check as they have thrown up an error.
        '''

        # Get edge dictionary for optimised coordinates and max/min displacement coordinates.
        edge_dict_1 = general._get_edges(data_dict, 'coords1')
        edge_dict_2 = general._get_edges(data_dict, 'coords2')
        edge_dict_3 = general._get_edges(data_dict, 'coords3')

        reacting_atoms = {}
        for structure in edge_dict_1.keys():
            c_1_2_comp = [x for x in edge_dict_2[structure] if x not in edge_dict_1[structure]]
            c_1_3_comp = [x for x in edge_dict_3[structure] if x not in edge_dict_1[structure]]
            if len(c_1_2_comp) == 0:
                reacting_atoms[structure] = general._clean(c_1_3_comp)
            else:
                reacting_atoms[structure] = general._clean(c_1_2_comp)
        check = []
        for structure in reacting_atoms.keys():
            if len(reacting_atoms[structure]) != 4: # Check that the reaction is two centre - specific to DA/[3+2]
                check.append(structure)

        for structure in reacting_atoms.keys():
            for value in reacting_atoms[structure]:
                new_value = str(value.split('_')[-1])+'_'+str(value.split('_')[0])
                try:
                    reacting_atoms[structure].remove(new_value)
                except:
                    pass
        return reacting_atoms, check

    def _check(common_atoms, check):
        '''
        A function to check that the files are successful two reaction centre TS's.

        Arguments:
        common_atoms (Dictionary): A dictionary containing all the common atoms information for all structures.
        check (List): A list of structures to check as they have thrown up an error.

        Returns:
        common_atoms (Dictionary): A cleaned dictionary containing all the common atoms information for all structures.
        check (List): A cleaned list of structures to check as they have thrown up an error.
        '''
        for structure in list(common_atoms.keys()):
            if len(common_atoms[structure]['di']) == 0:
                del common_atoms[structure]
                check.append(str(structure))
            elif len(common_atoms[structure]['dp']) == 0:
                del common_atoms[structure]
                check.append(str(structure))
            else:
                pass
        check = list(set(check))
        return common_atoms, check

    def _get_masses(lst):

        mass_lookup = {'C':12,
               'H':1,
               'N':14,
               'O':16,
               'Cl':35,
               'F':19,
               'Br':79}
        temp_lst = []
        for value in lst:
            temp_lst.append(mass_lookup[value])
        return sum(temp_lst)

    @timeit
    def _get_structures(data_dict, connect_dict, mol_types):

        structure_d = {}
        for structure in connect_dict.keys():
            rxn = structure.split('_')[1].split('.out')[0]
            r1_atoms =  connect_dict[structure]
            r1_symbs = [data_dict[structure]['atomsymb'][x] for x in r1_atoms]
            r1_mass = general._get_masses(r1_symbs)
            r2_atoms =  list(set(list(range(0, len(data_dict[structure]['coords1']))))- set(connect_dict[structure]))
            r2_symbs = [data_dict[structure]['atomsymb'][x] for x in r2_atoms]
            r2_mass = general._get_masses(r2_symbs)
            
            d = {}
            for gs in mol_types[structure]:
                if int(mol_types[structure][gs][0]) in r1_atoms:
                    d[gs] = r1_mass
                elif int(mol_types[structure][gs][0]) in r2_atoms:
                    d[gs] = r2_mass
            structure_d[rxn] = d
        return structure_d

class match():

    def __init__():
        pass

    @timeit
    def _match_to_gs(setup, structure_d):

        mass_lookup = {6:12,
               1:1,
               7:14,
               8:16,
               17:35,
               9:19,
               35:79}
        
        os.chdir(setup['path'])
        files = os.listdir()
        files = [val for val in files if not val.endswith('.pkl')]
        files = [val for val in files if not val.endswith('.csv')]

        file_d = {}
        for name in files:
            rxn = name.split('_')[1]
            ccobj = cclib.io.ccread(name)
            if ccobj == None:
                print('Error in file - ', name)
                pass
            else:
                mass = []
                for atom in list(ccobj.atomnos):
                    mass.append(mass_lookup[atom])
                file_d[name] = int(sum(mass))

        rxn_d = {}
        for rxn_num in structure_d.keys():
            rxn = '_'+rxn_num+'_'
            rxn_lst = []
            for structure in files:
                if rxn not in structure:
                    pass
                elif rxn in structure:
                    rxn_lst.append(structure)
            rxn_d[rxn_num] = rxn_lst
        
        mapped_d = {}
        for rxn in rxn_d.keys():
            if rxn in redo.keys():
                pass
            else:
                d= {}
                for molecule in rxn_d[rxn]:
                    mass = file_d[molecule]
                    if mass not in list(structure_d[rxn].values()):
                        print('Check rxn - ', rxn, molecule)
                        pass
                    else:
                        mol_type = list(structure_d[rxn].keys())[list(structure_d[rxn].values()).index(mass)]
                        d[molecule] = mol_type
                mapped_d[rxn] = d

        return mapped_d

    def _save_map(mapped_d):
        
        if os.path.isfile('molecule_type.pkl') == True:
            f = open('molecule_type.pkl', 'rb')
            loaded_d = pickle.load(f)
            combined_d = {**loaded_d, **mapped_d}
            new_file = input('Save as new file [y/n]? ')
            outcome = False
            while outcome == False:
                if new_file == 'y':
                    outcome = True
                    with open('comb_molecule_type.pkl', 'wb') as new_f:
                        pickle.dump(combined_d, new_f)
                        print('Saved as new file to comb_molecule_type.pkl')
                elif new_file == 'n':
                    outcome = True
                    with open('molecule_type.pkl', 'wb+') as new_f:
                        pickle.dump(combined_d, new_f)
                        print('File saved to molecule_type.pkl')
                else:
                    outcome = False
                    print('Not a valid decision - choose y or n.')
                    new_file = input('Save as new file [y/n]? ')

        else:
            with open('molecule_type.pkl', 'wb') as f:
                pickle.dump(mapped_d, f)
            print('File saved to molecule_type.pkl')

class runner():
    
    def __init__():
        pass
    
    @timeit
    def _runner():

        global redo
        # Load
        setup = general._load_config()
        data_dict = general._get_freqs()

        # Get reacting atoms
        dist_diff, vect_diff, react_dict = general._get_differences(data_dict)
        connect_dict = general._get_connected_atoms(data_dict)
        reacting_atoms, check = general._get_reacting_atoms(data_dict)

        # Get masses for each reactant
        edge_dict = general._get_edges(data_dict, coords='coords1')
        mol_types, redo = general._get_molecule_type(edge_dict, reacting_atoms)
        structure_d = general._get_structures(data_dict, connect_dict, mol_types)

        # Match to structures
        mapped_d = match._match_to_gs(setup, structure_d)
        if setup['view'] == True:
            print(mapped_d)
        match._save_map(mapped_d)

        if len(redo) > 0:
            print('Errors in ', len(redo), ' files.')
            disp = input('Display file names [y]? ')
            if disp == 'y':
                for rxn in redo: 
                    print(redo[rxn])
            else:
                pass

@timeit
def main():
    runner._runner()

if __name__ == '__main__':
    main()