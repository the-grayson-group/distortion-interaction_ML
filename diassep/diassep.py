"""
#####################################################################################
#                                    diassep.py                                     #
#                                                                                   #
#   This code takes a frequency calculation from Gaussian and separates the         #
#   species involved in the TS. The idea is to look for the frequency               #
#   information in the .out file to get the two vectors that have the greatest      #
#   magnitude change as well as the biggest bond distance difference change.        #
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
import copy
import cclib
import pickle
import argparse
import numpy as np
import xyz_py as xyzp
from math import sqrt
from molml.utils import get_connections
from molml.constants import BOND_LENGTHS

BOND_LENGTHS['Br'] = {"1":1.3}

class DIASSep:

    def __init__(self):

        DIASSep._output_structures(DIASSep._get_freqs())

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
            if options.xyz:
                if filename.endswith(".xyz"):
                    files.append(filename)
                else:
                    continue
            else:    
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
        dct = DIASSep._get_data() # Use _get_data to parse files
        atomic_masses = {1:'H', 6:'C', 7:'N', 8:'O', 9:'F', 14:'Si', 17:'Cl', 35:'Br'}

        for file in dct:
            f_check = []
            if options.xyz:
                filename = file.split('.xyz')[0]
            else:
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
            if options.xyz:
                pass
            else:
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

    def _get_adjacency(data_dict):
        '''
        Function to get the adjacency matrix for both displacement coordinates and determine the
        bond forming atoms.

        Arguments:
        data_dict (Dictionary): Dictionary with filename (key) and parsed data (value) pairs.

        Returns:
        react_dict (Dictionary): A dictionary of the structures and the reaction centres.
        '''
    
        # Build empty dictionary.
        react_dict = {}
        # Loop through all structures.
        for structure in data_dict.keys():
            # Get elements and coordinates.
            e_l = list(data_dict[structure]['atomsymb'])
            c1_l = [list(i) for i in data_dict[structure]['coords2']]
            c2_l = [list(i) for i in data_dict[structure]['coords3']]
            # Get the adjacency matrices.
            m1 = xyzp.get_adjacency(e_l, c1_l)
            m2 = xyzp.get_adjacency(e_l, c2_l)
            # Track the changes from one matrix to another.
            changes = []
            for i in range(0, len(m1)):
                if list(m1[i]) != list(m2[i]):
                    for index, (fi, se) in enumerate(zip(list(m1[i]), list(m2[i]))):
                        if fi != se:
                            changes.append(f'{i}_{index}')
            # Pull out the first instance and use that. Potentially may have to alter this depending upon the
            # if the structure has unusual behaviour but, this should conistently work.
            react_dict[structure] = changes[0]
        
        return react_dict

    def _get_differences(data_dict):
        '''
        Function to get the differences in distances of before and after the vibrational displacements.
        Also obtains the vector pair directions for each of the biggest changing atom distances.
        
        Arguments:
        data_dict (dictionary): Dictionary with filename (key) and parsed data (value) pairs.

        Returns:

        diff_dict (Dictionary): A dictionary of the largests atom-atom distance differences 
        vect_dict (Dictionary): A dictionary of the paired atoms and their displacement vector directions (cos(theta))
        react_dict (Dictionary): A dictionary of the structures and the reaction centres.
        '''
        before = DIASSep._get_distances(data_dict, 'coords1')
        after = DIASSep._get_distances(data_dict, 'coords2')

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

    def _get_connected_atoms(data_dict):
        '''
        Function to get the full connectivity of the whole molecule from the perspective of the reacting atom.

        Arguments:
        data_dict (Dictionary): Dictionary with filename (key) and parsed data (value) pairs.

        Returns:

        connect_dict (Dictionary): A dictionary containing the list of atoms that are connected to the reacting atom.
        '''
        # Create the distance/vector dictionaries and the reacting atoms for every structure.
        if options.adjacency:
            react_dict = DIASSep._get_adjacency(data_dict)
        else:
            dist_diff, vect_diff, react_dict = DIASSep._get_differences(data_dict)
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

    def _get_connected_break(data_dict, atoms):
            '''
            Function to get the full connectivity of the whole molecule from the perspective of the reacting atoms as given by the breakbonds argument.

            Returns:

            connect_dict (Dictionary): A dictionary containing the list of atoms that are connected to the reacting atom.
            '''
            # Create the distance/vector dictionaries and the reacting atoms for every structure.
            dist_diff, vect_diff, react_dict = DIASSep._get_differences(data_dict)

            connect_dict = {}

            # Loop through every structure
            for structure, atoms in react_dict.items():
                connected_atoms = []
                already_checked = []
                # Get the first reacting atom for the current structure and calculate its connectivity storing it in connected_atoms/already_checked
                atom_one = atoms[0]
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

    def _get_multi_connected_break(data_dict):
        '''
        Function to get the connectivity based upon a dictionary of common atoms. The dictionary must have the following structure:

        {structure1: {fragment1: [1,2,3], fragment2:[4,5], reacting:[1,4,3,5]}, ...}

        Arguments:

        data_dict (dictionary): Dictionary with filename (key) and parsed data (value) pairs.

        Returns:

        connect_dict (Dictionary): A dictionary containing the list of atoms that are connected to the reacting atom.

        '''
        # Read in the common atoms file and convert it to a dictionary
        c = str(options.commonatoms)
        with open(c, 'rb') as common_atoms:
            if options.xyz:
                load_dict = pickle.load(common_atoms)
                common_atoms_dict = DIASSep._clean_common_atoms(load_dict)
            else:
                load_dict = pickle.load(common_atoms)
                new_1 = {k.replace('_reopt.','.'):v for k,v in load_dict.items()}
                new_2 = {k.replace('_reopt2.','.'):v for k,v in new_1.items()}
                new_3 = {k.replace('_reopt3.','.'):v for k,v in new_2.items()}
                common_atoms_dict = {k.replace('_reopt4.','.'):v for k,v in new_3.items()}

        connect_dict = {}
        atom_dict = {}
        for structure in data_dict.keys():

            name = data_dict[structure]['name']
            if options.xyz:
                name = name
            else:
                name = structure
            connected_atoms = []
            already_checked = []

            # Add check that looks for reopts in file that wouldn't match atom dict
            if 'reopt' in structure:
                alt_name = DIASSep._remove_reopts(structure)
            else:
                alt_name = name

            atom_dict[structure] = [common_atoms_dict[alt_name]['reacting'][0], common_atoms_dict[alt_name]['reacting'][1]]

            # Get the first reacting atom for the current structure and calculate its connectivity storing it in connected_atoms/already_checked
            atom_one = common_atoms_dict[alt_name]['reacting'][0]
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

    def _remove_reopts(string):
        '''
        Function to remove reopts from filenames for the purpose of matching to common_atoms.pkl file

        Arguments:

        string (String): A string of the filename.

        Returns:

        new_string (String): A new string removing reopts for the purpose of matching files.
        '''

        if '_reopt.' in string:
            new_string = string.replace('_reopt', '')
        elif '_reopt2.' in string:
            new_string = string.replace('_reopt2', '')
        elif '_reopt3.' in string:
            new_string = string.replace('_reopt3', '')
        return new_string

    def _output_structures(data_dict):
        '''
        Function to output both distorted structures extracted from the optimised TS structure. Works by pulling the coordinates from the 
        list of indexed coordinates and the coordinates not used make up the other species.

        Output:
        
        <filename>_reactant1.gjf: A .gjf file of the first reactant.
        <filename>_reactant2.gjf: A .gjf file of the second reactant.

        mapping.pkl: A pickle file containing the TS -> distorted structure mapping. Created if -m argument parsed.
        '''
        # Get the connectivity dictionary for every structure
        if options.commonatoms:
            connect_dict = DIASSep._get_multi_connected_break(data_dict)
        elif options.breakbonds is None:
            connect_dict = DIASSep._get_connected_atoms(data_dict)
        else:
            connect_dict = DIASSep._get_connected_break(data_dict, connect_break)
        
        atom_map = {}
        # Loop through each structure in the connect_dict
        for structure in connect_dict.keys():
            reactant_1_new_num_dict = {}
            name1 = structure.split('.')[0]+'_reactant_1.gjf'
            file_1 = open(name1, 'w')
            file_1.write('#Insert Method Here\n\nTitle\n\nChargeMultiplicity\n')
        # Loop through all atoms in the connect_dict for the current structure and pull the coordinates from data_dict
            for atom, new_atom in zip(connect_dict[structure], range(0, len(connect_dict[structure]))):
                coords = str(data_dict[structure]['coords1'][atom])
                coords = coords.replace('[', str(data_dict[structure]['atomsymb'][atom])+' ')
                coords = coords.replace(']', '\n')
                file_1.write(coords)
                reactant_1_new_num_dict[atom+1] = new_atom+1
            file_1.close()

        # Here we want to get all the atoms in the TS in a list to remove reactant_1 atoms to leave us with reactant_2 atoms
            name2 = structure.split('.')[0]+'_reactant_2.gjf'
            all_atoms = list(range(0, len(data_dict[structure]['coords1'])))
            second_connect = list(set(all_atoms)- set(connect_dict[structure]))

        # Repeat above but this time for reactant_2
            reactant_2_new_num_dict = {}
            file_2 = open(name2, 'w')
            file_2.write('#Insert Method Here\n\nTitle\n\nChargeMultiplicity\n')
            for atom, new_atom in zip(second_connect, range(0, len(second_connect))):
                coords = str(data_dict[structure]['coords1'][atom])
                coords = coords.replace('[', str(data_dict[structure]['atomsymb'][atom])+' ')
                coords = coords.replace(']', '\n')
                file_2.write(coords)
                reactant_2_new_num_dict[atom+1] = new_atom+1
            file_2.close()
            comb = {'reactant_1':reactant_1_new_num_dict, 'reactant_2':reactant_2_new_num_dict}
            atom_map[structure] = comb

        if options.mapping:
            with open('mapping.pkl', 'wb') as f:
                pickle.dump(atom_map, f)
                print('Atom mapping stored in mapping.pkl')

class DIntraSep:

    def __init__(self):

        atom1, atom2 = atoms[0], atoms[1]
        DIntraSep._output_intra(atom1, atom2)

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

    def _get_intra_split(atom1, atom2):

        '''
        Function to get the split of the molecule into two reactants and the linker. 

        Arguments:
        atom1 (Int): The first atom at the reacting centre (on reactant 1) and start of linker.
        atom2 (Int): The second atom at the reacting centre (on reactant 2) and end of linker.

        Returns:
        intra_dict (Dictionary): A dictionary of all structures and the molecules atoms after the split.
        data_dict (Dictionary): A dictionary containing the data parsed using cclib (e.g., coordinates ands displacements)

        '''

        # Get data from all structures in current working directory
        path_dict = {}
        split_dict = {}
        data_dict = DIASSep._get_freqs()
        intra_dict = {}
        # Loop through every structure and get connections in a dictionary form
        for structure in data_dict:
            dct = get_connections(data_dict[structure]['atomsymb'].tolist(),
                                data_dict[structure]['coords1'].tolist())
        # Use dct to generate a list of list of edges in the system
            edges = []
            for first_atom in dct.keys():
                for second_atom in dct[first_atom].keys():
                    edges.append(list([first_atom,second_atom]))
        # For the current structure, find the shortest path between atom1 and atom2 from the edges list
            path_dict[structure] = DIntraSep._get_shortest_path(edges, atom1, atom2)


        # Now we want to split the molecule based on these edges to get three molecules.
            split_dict[structure] = {'m_1': [path_dict[structure][0], path_dict[structure][1]],
                                    'm_2': [path_dict[structure][-1], path_dict[structure][-2]],
                                    'link': path_dict[structure]}

            molecule1, molecule2, linker = copy.deepcopy(dct), copy.deepcopy(dct), copy.deepcopy(dct)

        # Need to delete these specific connections from dct
            for split, diction in zip(split_dict[structure].keys(), [molecule1, molecule2, linker]):
                if split == 'link':
                    pass
                elif split == 'm_1' or 'm_2':
                    del diction[split_dict[structure][split][0]][split_dict[structure][split][1]]
                    del diction[split_dict[structure][split][1]][split_dict[structure][split][0]]
            
        # Get the connectivity of the atoms specified for each of the reactants.
            connect_dict = {}  
            for split, partition in zip(split_dict[structure].keys(), [molecule1, molecule2, linker]):
                connected_atoms = []
                already_checked = [] 
                
                atom_one = split_dict[structure][split][0]
                for connected_atom in partition[int(atom_one)].keys():
                    connected_atoms.append(connected_atom)
                connected_atoms.append(int(atom_one))
                already_checked.append(int(atom_one))
                # Now loop through the connected atoms to get the next set of connected atoms checking if its in already_checked to avoid extra computation
                for new_atom in connected_atoms:
                    if new_atom not in already_checked:
                        for new_a in partition[int(new_atom)].keys():
                            connected_atoms.append(new_a)
                        already_checked.append(new_atom)
                connect_dict[split] = [*set(connected_atoms)] # Removes duplicate atoms in the list
                
                # Remove m_1 and m_2 from all atoms to get linkers
                if split == 'link':
                    rem_m_1 = [x for x in connect_dict['link'] if x not in connect_dict['m_1']]
                    rem_m_2 = [x for x in rem_m_1 if x not in connect_dict['m_2']]
                    connect_dict['link'] = rem_m_2


            # Add the X atoms back in to the respective systems
            connect_dict['m_1'].append(split_dict[structure]['m_1'][-1])
            connect_dict['m_2'].append(split_dict[structure]['m_2'][-1])
            for m in ['m_1', 'm_2']:
                connect_dict['link'].append(split_dict[structure][m][0])

            intra_dict[structure] = connect_dict
            
        return intra_dict, data_dict 

    def _output_intra(atom1, atom2):
            '''
            Function to output the three distorted structures extracted from the optimised TS structure. Works by pulling the coordinates from the 
            list of indexed coordinates and the coordinates not used make up the other species.
            Any valences lost are replaced with H as per Houk et. al. (http://dx.doi.org/10.1016/j.tetlet.2010.11.121)

            Output:
            
            <filename>_reactant1.gjf: A .gjf file of the first reactant.
            <filename>_reactant2.gjf: A .gjf file of the second reactant.
            <filename>_linker.gjf: A .gjf file of the linker.
            '''

            intra_dict, data_dict = DIntraSep._get_intra_split(atom1, atom2)
            # Create a change dictionary to make substitutions for H's.
            change_dict = {}
            for structure in intra_dict.keys():
                temp_dict = {}
                for mol in intra_dict[structure].keys():
                    if mol == 'link':
                        temp_dict[mol] = [intra_dict[structure]['m_1'][0], intra_dict[structure]['m_2'][0]]
                    else:
                        temp_dict[mol] = intra_dict[structure][mol][-1]
                change_dict[structure] = temp_dict
            # Loop through each structure in the connect_dict
            for structure in intra_dict.keys():
                name1 = structure.split('.')[0]+'_reactant_1.gjf'
                name2 = structure.split('.')[0]+'_reactant_2.gjf'
                linker = structure.split('.')[0]+'_linker.gjf'
                file_1 = open(name1, 'w')
                file_1.write('#Insert Method Here\n\nTitle\n\nChargeMultiplicity\n')
            # Loop through all atoms in the connect_dict for the current structure and pull the coordinates from data_dict
                for atom in intra_dict[structure]['m_1']:
                    coords = str(data_dict[structure]['coords1'][atom])
                    coords = coords.replace('[', str(data_dict[structure]['atomsymb'][atom])+' ')
                    coords = coords.replace(']', '\n')
                    if atom == change_dict[structure]['m_1']:
                        coords = 'H' + coords[1:]
                    file_1.write(coords)
                file_1.close()

            # Repeat above but this time for reactant_2 
                file_2 = open(name2, 'w')
                file_2.write('#Insert Method Here\n\nTitle\n\nChargeMultiplicity\n')
                for atom in intra_dict[structure]['m_2']:
                    coords = str(data_dict[structure]['coords1'][atom])
                    coords = coords.replace('[', str(data_dict[structure]['atomsymb'][atom])+' ')
                    coords = coords.replace(']', '\n')
                    if atom == change_dict[structure]['m_2']:
                        coords = 'H' + coords[1:]
                    file_2.write(coords)
                file_2.close()

            # Repeat above but this time for reactant_2 
                file_3 = open(linker, 'w')
                file_3.write('#Insert Method Here\n\nTitle\n\nChargeMultiplicity\n')
                for atom in intra_dict[structure]['link']:
                    coords = str(data_dict[structure]['coords1'][atom])
                    coords = coords.replace('[', str(data_dict[structure]['atomsymb'][atom])+' ')
                    coords = coords.replace(']', '\n')
                    if atom == atom1:
                        coords = 'H' + coords[1:]
                    elif atom == atom2:
                        coords = 'H' + coords[1:]
                    file_3.write(coords)
                file_3.close()

def main(c=None):
    if options.intrabreakbonds:
        DIntraSep()
    else:
        DIASSep()

if __name__ == '__main__':

    # Argument parse
    parser = argparse.ArgumentParser(
        prog='diassep',
        description='Python script for creating distorted reactant species from optimised TS Gaussian output files.')
    parser.add_argument('-b', dest='breakbonds', default=None, required=False, help='Comma separated list of numbers of the two atoms that connectivity is to be broken between.')
    parser.add_argument('-i', dest='intrabreakbonds', default=None, required=False, help='Comma separated list of numbers of the two atoms that start and end the linker - code may be specific for Diels-Alder reaction.')
    parser.add_argument('-a', dest='adjacency', action='store_true', help='Utilise the adjacency matrix approach rather than the distance approach.')
    parser.add_argument('-m', dest='mapping', action='store_true', help='Provides a pkl file containing a dictionary of all the mapped atoms numbers from the TS to the distorted structures.')
    parser.add_argument('-c', dest='commonatoms', default=None, required=False, help='Utilises a dictionary of common atoms to separate structures.')
    parser.add_argument('-x', dest='xyz', action='store_true', help='Perform the DIAS separation for xyz files - requires commonatoms (-c) to be parsed.')
    
    (options, args) = parser.parse_known_args()
    # Deal with bond breaking argument
    if options.intrabreakbonds:
        atoms = [int(x) for x in options.intrabreakbonds.split(',')]
        main(atoms)
    elif options.breakbonds:
        connect_break = [int(x) for x in options.breakbonds.split(',')]
        main(connect_break)
    else:
        main()