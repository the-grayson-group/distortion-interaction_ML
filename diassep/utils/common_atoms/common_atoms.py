"""
#####################################################################################
#                                   common_atoms.py                                 #
#                                                                                   #
#  Script to extract the diene/dipolar and dienohphile/dipolarophile common atoms   #
#  from a Gaussian frequency calculation using the frequencies and displacement     #
#  vectors along with the distances. Currently only works for these two reaction    #
#  centre systems. Saves a common_atoms.pkl file in the working directory. Also,    #
#  will save a check_systems.txt file if there are structures which parse errors.   #                                                
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
import cclib
import pickle
import numpy as np
from molml.utils import get_connections

class general:
 
     def __init__(self):

        data_dict = general._get_data_cclib()
        com, che = general._get_common_atoms(data_dict)
        common_atoms, check = general._check(com, che)
        # Save common atoms and names of structures needed to check
        general._save_files(common_atoms, check)

     def _save_files(common_atoms, check):
        '''
        Function to save the appropriate files.

        Arguments:
        common_atoms (Dictionary): A dictionary containing all the common atoms information for all structures.
        check (List): A list of structures to check as they have thrown up an error.
        '''
        with open('common_atoms.pkl', 'wb') as f1:
            pickle.dump(common_atoms, f1)
            print('Common atoms saved to common_atoms.pkl')
            f1.close()
        if len(check) != 0:
            with open('check_systems.txt', 'w') as f2:
                for line in check:
                    f2.write(str(line+'\n'))
                print('Check structures in check.txt - ',len(check),' files.')
                f2.close
        else:
            pass

     def _get_data_cclib():
        '''
        Function to take output from _get_data and specify the values needed to calculate the
        separate species.

        Returns:

        data_dict (dictionary of dictionaries): DoD with filename (key) and coords1, coords2 and 
        vectors (nested keys and values).
        '''
        data_dict = {}
        dct = {}
        files = []
        for filename in os.listdir():
            if filename.endswith(".out") or filename.endswith(".log"):
                files.append(filename)
            else:
                continue
        
        for file in files:
            dct[file] = cclib.io.ccread(file)

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

            if '_reopt' in file:
                    filename = file.split('_reopt')[0]
            else:
                    filename = file.split('.out')[0]   

            data_dict[file] = {} # Create nested dictionary
            data_dict[file]['coords1'] = dct[file].atomcoords[-1]
            data_dict[file]['coords2'] = dct[file].atomcoords[-1] - dct[file].vibdisps[position]
            data_dict[file]['coords3'] = dct[file].atomcoords[-1] + dct[file].vibdisps[position]
            data_dict[file]['disp_vect'] = dct[file].vibdisps[position]
            data_dict[file]['atomnos'] = dct[file].atomnos
            for charge in dct[file].atomcharges.keys():
                data_dict[file][charge] = dct[file].atomcharges[charge]

            atom_symbs = []
            for atom in data_dict[file]['atomnos']:
                atom_symbs.append(atomic_masses[atom])
            
            data_dict[file]['atomsymb'] = np.array(atom_symbs)
            data_dict[file]['name'] = filename

        return data_dict
    
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
    
        return reacting_atoms, check
     
     def _get_common_atoms(data_dict):
        '''
        Function to get the common atoms for all structures

        Returns:
        common_atoms (Dictionary): A dictionary containing all the common atoms information for all structures.
        check (List): A list of structures to check as they have thrown up an error.
        ''' 
        reacting_centres, check = general._get_reacting_atoms(data_dict)
        edge_dict = general._get_edges(data_dict) 
        reacting_centres = general._get_list(general._remove_dupes(reacting_centres))
        common_atoms = {}
        for structure in edge_dict.keys():
            dp_atoms = []
            di_atoms = []
            for atom_one in reacting_centres[structure]:
                for atom_two in reacting_centres[structure]:
                    if atom_one == atom_two:
                        pass
                    else:
                        path = general._get_shortest_path(edge_dict[structure], int(atom_one), int(atom_two))
                        if len(path) == 0:
                            pass
                        elif len(path) == 2:
                            dp_atoms.extend(path)
                        else:
                            di_atoms.extend(path)
            common_atoms[structure] = {'dp':list(set(dp_atoms)), 'di':list(set(di_atoms)), 'reacting':reacting_centres[structure], 'name':data_dict[structure]['name']}

        return common_atoms, check
     
     def _get_list(reacting_centres):
        '''
        Function to convert reacting_centre values to a list.

        Arguments:
        reacting_centres (Dictionary) = A dictionary of the reacting centres for all reactions.

        Returns:
        new_reacting_centres (Dictionary) = A cleaned version of reacting_centres
        ''' 
        new_reacting_centre = {} 
        for structure in reacting_centres:
            atom_list = []
            for pair in reacting_centres[structure]:
                atom_list.append(pair.split('_')[0])
                atom_list.append(pair.split('_')[-1])
            new_reacting_centre[structure] = atom_list
        return new_reacting_centre

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

def main():

    general()

if __name__ == '__main__':

    main()
