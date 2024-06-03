'''
#####################################################################################
#                                 get_energies.py                                   #
#   Code to extract energies from multiple different paths and calculate barriers.  #
#   Requires a .yaml file in the working directory that has the format as below:    #
#                                                                                   #
#   Usage:                                                                          #
#   Ensure that there is a config.yaml file in the working directory. This needs    #
#   to contain the information below (example).                                     #
#                                                                                   #
#      dataset: '3_2'                                                               #
#      paths:                                                                       #
#          ts: 'ts/Goodvibes_am1_spe_3_2_ts.csv'                                    #
#          gs: 'gs/Goodvibes_am1_spe_3_2_gs.csv'                                    #
#          dist_gs: 'dist_gs/Goodvibes_am1_spe_3_2_dist_gs.csv'                     #
#      display: False                                                               #
#      filename: 'am1_spe_three_two_barriers'                                       #
#                                                                                   #
#   To run:                                                                         #
#      python get_energies.py                                                       #
#                                                                                   #
#####################################################################################
#                                       Authors                                     #
#                                                                                   #
#--------------------------------- Samuel G. Espley --------------------------------#
#                                                                                   #
#####################################################################################
'''

# Imports
import yaml
import pandas as pd

class General:

    def __init__():
        pass

    def _load_config():
        
        '''
        Function to load in the configuration file for energy extraction.

        Returns
        setup (Dictionary): A dictionary containing setup information for extracting the energies.
        '''
        with open('config.yaml', 'r') as file:
            setup = yaml.safe_load(file)
        return setup

    def _load_data(name):
        
        '''
        Function to load data from a Goodvibes file.

        Arguments:
        name (String): A string of the name of the file to extract.

        Returns:
        df (DataFrame): A DataFrame containing the Goodvibes data.
        '''

        df = pd.DataFrame()
            
        # Open file
        file = open(name, 'r')
        lines = file.readlines()
        for line in lines:
            if str(line).startswith('   Structure'):
                ind = lines.index(line)
        lines = lines[ind:]
        
        # Remove separating line
        for line in lines:
            if str(line).startswith('   *'):
                lines.pop(lines.index(line))
        
        # Extract headers and values
        headers = lines[0].split(',')[:-1]
        vals = lines[1:]
        values = []
        for row in vals:
            new_row = row.split(',')
            values.append(new_row[:len(headers)])
            #values.append(row.split(','))

        # Create DataFrame
        df = pd.DataFrame(values, columns=headers)
        
        # Tidy DataFrame
        df['   Structure'] = df['   Structure'].str.replace('o  ','') # Remove bullets
        df = df.rename(columns={'   Structure': 'Structure'}) # Rename columns

        return df

    def _get_dataframes(setup):
        
        '''
        Function to get the data extracted from the Goodvibes files stated in the setup file.

        Arguments:
        setup (Dictionary): A dictionary containing setup information for extracting the energies.

        Returns:
        gs (DataFrame): A DataFrame containing the ground state reactant energies extracted from Goodvibes.
        ts (DataFrame): A DataFrame containing the transition energies extracted from Goodvibes.
        dist_gs (DataFrame): A DataFrame containing the distorted ground state reactant energies extracted from Goodvibes.
        '''

        # Create initial DataFrames
        gs, ts, dist_gs = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        # Loop through the paths in setup
        for path in setup['paths']:
            if path =='gs':
                df_temp = General._load_data(setup['paths'][path])
                gs = pd.concat([gs, df_temp])
            if path =='ts':
                df_temp = General._load_data(setup['paths'][path])
                ts = pd.concat([ts, df_temp])
            if path =='dist_gs':
                df_temp = General._load_data(setup['paths'][path])
                dist_gs = pd.concat([dist_gs, df_temp])
        return gs, ts, dist_gs

    def _quick_check(gs, dist_gs):
        
        '''
        Function to do a quick match up check between gs and distorted structures. Highlights gs structures that are not needed.

        Arguments:
        gs (DataFrame): A DataFrame containing the ground state reactant energies extracted from Goodvibes.
        dist_gs (DataFrame): A DataFrame containing the distorted ground state reactant energies extracted from Goodvibes.

        '''
        lst = []
        for id in gs['id']:
            if dist_gs['E'][dist_gs['id'] == id].empty == True:
                print(str(gs['Structure'][gs['id'] == id].iloc[0]))
                #print('gs_'+id+'.out ')
                lst.append(id)
        print(len(lst))

    def _get_distortion_energies(gs, dist_gs):
        
        '''
        Function to get the distorted energies from the gs and dist_gs dataframes.

        Arguments:
        gs (DataFrame): A DataFrame containing the ground state reactant energies extracted from Goodvibes.
        dist_gs (DataFrame): A DataFrame containing the distorted ground state reactant energies extracted from Goodvibes.

        Returns:
        df (DataFrame): A dataframe of the distortion energies with a reaction id.
        df_sum (DataFrame): A DataFrame of the summed distortion energies.

        Note:

        The reaction id (id) in the three_two reactions has the format x_y where x is the reaction id and y is the reactant
        number (e.g., 1 or 2.)
        '''

        # Check that the length of both gs and dist_gs are the same.
        #assert len(gs['rxn_id'].unique()) == len(dist_gs['rxn_id'].unique()), 'The number of ground states does not equal distorted structures'
        
        # Extract the energies of each molecule
        d_energies = {}
        for id in gs['id']:
            #gs_e = float(gs['E_SPC'][gs['id'] == id])
            #dgs_e = float(dist_gs['E_SPC'][dist_gs['id'] == id])
            gs_e = float(gs['E'][gs['id'] == id])
            dgs_e = float(dist_gs['E'][dist_gs['id'] == id])

            dist_e = (dgs_e - gs_e)*627.5
            d_energies[id] = float(dist_e)
            
        df = pd.DataFrame(d_energies, index=[1]).transpose().reset_index().rename(columns={'index':'id'})
        df = df.rename(columns={1:'distortion_energies'})
        
        # Save sum of each reactions energies
        sum_d_energies = {}
        for rxn_id in gs['rxn_id'].unique().tolist():
            if 'd' not in str(gs['id'][0]):
                temp_id = str(rxn_id)+'_'
                temp_lst = df['distortion_energies'].loc[df['id'].str.startswith(temp_id)].tolist()
                sum_d_energies[rxn_id] = sum(temp_lst)
            elif 'd' in str(gs['id'][0]):
                temp_lst = df['distortion_energies'].loc[df['id'].str.endswith('_'+str(rxn_id))].tolist()
                sum_d_energies[rxn_id] = sum(temp_lst)

        sum_df = pd.DataFrame(sum_d_energies, index=[1]).transpose().reset_index().rename(columns={'index':'id'})
        sum_df = sum_df.rename(columns={1:'sum_distortion_energies'})

        
        # Save individual contributions to dataframe
        if dataset == '3_2':
            d_contributions = {}
            for rxn_num in sum_df['id']:
                d_rxn = {}
                temp_df = df[df['id'].str.startswith(str(rxn_num)+'_')]
                for id, energy in zip(temp_df['id'], temp_df['distortion_energies']):
                    d_rxn[id] = energy
                d_contributions[rxn_num] = d_rxn
                assert len(d_contributions[rxn_num].keys()) == 2, d_contributions
        else:
            d_contributions = {}
            for rxn_num in sum_df['id']:
                d_rxn = {}
                temp_df = df[df['id'].str.endswith(str('_'+str(rxn_num)))]
                for id, energy in zip(temp_df['id'], temp_df['distortion_energies']):
                    d_rxn[id] = energy
                d_contributions[rxn_num] = d_rxn
                assert len(d_contributions[rxn_num].keys()) == 2, d_contributions

        
        contributions = pd.DataFrame([d_contributions]).transpose().reset_index().rename(columns={'index':'id', 0:'distortion_contributions'})
        sum_df = sum_df.merge(contributions, on='id')

        return df, sum_df

    def _get_barrier(gs, ts):
        
        '''
        Function to get the transition state reaction barriers.
        
        Arguments:
        gs (DataFrame): A DataFrame containing the ground state reactant energies extracted from Goodvibes.
        ts (DataFrame): A DataFrame containing the transition energies extracted from Goodvibes.
        
        Returns:
        df (DataFrame): A dataframe of the transition state barrier energies with a reaction id.
        '''
        # Check that the length of both gs and ts are the same.
        #assert len(gs['rxn_id'].unique()) == len(ts['rxn_id'].unique()), 'The number of ground states does not equaly distorted structures'

        q_d_energies = {}
        d_energies = {}
        for id in gs['rxn_id'].unique():
            
            # gs_energies= gs['E_SPC'][gs['rxn_id']==id].tolist()
            # q_gs_energies= gs['qh-G(T)_SPC'][gs['rxn_id']==id].tolist()
            gs_energies= gs['E'][gs['rxn_id']==id].tolist()
            q_gs_energies= gs['qh-G(T)'][gs['rxn_id']==id].tolist()

            gs_sum = sum([float(i) for i in gs_energies])
            q_gs_sum = sum([float(i) for i in q_gs_energies])

            # ts_energy = float(ts['E_SPC'][ts['rxn_id']==id])
            # q_ts_energy = float(ts['qh-G(T)_SPC'][ts['rxn_id']==id])
            ts_energy = float(ts['E'][ts['rxn_id']==id])
            q_ts_energy = float(ts['qh-G(T)'][ts['rxn_id']==id])

            barrier = (ts_energy - gs_sum)*627.5
            q_barrier = (q_ts_energy - q_gs_sum)*627.5

            d_energies[id] = float(barrier)
            q_d_energies[id] = float(q_barrier)
            
        e_barrier_df = pd.DataFrame(d_energies, index=[1]).transpose().reset_index().rename(columns={'index':'id'})
        e_barrier_df = e_barrier_df.rename(columns={1:'e_barrier'})

        q_barrier_df = pd.DataFrame(q_d_energies, index=[1]).transpose().reset_index().rename(columns={'index':'id'})
        q_barrier_df = q_barrier_df.rename(columns={1:'q_barrier'})

        barrier_df = e_barrier_df.merge(q_barrier_df, on='id')

        return barrier_df

    def _get_interaction(sum_dist_energies, barriers):
        
        '''
        Function to get the interaction energy from the transition state and distorted structures.

        Arguments:
        sum_dist_energies (DataFrame): A DataFrame containing the summed distortion energies.
        barriers (DataFrame): A DataFrame containing the reaction barriers.

        Returns:
        intera_df (DataFrame): A DataFrame containing the interaction energies.
        '''
        intera_energies = {}
        not_needed = []
        for rxn_id in barriers['id']:
            if sum_dist_energies[sum_dist_energies['id'] == rxn_id].empty == True:
                not_needed.append(rxn_id)
            else:
                dist_energy = float(sum_dist_energies['sum_distortion_energies'][sum_dist_energies['id'] == rxn_id])
                barrier = float(barriers['e_barrier'][barriers['id'] == rxn_id])
                intera_energy = dist_energy - barrier
                intera_energies[rxn_id] = intera_energy
            

        intera_df = pd.DataFrame(intera_energies, index=[1]).transpose().reset_index().rename(columns={'index':'id'})
        intera_df = intera_df.rename(columns={1:'interaction_energies'})

        return intera_df

    def _combine_save(barriers, sum_distorted, interactions):
        
        '''
        Function to combine energies and save to a pickle file.

        Arguments:
        barriers (DataFrame): A DataFrame containing the reaction barriers.
        sum_distorted (DataFrame): A DataFrame containing the summed distortion energies.
        interactions (DataFrame): A DataFrame containing the interaction energies.

        Returns:
        df (DataFrame): A DataFrame containing the outputted data.
        '''

        df = barriers.merge(sum_distorted, on='id').merge(interactions, on='id')
        f = sve_name+'.pkl'
        df.to_pickle(f)
        print(f'Energies DataFrame saved to {sve_name}')
        
        return df

class DA:

    def __init__():
        pass

    def _clean_dfs(df):

        '''
        Function to clean Diels-Alder DataFrames.

        Arguments:
        df (DataFrame): A DataFrame containing the given Diels-Alder data.

        Returns:
        df (DataFrame): A DataFrame containing the given cleaned Diels-Alder data.
        
        Note:
        ### reactant_1 is always the diene due to the way the atoms are numbered in the TS structures  ###
        
        '''
        # Get reaction number
        num_lst = []
        for name in df['Structure']:
            if 'reactant' in str(df['Structure'][0]):
                if 'endo' not in str(name) and 'exo' not in str(name):
                    num = name.split('_')[3]
                    num_lst.append(int(num))
                else:
                    num = name.split('_')[2]
                    num_lst.append(int(num))
            else:
                num = name.split('_')[-1]
                num_lst.append(int(num))
        df['rxn_id'] = num_lst

        if 'reactant' in str(df['Structure'][0]):
            rct_id_lst = []
            for name in df['Structure']:
                name_split = name.split('_')
                if name_split[-1] == str(1):
                    rct = 'di_'
                    if str(name_split[2]) == 'reactant':
                        rct_id_lst.append(rct+str(name_split[1]))
                    else:
                        rct_id_lst.append(rct+str(name_split[2]))
                elif name_split[-1] == str(2):
                    rct = 'dp_'
                    if str(name_split[2]) == 'reactant':
                        rct_id_lst.append(rct+str(name_split[1]))
                    else:
                        rct_id_lst.append(rct+str(name_split[2]))
                else:
                    print('Error in file: ', name)
            df['id'] = rct_id_lst

        elif 'gs' in str(df['Structure'][0]):
            rct_id_lst = []
            for name in df['Structure']:
                name_split = name.split('_')
                rct = name_split[1]+'_'+str(name_split[-1])
                rct_id_lst.append(rct)
            df['id'] = rct_id_lst   
        else:
            pass

        return df

    def _runner(gs, ts, dist_gs):
        
        '''
        Function to run the energy extraction. 
        
        Arguments:
        gs (DataFrame): A DataFrame containing the ground state reactant energies extracted from Goodvibes.
        ts (DataFrame): A DataFrame containing the transition energies extracted from Goodvibes.
        dist_gs (DataFrame): A DataFrame containing the distorted ground state reactant energies extracted from Goodvibes.

        Returns:
        df (DataFrame): A DataFrame containing the given Diels-Alder data.

        '''
        gs, ts, dist_gs = DA._clean_dfs(gs), DA._clean_dfs(ts), DA._clean_dfs(dist_gs) 
        #General._quick_check(gs, dist_gs)
        dist_energies, sum_dist_energies = General._get_distortion_energies(gs, dist_gs)
        barriers = General._get_barrier(gs, ts)
        intera_energies = General._get_interaction(sum_dist_energies, barriers)
        df = General._combine_save(barriers, sum_dist_energies, intera_energies)

        return df

class LA_DA:
    
    def __init__(self):

        self.cat_d = {
                    'alcl3_1_3':0,
                    'alcl3_1_4':1,
                    'bf3_1_3':2, 
                    'bf3_1_4':3,
                    'i2_1_3':4,
                    'i2_1_4':5, 
                    'zncl2_1_3':6,
                    'zncl2_1_4':7
                }
    
    def _get_barrier(self, gs, ts):
        '''
        Function to get the reaction barriers from GS reactants and TS structures.

        Arguments:
        gs (DataFrame): A dataframe containing the GS reactant energies.
        ts (DataFrame): A dataframe containing the TS energies.

        Returns:
        barriers (DataFrame): A dataframe containing all the barriers for each LA-DA reaction.
        '''
        q_d_energies = {}
        d_energies = {}
        for structure in ts['Structure']:
            cat = structure.split('_')[0]
            rxn = structure.split('_', 3)[-1]
            id = self.cat_d[f'{cat}_{rxn}']
            diene = f'{cat}_diene_{rxn}'
            dienophile = f'{cat}_dienophile_{rxn}'
            cat_name = f'{cat}_{cat}'
            e_d = {}
            # Loop through each of the two energies that we want to extract. 
            for E, name in zip(['E', 'qh-G(T)'], ['e_barrier', 'q_barrier']):
                ts_e = float(ts[E][ts['Structure'] == structure].iloc[0])
                cat_e = float(gs[E][gs['Structure'] == cat_name].iloc[0])
                diene_e = float(gs[E][gs['Structure'] == diene].iloc[0])
                dienophile_e = float(gs[E][gs['Structure'] == dienophile].iloc[0])
                barrier = (ts_e - (cat_e+diene_e+dienophile_e))*627.5095
                e_d[name] = barrier
            q_d_energies[id] = float(e_d['q_barrier'])
            d_energies[id] = float(e_d['e_barrier'])
        
        e_barrier_df = pd.DataFrame(d_energies, index=[1]).transpose().reset_index().rename(columns={'index':'id'})
        e_barrier_df = e_barrier_df.rename(columns={1:'e_barrier'})

        q_barrier_df = pd.DataFrame(q_d_energies, index=[1]).transpose().reset_index().rename(columns={'index':'id'})
        q_barrier_df = q_barrier_df.rename(columns={1:'q_barrier'})

        barrier_df = e_barrier_df.merge(q_barrier_df, on='id')

        return barrier_df
            
    def _get_distortion_energies(self, gs, dist_gs):
        '''
        Function to determine the distortion energies.

        Arguments:
        gs (DataFrame): A dataframe containing the GS reactant energies.
        dist_gs (DataFrame): A dataframe containing the distorted GS reactant energies.

        Returns:
        sum_df (DataFrame): A dataframe containing the distorted energies.
        '''
        # Create dictionary to contain data.
        d_energies = {'diene':{}, 'dienophile':{}, 'cat':{}}
        # Loop through every distorted strucutre. 
        for dist_gs_structure in dist_gs['Structure']:
            # Pull out catalyst, reaction and id.
            cat = dist_gs_structure.split('_')[0]
            rxn = dist_gs_structure.split('_', 4)[-1]
            id = self.cat_d[f'{cat}_{rxn}']
            gs_structure = dist_gs_structure.replace('_dist_gs_', '_')
            if gs_structure.split('_')[0] == gs_structure.split('_')[1]:
                gs_structure = gs_structure.split('_')[0]+'_'+gs_structure.split('_')[1]
            dist_e = float(dist_gs['E'][dist_gs['Structure'] == dist_gs_structure].iloc[0])
            gs_e = float(gs['E'][gs['Structure'] == gs_structure].iloc[0])
            distorted_e = (dist_e - gs_e)*627.5095
            # Deal with every possible structure in the LA-DA dataset.
            if 'diene' in gs_structure:
                d_energies['diene'][id] = distorted_e
            elif 'dienophile' in gs_structure:
                d_energies['dienophile'][id] = distorted_e
            else:
                d_energies['cat'][id] = distorted_e
        
        # Create DataFrame.
        sum_df = pd.DataFrame()
        for key in d_energies['diene']:
            d = {
                'id':key,
                'diene_distortion_energy':d_energies['diene'][key],
                'dienophile_distortion_energy':d_energies['dienophile'][key],
                'cat_distortion_energy':d_energies['cat'][key],
                'sum_distortion_energies':float(d_energies['diene'][key]+d_energies['dienophile'][key]+d_energies['cat'][key])
            }
            df = pd.DataFrame(d, index=[1])
            sum_df = pd.concat([sum_df, df], axis=0)

        return sum_df

    def _get_interaction_energies(self, barriers, dist_energies):
        '''
        Function to calculate the interaction energies and to merge all the data into one collective dataframe.
        
        Arguments:
        barriers (DataFrame): A dataframe containing the reaction barriers.
        dist_energies (DataFrame): A dataframe containing the distortion energies.

        Returns:
        df (DataFrame): A dataframe combining the barriers, distortion energies and interaction energies.
        '''
        # Merge dataframes.
        df = barriers.merge(dist_energies, on='id')
        # Create interaction energies column.
        df['interaction_energies'] = df['sum_distortion_energies'] - df['e_barrier']
        # Return df.
        return df

    def _runner(gs, ts, dist_gs):
        
        la = LA_DA()
        barriers = la._get_barrier(gs, ts) 
        dist_energies = la._get_distortion_energies(gs, dist_gs) 
        df = la._get_interaction_energies(barriers, dist_energies)
        # Save the energies dataframe.
        f = sve_name+'.pkl'
        df.to_pickle(f)
        print(f'Energies DataFrame saved to {sve_name}')
        return df

class TT:

    def __init__():
        pass

    def _get_dft_dataframes(setup):
        
        '''
        Function to get the data extracted from the Goodvibes files stated in the setup file.

        Arguments:
        setup (Dictionary): A dictionary containing setup information for extracting the energies.

        Returns:
        barriers (DataFrame): A DataFrame containing the extracted information from Goodvibes gs and tss.
        dist_gs (DataFrame): A DataFrame containing the extracted information from Goodvibes dist_gs.
        '''

        # Create initial DataFrames
        b_info, dist_gs = pd.DataFrame(), pd.DataFrame()
        # Loop through the paths in setup
        for path in setup['paths']:
            if path == 'barriers':
                b_info = pd.read_pickle(setup['paths'][path])
            if path =='dist_gs':
                dist_gs = pd.read_pickle(setup['paths'][path])
                # df_temp = General._load_data(setup['paths'][path])
                # dist_gs = pd.concat([dist_gs, df_temp])
        return b_info, dist_gs

    def _dft_get_distortion_energies(barriers, dist_gs):

        '''
        Function to get the DFT [3+2] Cycloaddition barriers.

        Arguements:
        barriers (DataFrame): A DataFrame with the reaction barriers and masses.
        dist_gs (DataFrame): A DataFrame containing the distorted ground state reactant energies extracted from Goodvibes.
        
        Returns:
        df (DataFrame): A dataframe of the distortion energies with a reaction id.
        df_sum (DataFrame): A DataFrame of the summed distortion energies.
        '''
        
        d_energies = {}
        for structure in dist_gs['Structure']:
            distorted_energy = float(dist_gs['E_SPC'][dist_gs['Structure']==structure])
            #distorted_energy = float(dist_gs['E'][dist_gs['Structure']==structure])
            mass = dist_gs['mass'][dist_gs['Structure'] == structure]
            rxn_num, rct_num = structure.split('_')[1], structure.split('_')[-1]
            id = str(rxn_num)+'_'+str(rct_num)
            b = barriers[barriers['rxn_id']==rxn_num]
            for col in ['r0_mass', 'r1_mass']:
                if mass in b[col].values:
                    en = col.split('_')[0]+'_e_energy'
                    if float((distorted_energy - float(b[en]))*627.5) < -1000:
                        pass
                    elif float((distorted_energy - float(b[en]))*627.5) > 1000:
                        pass
                    else:
                        energy = float(b[en])
            dist_energy = (distorted_energy - energy)*627.5
            d_energies[id] = dist_energy

        df = pd.DataFrame(d_energies, index=[1]).transpose().reset_index().rename(columns={'index':'id'})
        df = df.rename(columns={1:'distortion_energies'})

        sum_d_energies = {}
        d_contributions = {}
        for structure in dist_gs['Structure']:
            rxn_num = structure.split('_')[1]
            sum_d_energies[rxn_num] = sum(df['distortion_energies'][df['id'].str.startswith(str(rxn_num)+'_')])
            temp_df = df[df['id'].str.startswith(str(rxn_num)+'_')]
            d_rxn = {}
            for id in temp_df['id']:
                d_rxn[id] = float(temp_df['distortion_energies'][temp_df['id']==id])
            d_contributions[rxn_num] = d_rxn
            assert len(d_contributions[rxn_num].keys()) == 2, d_contributions

        sum_df = pd.DataFrame(sum_d_energies, index=[1]).transpose().reset_index().rename(columns={'index':'id'})
        sum_df = sum_df.rename(columns={1:'sum_distortion_energies'})

        contributions = pd.DataFrame([d_contributions]).transpose().reset_index().rename(columns={'index':'id', 0:'distortion_contributions'})
        sum_df = sum_df.merge(contributions, on='id')
        
        return df, sum_df
            
    def _dft_get_barriers(b_info):

        '''
        Function to manipulate a barriers DataFrame.

        Arguments:
        b_info (DataFrame): A DataFrame of reaction barriers that needs processing.

        Returns:
        barriers (DataFrame): A DataFrame of reaction barriers that has been processed.
        '''
        
        barriers = b_info[['rxn_id', 'dft_q_barrier', 'dft_e_barrier']].rename(columns={'rxn_id':'id', 'dft_q_barrier':'q_barrier', 'dft_e_barrier':'e_barrier'})
        
        return barriers

    def _clean_dfs(df):

        '''
        Function to clean [3+2] DataFrames.

        Arguments:
        df (DataFrame): A DataFrame containing the given [3+2] data.
        
        Returns:
        df (DataFrame): A DataFrame containing the given cleaned [3+2] data.
 
        '''
        df['Structure'] = df['Structure'].str.replace('_reopt3', '')
        df['Structure'] = df['Structure'].str.replace('_reopt2', '')
        df['Structure'] = df['Structure'].str.replace('_reopt', '')
        
        # Get reaction number
        num_lst = []
        for name in df['Structure']:
            num = name.split('_')[1]
            num_lst.append(int(num))
        df['rxn_id'] = num_lst

        if 'reactant' not in str(df['Structure'][0]):
            pass
        else:
            rct_num_lst = []
            for name in df['Structure']:
                rct_num = name.split('_')[3]
                rct_num_lst.append(int(rct_num))
            df['rct_id'] = rct_num_lst
        if 'reactant' in str(df['Structure'][0]):
            df['id'] = df['rxn_id'].astype(str) + '_' + df['rct_id'].astype(str)
        else:
            pass
        return df

    def _dft_runner(b_info, dist_gs):

        '''
        Function to run the energy extraction. 
        
        Arguments:
        gs (DataFrame): A DataFrame containing the ground state reactant energies extracted from Goodvibes.
        ts (DataFrame): A DataFrame containing the transition energies extracted from Goodvibes.
        dist_gs (DataFrame): A DataFrame containing the distorted ground state reactant energies extracted from Goodvibes.

        Returns:
        df (DataFrame): A DataFrame containing the given [3+2] DFT data.

        '''
        dist_energies, sum_dist_energies = TT._dft_get_distortion_energies(b_info, dist_gs)
        barriers = TT._dft_get_barriers(b_info)
        intera_energies = General._get_interaction(sum_dist_energies, barriers)
        df = General._combine_save(barriers, sum_dist_energies, intera_energies)

        return df

    def _runner(gs, ts, dist_gs):
        
        '''
        Function to run the energy extraction. 
        
        Arguments:
        gs (DataFrame): A DataFrame containing the ground state reactant energies extracted from Goodvibes.
        ts (DataFrame): A DataFrame containing the transition energies extracted from Goodvibes.
        dist_gs (DataFrame): A DataFrame containing the distorted ground state reactant energies extracted from Goodvibes.

        Returns:
        df (DataFrame): A DataFrame containing the given [3+2] data.

        '''
        gs, ts, dist_gs = TT._clean_dfs(gs), TT._clean_dfs(ts), TT._clean_dfs(dist_gs)
        #General._quick_check(gs, dist_gs)
        dist_energies, sum_dist_energies = General._get_distortion_energies(gs, dist_gs)
        barriers = General._get_barrier(gs, ts)
        intera_energies = General._get_interaction(sum_dist_energies, barriers)
        df = General._combine_save(barriers, sum_dist_energies, intera_energies)

        return df

class MA:

    def __init__():
        pass

    def _clean_dfs(df):
        
        '''
        Function to clean Michael addition DataFrames.

        Arguments:
        df (DataFrame): A DataFrame containing the given Michael addition data.

        Returns:
        df (DataFrame): A DataFrame containing the given cleaned Michael addition data.
        
        '''
        if 'ts' in df['Structure'].loc[0] and 'reactant' not in df['Structure'].loc[0]:
            df['Structure'] = df['Structure'].str[:-2]
        elif 'gs' in df['Structure'].loc[0]:
            df['Structure'] = df['Structure'].str.replace('nuc', 'nuc-0xx')
            df['Structure'] = df['Structure'].str[:-2]
        else:
            df['Structure'] = df['Structure'].str.replace('-5_', 'r')
            df['Structure'] = df['Structure'].str.replace('-4_', 'r')
            df['Structure'] = df['Structure'].str.replace('-3_', 'r')
            df['Structure'] = df['Structure'].str.replace('-2_', 'r')
            df['Structure'] = df['Structure'].str.replace('-1_', 'r')# Problem is here!!!
            df['Structure'] = df['Structure'].str.replace('rr', '_r') 
        
        # Get reaction number
        num_lst = []
        for name in df['Structure']:
            num = name.split('-')[1]
            if 'reactant' in num:
                num = num.split('_')[0]
            else:
                pass
            num_lst.append(int(num))
        df['rxn_id'] = num_lst


        if 'gs' in df['Structure'].loc[0]:
            rct_id_lst = []
            for name in df['Structure']:
                if 'gs' in name:
                    rct_id_lst.append(1)
                else:
                    rct_id_lst.append(0)
            df['rct_id'] = rct_id_lst
            df['id'] = df['rxn_id'].astype(str) + '_' + df['rct_id'].astype(str)

        if 'ts' in df['Structure'].loc[0]:
            pass

        if 'reactant' in df['Structure'].loc[0]:
            rct_id_lst = []
            for name in df['Structure']:
                rct_num = name.split('_')[-1]
                if float(rct_num) == 2:
                    # rxn = name.split('-')[1]
                    # rxn = rxn.split('_')[0]
                    # rct_num_lst.append(str(rxn+'_0'))
                    rct_id_lst.append(0)
                else:
                    rct_id_lst.append(int(rct_num))
            df['rct_id'] = rct_id_lst
            df['id'] = df['rxn_id'].astype(str) + '_' + df['rct_id'].astype(str)

        return df

    def _ma_distortion_energies(gs, dist_gs):

        '''
        Function to get the Michael addition distorted energies from the gs and dist_gs dataframes.

        Arguments:
        gs (DataFrame): A DataFrame containing the ground state reactant energies extracted from Goodvibes.
        dist_gs (DataFrame): A DataFrame containing the distorted ground state reactant energies extracted from Goodvibes.

        Returns:
        df (DataFrame): A dataframe of the distortion energies with a reaction id.
        df_sum (DataFrame): A DataFrame of the summed distortion energies.

        '''

        # Get individual distortion energies
        d_energies = {}
        for id in dist_gs['id']:
            if str(id).endswith(str(0)):
                gs_e  = float(gs['E_SPC'][gs['id'] == '0_0'])
            else:
                gs_e = float(gs['E_SPC'][gs['id'] == id])
            dgs_e =  float(dist_gs['E_SPC'][dist_gs['id'] == id])

            dist_e = (dgs_e - gs_e)*627.5
            d_energies[id] = float(dist_e)

            
        df = pd.DataFrame(d_energies, index=[1]).transpose().reset_index().rename(columns={'index':'id'})
        df = df.rename(columns={1:'distortion_energies'})

        # Get sum of each reactions distortion energy
        sum_d_energies = {}
        
        for rxn_num in set(list(dist_gs['rxn_id'])):
            energy = sum(df['distortion_energies'][df['id'].str.startswith(str(rxn_num)+'_')])
            sum_d_energies[rxn_num] = energy

        sum_df = pd.DataFrame(sum_d_energies, index=[1]).transpose().reset_index().rename(columns={'index':'id'})
        sum_df = sum_df.rename(columns={1:'sum_distortion_energies'})

        d_contributions = {}
        for rxn_num in set(list(dist_gs['rxn_id'])):
            d_rxn = {}
            temp_df = df[df['id'].str.startswith(str(rxn_num)+'_')]
            for id, energy in zip(temp_df['id'], temp_df['distortion_energies']):
                d_rxn[id] = energy
            d_contributions[rxn_num] = d_rxn
            assert len(d_contributions[rxn_num].keys()) == 2, d_contributions
        
        contributions = pd.DataFrame([d_contributions]).transpose().reset_index().rename(columns={'index':'id', 0:'distortion_contributions'})
        sum_df = sum_df.merge(contributions, on='id')

        return df, sum_df

    def _ma_barriers(gs, ts):
        
        '''
        Function to get the Michael addition transition state reaction barriers.
        
        Arguments:
        gs (DataFrame): A DataFrame containing the ground state reactant energies extracted from Goodvibes.
        ts (DataFrame): A DataFrame containing the transition energies extracted from Goodvibes.
        
        Returns:
        df (DataFrame): A dataframe of the transition state barrier energies with a reaction id.
        '''
        q_d_barriers = {}
        d_barriers = {}
        for rxn in ts['rxn_id']:
            ts_e = ts['E_SPC'][ts['rxn_id'] == rxn]
            q_ts_e = ts['qh-G(T)_SPC'][ts['rxn_id'] == rxn]

            gs_e = gs['E_SPC'][gs['rxn_id'] == rxn]
            q_gs_e = gs['qh-G(T)_SPC'][gs['rxn_id'] == rxn]

            nuc_e = gs['E_SPC'][gs['rxn_id'] == 0]
            q_nuc_e = gs['qh-G(T)_SPC'][gs['rxn_id'] == 0]

            gs_sum = float(gs_e) + float(nuc_e)
            q_gs_sum = float(q_gs_e) + float(q_nuc_e)

            barrier = (float(ts_e) - gs_sum)*627.5
            d_barriers[rxn] = float(barrier)
            q_barrier = (float(q_ts_e) - q_gs_sum)*627.5
            q_d_barriers[rxn] = float(q_barrier)

        e_barrier_df = pd.DataFrame(d_barriers, index=[1]).transpose().reset_index().rename(columns={'index':'id'})
        e_barrier_df = e_barrier_df.rename(columns={1:'e_barrier'})
        q_barrier_df = pd.DataFrame(q_d_barriers, index=[1]).transpose().reset_index().rename(columns={'index':'id'})
        q_barrier_df = q_barrier_df.rename(columns={1:'q_barrier'})

        barrier_df = e_barrier_df.merge(q_barrier_df, on='id')
        
        return barrier_df

    def _runner(gs, ts, dist_gs):
        
        '''
        Function to run the energy extraction. 
        
        Arguments:
        gs (DataFrame): A DataFrame containing the ground state reactant energies extracted from Goodvibes.
        ts (DataFrame): A DataFrame containing the transition energies extracted from Goodvibes.
        dist_gs (DataFrame): A DataFrame containing the distorted ground state reactant energies extracted from Goodvibes.

        Returns:
        df (DataFrame): A DataFrame containing the given Michael addition data.

        '''
        gs, ts, dist_gs = MA._clean_dfs(gs), MA._clean_dfs(ts), MA._clean_dfs(dist_gs)
        dist_energies, sum_dist_energies = MA._ma_distortion_energies(gs, dist_gs)
        barriers = MA._ma_barriers(gs, ts)
        intera_energies = General._get_interaction(sum_dist_energies, barriers)
        df = General._combine_save(barriers, sum_dist_energies, intera_energies)

        return df

class MAL:

    def __init__():
        pass

    def _clean_dfs(df):
        
        '''
        Function to clean Michael addition DataFrames.

        Arguments:
        df (DataFrame): A DataFrame containing the given Michael addition data.

        Returns:
        df (DataFrame): A DataFrame containing the given cleaned Michael addition data.
        
        '''
        if 'ts' in df['Structure'].loc[0] and 'reactant' not in df['Structure'].loc[0]:
            df['Structure'] = df['Structure'].str[:-2]
        elif 'gs' in df['Structure'].loc[0]:
            df['Structure'] = df['Structure'].str.replace('nuc', 'nuc_0xx')
            df['Structure'] = df['Structure'].str[:-2]
        else:
            df['Structure'] = df['Structure'].str.replace('-5_', 'r')
            df['Structure'] = df['Structure'].str.replace('-4_', 'r')
            df['Structure'] = df['Structure'].str.replace('-3_', 'r')
            df['Structure'] = df['Structure'].str.replace('-2_', 'r')
            df['Structure'] = df['Structure'].str.replace('-1_', 'r')# Problem is here!!!
            df['Structure'] = df['Structure'].str.replace('rr', '_r') 
        
        # Get reaction number
        num_lst = []
        for name in df['Structure']:
            num = name.split('_')[1]
            if 'reactant' in num:
                num = num.split('_')[0]
            else:
                pass
            num_lst.append(int(num))
        df['rxn_id'] = num_lst


        if 'gs' in df['Structure'].loc[0]:
            rct_id_lst = []
            for name in df['Structure']:
                if 'gs' in name:
                    rct_id_lst.append(1)
                else:
                    rct_id_lst.append(0)
            df['rct_id'] = rct_id_lst
            df['id'] = df['rxn_id'].astype(str) + '_' + df['rct_id'].astype(str)

        if 'ts' in df['Structure'].loc[0]:
            pass

        if 'reactant' in df['Structure'].loc[0]:
            rct_id_lst = []
            for name in df['Structure']:
                rct_num = name.split('_')[-1]
                if float(rct_num) == 2:
                    # rxn = name.split('-')[1]
                    # rxn = rxn.split('_')[0]
                    # rct_num_lst.append(str(rxn+'_0'))
                    rct_id_lst.append(0)
                else:
                    rct_id_lst.append(int(rct_num))
            df['rct_id'] = rct_id_lst
            df['id'] = df['rxn_id'].astype(str) + '_' + df['rct_id'].astype(str)
        return df

    def _ma_distortion_energies(gs, dist_gs):

        '''
        Function to get the Michael addition distorted energies from the gs and dist_gs dataframes.

        Arguments:
        gs (DataFrame): A DataFrame containing the ground state reactant energies extracted from Goodvibes.
        dist_gs (DataFrame): A DataFrame containing the distorted ground state reactant energies extracted from Goodvibes.

        Returns:
        df (DataFrame): A dataframe of the distortion energies with a reaction id.
        df_sum (DataFrame): A DataFrame of the summed distortion energies.

        '''

        # Get individual distortion energies
        d_energies = {}
        for id in dist_gs['id']:
            if str(id).endswith(str(0)):
                gs_e  = float(gs['E_SPC'][gs['id'] == '0_0'])
            else:
                gs_e = float(gs['E_SPC'][gs['id'] == id])
            dgs_e =  float(dist_gs['E_SPC'][dist_gs['id'] == id])

            dist_e = (dgs_e - gs_e)*627.5
            d_energies[id] = float(dist_e)

            
        df = pd.DataFrame(d_energies, index=[1]).transpose().reset_index().rename(columns={'index':'id'})
        df = df.rename(columns={1:'distortion_energies'})

        # Get sum of each reactions distortion energy
        sum_d_energies = {}
        
        for rxn_num in set(list(dist_gs['rxn_id'])):
            energy = sum(df['distortion_energies'][df['id'].str.startswith(str(rxn_num)+'_')])
            sum_d_energies[rxn_num] = energy

        sum_df = pd.DataFrame(sum_d_energies, index=[1]).transpose().reset_index().rename(columns={'index':'id'})
        sum_df = sum_df.rename(columns={1:'sum_distortion_energies'})

        d_contributions = {}
        for rxn_num in set(list(dist_gs['rxn_id'])):
            d_rxn = {}
            temp_df = df[df['id'].str.startswith(str(rxn_num)+'_')]
            for id, energy in zip(temp_df['id'], temp_df['distortion_energies']):
                d_rxn[id] = energy
            d_contributions[rxn_num] = d_rxn
            assert len(d_contributions[rxn_num].keys()) == 2, d_contributions
        
        contributions = pd.DataFrame([d_contributions]).transpose().reset_index().rename(columns={'index':'id', 0:'distortion_contributions'})
        sum_df = sum_df.merge(contributions, on='id')

        return df, sum_df

    def _ma_barriers(gs, ts):
        
        '''
        Function to get the Michael addition transition state reaction barriers.
        
        Arguments:
        gs (DataFrame): A DataFrame containing the ground state reactant energies extracted from Goodvibes.
        ts (DataFrame): A DataFrame containing the transition energies extracted from Goodvibes.
        
        Returns:
        df (DataFrame): A dataframe of the transition state barrier energies with a reaction id.
        '''
        q_d_barriers = {}
        d_barriers = {}
        for rxn in ts['rxn_id']:
            ts_e = ts['E_SPC'][ts['rxn_id'] == rxn]
            q_ts_e = ts['qh-G(T)_SPC'][ts['rxn_id'] == rxn]

            gs_e = gs['E_SPC'][gs['rxn_id'] == rxn]
            q_gs_e = gs['qh-G(T)_SPC'][gs['rxn_id'] == rxn]

            nuc_e = gs['E_SPC'][gs['rxn_id'] == 0]
            q_nuc_e = gs['qh-G(T)_SPC'][gs['rxn_id'] == 0]

            gs_sum = float(gs_e) + float(nuc_e)
            q_gs_sum = float(q_gs_e) + float(q_nuc_e)

            barrier = (float(ts_e) - gs_sum)*627.5
            d_barriers[rxn] = float(barrier)
            q_barrier = (float(q_ts_e) - q_gs_sum)*627.5
            q_d_barriers[rxn] = float(q_barrier)

        e_barrier_df = pd.DataFrame(d_barriers, index=[1]).transpose().reset_index().rename(columns={'index':'id'})
        e_barrier_df = e_barrier_df.rename(columns={1:'e_barrier'})
        q_barrier_df = pd.DataFrame(q_d_barriers, index=[1]).transpose().reset_index().rename(columns={'index':'id'})
        q_barrier_df = q_barrier_df.rename(columns={1:'q_barrier'})

        barrier_df = e_barrier_df.merge(q_barrier_df, on='id')
        
        return barrier_df

    def _runner(gs, ts, dist_gs):
        
        '''
        Function to run the energy extraction. 
        
        Arguments:
        gs (DataFrame): A DataFrame containing the ground state reactant energies extracted from Goodvibes.
        ts (DataFrame): A DataFrame containing the transition energies extracted from Goodvibes.
        dist_gs (DataFrame): A DataFrame containing the distorted ground state reactant energies extracted from Goodvibes.

        Returns:
        df (DataFrame): A DataFrame containing the given Michael addition data.

        '''
        gs, ts, dist_gs = MAL._clean_dfs(gs), MAL._clean_dfs(ts), MAL._clean_dfs(dist_gs)
        dist_energies, sum_dist_energies = MAL._ma_distortion_energies(gs, dist_gs)
        print(dist_energies)
        barriers = MAL._ma_barriers(gs, ts)
        intera_energies = General._get_interaction(sum_dist_energies, barriers)
        df = General._combine_save(barriers, sum_dist_energies, intera_energies)

        return df

def main():

    setup = General._load_config()
    global sve_name, dataset
    sve_name = setup['filename']
    dataset = setup['dataset']
    if dataset == 'dft_3_2':
        b_info, dist_gs = TT._get_dft_dataframes(setup)
    else:
        gs, ts, dist_gs = General._get_dataframes(setup)
    # Main extraction
    if dataset == '3_2':
        print('[3+2]-Cycloadditions')
        print('------------------')
        df = TT._runner(gs, ts, dist_gs)
    elif dataset == 'dft_3_2':
        print('DFT [3+2]-Cycloadditions')
        print('------------------')
        df = TT._dft_runner(b_info, dist_gs)    
    elif dataset == 'da':
        print('Diels-Alder')
        print('------------------')
        df = DA._runner(gs, ts, dist_gs)
    elif dataset == 'ma':
        print('Michael Addition')
        print('------------------')
        df = MA._runner(gs, ts, dist_gs)
    elif dataset == 'mal':
        print('Michael Addition Malonate')
        print('------------------')
        df = MAL._runner(gs, ts, dist_gs)
    elif dataset == 'la_da':
        print('Catalytic Diels-Alder (LA)')
        print('------------------')
        df = LA_DA._runner(gs, ts, dist_gs)
    else:
        print('Not a valid input')

    # To display
    if setup['display'] == True:
        print(df)
    else:
        pass
        
if __name__ == '__main__':
    main()
