'''
Code to take a Goodvibes output and get the masses attached. 

Saves file as a pickle file
'''

import cclib
import pandas as pd

mass_lookup = {6:12,
               1:1,
               7:14,
               8:16,
               17:35,
               9:19,
               35:79}

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

def _extract_masses(df):
     
    data_dict = {}
    for structure in df['Structure']:
        name = structure+'.out'
        ccobj = cclib.io.ccread(name)
        mass =[]
        for atom in list(ccobj.atomnos):
            print(structure)
            mass.append(mass_lookup[atom])
        data_dict[structure] = float(sum(mass))
    return data_dict

df = _load_data('Goodvibes_dft_solvent_3_2_dist_gs.csv')
data_dict = _extract_masses(df)

mass_df = pd.DataFrame.from_dict(data_dict, orient='index').reset_index().rename(columns={'index':'Structure', 0:'mass'})
df = df.merge(mass_df, on='Structure')
df.to_pickle('Goodvibes_dft_solvent_3_2_dist_gs_w_masses.pkl')
