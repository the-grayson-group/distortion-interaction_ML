
import os
import pandas as pd
"""
#####################################################################################
#                               _get_3_2_goodvibes.py                               #
#                                                                                   #
#          Code to get the energies from the Coley paper - Instead this             #
#          extracts the energy from respective Goodvibes_X.csv files                #
#                                                                                   #
#####################################################################################

"""


mass_lookup = {'C':12,
               'H':1,
               'N':14,
               'O':16,
               'Cl':35,
               'F':19,
               'Br':79}

def _get_mass(df):

    df['Formula'] = df['Structure'].str.replace('_hess_g16', '.')
    df['Formula'] = df['Formula'].str.replace('_alt', '')
    

    structures = df['Formula'].to_list()

    #df['Structure'] = df['Structure'].str.replace('_hess_g16', '.')
    mass_d = {}
    for structure in structures:
        
        if not str(structure).startswith('TS') and not str(structure).startswith('p0'):
            formula = structure.split('_')[1]
            characters = [character for character in formula]

            
            d = {}
            mass_list = []
            for character in characters:

                if character == 'C':
                    if characters[characters.index(character)+1] == 'l':
                        character = 'l'
                elif character == 'B':
                    if characters[characters.index(character)+1] == 'r':
                        character = 'r'
                if character.isalpha() == True:
                    if characters[characters.index(character)+1].isdigit() == False:
                        d[character] = 1
                    elif characters[characters.index(character)+1].isdigit() == True:
                        if characters[characters.index(character)+2].isdigit() == True:
                            value = float(str(characters[characters.index(character)+1])+ str(characters[characters.index(character)+2]))
                            d[character] = value
                        else:
                            d[character] = characters[characters.index(character)+1] 

            if 'l' in d:
                d['Cl'] = d.pop('l')
            if 'r' in d:
                d['Br'] = d.pop('r')
            
            for atom, value in d.items():
                mass_list.append(float(mass_lookup[atom])*float(value))
            mass_d[structure.split('.')[0]] = sum(mass_list)
        else:
            continue
    return mass_d   

def _file_split(file):
    t1_file = file.replace('_hess_g16', '')
    t2_file = t1_file.replace('_alt', '')
    return t2_file

def _load_data(name, directory):
    df = pd.DataFrame()
    data = pd.DataFrame()
    
        
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
    del df['im'], df['freq'] # Delete unwanted im freq columns

    # Get filenames
    r0_files = list(df['Structure'][df['Structure'].str.startswith('r0_')])
    r1_files = list(df['Structure'][df['Structure'].str.startswith('r1_')])
    ts_file = list(df['Structure'][df['Structure'].str.startswith('TS_')])[0]
    
    # Check for alt files for r0 and r1
    if len(r0_files) > 1:
        r0_file = str([i for i in r0_files if 'alt' in i][0])
    else:
        r0_file = r0_files[0]
    if len(r1_files) > 1:
        r1_file = str([i for i in r1_files if 'alt' in i][0])
    else:
        r1_file = r1_files[0]

    # Extract the quasi-harmonic energies from the Goodvibes output and calculate the barrier
    r0_q_energy= float(df['qh-G(T)_SPC'][df['Structure'] == r0_file])
    r1_q_energy= float(df['qh-G(T)_SPC'][df['Structure'] == r1_file])
    ts_q_energy = float(df['qh-G(T)_SPC'][df['Structure'] == ts_file])
    dft_q_barrier = (ts_q_energy - (r0_q_energy + r1_q_energy))*627.5094740631

    # Extract energy values from the Goodvibes output.
    r0_e_energy= float(df['E_SPC'][df['Structure'] == r0_file])
    r1_e_energy= float(df['E_SPC'][df['Structure'] == r1_file])
    ts_e_energy = float(df['E_SPC'][df['Structure'] == ts_file])
    dft_e_barrier = (ts_e_energy - (r0_e_energy + r1_e_energy))*627.5094740631

    # Get the masses from the formula
    d_masses = _get_mass(df)

    # Final tidy of DataFrame and creat new DataFrame
    df['Formula'] = df['Formula'].str.replace('.','', regex=False)
    df['Formula'] = df['Formula'].str.split('_')
    df['Type'] = df['Formula'].str[0]
    df['Formula'] = df['Formula'].str[1]

    
    r0_file_clean = _file_split(r0_file)
    r1_file_clean = _file_split(r1_file)


    rxn = {'rxn_id':str(directory),
            'ts_file':ts_file, 'ts_q_energy':ts_q_energy, 'dft_q_barrier':dft_q_barrier, 
            'ts_e_energy':ts_e_energy, 'dft_e_barrier':dft_e_barrier,
            'r0_file':r0_file, 'r0_q_energy':r0_q_energy, 'r0_e_energy':r0_e_energy, 'r0_mass':d_masses[r0_file_clean], 
            'r1_file':r1_file, 'r1_q_energy':r1_q_energy, 'r1_e_energy':r1_e_energy, 'r1_mass':d_masses[r1_file_clean]
            }
    data = data.append(rxn, ignore_index=True)
    
    return(data)

def _save_data(data):

    data.to_pickle('3_2_energies.pkl')

def main():
    data = pd.DataFrame()
    for directory in next(os.walk('.'))[1]:
        path = directory+'/frequency_logs/'
        name = 'Goodvibes_'+str(directory)+'.csv'
        print(directory)
        os.chdir(path)
        d = _load_data(name, directory)
        data = pd.concat([data, d])
        os.chdir('../../')
    _save_data(data)
    print('Saved to pkl file')
    #print(data)

if __name__ == '__main__':
    main()

