# common_atoms

This directory contains the code to perform the common atoms analysis of the [3+2] cycloaddition dataset.

### [common_atoms.py](common_atoms.py)

This script generates the common atoms for the [3+2] cycloaddition dataset. This dataset does not have a consistent atom numbering across the entire dataset therefore, extracting the common atoms for each individual reaction is crucial to performing atom specific machine learning techniques. Saves a ```common_atoms.pkl``` file in the working directory. Also, will save a ```check_systems.txt``` file if there are structures which parse errors. 

### [common_atoms.pkl](common_atoms.pkl)

This file contains the common atoms for each reactant in the dataset (in the case of the [3+2] this is the reacting atoms and the atom on the backbone of the dipole). The reacting atoms list is formatted so that index 0 and 1 react with eachother. Same is for index 2 and 3. It also contains the reacting atoms and file name. The format is as follows:

``` 'ts_206.out': {'dp': [2, 3], 'di': [5, 12, 13], 'reacting': [2, 13, 3, 5], 'name': 'ts_206'} ```