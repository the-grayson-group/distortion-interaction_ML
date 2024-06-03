# di_dp

This directory contains the code and the resulting files to generate the reactant types from the [3+2] cycloaddition dataset.

### [_check_distances.py](_check_distances.py)

This script checks the distances between atoms in a file. It requires a [common_atoms.pkl](data/common_atoms_combined.pkl) file to run.

### [_get_3_2_goodvibes.py](_get_3_2_goodvibes.py)

This script scrapes through the directories from the [Coley paper](https://doi.org/10.1038/s41597-023-01977-8) and extracts the thermochemical data for all reactions. It saves it in a [pickle file](data/3_2_energies.pkl). Specific script for the [3+2] data structure.

### [get_di_dp.py](get_di_dp.py)

This code generates a ```molecule_type.pkl``` file that contains, for each reaction passed, the type of each of the reactants in the reaction are stored in a dictionary with the format as follows:

``` {1:{'<filename>_reactant_1': 'di', '<filename>_reactant_1': 'dp'}, ....}```

This should be accurate however, checking that the files are correctly labelled is important. Thius can either be done manually or, checked via python.

### [tt_mol_types.pkl](tt_mol_types.pkl)

This file is molecule types output explained in the previous section.

