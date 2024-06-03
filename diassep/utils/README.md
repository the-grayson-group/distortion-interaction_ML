# utils directory

This directory contains various scripts that have been used throughout the project to either check structures or to extract data from an external dataset.

## common_atoms

This directory contains the scripts/files associated with extracting the common atoms from the [3+2] cycloaddition dataset. For more information see the help associated with the [common_atoms.py](common_atoms/common_atoms.py).

## data

This directory contains data files created as a part of the workflow as well as a temporary ipynb to view the data. These include:

- [full_dataset.csv](data/full_dataset.csv) - The full dataset from the Coley paper.

- [3_2_energies.pkl](data/3_2_energies.pkl) - The extracted [3+2] energies using Goodvibes.

- [common_atoms_combined.pkl](data/common_atoms_combined.pkl) - The common atoms extracted for the [3+2] cycloaddition reactions.

- [read.ipynb](data/read.ipynb) - A temporary ipynb file to read data.

## di_dp

This directory contains the code and information required for generating the reaction types for reactants in the [3+2] cycloaddition dataset.