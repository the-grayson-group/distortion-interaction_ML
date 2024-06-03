# three_two_match

This directory contains a collection of scripts to perform the [3+2] Cycloaddition matching.

### [add_masses_goodvibes.py](add_masses_goodvibes.py)

This code will take a Goodvibes.csv and generate the masses for each species in the file. This must be run in the directory containing the .out files.

### [match_gss.py](match_gss.py)

This python script will match the [3+2] ground state reactant structures to their equivalent distorted structures. Depending on usage, it will save a file ```rxn_rct_masses.pkl``` which is the matched masses.

This code exists as tools to check that the correct reactnat types have been assigned.
