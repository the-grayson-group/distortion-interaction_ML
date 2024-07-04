# get_energies.py

This is the script used to extract the distortion, interaction and reaction barrier energies. It is designed to be specific to the four datasets we are interested in (Diels-Alder, [3+2] Cycloadditions, nitro-Michael addition, and dimethyl malonate Michael addition).

To run this code, a ```setup.yaml``` file is required as it provides suitable flexibility and control over the energy extraction process. Below is an example of the contents:

```
dataset: '3_2'
paths:
  ts: 'ts/Goodvibes_ts.csv'
  gs: 'gs/Goodvibes_gs.csv'
  dist_gs: 'dist_gs/Goodvibes_dist_gs.csv'
display: True
filename: 'three_two_barriers'
```

To run, navigate to the directory you want to perform the extraction and run ```python get_energies.py```.

## barriers

A directory containing all the barriers for all of the datasets tested.

## three_two_match

This directory contains code to check that the mapping and assignment of reactants was correct.