# [3+2] Cycloaddition Energy Extraction Workflow

Thie data is readily available from the following [link](https://figshare.com/articles/dataset/dipolar_cycloaddition_dataset/21707888). There are two files of interest in this data archive. 

The first being [full_dataset.csv](full_dataset.csv), which can be found in this directory, contains the full dataset (SMILES, G<sub>act</sub>, and G<sub>r</sub>). 

The second file is a zip file called full_dataset_profiles.tar.gz. This contains each reaction in separate directories (e.g., 0, 1, 2, ..., 5269) inside are two further zip folders containing the Gaussian .logs. The following bash commands are useful in extracting these and making the directory ready for Goodvibes extractions with the [_get_3_2_goodvibes.py](../_get_3_2_goodvibes.py) script. 

***Unzip files***
```
for i in *;do cd $i; for file in *.tar.gz; do tar -xvzf $file; done; cd ../; done
```
***Rename files***

*Note: this step may require more filenames to be slightly altered manually as there is inconsistency in the naming conventions*
```
for i in *; do cd $i; cd single_point_logs; rename sp_g16 hess_g16_SPE *.log; cd ../../; done
```
```
for i in *; do cd $i; cd frequency_logs; rename optts_g16 hess_g16 *.log; cd ../../; done
```
***Rename SPE files***

*Note: Only required in the 3.0.1 version of Goodvibes - fixed in 3.0.2.*
```
for i in *; do cd $i; cd single_point_logs; rename .log .out *.log; cd ../../; done
```
***Move SPE files to frequency directory***

```
for i in *; do cd $i; cd single_point_logs; cp *.out ../frequency_logs; cd ../../; done
```

As mentioned, there is every possibility that some files may need manually altering to match the names. Once that is complete, you can then return to the main directory in which you can see all the reaction directories and run the [_get_3_2_goodvibes.py](../_get_3_2_goodvibes.py) script.


## Files in this directory.

### [common_atoms.pkl](common_atoms.pkl)

This file contains the common atoms for each reactant in the dataset (in the case of the [3+2] this is the reacting atoms and the atom on the backbone of the dipole). The reacting atoms list is formatted so that index 0 and 1 react with eachother. Same is for index 2 and 3. It also contains the reacting atoms and file name. The format is as follows:

``` 'ts_206.out': {'dp': [2, 3], 'di': [5, 12, 13], 'reacting': [2, 13, 3, 5], 'name': 'ts_206'} ```

### [mapping.pkl](mapping.pkl)

This file contains the mapping for the [3+2] cycloaddition dataset from TS to the reactants. The file has the following format:

```'ts_6_reopt.out': {'reactant_1': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 12: 8, 13: 9, 14: 10, 15: 11}, 'reactant_2': {8: 1, 9: 2, 10: 3, 11: 4, 16: 5, 17: 6, 18: 7, 19: 8}}```

### [3_2_energies.pkl](3_2_energies.pkl)

This file contains all the energies for the [3+2] cycloaddition dataset. This is extracted using the [_get_3_2_goodvibes.py](../_get_3_2_goodvibes.py) script.