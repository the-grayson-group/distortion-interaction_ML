# Feature Extraction 

The feature extraction for the Distortion-Interaction project can be performed using the `f_extract.py` script in this directory. It utilises a `config.yaml` file to pass key information required for performing the extraction. An example can be found in this directory.

The directory [_tt](_tt) contains extra files required for running the feature extraction for the [3+2] dataset.

## [features](features)

This directory contains all the features for each of the datasets. These are the raw features prior to feature selection. Each sub directory for a given dataset contains the individual .pkl files for reactant (gs), distorted reactant (dist_gs), and transition structure (ts). The files that end with ```<dataset>_features.pkl``` are the final features before feature selection.

## Usage

The `config.yaml` setup will change depending upon the dataset.

### [3+2] Cycloaddition

```
# [3+2]
dataset : tt
type : ts
path : X:\Chemistry\ResearchProjects\MNGrayson\EG-CH1463\year_3\datasets\three_two\gs_ts_three_two\am1\solvent\ts
spe : False
ca_from_file : True
ca_file : common_atoms.pkl
mapping_file : mapping.pkl
mol_types : tt_mol_types.pkl
verbose : 0
```

*dataset* - Either **tt**, **da**, **ma** or **mal** for the three datasets working with.

*type* - Either **ts**, **gs** or **dist_gs** for the different species.

*path* - Path to the .out files.

*spe* - Whether or not the directory contains SPE files **True** or **False**.

*ca_from_file* - Whether or not to take the common atoms from a given file **True** or **False**.

*ca_file* - If *ca_from_file* is **True**, then this is the path to the common atoms pickle file. ***For [3+2] dataset***

*mapping_file* - If *ca_from_file* is **True**, then this is the path to the mappings for common atoms pickle file. ***For [3+2] dataset***

*mol_types* - If *ca_from_file* is **True**, then this is the path to the molecule types pickle file. ***For [3+2] dataset***

*verbose* - Either **0** to not print to terminal or **1** which does print the logger to the terminal.

---

### Diels-Alder

```
# Diels-Alder
dataset : da
type : ts
path: X:\Chemistry\ResearchProjects\MNGrayson\EG-CH1463\year_3\datasets\diels_alder\gs_ts_da\am1\all\exo\ts
spe : False
common_atoms :
  gs : {dp : [2,4], di : [1,5,6,3]}
  dist_gs : {dp : [2,4], di : [1,5,6,3]}
  ts : [1,2,3,4,5,6]
ca_from_file : False
verbose : 0
```

*dataset* - Either **tt**, **da**, **ma** or **mal** for the three datasets working with.

*type* - Either **ts**, **gs** or **dist_gs** for the different species.

*path* - Path to the .out files.

*spe* - Whether or not the directory contains SPE files **True** or **False**.

*common_atoms* - A dictionary structure giving the common atoms for **gs**, **dist_gs** and **ts** structures.

*ca_from_file* - Whether or not to take the common atoms from a given file **True** or **False**.

*verbose* - Either **0** to not print to terminal or **1** which does print the logger to the terminal.

---

### Michael Addition

```
# Michael Additions
dataset : ma
type : dist_gs
path: X:\Chemistry\ResearchProjects\MNGrayson\EG-CH1463\year_3\datasets\michael_addition\splits_ma\am1
spe : True
common_atoms :
  gs : {ma : [1,2,3,4], nu : [1]}
  dist_gs : {ma : [1,2,3,4], nu : [1]}
  ts : [1,2,3,4,5,6,7,8,9,10]
ca_from_file : False
verbose : 0
```

*dataset* - Either **tt**, **da**, **ma** or **mal** for the three datasets working with.

*type* - Either **ts**, **gs** or **dist_gs** for the different species.

*path* - Path to the .out files.

*spe* - Whether or not the directory contains SPE files **True** or **False**.

*common_atoms* - A dictionary structure giving the common atoms for **gs**, **dist_gs** and **ts** structures.

*ca_from_file* - Whether or not to take the common atoms from a given file **True** or **False**.

*verbose* - Either **0** to not print to terminal or **1** which does print the logger to the terminal.

---

### Michael Addition - Malonate

```
# Malonate
dataset : mal
type : ts
path: X:\Chemistry\ResearchProjects\MNGrayson\EG-CH1463\year_3\datasets\michael_addition_mal\gs_ts_ma_mal\am1\ts
spe : True
common_atoms :
  gs : {ma : [1,2,3,4,5], nu : [1,2,3,4,5,6,7,8,9,10]}
  dist_gs : {ma : [1,2,3,4,5], nu : [1,2,3,4,5,6,7,8,9,10]}
  ts : [1,2,3,4,5,6,7,8,9,10]
ca_from_file : False
verbose : 0
```

*dataset* - Either **tt**, **da**, **ma** or **mal** for the three datasets working with.

*type* - Either **ts**, **gs** or **dist_gs** for the different species.

*path* - Path to the .out files.

*spe* - Whether or not the directory contains SPE files **True** or **False**.

*common_atoms* - A dictionary structure giving the common atoms for **gs**, **dist_gs** and **ts** structures.

*ca_from_file* - Whether or not to take the common atoms from a given file **True** or **False**.

*verbose* - Either **0** to not print to terminal or **1** which does print the logger to the terminal.

---

The running of this script will create a `logger` file that will provide timings and other information about errors in the dataset.

All features for each dataset can be found in the [features](features) directory. There is combined feature sets as well as the individual feature sets.

This script creates a large amount of features in a good time frame that will need ***feature selection***.

---

### Collating Features

The code has functionality to take the dataframes created from each run and combine them with the following line and subsequent ```config.yaml``` file:

```python f_extract.py -c```

```
# Diels-Alder
gs_path : H:\dist_inter\code\feature_extraction\features\da\da_exo_gs.pkl
ts_path : H:\dist_inter\code\feature_extraction\features\da\da_exo_ts.pkl
dist_gs_path : H:\dist_inter\code\feature_extraction\features\da\da_exo_dist_gs.pkl
am1_barriers_path : H:\dist_inter\code\energy_extraction\barriers\da\new_am1_da_exo_barriers.pkl
dft_barriers_path : H:\dist_inter\code\energy_extraction\barriers\da\new_dft_da_exo_barriers.pkl
filename : da_exo_features.pkl
```