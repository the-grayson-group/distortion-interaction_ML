# feature_selection

This directory contains the code required for feature selection for the datasets. Varying different approaches are usable however, data is available for the method used in the paper to reduce space occupied in the repository.

## _f_selection

This directory contains all the pickle files for different feature selection methods.


## pre_ml.py

This script generates the ```pre_ml_mae.csv``` and ```pre_ml_rmse.csv``` which contain the pre-ML metrics for AM1-DFT barriers for the datasets used in this work.

## f_select.py

This script performs feature selection for a given set of features. It has flexibility in what method can be used and will ask the user for certain inputs. User input is not ideal but, there are a lot of different combinations thus, allowing the user control on which dataset to perform the feature selection for is best currently. Could likely be rewritten to take in a parsed file instead.

### Usage

This script will be in the same directory as the feature pickle files.
These can be located [here](../feature_extraction/features/). In this directory there are all the combined features and in the subdirectories, there are the features prior to being combined (individual features for GS, dist GS and TS).

```python f_select.py```

You will then be asked which dataset you which to perform feature extraction for.

After selection, you will then be asked to choose a feature selection method:

SFS
PCA
RFECV

Once selected, the code will run and information will be stored in the log file created.
