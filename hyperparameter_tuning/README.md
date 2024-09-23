# hyperparameter_tuning

This directory contains the script to perform the hyperparameter tuning. This code is designed to work across the four datasets utilised within this research. Minor adaptations may be required to ensure that this code works on alternative datasets. 
Each sub-directory contains the tuned hyperparameters for the given dataset (either [da](da), [ma](ma), [mal](mal), or [tt](tt)).

### [hyp_tuning.py](hyp_tuning.py)

This is the script to perform the hyperparameter tuning. This code require the user to parse the path to the dataset. The following models will be tuned:

- ``` Ridge Regression ```
- ``` Kernel Ridge (Polynomial Kernel) ```
- ``` Support Vector Regression (Radial Basis Function Kernel) ```
- ``` Neural Network with 2 Hidden Layers ```
- ``` Neural Network with 4 Hidden Layers ``` 
- ``` Random Forest ```

The usage of the code is as follows:

``` python hyp_tuning.py path/to/dataset.pkl ```

This will generate a ```hps.pkl``` file that contains the tuned hyperparameters for the models tested. The format is as follows:

```{'ridge_e_barrier_dft': {'b_hps': {'alpha': 0.8, 'tol': 0.1}, 'train_mae': 1.31841643257262}, ...} ```

In addition to testing these models, multiple targets are tested. This code is not optimised to running on GPU (due to the fact that the size of the datasets used in these projects is moderate and that there is a few sklearn models) however, the code could be adjusted to change usage. 