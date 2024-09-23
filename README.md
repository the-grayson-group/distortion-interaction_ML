# [Distortion/Interaction Analysis via Machine Learning](https://doi.org/10.26434/chemrxiv-2024-rpk9q)

![toc](toc.png "TOC")

This repository contains the code associated with the paper titled "Distortion/Interaction Analysis via Machine Learning". In each sub directory there is a ```README.md``` file which explains more regarding the code in that specific directory. 

### [Requirements](requirements.txt)

The code in the project works with the following packages. If using slightly different versions, adaptations may be required in the code.

```
molml==0.9.0
cclib==1.8
keras==3.3.3
morfeus==0.7.2
scikit-learn==1.3.1
numpy==1.26.0
pandas==2.1.1
xyz_py==5.9.3
tensorflow==2.16.1
matplotlib==3.8.4
keras_tuner==1.4.5
ipykernel==6.29.5
yaml==6.0.1
inquirer==3.2.4
```

## Directories

Below is a brief description of each directory in the order that it would likely be used in a workflow.

### [diassep](diassep)

Code to perform the Distortion/Interaction separation. For new datasets, there may be issues with the code specific to the new dataset - raise an issue if so.

### [energy_extraction](energy_extraction)

Code to extract energies for each dataset.

### [feature_extraction](feature_extraction)

Code to extract features for each dataset - for new datasets, adaptations may be required.

### [feature_selection](feature_selection)

Code to perform feature selection for a given dataset after all possible features have been extracted.

### [hyperparameter_tuning](hyperparameter_tuning)

Code to perform hyperparameter tuning for sklearn and TensorFlow models on a given dataset. Alterations to the code may be required depending upon versions of python packages used.

### [machine_learning](machine_learning)

Code to run the machine learning models after hyperparameter tuning. This code requires hyperparameter tuning to be completed and can be ran in batch across multiple different random states. Alterations to the code may be required depending upon versions of python packages used.
