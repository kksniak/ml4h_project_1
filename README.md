# Cardiac Abnormalities Prediction - Machine Learning for Healthcare Project 1

## Table of Contents

- [Usage](#usage)
- [Description](#description)
- [References](#references)

## Usage

### Setup Recommended Steps:

All data should be in the data directory. (Maybe draw diagram??)

Create and activate an environment in conda by running the following commands:
```
$ conda env create -f env_name.yml
$ conda activate ml4h
```

### Reproducibility of results
Run main.py or sth like that

## Description
Description of files

- `models/attention_model.py` - Provides a class implementing an Attention-based model. Class initialisation takes one argument as an input: 'dataset' can be either set to "mithb" or "ptbdb" to be trained and tested on the corresponding dataset. Class method `train()` trains the models; however, due to the significant training time (around a few hours on CPU or half an hour on GPU), we provided the argument `load_model` that if sets to True loads the pre-trained models. Class method `transfer_learning_method_1()` trains the model using transfer learning with frozen layers. Class method `transfer_learning_method_1()` trains the model using transfer learning without frozen layers. Class method `predict()` creates variables with test predictions. 
- `models/attention_model_checkpoints`
- `models/ensemble.py` - provides a class implementing an Ensemble of our models. 

## References
