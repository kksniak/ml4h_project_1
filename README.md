# Cardiac Abnormalities Prediction - Machine Learning for Healthcare Project 1

## Table of Contents

- [Usage](#usage)
- [Description](#description)
- [References](#references)

## Usage

### Environment Setup

1. Create a top-level directory named `data` and put the csv files in it. A detailed diagram is included in the [repository structure](#repository-structure).
1. Create a conda environment with `conda env create -f [ENV_NAME].yml`, using the desired environment file. There MiniConda environment is recommended unless support for hardware acceleration with Apple's [Metal API](https://developer.apple.com/metal/) is desired.
1. Activate the conda environment with `conda activate ml4h`

### Running the Code
For reproducing the results from the report, a script has been provided that trains and evaluates the models, and saves the results in the `results/` directory.


## Repository structure

    .
    ├── code                                
    │   ├── mock                        # Static mock data
    │   │   └── sample_predictions.py
    │   ├── models                      # Classification models
    │   │   ├── attention_model.py
    │   │   ├── autoencoder_tree.py
    │   │   ├── baselines.py
    │   │   ├── ensemble.py
    │   │   ├── resnet1d.py
    │   │   ├── tree.py
    │   │   ├── vanilla_cnn.py
    │   │   └── vanilla_rnn.py
    │   ├── config.py                   # Config (e.g. GPU)
    │   ├── datasets.py                 # Utils for loading datasets
    │   ├── evaluation.py               # Utils for evaluating model performance
    │   ├── main.py                     # Script for training and evaluating all models
    │   └── utils.py
    ├── data                            # Put CSV data here
    ├── results                         # Plots and evaluation metrics
    ├── .gitignore
    ├── .pylintrc                       # Linting config
    ├── .style.yapf                     # Formatter config
    ├── .environment-m1.yml             # MiniForge environment
    └── .environment.yml                # MiniConda environment

## Description
Description of files

- `models/attention_model.py` - Provides a class implementing an Attention-based model. Class initialisation takes one argument as an input: `dataset` can be either set to "mithb" or "ptbdb" to be trained and tested on the corresponding dataset. Class method `train()` trains the models; however, due to the significant training time (around a few hours on CPU or half an hour on GPU), we provided the argument `load_model` that if sets to True loads the pre-trained models. Class method `transfer_learning_method_1()` trains the model using transfer learning with frozen layers. Class method `transfer_learning_method_2()` trains the model using transfer learning without frozen layers.
- `models/attention_model_checkpoints` - Folder with checkopoints for attention models. Used by Attention class if `train(load_model = True)`.
- `models/ensemble.py` - Provides a class implementing an Ensemble of our models. Class initialisation takes two arguments as input: `dataset` can be either set to "mithb" or "ptbdb", and `method` can be set to either "probs" or "feats" corresponding to methods Ensemble softmax or hidden layer described in the report. Class method `train()` trains the model, and it takes as an argument the dictionary of already trained models (restricted to categories attention, autoencoder, CNN, and ResNet models). If no model is passed within the category, the ensemble `train()` will train the missing model(s).

## References
