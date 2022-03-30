# Cardiac Abnormality Classification - Machine Learning for Health Care Project 1

## Table of Contents

- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Content Overview](#content-overview)
- [References](#references)

## Usage

### Environment Setup

1. Create a top-level directory named `data` and put the csv files in it. A detailed diagram is included in the [repository structure](#repository-structure).
1. Create a conda environment with `conda env create -f [ENV_NAME].yml`, using the desired environment file. The `environment.yml` is recommended unless support for hardware acceleration with Apple's [Metal API](https://developer.apple.com/metal/) is desired.
1. Activate the conda environment with `conda activate ml4h-base`

### Running the Code
For reproducing the results from the report, a script has been provided that trains and evaluates the models, and saves the results in the `results/` directory.

**Note that complete reproducibility is not guaranteed across platforms, as described in the [PyTorch documentation](https://pytorch.org/docs/stable/notes/randomness.html). Thus, you may observe slight deviations in the results compared to those of the report.**

Run this script from the root directory with `python code/main.py`. 

The script uses the following options from `config.py`:
- `USE_GPU` (default value `-1`): -1 to use all available GPUs, 0 to use no GPU.
- `DRY_RUN` (default value `False`): do not save results if set to True

## Repository structure

    .
    ├── code                                
    │   ├── models                      # Classification models
    │   │   ├── attention_model_checkpoints
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
    ├── .environment-m1.yml             # M1 Mac environment
    └── .environment.yml                # Intel environment

## Content Overview
The following is an overview of the contents of this repository.

- `models/`
    - `attention_model_checkpoints/` – Contains checkpoints for attention models. Used to avoid retraining.
    - `attention_model.py` – Module implementing an attention-based model. 
    - `autoencoder_tree.py` – Module implementing a model based on autoencoder featurization and extremely randomized tree boosting.
    - `baselines.py` – Contains functions to instantiate and train baseline models.
    - `ensemble.py` – Module implementing an ensemble of the other models.
    - `resnet1d.py` – Module implementing a residual neural network model.
    - `tree.py` – Module implementing an extremely randomized tree boosting model. Only used for comparison to the `autoencoder_tree` model.
    - `vanilla_cnn.py` – Module implementing a vanilla CNN model.
    - `vanilla_rnn.py` – Module implementing a vanilla RNN model.
- `config.py` – Contains configuration options (see [Running the Code](#running-the-code)).
- `datasets.py` – Contains utility functions for loading datasets.
- `evaluation.py` – Contains utility functions for evaluating models.
- `main.py` – Script for training and evaluating all models (see [Running the Code](#running-the-code))
- `utils.py` – Various utilities.
- `data/` – Directory for raw data.
- `results/` – Directory where plots and metrics are saved by the evaluation utils.

