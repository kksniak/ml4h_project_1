import imp
import numpy as np
from sklearn import ensemble
import torch

from models.baselines import (
    test_PTBDB_baseline,
    test_mitbih_baseline,
)
from models.vanilla_cnn import train_vanilla_cnn
from models.vanilla_rnn import train_vanilla_rnn
from models.resnet1d import train_resnet, perform_transfer_learning
from models.attention_model import Attention
from models.autoencoder_tree import AutoencoderTree
from models.ensemble import Ensemble
from datasets import load_arrhythmia_dataset, load_PTB_dataset
from utils import get_preds_from_numpy
from evaluation import evaluate

SEED = 2137
torch.manual_seed(SEED)
np.random.seed(SEED)


x_mit, y_mit, x_test_mit, y_test_mit = load_arrhythmia_dataset()
x_ptbdb, y_ptbdb, x_test_ptbdb, y_test_ptbdb = load_PTB_dataset()

### BASELINES

test_mitbih_baseline(x_test_mit, y_test_mit)
test_PTBDB_baseline(x_test_ptbdb, y_test_ptbdb)

### TASK 1
## Vanilla CNN
# MIT dataset
v_cnn_model_mit, v_cnn_trainer_mit = train_vanilla_cnn(
    channels=[1, 20, 20, 40],
    kernel_size=10,
    cnn_output_size=160,
    dataset="mithb",
    max_epochs=1,
)
preds_vcnn_mit = get_preds_from_numpy(v_cnn_model_mit, v_cnn_trainer_mit, x_test_mit)
evaluate("VanillaCNN-MIT", preds_vcnn_mit, y_test_mit, save_results=True)

# PTBDB dataset
v_cnn_model_ptbdb, v_cnn_trainer_ptbdb = train_vanilla_cnn(
    channels=[1, 20, 20, 40],
    kernel_size=10,
    cnn_output_size=160,
    dataset="ptbdb",
    max_epochs=1,
)
preds_vcnn_ptbdb = get_preds_from_numpy(
    v_cnn_model_ptbdb, v_cnn_trainer_ptbdb, x_test_ptbdb
)
evaluate("VanillaCNN-PTB", preds_vcnn_ptbdb, y_test_ptbdb, save_results=True)

## Vanilla RNN
# MIT dataset
v_rnn_model_mit, v_rnn_trainer_mit = train_vanilla_rnn(
    no_hidden=512, dataset="mitdb", num_layers=1
)
preds_vrnn_mit = get_preds_from_numpy(v_rnn_model_mit, v_rnn_trainer_mit, x_test_mit)
evaluate("VanillaRNN-MIT", preds_vrnn_mit, y_test_mit, save_results=True)

# PTBDB dataset
v_rnn_model_ptbdb, v_rnn_trainer_ptbdb = train_vanilla_rnn(
    no_hidden=512, dataset="ptbdb", num_layers=1
)
preds_vrnn_ptbdb = get_preds_from_numpy(
    v_rnn_model_ptbdb, v_rnn_trainer_ptbdb, x_test_ptbdb
)
evaluate("VanillaRNN-PTB", preds_vrnn_ptbdb, y_test_ptbdb, save_results=True)


### TASK 2
## Residual Neural Network
# MIT dataset
resnet_model_mit, resnet_trainer_mit = train_resnet(
    channels=[10, 20, 20, 40], dataset="mithb", max_epochs=1
)
preds_resnet_mit = get_preds_from_numpy(
    resnet_model_mit, resnet_trainer_mit, x_test_mit
)
evaluate("ResNet-MIT", preds_resnet_mit, y_test_mit, save_results=True)

# PTBDB dataset
resnet_model_ptbdb, resnet_trainer_ptbdb = train_resnet(
    channels=[10, 20, 20, 40], dataset="ptbdb", max_epochs=1
)
preds_resnet_ptbdb = get_preds_from_numpy(
    resnet_model_ptbdb, resnet_trainer_ptbdb, x_test_ptbdb
)
evaluate("ResNet-PTB", preds_resnet_ptbdb, y_test_ptbdb, save_results=True)

# Transer learning for ResNet
resnet_model_transfer, resnet_trainer_transfer = perform_transfer_learning(1)
preds_resnet_transfer = get_preds_from_numpy(
    resnet_model_transfer, resnet_trainer_transfer, x_test_ptbdb
)
evaluate("ResNetTransfer-PTB", preds_resnet_transfer, y_test_ptbdb, save_results=True)

# Attention Model
# MIT dataset

attention_model_mit = Attention(dataset="mithb")
attention_model_mit.train(load_model=True)
attention_model_mit.predict()
preds_attention_mit = attention_model_mit.y_pred
evaluate("Attention-MIT", preds_attention_mit, y_test_mit, save_results=True)

# PTBDB dataset

attention_model_ptbdb = Attention(dataset="ptbdb")
attention_model_ptbdb.train(load_model=True)
attention_model_ptbdb.predict()
preds_attention_ptbdb = attention_model_ptbdb.y_pred
evaluate("Attention-PTB", preds_attention_ptbdb, y_test_ptbdb, save_results=True)

# Transfer Learning for Attention (freezed layers)

attention_model_transfer_1 = Attention(dataset="ptbdb")
attention_model_transfer_1.transfer_learning_method_1()
attention_model_transfer_1.predict()
preds_attention_transfer_1 = attention_model_transfer_1.y_pred
evaluate(
    "AttentionFreezed-PTB", preds_attention_transfer_1, y_test_ptbdb, save_results=True
)

# Transfer Learning for Attention (training entire model)

attention_model_transfer_2 = Attention(dataset="ptbdb")
attention_model_transfer_2.transfer_learning_method_2()
attention_model_transfer_2.predict()
preds_attention_transfer_2 = attention_model_transfer_2.y_pred
evaluate(
    "AttentionUnfreezed-PTB",
    preds_attention_transfer_2,
    y_test_ptbdb,
    save_results=True,
)

## Autoencoder + ExtraTrees
# MIT dataset

autoencoder_model_mit = AutoencoderTree(dataset="mithb", train_ae_on="same")
autoencoder_model_mit.train()
autoencoder_model_mit.predict()
preds_autoencoder_mit = autoencoder_model_mit.y_pred_proba
evaluate("AutoencoderTree-MIT", preds_autoencoder_mit, y_test_mit, save_results=True)

# PTBDB dataset

autoencoder_model_ptbdb = AutoencoderTree(dataset="ptbdb", train_ae_on="same")
autoencoder_model_ptbdb.train()
autoencoder_model_ptbdb.predict()
preds_autoencoder_ptbdb = autoencoder_model_ptbdb.y_pred_proba
evaluate(
    "AutoencoderTree-PTB", preds_autoencoder_ptbdb, y_test_ptbdb, save_results=True
)

# Transfer Learning

autoencoder_model_transfer = AutoencoderTree(dataset="ptbdb", train_ae_on="full")
autoencoder_model_transfer.train()
autoencoder_model_transfer.predict()
preds_autoencoder_transfer = autoencoder_model_transfer.y_pred_proba
evaluate(
    "AutoencoderTreeTransfer-PTB",
    preds_autoencoder_transfer,
    y_test_ptbdb,
    save_results=True,
)


### Task 3

models_for_ensemble_mit = {
    "cnn": [v_cnn_model_mit, v_cnn_trainer_mit],
    "resnet": [resnet_model_mit, resnet_trainer_mit],
    "attention": attention_model_mit,
    "autoencoder": autoencoder_model_mit,
}

models_for_ensemble_ptbdb = {
    "cnn": [v_cnn_model_ptbdb, v_cnn_trainer_ptbdb],
    "resnet": [resnet_model_transfer, resnet_trainer_transfer],
    "attention": attention_model_ptbdb,
    "autoencoder": autoencoder_model_transfer,
}

## Method 1: Combined outputs from softmax layers

# MIT dataset
ensemble_model_method1_mit = Ensemble(dataset="mithb", method="probs")
ensemble_model_method1_mit.train(models=models_for_ensemble_mit)
ensemble_model_method1_mit.predict()
preds_ensemble_method1_mit = ensemble_model_method1_mit.y_pred_proba
evaluate(
    "EnsembleCombineSoftmax-MIT",
    preds_ensemble_method1_mit,
    y_test_mit,
    save_results=True,
)

# PTBDB dataset
ensemble_model_method1_ptbdb = Ensemble(dataset="ptbdb", method="probs")
ensemble_model_method1_ptbdb.train(models=models_for_ensemble_ptbdb)
ensemble_model_method1_ptbdb.predict()
preds_ensemble_method1_ptbdb = ensemble_model_method1_ptbdb.y_pred_proba
evaluate(
    "EnsembleCombineSoftmax-PTB",
    preds_ensemble_method1_ptbdb,
    y_test_ptbdb,
    save_results=True,
)


## Method 2: Combined features from hidden layers


# PTBDB dataset
ensemble_model_method2_ptbdb = Ensemble(dataset="ptbdb", method="feats")
ensemble_model_method2_ptbdb.train(models=models_for_ensemble_ptbdb)
ensemble_model_method2_ptbdb.predict()
preds_ensemble_method2_ptbdb = ensemble_model_method2_ptbdb.y_pred_proba
evaluate(
    "EnsembleCombineHidden-PTB",
    preds_ensemble_method2_ptbdb,
    y_test_ptbdb,
    save_results=True,
)

# MIT dataset
ensemble_model_method2_mit = Ensemble(dataset="mithb", method="feats")
ensemble_model_method2_mit.train(models=models_for_ensemble_mit)
ensemble_model_method2_mit.predict()
preds_ensemble_method2_mit = ensemble_model_method2_mit.y_pred_proba
evaluate(
    "EnsembleCombineHidden-MIT",
    preds_ensemble_method2_mit,
    y_test_mit,
    save_results=True,
)
