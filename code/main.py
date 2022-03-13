from cgi import test
from gc import callbacks
import torch
from utils import CNN_output_shape, get_predictions
from models import vanillaCNN
from baselines import (
    train_mitbih_baseline,
    test_mitbih_baseline,
    train_PTBDB_baseline,
    test_PTBDB_baseline,
)
from datasets import load_arythmia_dataset, load_PTB_dataset
from utils import CNN_output_shape
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, accuracy_score


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
cnn_channels = [1, 20, 20, 40]
kernel_size = 10

cnn_out_shape = 187


for i in range(1, len(cnn_channels)):
    cnn_out_shape = CNN_output_shape(cnn_out_shape, 1, kernel_size)

# ##Arythmia DATASET
# x, y, x_test, y_test = load_arythmia_dataset()


# dataset = TensorDataset(
#     torch.tensor(x, dtype=torch.float).squeeze(),
#     torch.tensor(y, dtype=torch.long).squeeze(),
# )
# train_dataset, val_dataset = random_split(
#     dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))]
# )
# test_dataset = TensorDataset(
#     torch.tensor(x_test, dtype=torch.float).squeeze(),
#     torch.tensor(y_test, dtype=torch.long).squeeze(),
# )

# train_loader = DataLoader(train_dataset, batch_size=32)
# val_loader = DataLoader(val_dataset, batch_size=64)
# test_loader = DataLoader(test_dataset, batch_size=64)
# vCNN = vanillaCNN(cnn_channels, kernel_size, cnn_channels[-1] * cnn_out_shape, 5)
# trainer = pl.Trainer(
#     gpus=1, max_epochs=15, callbacks=[EarlyStopping(monitor="val_loss", mode="min")]
# )
# trainer.fit(model=vCNN, train_dataloaders=train_loader, val_dataloaders=val_loader)

# test_preds = get_predictions(vCNN, test_loader, trainer)
# # print("preds: ", test_preds.shape, type(test_preds))
# # print(test_preds)
# # print(np.argmax(test_preds, axis=-1).shape)
# print("Vanilla CNN acc: ", accuracy_score(y_test, np.argmax(test_preds, axis=-1)))


# # train_mitbih_baseline(x, y)
# baseline_mitbih_preds = test_mitbih_baseline(x_test, y_test)


# ##PTBDB DATASET


x, y, x_test, y_test = load_PTB_dataset()
dataset = TensorDataset(
    torch.tensor(x, dtype=torch.float).squeeze(),
    torch.tensor(y, dtype=torch.float).squeeze(),
)
train_dataset, val_dataset = random_split(
    dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))]
)
test_dataset = TensorDataset(
    torch.tensor(x_test, dtype=torch.float).squeeze(),
    torch.tensor(y_test, dtype=torch.float).squeeze(),
)

train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

vCNN_PTBDB = vanillaCNN(cnn_channels, kernel_size, cnn_channels[-1] * cnn_out_shape, 2)
trainer = pl.Trainer(
    gpus=1, max_epochs=15, callbacks=[EarlyStopping(monitor="val_loss", mode="min")]
)
trainer.fit(
    model=vCNN_PTBDB, train_dataloaders=train_loader, val_dataloaders=val_loader
)

test_preds = get_predictions(vCNN_PTBDB, test_loader, trainer)
print("Vanilla CNN acc: ", accuracy_score(y_test, np.round(test_preds)))
print("ROC AUC vanilla CNN: ", roc_auc_score(y_test, test_preds))

# # train_PTBDB_baseline(x, y)
# baseline_PTBDB_preds = test_PTBDB_baseline(x_test, y_test)
# print("ROC AUC: ", roc_auc_score(y_test, baseline_PTBDB_preds))
# fpr, tpr, thresholds = roc_curve(y_test, baseline_PTBDB_preds, pos_label=1)
# plt.figure()
# lw = 2
# plt.plot(
#     fpr, tpr, color="darkorange", lw=lw, label="ROC curve",
# )
# plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Receiver operating characteristic example")
# plt.legend(loc="lower right")
# plt.show()
# display = PrecisionRecallDisplay.from_predictions(
#     y_test, baseline_PTBDB_preds, name="Baseline"
# )
# _ = display.ax_.set_title("2-class Precision-Recall curve")
# plt.show()
