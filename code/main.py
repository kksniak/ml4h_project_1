import torch
from utils import CNN_output_shape
from models import vanillaCNN
from datasets import load_arythmia_dataset, load_PTB_dataset
from utils import CNN_output_shape
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pytorch_lightning as pl

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
cnn_channels = [1, 20, 20, 40]
kernel_size = 10

cnn_out_shape = 187

for i in range(1, len(cnn_channels)):
    cnn_out_shape = CNN_output_shape(cnn_out_shape, 1, kernel_size)


x, y, x_test, y_test = load_arythmia_dataset()


dataset = TensorDataset(
    torch.tensor(x, dtype=torch.float).squeeze(),
    torch.tensor(y, dtype=torch.long).squeeze(),
)
test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))

train_loader = DataLoader(dataset, batch_size=32)
vCNN = vanillaCNN(cnn_channels, kernel_size, cnn_channels[-1] * cnn_out_shape, 5)
trainer = pl.Trainer(gpus=1,max_epochs=2)
trainer.fit(model=vCNN, train_dataloader=train_loader)
