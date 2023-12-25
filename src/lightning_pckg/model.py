from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric


class NN(pl.LightningModule):
    def __init__(self, input_size, learning_rate, num_classes):
        super().__init__()

        self.FC1 = nn.Linear(input_size, 500)
        self.FC2 = nn.Linear(500, 200)
        self.FC3 = nn.Linear(200, 50)
        self.FC4 = nn.Linear(50, num_classes)
        self.bn1 = nn.BatchNorm1d(500)
        self.bn2 = nn.BatchNorm1d(200)
        self.bn3 = nn.BatchNorm1d(50)
        self.drops = nn.Dropout(0.3)

        self.lr = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = F.relu(self.FC1(x))
        x = self.bn1(x)
        x = F.relu(self.FC2(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = F.relu(self.FC3(x))
        x = self.bn3(x)
        x = self.FC4(x)
        return x

    def training_step(self, batch):
        loss, scores, y = self._common_step(batch)
        self.log_dict(
            {
                "train_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "scores": scores, "y": y}

    def training_epoch_end(self, outputs):
        scores = torch.cat([x["scores"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])
        self.log_dict(
            {
                "train_acc": self.accuracy(scores, y),
                "train_f1": self.f1_score(scores, y),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        self.log("test_loss", loss)
        return loss

    def _common_step(self, batch):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def predict_step(self, batch):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
