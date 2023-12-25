import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as torch_optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from typing import Union
import numpy as np
import itertools
from sklearn.metrics import f1_score
from src.utils.common import softmax_func
import mlflow
import onnxruntime


class IoT_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


class IoTdeviceModel(nn.Module):
    def __init__(self, input_length, output_length):
        super().__init__()
        self.lin1 = nn.Linear(input_length, 500)
        self.lin2 = nn.Linear(500, 200)
        self.lin3 = nn.Linear(200, 50)
        self.lin4 = nn.Linear(50, output_length)
        self.bn1 = nn.BatchNorm1d(500)
        self.bn2 = nn.BatchNorm1d(200)
        self.bn3 = nn.BatchNorm1d(50)
        self.drops = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.bn1(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = F.relu(self.lin3(x))
        x = self.bn3(x)
        x = self.lin4(x)
        return x


class ModelTrainingAndEval:
    def __init__(self, model, train_dl, valid_dl, epochs, lr=0.001, wd=0.0):
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.epochs = epochs
        self.lr = lr
        self.wd = wd

    def _get_optimizer(self):
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optim = torch_optim.Adam(parameters, lr=self.lr, weight_decay=self.wd)
        return optim

    def train_model(self):
        self.optim = self._get_optimizer()
        self.epoch_count = []
        self.train_loss_value = []
        self.test_loss_value = []
        self.accuracy = []
        self.y_indices = []
        self.predicted_indices = []

        for epoch in range(self.epochs):
            self.model.train()
            total_training = 0
            sum_training_loss = 0
            for x, y in self.train_dl:
                batch = y.shape[0]
                output = self.model(x)
                training_loss = F.cross_entropy(output, y / 1.0)
                self.optim.zero_grad()
                training_loss.backward()
                self.optim.step()
                total_training += batch
                sum_training_loss += batch * (training_loss.item())
            # print("training loss: ", sum_training_loss/total_training)

            self.model.eval()
            total_valid = 0
            sum_valid_loss = 0
            correct = 0
            for x, y in self.valid_dl:
                current_batch_size = y.shape[0]
                with torch.inference_mode():
                    out = self.model(x)
                    valid_loss = F.cross_entropy(out, y / 1.0)
                    sum_valid_loss += current_batch_size * (valid_loss.item())
                    total_valid += current_batch_size
                    probabilities = F.softmax(out, dim=1)
                    predicted_indices = torch.max(probabilities, 1)[1]
                    y_indices = torch.max(y, 1)[1]
                    correct += (predicted_indices == y_indices).float().sum().item()
                    self.predicted_indices.extend(predicted_indices.tolist())
                    self.y_indices.extend(y_indices.tolist())
            self.epoch_count.append(epoch)
            self.test_loss_value.append(valid_loss.item())
            self.train_loss_value.append(training_loss.item())
            self.accuracy.append((correct / total_valid))
            if epoch % 10 == 0:
                print(
                    f"Epoch: {epoch} || Training Loss: {(sum_training_loss/total_training):0,.2f} || Validation loss : {(sum_valid_loss/total_valid):0,.2f} || Accuracy: {(correct/total_valid):0,.2f}"
                )

    def plot_loss(self, img_path: str):
        Path(img_path).parent.absolute().mkdir(parents=True, exist_ok=True)
        plt.plot(self.epoch_count, self.train_loss_value, label="Training Loss")
        plt.plot(self.epoch_count, self.test_loss_value, label="Test Loss")
        plt.legend(loc="best")
        plt.title("Train and Test Loss curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(img_path)
        plt.clf()

    def plot_accuracy(self, img_path: str):
        Path(img_path).parent.absolute().mkdir(parents=True, exist_ok=True)
        plt.plot(self.epoch_count, self.accuracy, label="Accuracy")
        plt.title("Accuracy plot")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig(img_path)
        plt.clf()

    def plot_confusion_matrix(
        self,
        img_path: str,
        column_label: Union[list[int], list[str]],
        index_label: Union[list[int], list[str]],
    ) -> None:
        confusion_matrix_df = pd.DataFrame(
            confusion_matrix(self.y_indices, self.predicted_indices, normalize="true")
        )
        confusion_matrix_df.columns = column_label
        confusion_matrix_df.index = index_label
        ax = sns.heatmap(confusion_matrix_df, annot=False)
        Path(img_path).parent.absolute().mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(img_path)
        plt.clf()

    def get_classification_report(self, target_names):
        return classification_report(
            self.y_indices, self.predicted_indices, target_names=target_names
        )

    def save_model(self, file_path: str):
        Path(file_path).parent.absolute().mkdir(parents=True, exist_ok=True)
        torch.save(self.model, file_path)


def plot_confusion_matrix(
    img_path: str,
    y: list[int],
    predicted_y: list[int],
    label_names: Union[list[int], list[str]] = None,
) -> None:
    confusion_matrix_df = pd.DataFrame(
        confusion_matrix(y, predicted_y, normalize="true")
    )
    if label_names is not None:
        confusion_matrix_df.columns = label_names
        confusion_matrix_df.index = label_names
    ax = sns.heatmap(confusion_matrix_df, annot=False)
    Path(img_path).parent.absolute().mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(img_path)


def get_f1_score_for_model(
    session: onnxruntime.capi.onnxruntime_inference_collection.InferenceSession,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    y_actual = []
    y_predicted_list = []
    for X, y in itertools.zip_longest(X_test, y_test):
        input_array = np.expand_dims(X, axis=0).astype(np.float32)
        input_name = session.get_inputs()[0].name
        input_data = {input_name: input_array}
        prediction = session.run(None, input_data)
        prediction = prediction[0]
        probabilities = softmax_func(prediction[0])
        y_predicted = np.argmax(probabilities)
        y_predicted_list.append(y_predicted)
        y_actual.append(y)
    return f1_score(y_actual, y_predicted_list, average="micro")


def is_blessed(
    blessing_threshold: Union[int, float],
    staged_session: onnxruntime.capi.onnxruntime_inference_collection.InferenceSession,
    production_session: onnxruntime.capi.onnxruntime_inference_collection.InferenceSession,
    X_test: np.ndarray,
    y_test: np.ndarray,
    logger,
):
    staged_model_f1_score = get_f1_score_for_model(staged_session, X_test, y_test)
    production_model_f1_score = get_f1_score_for_model(
        production_session, X_test, y_test
    )
    logger.info(
        f"F1 Score with trained model: {staged_model_f1_score}, F1 Score with production model: {production_model_f1_score}"
    )
    performance_difference = staged_model_f1_score - production_model_f1_score
    if performance_difference >= blessing_threshold:
        logger.info(
            f"The score improvement is {performance_difference}, and the threshold is {blessing_threshold} so returning the model blessing as True"
        )
        return True
    else:
        logger.info(
            f"The score improvement is {performance_difference}, and the threshold is {blessing_threshold} so returning the model blessing as False"
        )
        return False
