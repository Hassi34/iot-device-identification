import argparse
import time
from src.utils.common import read_yaml
from src.utils.sys_logging import get_logger
from src.utils import MLFlowManager
from src.utils.data_ops import load_np_arr_from_gz
from sklearn.metrics import classification_report
from src.lightning_pckg.model import NN
import torch.nn.functional as F
from src.utils.ml import plot_confusion_matrix
from src.lightning_pckg.training_callbacks import PrintingCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
import io
from pathlib import Path

# from pytorch_lightning.loggers import MLFlowLogger

from src.utils.ml import (
    IoT_Dataset,
    get_default_device,
    to_device,
    DeviceDataLoader,
)
from torch.utils.data import DataLoader
from datetime import datetime
import pytorch_lightning as pl
import mlflow
import onnxruntime as rt
import numpy as np
import torch
import sys

STAGE = "Training and Evaluation"


def train_eval():
    torch.set_float32_matmul_precision("medium")
    X_train = load_np_arr_from_gz(X_TRAIN_FILE_PATH)
    y_train = load_np_arr_from_gz(Y_TRAIN_FILE_PATH)
    X_test = load_np_arr_from_gz(X_TEST_FILE_PATH)
    y_test = load_np_arr_from_gz(Y_TEST_FILE_PATH)
    logger.info(
        f"Numpy Arrays have been loaded having the shapes : {X_train.shape,  y_train.shape, X_test.shape, y_test.shape}"
    )
    num_classes = len(np.unique(y_train))
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train).type(torch.LongTensor)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test).type(torch.LongTensor)

    logger.info("Convertd numpy arrays to the pytorch tensors")
    mlflow_service = MLFlowManager()
    experiment_id = mlflow_service.get_or_create_an_experiment(EXPERIMENT_NAME)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = config["mlflow"]["RUN_ID_PREFIX"] + "-" + timestamp

    mlflow.pytorch.autolog(
        silent=False, log_models=False, registered_model_name=PYTORCH_MODEL_NAME
    )
    train_ds = IoT_Dataset(X_train, y_train)
    test_ds = IoT_Dataset(X_test, y_test)
    logger.info("Train and Test Datasets created ")
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE_FOR_DATA_LOADER, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE_FOR_DATA_LOADER, shuffle=False)
    logger.info("Train and Test DataLoaders created ")
    device = get_default_device()
    logger.info(f'The default device is "{device}"')
    for x, y in train_dl:
        input_tensor_length = x.shape[1]
        sample_input = x[0].unsqueeze(0)
        break
    logger.info(
        f"Input and output tensor length is going to be {input_tensor_length} and {num_classes} respectively"
    )
    model = NN(
        input_size=input_tensor_length,
        learning_rate=LEARNING_RATE,
        num_classes=num_classes,
    )
    to_device(model, device)
    train_dl = DeviceDataLoader(train_dl, device)
    test_dl = DeviceDataLoader(test_dl, device)
    trainer = pl.Trainer(
        logger=TensorBoardLogger(TENSORBOARD_LOGS_DIR, name=EXPERIMENT_NAME),
        # logger=MLFlowLogger(experiment_name="my_experiment", save_dir="my_logs"),
        # profiler= 'simple', #find the bottleneck then commitout
        accelerator=ACCELERATOR,
        devices=DEVICES,
        min_epochs=MIN_NUMBER_OF_EPOCHS,
        max_epochs=MAX_NUMBER_OF_EPOCHS,
        resume_from_checkpoint=CKPT_FILE_PATH_FOR_TRAINING,
        precision=PRECISION,
        callbacks=[
            PrintingCallback(),
            EarlyStopping(monitor=METRICS_TO_TRACK, patience=EARLY_STOPPING_PATIENCE),
            ModelCheckpoint(
                dirpath=CKPT_DIR,
                save_top_k=TOP_K_CKPT_TO_BE_SAVED,
                mode=OPTIMIZATION_MODE,
                monitor=METRICS_TO_TRACK,
                filename=f"{time.strftime('%Y%m%d%H%M%S')}-" + "{epoch}-{val_loss:.2f}",
                verbose=True,
            ),
        ],
    )
    logger.info("Starting mlflow experiment...")
    valid_dl = test_dl  # This is just for the demo purpose, you should always specify the different data for test and validation
    mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
    trainer.fit(model, train_dl, valid_dl)
    trainer.validate(model, valid_dl)
    trainer.test(model, test_dl)
    # trainer.save_checkpoint
    y_actual = []
    y_predicted_list = []
    for x, y in valid_dl:
        to_device(model, device)
        out = model(x)
        probabilities = F.softmax(out, dim=1)
        y_predicted = torch.max(probabilities, 1)[1]
        y_predicted_list.extend(y_predicted.tolist())
        y_actual.extend(y.tolist())
    labels = list(params["labels_mapping"].values())
    if len(y_actual) > 0 and len(y_predicted_list) > 0:
        plot_confusion_matrix(
            CONFUSION_MATRIX_PLOT_FILE_PATH,
            y=y_actual,
            predicted_y=y_predicted_list,
            label_names=labels,
        )

    logger.info("Model has been trained successfully")
    logger.info(classification_report(y_actual, y_predicted_list, target_names=labels))
    logger.info("Converting model to onnx format")
    model.eval()
    Path(ONNX_TRAINED_MODEL_FILE_PATH).parent.absolute().mkdir(
        parents=True, exist_ok=True
    )
    model.to_onnx(
        file_path=ONNX_TRAINED_MODEL_FILE_PATH,
        input_sample=sample_input,
        input_names=["input"],
        verbose=True,
    )
    logger.info(f'ONNX model has been exported to "{ONNX_TRAINED_MODEL_FILE_PATH}"')
    logger.info("Starting model validation...")
    onnx_sess = rt.InferenceSession(
        ONNX_TRAINED_MODEL_FILE_PATH,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = onnx_sess.get_inputs()[0].name
    input_data = {input_name: sample_input.cpu().numpy()}
    output = onnx_sess.run(None, input_data)
    if isinstance(output[0], np.ndarray) and len(output[0][0]) == num_classes:
        logger.info("Model validation passed")
    else:
        logger.critical("Model validation failed!")
        sys.exit(1)

    logger.info("Exporting ONNX model for buffering...")
    buffer = io.BytesIO()
    torch.onnx.export(model.cpu(), sample_input.cpu(), f=buffer)
    buffer.seek(0)
    onnx_model = buffer.read()
    logger.info("Loaded bytes string from buffer which holds the ONNX model")
    logger.info("Started logging the ONNX model to MLFlow model repository...")
    mlflow.onnx.log_model(
        onnx_model=onnx_model,
        artifact_path=ONNX_LOGGED_MODEL_DIR,
        pip_requirements=mlflow.onnx.get_default_pip_requirements(),
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        code_paths=["src/stage_03_training_and_eval.py"],
        registered_model_name=ONNX_MODEL_NAME,
    )
    logger.info("ONNX has been saved in MLFlow models repo...")
    latest_model_version = mlflow_service.latest_model_version(
        model_name=ONNX_MODEL_NAME
    )

    versions = mlflow_service.client.search_model_versions(f"name='{ONNX_MODEL_NAME}'")
    for version in versions:
        if version.current_stage == "Staging":
            mlflow_service.transition_model_version_stage(
                model_name=ONNX_MODEL_NAME,
                model_version=version.version,
                stage="Archived",
            )
            logger.info(
                f"Model previous version # {version.version} has been transitioned from Staging to Archive"
            )

    mlflow_service.transition_model_version_stage(
        model_name=ONNX_MODEL_NAME, model_version=latest_model_version, stage="Staging"
    )
    logger.info(
        f"Model latest version # {latest_model_version} has been transitioned to MLFlow Staging"
    )
    mlflow.log_artifact(f"{CONFUSION_MATRIX_PLOT_FILE_PATH}")
    logger.info("Logged the confusion metrics artifact to MLflow artifacts repo")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/system.yaml")
    args.add_argument("--params", "-p", default="configs/params.yaml")
    parsed_args = args.parse_args()
    config = read_yaml(parsed_args.config)
    params = read_yaml(parsed_args.params)
    LOGS_FILE_PATH = config["logs"]["RUNNING_LOGS_FILE_PATH"]
    X_TRAIN_FILE_PATH = config["data"]["PREPROCESSED_DATA_FILE_PATHS"][0]
    Y_TRAIN_FILE_PATH = config["data"]["PREPROCESSED_DATA_FILE_PATHS"][1]
    X_TEST_FILE_PATH = config["data"]["PREPROCESSED_DATA_FILE_PATHS"][2]
    Y_TEST_FILE_PATH = config["data"]["PREPROCESSED_DATA_FILE_PATHS"][3]

    EXPERIMENT_NAME = config["mlflow"]["EXPERIMENT_NAME"]
    PYTORCH_MODEL_NAME = config["mlflow"]["PYTORCH_MODEL_NAME"]
    ONNX_MODEL_NAME = config["mlflow"]["ONNX_MODEL_NAME"]
    ONNX_LOGGED_MODEL_DIR = config["mlflow"]["ONNX_LOGGED_MODEL_DIR"]

    BATCH_SIZE_FOR_DATA_LOADER = params["data_preprocessing"][
        "BATCH_SIZE_FOR_DATA_LOADER"
    ]
    NUMBER_OF_EPOCHS = params["ml"]["MAX_NUMBER_OF_EPOCHS"]
    MODEL_LOSS_PLOT_FILE_PATH = config["artifacts"]["MODEL_LOSS_PLOT_FILE_PATH"]
    MODEL_ACCURACY_PLOT_FILE_PATH = config["artifacts"]["MODEL_ACCURACY_PLOT_FILE_PATH"]
    CONFUSION_MATRIX_PLOT_FILE_PATH = config["artifacts"][
        "CONFUSION_MATRIX_PLOT_FILE_PATH"
    ]

    TRAINED_MODEL_FILE_PATH = config["artifacts"]["TRAINED_MODEL_FILE_PATH"]
    ONNX_TRAINED_MODEL_FILE_PATH = config["artifacts"]["ONNX_TRAINED_MODEL_FILE_PATH"]
    # LIGHTNING_MODEL_CKPT_FILE_PATH = config["artifacts"]["LIGHTNING_MODEL_CKPT_FILE_PATH"]
    TENSORBOARD_LOGS_DIR = config["logs"]["TENSORBOARD_LOGS_DIR"]
    ACCELERATOR = params["ml"]["ACCELERATOR"]
    DEVICES = params["ml"]["DEVICES"]
    MAX_NUMBER_OF_EPOCHS = params["ml"]["MAX_NUMBER_OF_EPOCHS"]
    EARLY_STOPPING_PATIENCE = params["ml"]["EARLY_STOPPING_PATIENCE"]
    METRICS_TO_TRACK = params["ml"]["METRICS_TO_TRACK"]
    MIN_NUMBER_OF_EPOCHS = params["ml"]["MIN_NUMBER_OF_EPOCHS"]
    PRECISION = params["ml"]["PRECISION"]
    LEARNING_RATE = params["ml"]["LEARNING_RATE"]

    CKPT_DIR = config["ckpt"]["CKPT_DIR"]
    TOP_K_CKPT_TO_BE_SAVED = params["ckpt"]["TOP_K_CKPT_TO_BE_SAVED"]
    OPTIMIZATION_MODE = params["ckpt"]["OPTIMIZATION_MODE"]
    CKPT_FILE_PATH_FOR_TRAINING = config["ckpt"]["CKPT_FILE_PATH_FOR_TRAINING"]

    logger = get_logger(LOGS_FILE_PATH)
    with logger.catch():
        logger.info("\n********************")
        logger.info(f'>>>>> stage "{STAGE}" started <<<<<')
        train_eval()
        logger.success(f'>>>>> stage "{STAGE}" completed!<<<<<\n')
