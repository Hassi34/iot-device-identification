import argparse
from src.utils.common import read_yaml
from src.utils.sys_logging import get_logger
from src.utils import MLFlowManager
from src.utils.data_ops import load_np_arr_from_gz
from pathlib import Path
from src.utils.ml import is_blessed

import mlflow
import onnxruntime as rt
import numpy as np
import torch
import sys

STAGE = "Model Blessing"


def model_blessing():
    mlflow_service = MLFlowManager()
    latest_model_version = mlflow_service.latest_model_version(ONNX_MODEL_NAME)

    Path(STAGING_MODEL_DIR).absolute().mkdir(parents=True, exist_ok=True)
    Path(PRODUCTION_MODEL_DIR).absolute().mkdir(parents=True, exist_ok=True)

    mlflow.onnx.load_model(
        f"models:/{ONNX_MODEL_NAME}/staging", dst_path=STAGING_MODEL_DIR
    )
    onnx_sess_staging = rt.InferenceSession(
        ONNX_STAGED_MODEL_FILE_PATH,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    logger.info("ONNX runtime for staging has been initialized")
    try:
        mlflow.onnx.load_model(
            f"models:/{ONNX_MODEL_NAME}/production", dst_path=PRODUCTION_MODEL_DIR
        )
    except mlflow.exceptions.MlflowException:
        mlflow_service.transition_model_version_stage(
            model_name=ONNX_MODEL_NAME,
            model_version=latest_model_version,
            stage="Production",
        )
        logger.info(
            f"As there is no model available with production tag so the latest version # {latest_model_version} has been transitioned to Production"
        )
        mlflow.onnx.load_model(
            f"models:/{ONNX_MODEL_NAME}/production", dst_path=PRODUCTION_MODEL_DIR
        )

    onnx_sess_production = rt.InferenceSession(
        ONNX_PRODUCTION_MODEL_FILE_PATH,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    logger.info("ONNX runtime for production has been initialized")
    logger.info("Started preparing data for performance comparison")
    torch.set_float32_matmul_precision("medium")
    X_test = load_np_arr_from_gz(X_TEST_FILE_PATH)
    y_test = load_np_arr_from_gz(Y_TEST_FILE_PATH)
    logger.info(
        f"Numpy Arrays for test data have been loaded to RAM having the shapes : { X_test.shape, y_test.shape}"
    )

    if MODEL_BLESSING_THRESHOLD_COMPARISION:
        logger.info("Initializing model blessing comparision...")
        proceed_blessing = is_blessed(
            MODEL_BLESSING_THRESHOLD,
            staged_session=onnx_sess_production,
            production_session=onnx_sess_staging,
            X_test=X_test,
            y_test=y_test,
            logger=logger,
        )
        if not proceed_blessing:
            logger.critical(
                "Current model is not better than the production model, terminating the pipeline..."
            )
            sys.exit(1)
        logger.success(
            "Current model is better than the production model, proceeding with model blessing..."
        )
        logger.info("Performing pre-blessing model validations...")
        input_array = np.expand_dims(X_test[0], axis=0).astype(np.float32)
        input_name = onnx_sess_production.get_inputs()[0].name
        input_data = {input_name: input_array}
        prediction = onnx_sess_production.run(None, input_data)
        prediction = prediction[0][0]

        if isinstance(prediction, np.ndarray) and len(prediction) == 10:
            logger.info("Model has been validated successfully")

        versions = mlflow_service.client.search_model_versions(
            f"name='{ONNX_MODEL_NAME}'"
        )
        for version in versions:
            if version.current_stage == "Production":
                mlflow_service.transition_model_version_stage(
                    model_name=ONNX_MODEL_NAME,
                    model_version=version.version,
                    stage="Archived",
                )
                logger.info(
                    f"Model previous version # {version.version} has been transitioned from Production to Archive"
                )

        mlflow_service.transition_model_version_stage(
            model_name=ONNX_MODEL_NAME,
            model_version=latest_model_version,
            stage="Production",
        )
        logger.info(
            f"Model version {latest_model_version} has been transitioned from staging to Production"
        )

    else:
        logger.critical("Skipping model blessing as it is set to False...")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/system.yaml")
    args.add_argument("--params", "-p", default="configs/params.yaml")
    parsed_args = args.parse_args()
    config = read_yaml(parsed_args.config)
    params = read_yaml(parsed_args.params)
    LOGS_FILE_PATH = config["logs"]["RUNNING_LOGS_FILE_PATH"]
    X_TEST_FILE_PATH = config["data"]["PREPROCESSED_DATA_FILE_PATHS"][2]
    Y_TEST_FILE_PATH = config["data"]["PREPROCESSED_DATA_FILE_PATHS"][3]

    ONNX_MODEL_NAME = config["mlflow"]["ONNX_MODEL_NAME"]

    STAGING_MODEL_DIR = config["artifacts"]["STAGING_MODEL_DIR"]
    PRODUCTION_MODEL_DIR = config["artifacts"]["PRODUCTION_MODEL_DIR"]
    ONNX_STAGED_MODEL_FILE_PATH = config["artifacts"]["ONNX_STAGED_MODEL_FILE_PATH"]
    ONNX_PRODUCTION_MODEL_FILE_PATH = config["artifacts"][
        "ONNX_PRODUCTION_MODEL_FILE_PATH"
    ]
    MODEL_BLESSING_THRESHOLD = params["ml"]["MODEL_BLESSING_THRESHOLD"]
    MODEL_BLESSING_THRESHOLD_COMPARISION = params["ml"][
        "MODEL_BLESSING_THRESHOLD_COMPARISION"
    ]

    logger = get_logger(LOGS_FILE_PATH)
    with logger.catch():
        logger.info("\n********************")
        logger.info(f'>>>>> stage "{STAGE}" started <<<<<')
        model_blessing()
        logger.success(f'>>>>> stage "{STAGE}" completed!<<<<<\n')
