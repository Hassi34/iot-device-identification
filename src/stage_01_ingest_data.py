import argparse
from src.utils.common import read_yaml
from src.utils import MongoDBOps
from src.utils.sys_logging import get_logger
from src.utils import MLFlowManager
import mlflow
from pathlib import Path
import os

STAGE = "Ingest Data"


def ingest_data():
    logger.info("Pulling data from the source...")
    mongo_db = MongoDBOps(database_name=MONGO_DATABSE_NAME)
    complete_df = mongo_db.export_collection_as_dataframe(
        collection_name=MONGO_COLLECTION_NAME,
        rows_to_load=MONGO_NUMBER_OF_ROWS_TO_INGEST,
    )
    logger.info(
        f'The collection has been exported as a pandas dataframe with the shape "{complete_df.shape}"'
    )
    Path(RAW_DATA_FILE_PATH).parent.absolute().mkdir(parents=True, exist_ok=True)
    complete_df.to_parquet(RAW_DATA_FILE_PATH, compression="gzip")
    logger.info(f'Data has been saved locally at "{RAW_DATA_FILE_PATH}"')
    mlflow_service = MLFlowManager()
    mlflow.set_experiment(EXPERIMENT_NAME)
    runs = mlflow.search_runs(order_by=["attribute.start_time DESC"])
    if runs.empty:
        logger.warning("This is a new experiment, skipping the data drift check...")
    recent_run = runs[0:1]
    recent_run_id = recent_run.run_id[0]
    Path(LAST_EXP_DATA_DIR).absolute().mkdir(parents=True, exist_ok=True)
    file_name = Path(RAW_DATA_FILE_PATH).resolve().name
    mlflow_artifact_path = MLFLOW_ARTIFACT_DIR + "/" + file_name

    last_experiment_data_file_path = os.path.join(
        LAST_EXP_DATA_DIR, MLFLOW_ARTIFACT_DIR, file_name
    )

    try:
        mlflow_service.client.download_artifacts(
            recent_run_id, mlflow_artifact_path, LAST_EXP_DATA_DIR
        )
        logger.info(
            f"The last data version has been downloaded and saved to {last_experiment_data_file_path}"
        )
    except Exception as e:
        logger.error("Could not download the last data version")
        raise e


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/system.yaml")
    parsed_args = args.parse_args()
    config = read_yaml(parsed_args.config)
    LOGS_FILE_PATH = config["logs"]["RUNNING_LOGS_FILE_PATH"]
    RAW_DATA_FILE_PATH = config["data"]["RAW_DATA_FILE_PATH"][0]
    LAST_EXP_DATA_DIR = config["data"]["LAST_EXP_DATA_DIR"]
    MONGO_DATABSE_NAME = config["data"]["MONGO_DATABSE_NAME"]
    MONGO_COLLECTION_NAME = config["data"]["MONGO_COLLECTION_NAME"]
    MONGO_NUMBER_OF_ROWS_TO_INGEST = config["data"]["MONGO_NUMBER_OF_ROWS_TO_INGEST"]
    MLFLOW_ARTIFACT_DIR = config["mlflow"]["ARTIFACT_DIR"]
    EXPERIMENT_NAME = config["mlflow"]["EXPERIMENT_NAME"]
    logger = get_logger(LOGS_FILE_PATH)
    with logger.catch():
        logger.info("\n********************")
        logger.info(f'>>>>> stage "{STAGE}" started <<<<<')
        ingest_data()
        logger.success(f'>>>>> stage "{STAGE}" completed!<<<<<\n')
