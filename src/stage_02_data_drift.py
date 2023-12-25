import argparse
from src.utils.common import read_yaml
from src.utils.sys_logging import get_logger
from pathlib import Path
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import json
import sys
import os

STAGE = "Data Drift"


def check_data_drift():
    if CHECK_DATA_DRIFT:
        current_data = pd.read_parquet(RAW_DATA_FILE_PATH)
        logger.info(f"The shape of the current data is {current_data.shape}")

        file_name = Path(RAW_DATA_FILE_PATH).resolve().name
        last_experiment_data_file_path = os.path.join(
            LAST_EXP_DATA_DIR, MLFLOW_ARTIFACT_DIR, file_name)
        reference_data = pd.read_parquet(last_experiment_data_file_path)
        logger.info(
            f"The shape of the reference data is {reference_data.shape}")

        data_drift_report = Report(metrics=[
            DataDriftPreset(),
        ])
        data_drift_report.run(current_data=current_data,
                              reference_data=reference_data,
                              column_mapping=None)
        Path(DATA_DRIFT_REPORT_PATH).parent.absolute().mkdir(
            parents=True, exist_ok=True)
        with open(DATA_DRIFT_REPORT_PATH, 'w') as f:
            json.dump(data_drift_report.as_dict(), f)
        logger.info(
            f"The data drift report has been saved to {DATA_DRIFT_REPORT_PATH}")

        with open(DATA_DRIFT_REPORT_PATH, 'r') as f:
            report = json.load(f)

        logger.info(
            f"Loaded the data drift report from {DATA_DRIFT_REPORT_PATH}")

        drifted_columns = {}
        for feature, values in report['metrics'][1]['result']['drift_by_columns'].items():
            if not values['drift_detected']:
                drifted_columns.update({feature: values})
        if len(drifted_columns) == 0:
            logger.warning("No data drift detected")
            sys.exit(1)
        logger.warning(
            f"The data drif is found in {len(drifted_columns)} columns found, proceeding to model training")
        with open(DRIFTED_COL_REPORT_PATH, 'w') as f:
            json.dump(drifted_columns, f)
        logger.info(
            f"Report with drifted columns has been saved to {DRIFTED_COL_REPORT_PATH}")

    else:
        logger.warning(
            "The CHECK_DATA_DRIFT parameter has been set to False, skipping data drift check...")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/system.yaml")
    args.add_argument("--params", "-p", default="configs/params.yaml")
    parsed_args = args.parse_args()
    config = read_yaml(parsed_args.config)
    params = read_yaml(parsed_args.params)
    LOGS_FILE_PATH = config["logs"]["RUNNING_LOGS_FILE_PATH"]
    RAW_DATA_FILE_PATH = config["data"]["RAW_DATA_FILE_PATH"][0]
    LAST_EXP_DATA_DIR = config["data"]["LAST_EXP_DATA_DIR"]
    CHECK_DATA_DRIFT = params["model_monitoring"]["CHECK_DATA_DRIFT"]
    DATA_DRIFT_P_VALUE_THRESHOLD = params["model_monitoring"]["DATA_DRIFT_P_VALUE_THRESHOLD"]
    MLFLOW_ARTIFACT_DIR = config['mlflow']["ARTIFACT_DIR"]
    TARGET_COLUMN_NAME = params["data_preprocessing"]["TARGET_COLUMN_NAME"]
    DATA_DRIFT_REPORT_PATH = config["model_monitoring"]["DATA_DRIFT_REPORT_PATH"]
    DRIFTED_COL_REPORT_PATH = config['model_monitoring']['DRIFTED_COL_REPORT_PATH']
    logger = get_logger(LOGS_FILE_PATH)
    with logger.catch():
        logger.info("\n********************")
        logger.info(f'>>>>> stage "{STAGE}" started <<<<<')
        check_data_drift()
        logger.success(f'>>>>> stage "{STAGE}" completed!<<<<<\n')
