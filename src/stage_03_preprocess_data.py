import argparse
from src.utils.common import read_yaml
from src.utils.sys_logging import get_logger
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.utils.common import write_dict_to_yaml
from src.utils.data_ops import gzip_np_arr
from sklearn.model_selection import train_test_split
from src.utils.data_ops import get_fitted_pipeline
from pathlib import Path

STAGE = "Preprocess Data"


def preprocess_data():
    complete_df = pd.read_parquet(RAW_DATA_FILE_PATH)
    logger.info(
        f'The raw data file has been loaded from "{RAW_DATA_FILE_PATH}" with the shape "{complete_df.shape}"'
    )
    duplicate_rows = complete_df.duplicated().sum()
    if duplicate_rows > 0:
        logger.warning(
            f"Found {duplicate_rows} duplicate rows, removing duplicate rows..."
        )
        complete_df = complete_df.drop_duplicates(keep="first")
    X = complete_df.drop([TARGET_COLUMN_NAME], axis=1)
    y = complete_df[TARGET_COLUMN_NAME]
    feature_cols = params["input_features_schema"]
    feature_cols = list(feature_cols.keys())
    logger.info(f"Read {len(feature_cols)} feature columns from params")
    data_processing_pipeline = get_fitted_pipeline(
    X, feature_cols, KNN_IMPUTER_NEIGHBORS=KNN_IMPUTER_NEIGHBORS
    )
    Path(DATA_PREPROCESSING_PIPELINE_FILE_PATH).parent.absolute().mkdir(parents=True, exist_ok=True)
    joblib.dump(data_processing_pipeline, DATA_PREPROCESSING_PIPELINE_FILE_PATH, compress=1)
    logger.info(f"Saved the preprocessing pipeline to {DATA_PREPROCESSING_PIPELINE_FILE_PATH}")
    data_processing_pipeline = joblib.load(DATA_PREPROCESSING_PIPELINE_FILE_PATH)
    data_processing_pipeline
    data_processing_pipeline = joblib.load(DATA_PREPROCESSING_PIPELINE_FILE_PATH)
    logger.info(
        f'Loaded sklearn data preprocessing pipeline from "{DATA_PREPROCESSING_PIPELINE_FILE_PATH}"'
    )
    X_transformed = data_processing_pipeline.transform(X)
    logger.info(f'Dataframe shape after transformation is "{X_transformed.shape}"')

    le = LabelEncoder()
    le.fit(y)
    labels_mapping_dict = {"labels_mapping": ""}
    le_dict = dict(zip(le.transform(le.classes_), le.classes_))
    le_dict = {int(k): v for k, v in le_dict.items()}

    labels_mapping_dict["labels_mapping"] = le_dict
    logger.info(f"Label encoding map has the dictionary: {le_dict}")
    write_dict_to_yaml(labels_mapping_dict, parsed_args.params)
    logger.info(f'Updated the label encoding map in the file at "{parsed_args.params}"')
    labels_dict = read_yaml(parsed_args.params)["labels_mapping"]
    reverse_dict = {v: k for k, v in labels_dict.items()}
    y = y.map(reverse_dict)
    logger.info("Successfully mapped the target column")
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=DATA_SPLIT_RANDOM_STATE,
    )
    logger.info(
        f"Data split for training completed with the shapes : {X_train.shape,  y_train.shape, X_test.shape, y_test.shape}"
    )
    Path(X_TRAIN_FILE_PATH).parent.absolute().mkdir(
    parents=True, exist_ok=True)
    gzip_np_arr(X_train, X_TRAIN_FILE_PATH)
    gzip_np_arr(y_train, Y_TRAIN_FILE_PATH)
    gzip_np_arr(X_test, X_TEST_FILE_PATH)
    gzip_np_arr(y_test, Y_TEST_FILE_PATH)
    logger.info(
        f"The processed data has been saved to the following paths: {X_TRAIN_FILE_PATH, Y_TRAIN_FILE_PATH, X_TEST_FILE_PATH, Y_TEST_FILE_PATH}"
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/system.yaml")
    args.add_argument("--params", "-p", default="configs/params.yaml")
    parsed_args = args.parse_args()
    # cloud_sync = CloudSync()
    config = read_yaml(parsed_args.config)
    params = read_yaml(parsed_args.params)
    LOGS_FILE_PATH = config["logs"]["RUNNING_LOGS_FILE_PATH"]
    RAW_DATA_FILE_PATH = config["data"]["RAW_DATA_FILE_PATH"][0]
    DATA_PREPROCESSING_PIPELINE_FILE_PATH = config["artifacts"][
        "DATA_PREPROCESSING_PIPELINE_FILE_PATH"
    ]
    TARGET_COLUMN_NAME = params["data_preprocessing"]["TARGET_COLUMN_NAME"]
    TEST_SIZE = params["data_preprocessing"]["TEST_SIZE"]
    DATA_SPLIT_RANDOM_STATE = params["data_preprocessing"]["DATA_SPLIT_RANDOM_STATE"]
    X_TRAIN_FILE_PATH = config["data"]["PREPROCESSED_DATA_FILE_PATHS"][0]
    Y_TRAIN_FILE_PATH = config["data"]["PREPROCESSED_DATA_FILE_PATHS"][1]
    X_TEST_FILE_PATH = config["data"]["PREPROCESSED_DATA_FILE_PATHS"][2]
    Y_TEST_FILE_PATH = config["data"]["PREPROCESSED_DATA_FILE_PATHS"][3]
    KNN_IMPUTER_NEIGHBORS = params['data_preprocessing']['KNN_IMPUTER_NEIGHBORS']
    logger = get_logger(LOGS_FILE_PATH)
    with logger.catch():
        logger.info("\n********************")
        logger.info(f'>>>>> stage "{STAGE}" started <<<<<')
        preprocess_data()
        logger.success(f'>>>>> stage "{STAGE}" completed!<<<<<\n')
