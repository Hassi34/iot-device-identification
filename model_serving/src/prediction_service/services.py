from fastapi import HTTPException, status
import sklearn.pipeline
import pandas as pd
from typing import Iterable
import onnxruntime as rt
import numpy as np
from src.utils.common import softmax_func


class Services:
    def __init__(self, request_data: dict, params=dict):
        self.request_data: dict = request_data
        self.params: dict = params
        self.labels_mapping: dict = params["labels_mapping"]
        self.input_features_schema: dict = self.params["input_features_schema"]
        self.required_input_features: Iterable[str] = self.input_features_schema.keys()
        self.requested_input_features: Iterable[str] = self.request_data.keys()

    def _req_dict_to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame([self.request_data])

    def min_num_of_cols_validation(self):
        min_num_of_columns_required = len(self.required_input_features)
        num_of_cols_in_request = len(self.requested_input_features)
        if num_of_cols_in_request < min_num_of_columns_required:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"The minimum number of columns required are {min_num_of_columns_required}, while you passed {num_of_cols_in_request} columns",
            )

    def col_names_validation(self):
        for feature in self.required_input_features:
            if feature not in self.requested_input_features:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"The column name: {feature} is not available in the current request",
                )

    def preprocess_json_input(self, preprocessor: sklearn.pipeline.Pipeline):
        input_df = self._req_dict_to_pandas()
        self.X_scaled = preprocessor.transform(input_df)

    def predict(self, model_path: str):
        onnx_sess = rt.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        input_name = onnx_sess.get_inputs()[0].name
        input_data = {input_name: self.X_scaled.astype(np.float32)}
        output = onnx_sess.run(None, input_data)
        probabilities = softmax_func(output[0][0])
        max_index = np.argmax(probabilities, axis=0)
        predicted_label = self.labels_mapping[max_index]
        return predicted_label, probabilities.tolist()
