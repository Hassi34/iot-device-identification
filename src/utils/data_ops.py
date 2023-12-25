import gzip
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def gzip_np_arr(np_array: np.ndarray, filepath: str):
    with gzip.GzipFile(filepath, "w") as f:
        np.save(file=f, arr=np_array)


def load_np_arr_from_gz(filepath: str) -> np.ndarray:
    with gzip.GzipFile(filepath, "r") as f:
        return np.load(f)


def get_fitted_pipeline(df, columns, KNN_IMPUTER_NEIGHBORS: int = 3):
    ct = ColumnTransformer(
        transformers=[("input_features", "passthrough", columns)], remainder="drop"
    )
    imputer = KNNImputer(n_neighbors=KNN_IMPUTER_NEIGHBORS)
    scaler = StandardScaler()

    pipeline = Pipeline(
        steps=[("select_columns", ct), ("imputer", imputer), ("scaler", scaler)]
    )

    return pipeline.fit(df)
