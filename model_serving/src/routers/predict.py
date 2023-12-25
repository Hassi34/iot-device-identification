from src.utils.common import read_yaml
import src.schemas.schema as SCHEMA
from fastapi import status, Body
from src.prediction_service import Services
import joblib
from fastapi import APIRouter, Depends, HTTPException
from typing import Annotated
from src.routers.users import get_current_user

config = read_yaml("src/configs/system.yaml")
params = read_yaml("src/configs/params.yaml")

PRODUCTION_MODEL_FILE_PATH = config["model_serving"]["PRODUCTION_MODEL_FILE_PATH"]
DATA_PREPROCESSOR_FILE_PATH = config["model_serving"]["DATA_PREPROCESSOR_FILE_PATH"]
LABELS_MAPPING = params["labels_mapping"]

router = APIRouter(prefix="/predict", tags=["Prediction"])

user_dependency = Annotated[dict, Depends(get_current_user)]


@router.post(
    "/iot-devices",
    response_model=SCHEMA.ShowResults,
    status_code=status.HTTP_200_OK,
)
async def predict_route(user: user_dependency, inputParam: SCHEMA.Prediction):
    if user:
        request_data = inputParam.request_data
        prediction_services = Services(request_data=request_data, params=params)
        prediction_services.min_num_of_cols_validation()
        prediction_services.col_names_validation()
        preprocessor = joblib.load(DATA_PREPROCESSOR_FILE_PATH)
        prediction_services.preprocess_json_input(preprocessor=preprocessor)
        predicted_label, probabilites = prediction_services.predict(
            model_path=PRODUCTION_MODEL_FILE_PATH
        )

        return {
            "predicted_class_label": predicted_label,
            "class_indices": LABELS_MAPPING,
            "prediction_probabilities": probabilites,
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed"
        )
