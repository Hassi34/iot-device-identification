from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run as app_run
from src.utils.common import read_yaml
from src.routers import home, predict, authentication, users

config = read_yaml("src/configs/system.yaml")

APP_HOST = config["model_serving"]["APP_HOST"]
APP_PORT = config["model_serving"]["APP_PORT"]

API_TITLE = config["model_serving"]["API_TITLE"]
API_DESCRIPTION = config["model_serving"]["API_DESCRIPTION"]
API_VERSION = config["model_serving"]["API_VERSION"]

app = FastAPI(title=API_TITLE, description=API_DESCRIPTION, version=API_VERSION)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(home.router)
app.include_router(authentication.router)
app.include_router(users.router)
app.include_router(predict.router)


if __name__ == "__main__":
    app_run(app=app, host=APP_HOST, port=APP_PORT)
