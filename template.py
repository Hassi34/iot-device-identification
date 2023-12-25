import os


def create_structure():
    dirs = [
        os.path.join("logs"),
        os.path.join("scripts"),
        os.path.join("artifacts", "pipelines"),
        os.path.join("artifacts", "models"),
        os.path.join("artifacts", "model_evaluation"),
        os.path.join("model_serving", "src", "utils"),
        os.path.join("model_serving", "src", "prediction_service"),
        os.path.join("model_serving", "src", "schemas"),
        os.path.join("model_serving", "src", "artifacts"),
        os.path.join("model_serving", "src", "artifacts", "production_model"),
        os.path.join("model_serving", "src", "artifacts", "pipelines"),
        os.path.join("model_serving", "src", "artifacts", "model_monitoring"),
        os.path.join("src", "utils"),
    ]

    for dir_ in dirs:
        os.makedirs(dir_, exist_ok=True)
        with open(os.path.join(dir_, ".gitkeep"), "w") as f:
            pass

    files = [
        "dvc.yaml",
        ".gitignore",
        os.path.join("src", "__init__.py"),
        os.path.join("src", "README.md"),
        os.path.join("configs", "params.yaml"),
        os.path.join("configs", "system.yaml"),
        os.path.join("logs", "running_logs.log"),
        os.path.join("model_serving", "src", "__init__.py"),
        os.path.join("model_serving", "src", "utils", "__init__.py"),
        os.path.join("model_serving", "src", "schemas", "__init__.py"),
        os.path.join("model_serving", "src", "prediction_service", "__init__.py"),
        os.path.join("model_serving", "src", "prediction_service", "services.py"),
        os.path.join("model_serving", "src", "main.py"),
        os.path.join("model_serving", "requirements.txt"),
        os.path.join("model_serving", "README.md"),
        os.path.join("model_serving", "setup.py"),
    ]

    for file in files:
        if not os.path.exists(file):
            with open(file, "w") as f:
                pass


if __name__ == "__main__":
    create_structure()
