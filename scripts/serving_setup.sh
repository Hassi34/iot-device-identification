#!/bin/bash 
set -e
echo [$(date)]: ">>>>>>>>>>>>>>>>>> SERVING ENVIRONMENT SETUP >>>>>>>>>>>>>>>>>>"
echo [$(date)]: ">>>>>>>>>>>>>>>>>> Create project folder structure >>>>>>>>>>>>>>>>>>"
python3 template.py || python template.py
echo [$(date)]: ">>>>>>>>>>>>>>>>>> Copy production model to serving environmen >>>>>>>>>>>>>>>>>>"
cp -R artifacts/models/production/* model_serving/src/artifacts/production_model
echo [$(date)]: ">>>>>>>>>>>>>>>>>> Copy params to serving >>>>>>>>>>>>>>>>>>"
cp configs/params.yaml model_serving/src/configs/params.yaml
echo [$(date)]: ">>>>>>>>>>>>>>>>>> Copy data preprocessing pipeline >>>>>>>>>>>>>>>>>>"
cp artifacts/pipelines/data_preprocessing_pipeline.pkl model_serving/src/artifacts/pipelines/data_preprocessing_pipeline.pkl
echo [$(date)]: ">>>>>>>>>>>>>>>>>> Copy common utils to serving >>>>>>>>>>>>>>>>>>"
cp src/utils/common.py model_serving/src/utils/common.py
echo [$(date)]: ">>>>>>>>>>>>>>>>>> Copy setup.py >>>>>>>>>>>>>>>>>>"
cp setup.py model_serving/setup.py
echo [$(date)]: ">>>>>>>>>>>>>>>>>> SERVING SETUP COMPLETED >>>>>>>>>>>>>>>>>>"