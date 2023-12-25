#!/bin/bash 
set -e
echo [$(date)]: ">>>>>>>>>>>>>>>>>> TRAINING ENVIRONMENT SETUP >>>>>>>>>>>>>>>>>>"
echo [$(date)]: ">>>>>>>>>>>>>>>>>> Install Requirements >>>>>>>>>>>>>>>>>>"
pip3 install -r requirements.txt || pip install -r requirements.txt
echo [$(date)]: ">>>>>>>>>>>>>>>>>> START TRANING >>>>>>>>>>>>>>>>>>"
echo [$(date)]: ">>>>>>>>>>>>>>>>>> Create project folder structure >>>>>>>>>>>>>>>>>>"
python3 template.py || python template.py
echo [$(date)]: ">>>>>>>>>>>>>>>>>> Stage 01- Ingest Data >>>>>>>>>>>>>>>>>>"
python3 src/stage_01_ingest_data.py --config=configs/system.yaml || python src/stage_01_ingest_data.py --config=configs/system.yaml
echo [$(date)]: ">>>>>>>>>>>>>>>>>> Stage 02- Data Drift Check >>>>>>>>>>>>>>>>>>"
python3 src/stage_02_data_drift.py --config=configs/system.yaml --params=configs/params.yaml || python src/stage_02_data_drift.py --config=configs/system.yaml --params=configs/params.yaml
echo [$(date)]: ">>>>>>>>>>>>>>>>>> Stage 03- Preprocess Data >>>>>>>>>>>>>>>>>>"
python3 src/stage_03_preprocess_data.py --config=configs/system.yaml --params=configs/params.yaml || python src/stage_03_preprocess_data.py --config=configs/system.yaml --params=configs/params.yaml
echo [$(date)]: ">>>>>>>>>>>>>>>>>> Stage 04- ModelTraining&Evaluation >>>>>>>>>>>>>>>>>>"
python3 src/stage_04_training_and_eval.py --config=configs/system.yaml --params=configs/params.yaml || python src/stage_04_training_and_eval.py --config=configs/system.yaml --params=configs/params.yaml
echo [$(date)]: ">>>>>>>>>>>>>>>>>> Stage 05- ModelBlessing >>>>>>>>>>>>>>>>>>"
python3 src/stage_05_model_blessing.py --config=configs/system.yaml --params=configs/params.yaml || python src/stage_05_model_blessing.py --config=configs/system.yaml --params=configs/params.yaml