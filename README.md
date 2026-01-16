# MALLORN Astronomical Classification Challenge

Goal: Identify Tidal Disruption Events (TDEs) from simulated LSST lightcurves using machine learning.

This project uses a clean, script-based workflow (no notebooks) for reproducibility and strong Kaggle performance.

--------------------------------------------------

Folder Structure

mallorn_tde/
│
├── data/
│   ├── raw/         
│   └── processed/        
│
├── src/
│   ├── 01_explore.py    
│   ├── 02_features.py  
│   ├── 03_train.py      
│   └── 04_predict.py    
│
├── submissions/
│   └── submission_v1.csv
│
├── requirements.txt
└── README.md

--------------------------------------------------

Workflow

1. Place train_log.csv and test_log.csv in data/raw/
2. Run:
   python src/01_explore.py
3. Run:
   python src/02_features.py
4. Run:
   python src/03_train.py
5. Run:
   python src/04_predict.py

--------------------------------------------------

Evaluation

Metric: F1 Score  
Only TDE classification (1) is evaluated  
Class imbalance is high → threshold tuning is essential

--------------------------------------------------
