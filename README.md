# Formula 1 Pit Stop & Driver Ranking Predictions

**Author:** Aya Tarist  
**Course:** ANOP 330 (Prof. Bailey)  
**Date:** August 2023  

---

## 📖 Project Overview

This repository contains all materials for a two-part Formula 1 machine-learning capstone:

1. **Pit-Stop Classification (Belgian Grand Prix 2022)**  
   * Use FastF1 telemetry and timing data to engineer a binary target (`PitStopOccurred`) for each lap.  
   * Conduct Exploratory Data Analysis (EDA) to identify key predictors (LapTime, TyreLife, speed metrics, compound, etc.).  
   * Prepare features and visualize distributions, correlations, and class imbalance.  
   * *Next step:* build and evaluate a supervised classification model.

2. **Driver-Ranking Prediction (2009 – 2022 seasons)**  
   * Use Ergast API data (drivers, constructors, race results, points) to predict each driver’s finishing rank in a race.  
   * Pre-process dates, fill missing values, and engineer features (driver age, DNF flags, concatenated driver names, etc.).  
   * Perform EDA to uncover trends in team and driver performance, correlation structure, and feature importance.  
   * Train and compare three models (Logistic Regression, Decision Tree, Random Forest) with Stratified K-Fold cross-validation.  
   * Evaluate accuracy, F1-score, precision, recall, and balanced accuracy **before and after** hyper-parameter tuning.

---

## 🗂 File Structure

├── README.md ← This documentation
├── F1_Rankings.ipynb ← Jupyter notebook for driver-ranking prediction
├── Project Report - EDA.pdf ← PDF: EDA for Pit-Stop classification
├── Project Report - Results & Analysis.pdf ← PDF: data prep, EDA, modeling & results for driver rankings
├── data/ ← (optional) raw CSV exports or fetched data sets
│ ├── belgian_gp_2022.csv ← Telemetry & timing data for Belgian GP 2022
│ └── f1_2009_2022.csv ← Aggregated Ergast API data (2009–2022)
└── requirements.txt ← Python dependencies and versions

yaml
Copy
Edit

> **Note:** If `data/` is missing, fetch data via the **FastF1** package (for Belgian GP 2022) and the **Ergast API** (for multi-season data).

---

## 🚀 How to Run

### 1  Create and activate an environment

conda create -n f1_ml python=3.9
conda activate f1_ml
pip install -r requirements.txt

go
Copy
Edit

`requirements.txt`

fastf1>=2.4
pandas>=1.3
numpy>=1.21
matplotlib>=3.4
seaborn>=0.11
scikit-learn>=1.0
jupyterlab>=3.0

vbnet
Copy
Edit

*(Alternatively, install manually with `pip`.)*

### 2  Pit-Stop EDA (PDF + optional notebook)

jupyter lab

markdown
Copy
Edit

* Open **Project Report - EDA.pdf** to view findings.  
* If present, open `F1_PitStop_EDA.ipynb` to reproduce plots.

### 3  Driver-Ranking Prediction notebook

jupyter lab F1_Rankings.ipynb

yaml
Copy
Edit

The notebook covers:

* Data ingestion & cleaning  
* EDA (team trends, driver vs. constructor points, correlation matrix)  
* Feature engineering  
* Modeling (LogReg, DecisionTree, RandomForest)  
* Evaluation (pre- & post-tuning metrics)

---

## 📊 Key Insights

### Pit-Stop EDA

* **Class imbalance:** 90 % laps with no stop, 10 % with stop.  
* **Lap-time & speed patterns:** slower laps and lower speed sectors correlate with pit stops.  
* **Compound usage:** Medium > Hard > Soft.  
* **Next steps:** add weather data, address imbalance, train classifiers.

### Driver-Ranking Model (PDF)

* **EDA:** Mercedes dominates; older drivers score fewer points.  
* **Best model:** Random Forest (post-tuning) — accuracy ≈ 85 %, F1 ≈ 0.85, balanced accuracy ≈ 0.78.  
* **Interpretation:** Ensemble trees capture non-linear interactions better than Logistic Regression.

---

## 🤖 Possible Extensions

* **Pit-Stop model:** add weather features, SMOTE / class weights, compare XGBoost.  
* **Ranking model:** include constructor budgets, track length, laps; explore ordinal regression; deploy as a web app.

