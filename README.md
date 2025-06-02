Formula 1 Pit Stop & Driver Ranking Predictions
Author: Aya Tarist
Course: ANOP 330 (Prof. Bailey)
Date: August 2023

📖 Project Overview
This repository contains all materials for a two‐part Formula 1 Machine Learning capstone:

Pit Stop Classification (Belgian Grand Prix 2022)

Use FastF1 telemetry and timing data to engineer a binary target (PitStopOccurred) for each lap.

Conduct Exploratory Data Analysis (EDA) to identify key predictors (LapTime, TyreLife, Speed metrics, Compound, etc.).

Prepare features and visualize distributions, correlations, and class imbalance.

(Next steps: build and evaluate a supervised classification model.)

Driver Ranking Prediction (2009–2022 Seasons)

Use Ergast API data (drivers, constructors, race results, points) to predict each driver’s finishing rank in a race.

Preprocess dates, fill missing values, and engineer features (driver age, DNF flags, concatenated driver names, etc.).

Perform EDA to uncover trends in team and driver performance, correlation structure, and feature importance.

Train and compare three models (Logistic Regression, Decision Tree, Random Forest) using Stratified K-Fold CV.

Evaluate accuracy, F1-score, precision, recall, and balanced accuracy before and after hyperparameter tuning.

🗂 File Structure
rust
Copy
Edit
├── README.md                                  ← This documentation
├── F1_Rankings.ipynb                          ← Jupyter notebook with full workflow for driver ranking prediction
├── Project Report - EDA.pdf                   ← PDF: Exploratory Data Analysis for Pit Stop Classification
├── Project Report - Results & Analysis .pdf   ← PDF: Data Prep, EDA, Modeling & Results for Driver Rankings
├── data/                                      ← (Optional) Raw CSV exports or fetched datasets
│   ├── belgian_gp_2022.csv                    ← Telemetry/timing data for Belgian GP (2022)
│   └── f1_2009_2022.csv                       ← Aggregated Ergast API data for seasons 2009–2022
└── requirements.txt                           ← Python dependencies and versions
Note: If your local copy does not include data/, fetch data via the FastF1 package (for 2022 Belgian GP) and the Ergast API (for multi‐season data).

🚀 How to Run
1. Create a Conda (or virtualenv) Environment
bash
Copy
Edit
conda create -n f1_ml python=3.9
conda activate f1_ml
pip install -r requirements.txt
Contents of requirements.txt:

shell
Copy
Edit
fastf1>=2.4
pandas>=1.3
numpy>=1.21
matplotlib>=3.4
seaborn>=0.11
scikit-learn>=1.0
jupyterlab>=3.0
(If you prefer, install these packages manually with pip install fastf1 pandas numpy matplotlib seaborn scikit-learn jupyterlab.)

2. Pit Stop EDA Notebook
Launch JupyterLab (or Jupyter Notebook):

bash
Copy
Edit
jupyter lab
Open Project Report - EDA.pdf to read the write-up and view plots.

(Optional) If you have the Python‐based EDA notebook (not included in PDF), open it to explore code cells and replicate all visualizations:

bash
Copy
Edit
jupyter lab F1_PitStop_EDA.ipynb
Key Outputs:

Class distribution of PitStopOccurred (90% no stop, 10% stop).

Histograms and box plots for LapTime, TyreLife, Compound, Stint.

Correlation heatmap showing relationships between speed metrics, lap times, tyre life, and pit stops.

3. Driver Ranking Prediction Notebook
Open F1_Rankings.ipynb in JupyterLab:

bash
Copy
Edit
jupyter lab F1_Rankings.ipynb
Walk Through Sections:

Data Ingestion & Cleaning (2009–2022 Ergast data):

Converts driverDateOfBirth & raceDate to datetime.

Converts numeric columns (e.g., driverFastestLapSpeed) to numeric dtype.

Concatenates driverForename + driverSurname → driverName.

Calculates driverAge (today’s date minus birthdate).

Creates driverDnf/constructorDnf flags based on result status codes.

Fills missing numeric columns (points, grid positions) with zeros.

Exploratory Data Analysis:

Team points‐per‐race trends (e.g., Mercedes vs. Ferrari vs. Red Bull).

Driver vs. constructor championship point time series.

Correlation matrix (highlighting driverStartGridPos ↔ driverFinalGridPos, etc.).

Feature Engineering:

Select features with high correlation to driverFinalRank:
season, driverId, constructorId, driverStartGridPos, driverFinalGridPos, driverChampionshipStandingPosition, constructorChampionshipStandingPosition.

Modeling & Evaluation:

Splits data (80% train / 20% test).

Uses 10-fold Stratified K-Fold CV on training.

Trains three classifiers:

Logistic Regression

Decision Tree

Random Forest

Reports accuracy, F1-score, precision, recall, balanced accuracy before and after hyperparameter tuning (GridSearchCV).

Summary of best hyperparameters and model performance (e.g., Decision Tree ≈ 85% accuracy, Random Forest ≈ 85% accuracy and F1).

Key Outputs:

Table 1: Best hyperparameters after tuning (Logistic Regression: C=10, penalty=l1; Decision Tree: default; Random Forest: max_depth=30, n_estimators=200).

Table 2 / Table 3: Model metrics (accuracy, F1, precision, recall, balanced accuracy) pre- and post-tuning.

Scatter plots showing LapTime vs. TyreLife, boxplots of lap times, bar charts of compound usage, correlation heatmap, and time-series plots for team/driver points by season.

📊 Results & Key Insights
Pit Stop EDA (Project Report - EDA.pdf)
Class Imbalance: 714 laps without pit stops (90.15%), 78 laps with pit stops (9.85%).

Lap Time Distribution: Median ≈ 125 seconds; outliers likely due to safety cars or incidents.

TyreLife Distribution: Right-skewed—most tyres changed early, few long stints.

Compound Usage: “MEDIUM” used most often, followed by “HARD” and “SOFT.”

Correlation:

LapTime has moderate positive correlation (0.51) with PitStopOccurred.

SpeedI2 has strong negative correlation—higher speed laps rarely coincide with pit stops.

TyreLife correlation is low, unexpectedly.

Next Steps: Incorporate external weather data (e.g., manual merge of track & race conditions), refine feature engineering, and train classification models.

Driver Ranking Predictions (Project Report - Results & Analysis .pdf)
Data Cleaning:

Converted date strings to datetime.

Filled missing numeric values with zero (zero points or no grid position).

Engineered driverDnf and constructorDnf flags.

EDA Findings:

Team Trends: Mercedes dominates points per race; Ferrari & Red Bull in a close rivalry; midfield teams (McLaren, Renault, AlphaTauri/Williams) cluster together; smaller teams (Haas, Toyota, BMW Sauber, Honda) lag behind.

Driver vs. Constructor Points: Drivers’ success depends on team performance; older drivers tend to score fewer points.

Correlation Matrix:

driverStartGridPos ↔ driverFinalGridPos (r ≈ 0.59).

driverFinalGridPos ↔ driverRacePoints (r ≈ 0.44).

driverAge ↔ driverChampionshipStandingPoints (r ≈ −0.61).

Modeling Results:

Before Tuning:

Logistic Regression – Accuracy: 20.5%, F1: 0.168.

Decision Tree – Accuracy: 85.06%, F1: 0.772.

Random Forest – Accuracy: 81.77%, F1: 0.741.

After Tuning:

Logistic Regression – Accuracy: 21.16%, F1: 0.190.

Decision Tree – Accuracy: 82.27%, F1: 0.823.

Random Forest – Accuracy: 85.42%, F1: 0.851.

Best Model: Random Forest (post-tuning) achieved highest balanced performance (Accuracy: 85.4%, F1: 0.851, Balanced Accuracy: 0.777).

Interpretation:

Decision trees (single vs. ensemble) capture non‐linear interactions (track conditions, driver skill, team strategy) better than Logistic Regression.

Random Forest’s ensemble averaging reduces overfitting and yields the strongest generalization.

⚙️ Dependencies
Below is a template requirements.txt for easy environment setup. Adjust versions as needed.

shell
Copy
Edit
fastf1>=2.4
pandas>=1.3
numpy>=1.21
matplotlib>=3.4
seaborn>=0.11
scikit-learn>=1.0
jupyterlab>=3.0
To install:

bash
Copy
Edit
pip install -r requirements.txt
🤖 Extending the Work
Pit Stop Model:

Add weather features (temperature, precipitation, track conditions) by merging external APIs or CSVs.

Use class‐imbalance techniques (SMOTE, class weights) to improve classification.

Compare different classifiers (Random Forest, XGBoost, Logistic Regression with L1/L2 regularization).

Ranking Model:

Introduce advanced features (constructor budget, driver experience metrics, track length, lap count).

Explore multi‐class ranking as ordinal regression.

Deploy a model in a lightweight web app for end‐users to query predicted finishing positions given driver, team, and qualifying data.

