Fraud Detection for E-commerce and Bank Transactions
Project Overview
This repository contains the code and documentation for the Interim-2 and Final Submission of the 10 Academy Artificial Intelligence Mastery Week 8 & 9 Challenge (16 July - 29 July 2025). The project improves fraud detection for e-commerce (Fraud_Data.csv) and bank transactions (creditcard.csv) at Adey Innovations Inc. Task 1 involves data cleaning, EDA, and feature engineering. Tasks 2 and 3 involve train-test splitting, SMOTE, model training (Logistic Regression, Random Forest), evaluation, and SHAP explainability. The implementation addresses feedback from a previous submission (2.5/100).
Objectives

Task 1: Clean data, perform EDA, engineer features (e.g., device_id-based velocity), save datasets to data/processed/.
Task 2: Split data, apply SMOTE, train and evaluate Logistic Regression and Random Forest, using AUC-PR, F1-Score, and Confusion Matrix.
Task 3: Use SHAP to interpret the best model, generating Summary and Force Plots.
Code Structure: Modular, documented code with automated tests and CI.

Datasets

Input Datasets:
Fraud_Data.csv: E-commerce transactions.
IpAddress_to_Country.csv: IP-to-country mapping.
creditcard.csv: Bank transactions.


Preprocessed Datasets (in data/processed/): X_fraud.csv, y_fraud.csv, X_credit.csv, y_credit.csv.

Repository Structure

task1_preprocessing.py: Script for Task 1.
task2_3_modeling_explainability.py: Script for Tasks 2 and 3.
test_task1_preprocessing.py: Tests for Task 1.
test_task2_3_modeling_explainability.py: Tests for Tasks 2 and 3.
.github/workflows/ci.yml: GitHub Actions workflow for CI.
plots/: EDA and SHAP visualizations.
data/processed/: Preprocessed datasets.
models/: Trained model files.
results/: Model predictions and metrics.
fraud_detection_task1_report.md: Task 1 report.
fraud_detection_task2_3_report.md: Tasks 2 and 3 report.
requirements.txt: Python dependencies.
README.md: This file.

Setup Instructions
Prerequisites

Python 3.8+
Git

Installation

Clone the Repository:
git clone [Insert Your GitHub Repository URL]
cd [repository-name]


Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt

Contents of requirements.txt:
pandas==2.2.2
numpy==1.26.4
matplotlib==3.9.1
seaborn==0.13.2
scikit-learn==1.5.1
imbalanced-learn==0.12.3
joblib==1.4.2
shap==0.46.0


Place Input Datasets:

Copy Fraud_Data.csv, IpAddress_to_Country.csv, and creditcard.csv to the root directory (for Task 1).
Task 2/3 uses preprocessed datasets in data/processed/.



Running the Code

Run Task 1 (Preprocessing):
python task1_preprocessing.py


Outputs visualizations to plots/ and datasets to data/processed/.


Run Tasks 2 and 3 (Modeling and Explainability):
python task2_3_modeling_explainability.py


Outputs models to models/, predictions and metrics to results/, SHAP plots to plots/.


Run Tests:
python -m unittest test_task1_preprocessing.py -v
python -m unittest test_task2_3_modeling_explainability.py -v


View Reports:

Open fraud_detection_task1_report.md and fraud_detection_task2_3_report.md or convert to PDF:pandoc fraud_detection_task1_report.md -o fraud_detection_task1_report.pdf
pandoc fraud_detection_task2_3_report.md -o fraud_detection_task2_3_report.pdf





Continuous Integration
A GitHub Actions workflow (.github/workflows/ci.yml) automates testing:

Runs on push or pull requests to main.
Tests task1_preprocessing.py and task2_3_modeling_explainability.py.
Verifies output files in data/processed/, models/, results/, and plots/.Check the "Actions" tab in the GitHub repository for results.

Key Features

Task 1: Data cleaning, EDA, feature engineering (e.g., device_id-based velocity), datasets saved to data/processed/.
Task 2: Train-test split, SMOTE, Logistic Regression and Random Forest models, evaluation with AUC-PR, F1-Score, Confusion Matrix.
Task 3: SHAP Summary and Force Plots for the best model, identifying key fraud drivers.
Testing: Unit tests for all tasks ensure functionality.
CI: GitHub Actions validates code reliability.

Notes

Uses device_id-based transaction velocity due to unique user_id values.
SMOTE applied to training data only to avoid leakage.
No NaN values in preprocessed data, preventing errors.
Random Forest chosen as ensemble model for robust performance and SHAP compatibility.

Next Steps

Final Submission (29 July 2025): Finalize report with SHAP insights and polish documentation.

