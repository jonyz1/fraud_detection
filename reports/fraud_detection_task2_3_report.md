Tasks 2 and 3: Model Building, Training, and Explainability
Overview
Tasks 2 and 3 build on the preprocessed datasets from Task 1 (X_fraud.csv, y_fraud.csv, X_credit.csv, y_credit.csv in data/processed/) to develop, evaluate, and interpret fraud detection models for e-commerce (Fraud_Data.csv) and bank transactions (creditcard.csv). Task 2 involves train-test splitting, SMOTE, and training/evaluating Logistic Regression and Random Forest models. Task 3 uses SHAP to interpret the best model. The implementation addresses feedback from the Task 1 submission (2.5/100) with robust evaluation and explainability.
Task 2: Model Building and Training
Data Preparation

Features and Target: Separated features (X_fraud, X_credit) and targets (y_fraud as class, y_credit as Class).
Train-Test Split: Used train_test_split with 80-20 split and stratification to maintain class distribution (~2% fraud in Fraud_Data.csv, ~0.17% in creditcard.csv). Printed shapes and distributions for verification.

Class Imbalance Handling

SMOTE: Applied to training data only using imblearn.over_sampling.SMOTE to balance classes (50-50 post-SMOTE), avoiding data leakage.
Justification: SMOTE generates realistic synthetic samples, preserving data size and avoiding overfitting compared to random oversampling.

Model Selection

Logistic Regression: Chosen as a simple, interpretable baseline.
Random Forest: Selected as the ensemble model for its ability to capture non-linear patterns and provide feature importance, suitable for SHAP.

Model Training

Trained both models on SMOTE-balanced training data for both datasets.
Saved models to models/ (e.g., Logistic Regression_fraud.pkl, Random Forest_creditcard.pkl).

Model Evaluation

Metrics:
F1-Score: Balances precision and recall, critical for imbalanced data.
AUC-PR: Measures precision-recall trade-off, suitable for rare fraud events.
Confusion Matrix: Quantifies true positives (fraud detected), false positives, etc.


Results: Saved to results/metrics_fraud.txt and results/metrics_creditcard.txt. Predictions saved to results/predictions_[model]_[dataset].csv.
Best Model Selection:
Selected based on highest AUC-PR across datasets.
Random Forest typically outperforms Logistic Regression due to its ability to model complex patterns, especially with engineered features like device_id-based velocity.
Justification: AUC-PR prioritizes performance on the minority (fraud) class, critical for fraud detection where missing fraud (false negatives) is costly.



Task 3: Model Explainability

SHAP Analysis: Applied to the best model (likely Random Forest) using shap.TreeExplainer for fraud and creditcard datasets.
Plots Generated:
Summary Plot: Shows global feature importance, saved as plots/shap_summary_[model]_[dataset].png.
Fraud_Data.csv: Likely highlights avg_time_between_transactions (rapid transactions indicate fraud), time_since_signup (short times suggest account takeover), and device_transaction_count.
creditcard.csv: Likely emphasizes Amount (higher values linked to fraud) and PCA features like V1, V2.


Force Plot: Illustrates local feature contributions for the first test instance, saved as plots/shap_force_[model]_[dataset].png.


Insights:
Global: Features with high SHAP values (e.g., rapid transaction velocity, high purchase amounts) are key fraud drivers.
Local: Force plots show how specific feature values (e.g., low time_since_signup) push predictions toward fraud.
These align with Task 1 insights (e.g., higher purchase_value for fraud, unusual transaction hours).



Code Structure and Testing

Script: task2_3_modeling_explainability.py with modular functions for loading, splitting, SMOTE, training, evaluation, and SHAP.
Testing: test_task2_3_modeling_explainability.py verifies data loading, splitting, SMOTE, model training, evaluation, and SHAP plots.
CI: GitHub Actions (.github/workflows/ci.yml) tests Tasks 1, 2, and 3, ensuring output files exist.

Dependencies
Listed in requirements.txt:

pandas==2.2.2
numpy==1.26.4
matplotlib==3.9.1
seaborn==0.13.2
scikit-learn==1.5.1
imbalanced-learn==0.12.3
joblib==1.4.2
shap==0.46.0
