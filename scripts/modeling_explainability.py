import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import os

# Create directories if they don't exist
for folder in ['models', 'results', 'plots']:
    os.makedirs(folder, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# -----------------------------------
# 1. Load Preprocessed Data
# -----------------------------------
def load_preprocessed_data():
    """Load preprocessed datasets from data/processed/."""
    try:
        X_fraud = pd.read_csv('data/processed/X_fraud.csv')
        y_fraud = pd.read_csv('data/processed/y_fraud.csv').values.ravel()
        X_credit = pd.read_csv('data/processed/X_credit.csv')
        y_credit = pd.read_csv('data/processed/y_credit.csv').values.ravel()
    except FileNotFoundError as e:
        print(f"Error: Preprocessed file not found. Ensure 'data/processed/*.csv' files exist.")
        raise e
    
    print("Fraud Data Shapes:", X_fraud.shape, y_fraud.shape)
    print("Credit Card Data Shapes:", X_credit.shape, y_credit.shape)
    print("\nFraud Data Class Distribution:\n", pd.Series(y_fraud).value_counts(normalize=True))
    print("\nCredit Card Data Class Distribution:\n", pd.Series(y_credit).value_counts(normalize=True))
    
    return X_fraud, y_fraud, X_credit, y_credit

# -----------------------------------
# 2. Train-Test Split
# -----------------------------------
def split_data(X, y, test_size=0.2):
    """Perform train-test split with stratification."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    print(f"Train Class Distribution:\n", pd.Series(y_train).value_counts(normalize=True))
    return X_train, X_test, y_train, y_test

# -----------------------------------
# 3. Apply SMOTE
# -----------------------------------
def apply_smote(X_train, y_train):
    """Apply SMOTE to training data to handle class imbalance."""
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"SMOTE Applied - Train Shape: {X_train_smote.shape}")
    print(f"SMOTE Train Class Distribution:\n", pd.Series(y_train_smote).value_counts(normalize=True))
    return X_train_smote, y_train_smote

# -----------------------------------
# 4. Train and Evaluate Models
# -----------------------------------
def train_and_evaluate(X_train, y_train, X_test, y_test, dataset_name):
    """Train Logistic Regression and Random Forest, evaluate, and save results."""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name} on {dataset_name}...")
        model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(model, f'models/{model_name}_{dataset_name}.pkl')
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluation metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        auc_pr = average_precision_score(y_test, y_pred_proba)
        
        results[model_name] = {
            'classification_report': report,
            'confusion_matrix': cm,
            'auc_pr': auc_pr,
            'model': model,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        # Save predictions
        pd.DataFrame({'y_test': y_test, 'y_pred': y_pred}).to_csv(
            f'results/predictions_{model_name}_{dataset_name}.csv', index=False
        )
        
        # Print results
        print(f"\n{model_name} Results for {dataset_name}:")
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", cm)
        print(f"AUC-PR Score: {auc_pr:.4f}")
    
    # Save metrics to file
    with open(f'results/metrics_{dataset_name}.txt', 'w') as f:
        for model_name, metrics in results.items():
            f.write(f"{model_name}:\n")
            f.write("Classification Report:\n")
            f.write(str(metrics['classification_report']))
            f.write("\nConfusion Matrix:\n")
            f.write(str(metrics['confusion_matrix']))
            f.write(f"\nAUC-PR Score: {metrics['auc_pr']:.4f}\n\n")
    
    return results

# -----------------------------------
# 5. SHAP Explainability
# -----------------------------------
def generate_shap_plots(model, X_test, dataset_name, model_name):
    """Generate SHAP Summary and Force Plots for the model."""
    explainer = shap.TreeExplainer(model) if model_name == 'Random Forest' else shap.LinearExplainer(model, X_test)
    shap_values = explainer.shap_values(X_test)
    
    # Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[1] if model_name == 'Random Forest' else shap_values, 
                     X_test, show=False)
    plt.title(f'SHAP Summary Plot - {model_name} ({dataset_name})')
    plt.tight_layout()
    plt.savefig(f'plots/shap_summary_{model_name}_{dataset_name}.png')
    plt.close()
    
    # Force Plot for first test instance
    plt.figure(figsize=(10, 4))
    shap.force_plot(explainer.expected_value[1] if model_name == 'Random Forest' else explainer.expected_value, 
                    shap_values[1][0] if model_name == 'Random Forest' else shap_values[0], 
                    X_test.iloc[0], show=False, matplotlib=True)
    plt.title(f'SHAP Force Plot - {model_name} ({dataset_name}) - First Instance')
    plt.tight_layout()
    plt.savefig(f'plots/shap_force_{model_name}_{dataset_name}.png')
    plt.close()

# -----------------------------------
# 6. Select Best Model
# -----------------------------------
def select_best_model(fraud_results, credit_results):
    """Select the best model based on AUC-PR."""
    best_model = {'name': None, 'dataset': None, 'auc_pr': 0}
    
    for dataset_name, results in [('fraud', fraud_results), ('creditcard', credit_results)]:
        for model_name, metrics in results.items():
            auc_pr = metrics['auc_pr']
            if auc_pr > best_model['auc_pr']:
                best_model = {
                    'name': model_name,
                    'dataset': dataset_name,
                    'auc_pr': auc_pr,
                    'model': metrics['model'],
                    'X_test': metrics['X_test']
                }
    
    print(f"\nBest Model: {best_model['name']} on {best_model['dataset']} (AUC-PR: {best_model['auc_pr']:.4f})")
    print("Justification: Selected based on highest AUC-PR, which balances precision and recall for imbalanced data.")
    
    return best_model

# -----------------------------------
# Main Execution
# -----------------------------------
def main():
    """Execute Tasks 2 and 3: Model building, training, evaluation, and explainability."""
    print("Starting Tasks 2 and 3: Model Building, Training, and Explainability")
    
    # Load preprocessed data
    X_fraud, y_fraud, X_credit, y_credit = load_preprocessed_data()
    
    # Split data
    print("\nSplitting Fraud Data...")
    X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = split_data(X_fraud, y_fraud)
    print("\nSplitting Credit Card Data...")
    X_credit_train, X_credit_test, y_credit_train, y_credit_test = split_data(X_credit, y_credit)
    
    # Apply SMOTE
    print("\nApplying SMOTE to Fraud Data...")
    X_fraud_train_smote, y_fraud_train_smote = apply_smote(X_fraud_train, y_fraud_train)
    print("\nApplying SMOTE to Credit Card Data...")
    X_credit_train_smote, y_credit_train_smote = apply_smote(X_credit_train, y_credit_train)
    
    # Train and evaluate models
    print("\nEvaluating Models on Fraud Data...")
    fraud_results = train_and_evaluate(X_fraud_train_smote, y_fraud_train_smote, 
                                      X_fraud_test, y_fraud_test, 'fraud')
    print("\nEvaluating Models on Credit Card Data...")
    credit_results = train_and_evaluate(X_credit_train_smote, y_credit_train_smote, 
                                       X_credit_test, y_credit_test, 'creditcard')
    
    # Select best model
    best_model = select_best_model(fraud_results, credit_results)
    
    # Generate SHAP plots for best model
    print(f"\nGenerating SHAP Plots for {best_model['name']} on {best_model['dataset']}...")
    generate_shap_plots(best_model['model'], best_model['X_test'], 
                        best_model['dataset'], best_model['name'])
    
    print("\nTasks 2 and 3 Completed: Models trained, evaluated, SHAP plots saved to plots/, and results saved to results/.")

if __name__ == "__main__":
    main()