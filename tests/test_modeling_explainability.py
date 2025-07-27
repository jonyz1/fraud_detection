import unittest
import pandas as pd
import numpy as np
import os
import shutil
from task2_3_modeling_explainability import load_preprocessed_data, split_data, apply_smote, train_and_evaluate, generate_shap_plots, select_best_model

class TestTask2_3ModelingExplainability(unittest.TestCase):
    def setUp(self):
        """Set up synthetic preprocessed datasets and directories."""
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        
        # Synthetic X_fraud.csv and y_fraud.csv
        self.X_fraud = pd.DataFrame({
            'purchase_value': [0.1, -0.2, 0.3],
            'age': [-0.5, 0.0, 0.5],
            'user_transaction_count': [1, 1, 1],
            'device_transaction_count': [2, 2, 1],
            'avg_time_between_transactions': [0.5, 0.5, 1.0],
            'hour_of_day': [10, 15, 23],
            'day_of_week': [1, 2, 3],
            'time_since_signup': [1.0, 2.0, 0.5],
            'source_Direct': [0, 1, 0],
            'browser_Firefox': [0, 1, 0],
            'sex_M': [1, 0, 1],
            'country_USA': [1, 0, 0]
        })
        self.y_fraud = pd.Series([0, 1, 0])
        self.X_fraud.to_csv('data/processed/X_fraud.csv', index=False)
        self.y_fraud.to_csv('data/processed/y_fraud.csv', index=False)
        
        # Synthetic X_credit.csv and y_credit.csv
        self.X_credit = pd.DataFrame({
            'Time': [0.0, 1.0, 2.0],
            'V1': [0.1, -0.1, 0.2],
            'V2': [0.2, 0.3, -0.1],
            'Amount': [0.5, -0.5, 0.0]
        })
        self.y_credit = pd.Series([0, 1, 0])
        self.X_credit.to_csv('data/processed/X_credit.csv', index=False)
        self.y_credit.to_csv('data/processed/y_credit.csv', index=False)

    def tearDown(self):
        """Clean up temporary files and directories."""
        for folder in ['data/processed', 'models', 'results', 'plots']:
            if os.path.exists(folder):
                shutil.rmtree(folder)

    def test_load_preprocessed_data(self):
        """Test loading preprocessed datasets."""
        X_fraud, y_fraud, X_credit, y_credit = load_preprocessed_data()
        self.assertEqual(X_fraud.shape, (3, 12), "X_fraud shape incorrect")
        self.assertEqual(y_fraud.shape, (3,), "y_fraud shape incorrect")
        self.assertEqual(X_credit.shape, (3, 4), "X_credit shape incorrect")
        self.assertEqual(y_credit.shape, (3,), "y_credit shape incorrect")
        self.assertFalse(X_fraud.isna().any().any(), "X_fraud should have no NaN")
        self.assertFalse(X_credit.isna().any().any(), "X_credit should have no NaN")

    def test_split_data(self):
        """Test train-test split with stratification."""
        X_fraud, y_fraud, _, _ = load_preprocessed_data()
        X_train, X_test, y_train, y_test = split_data(X_fraud, y_fraud, test_size=0.33)
        self.assertEqual(X_train.shape[0], 2, "X_train should have 2 rows")
        self.assertEqual(X_test.shape[0], 1, "X_test should have 1 row")
        self.assertEqual(len(y_train), 2, "y_train should have 2 elements")
        self.assertEqual(len(y_test), 1, "y_test should have 1 element")

    def test_apply_smote(self):
        """Test SMOTE application."""
        X_fraud, y_fraud, _, _ = load_preprocessed_data()
        X_train, X_test, y_train, y_test = split_data(X_fraud, y_fraud, test_size=0.33)
        X_train_smote, y_train_smote = apply_smote(X_train, y_train)
        self.assertGreaterEqual(len(X_train_smote), len(X_train), "SMOTE should not reduce training data size")
        self.assertTrue(np.any(y_train_smote == 1), "SMOTE should generate fraudulent samples")

    def test_train_and_evaluate(self):
        """Test model training and evaluation."""
        X_fraud, y_fraud, X_credit, y_credit = load_preprocessed_data()
        X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = split_data(X_fraud, y_fraud, test_size=0.33)
        X_credit_train, X_credit_test, y_credit_train, y_credit_test = split_data(X_credit, y_credit, test_size=0.33)
        
        # Apply SMOTE
        X_fraud_train_smote, y_fraud_train_smote = apply_smote(X_fraud_train, y_fraud_train)
        X_credit_train_smote, y_credit_train_smote = apply_smote(X_credit_train, y_credit_train)
        
        # Train and evaluate
        fraud_results = train_and_evaluate(X_fraud_train_smote, y_fraud_train_smote, 
                                          X_fraud_test, y_fraud_test, 'fraud')
        credit_results = train_and_evaluate(X_credit_train_smote, y_credit_train_smote, 
                                           X_credit_test, y_credit_test, 'creditcard')
        
        # Check model files
        for model_name in ['Logistic Regression', 'Random Forest']:
            for dataset in ['fraud', 'creditcard']:
                self.assertTrue(os.path.exists(f'models/{model_name}_{dataset}.pkl'), 
                                f"{model_name} model for {dataset} should exist")
                self.assertTrue(os.path.exists(f'results/predictions_{model_name}_{dataset}.csv'), 
                                f"{model_name} predictions for {dataset} should exist")
        
        # Check metrics files
        self.assertTrue(os.path.exists('results/metrics_fraud.txt'), "Fraud metrics file should exist")
        self.assertTrue(os.path.exists('results/metrics_creditcard.txt'), "Creditcard metrics file should exist")

    def test_generate_shap_plots(self):
        """Test SHAP plot generation."""
        X_fraud, y_fraud, _, _ = load_preprocessed_data()
        X_train, X_test, y_train, y_test = split_data(X_fraud, y_fraud, test_size=0.33)
        X_train_smote, y_train_smote = apply_smote(X_train, y_train)
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train_smote, y_train_smote)
        generate_shap_plots(model, X_test, 'fraud', 'Random Forest')
        self.assertTrue(os.path.exists('plots/shap_summary_Random Forest_fraud.png'), 
                        "SHAP summary plot should exist")
        self.assertTrue(os.path.exists('plots/shap_force_Random Forest_fraud.png'), 
                        "SHAP force plot should exist")

    def test_select_best_model(self):
        """Test best model selection."""
        X_fraud, y_fraud, X_credit, y_credit = load_preprocessed_data()
        X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = split_data(X_fraud, y_fraud, test_size=0.33)
        X_credit_train, X_credit_test, y_credit_train, y_credit_test = split_data(X_credit, y_credit, test_size=0.33)
        X_fraud_train_smote, y_fraud_train_smote = apply_smote(X_fraud_train, y_fraud_train)
        X_credit_train_smote, y_credit_train_smote = apply_smote(X_credit_train, y_credit_train)
        fraud_results = train_and_evaluate(X_fraud_train_smote, y_fraud_train_smote, 
                                          X_fraud_test, y_fraud_test, 'fraud')
        credit_results = train_and_evaluate(X_credit_train_smote, y_credit_train_smote, 
                                           X_credit_test, y_credit_test, 'creditcard')
        best_model = select_best_model(fraud_results, credit_results)
        self.assertIn(best_model['name'], ['Logistic Regression', 'Random Forest'], 
                      "Best model should be Logistic Regression or Random Forest")
        self.assertIn(best_model['dataset'], ['fraud', 'creditcard'], 
                      "Best model dataset should be fraud or creditcard")

if __name__ == '__main__':
    unittest.main()