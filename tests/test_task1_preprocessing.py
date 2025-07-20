
import unittest
import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime
from task1_preprocessing import load_and_check_data, clean_data, perform_eda, merge_geolocation, engineer_features, transform_data

class TestTask1Preprocessing(unittest.TestCase):
    def setUp(self):
        """Set up synthetic datasets and directories for testing."""
        # Create temporary directories
        os.makedirs('plots', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        # Synthetic Fraud_Data.csv
        self.fraud_data = pd.DataFrame({
            'user_id': ['1', '2', '3'],
            'signup_time': ['2025-07-01 10:00:00', '2025-07-01 11:00:00', '2025-07-01 12:00:00'],
            'purchase_time': ['2025-07-01 10:30:00', '2025-07-01 11:15:00', '2025-07-01 12:05:00'],
            'purchase_value': [50, 100, np.nan],
            'device_id': ['device1', 'device1', 'device2'],
            'source': ['SEO', 'Ads', 'Direct'],
            'browser': ['Chrome', 'Firefox', 'Safari'],
            'sex': ['M', 'F', np.nan],
            'age': [25, 30, 35],
            'ip_address': [123456, 789012, 345678],
            'class': [0, 1, 0]
        })
        self.fraud_data.to_csv('Fraud_Data.csv', index=False)
        
        # Synthetic creditcard.csv
        self.creditcard_data = pd.DataFrame({
            'Time': [0, 1, 2],
            'V1': [1.0, -1.0, 0.5],
            'V2': [0.5, 0.2, -0.1],
            'Amount': [100, np.nan, 200],
            'Class': [0, 1, 0]
        })
        self.creditcard_data.to_csv('creditcard.csv', index=False)
        
        # Synthetic IpAddress_to_Country.csv
        self.ip_to_country = pd.DataFrame({
            'lower_bound_ip_address': [100000, 700000],
            'upper_bound_ip_address': [200000, 800000],
            'country': ['USA', 'Canada']
        })
        self.ip_to_country.to_csv('IpAddress_to_Country.csv', index=False)

    def tearDown(self):
        """Clean up temporary files and directories."""
        for file in ['Fraud_Data.csv', 'creditcard.csv', 'IpAddress_to_Country.csv']:
            if os.path.exists(file):
                os.remove(file)
        if os.path.exists('plots'):
            shutil.rmtree('plots')
        if os.path.exists('data/processed'):
            shutil.rmtree('data/processed')

    def test_load_and_check_data(self):
        """Test loading datasets and checking for errors."""
        fraud_data, creditcard_data, ip_to_country = load_and_check_data()
        self.assertEqual(len(fraud_data), 3, "Fraud Data should have 3 rows")
        self.assertEqual(len(creditcard_data), 3, "Credit Card Data should have 3 rows")
        self.assertEqual(len(ip_to_country), 2, "IP to Country Data should have 2 rows")
        self.assertTrue('class' in fraud_data.columns, "Fraud Data should have 'class' column")
        self.assertTrue('Class' in creditcard_data.columns, "Credit Card Data should have 'Class' column")

    def test_clean_data(self):
        """Test data cleaning: missing values, duplicates, and data types."""
        fraud_data, creditcard_data, ip_to_country = load_and_check_data()
        fraud_data, creditcard_data, ip_to_country = clean_data(fraud_data, creditcard_data, ip_to_country)
        
        # Check missing value imputation
        self.assertFalse(fraud_data['purchase_value'].isna().any(), "purchase_value should have no NaN")
        self.assertFalse(fraud_data['sex'].isna().any(), "sex should have no NaN")
        self.assertFalse(creditcard_data['Amount'].isna().any(), "Amount should have no NaN")
        
        # Check data types
        self.assertEqual(fraud_data['signup_time'].dtype, 'datetime64[ns]', "signup_time should be datetime")
        self.assertEqual(fraud_data['ip_address'].dtype, 'Int64', "ip_address should be Int64")
        self.assertEqual(creditcard_data['Time'].dtype, float, "Time should be float")
        
        # Check duplicates
        self.assertEqual(len(fraud_data), 3, "No duplicates should be removed in synthetic data")

    def test_perform_eda(self):
        """Test EDA: check if visualization files are created."""
        fraud_data, creditcard_data, ip_to_country = load_and_check_data()
        perform_eda(fraud_data, creditcard_data)
        expected_plots = [
            'plots/purchase_value_distribution.png',
            'plots/age_distribution.png',
            'plots/source_distribution.png',
            'plots/class_distribution_fraud.png',
            'plots/amount_distribution.png',
            'plots/class_distribution_creditcard.png',
            'plots/purchase_value_vs_class.png',
            'plots/age_vs_class.png',
            'plots/amount_vs_class.png',
            'plots/correlation_heatmap_creditcard.png'
        ]
        for plot in expected_plots:
            self.assertTrue(os.path.exists(plot), f"{plot} should exist")

    def test_merge_geolocation(self):
        """Test geolocation merging."""
        fraud_data, _, ip_to_country = load_and_check_data()
        fraud_data = merge_geolocation(fraud_data, ip_to_country)
        self.assertTrue('country' in fraud_data.columns, "country column should be added")
        self.assertEqual(fraud_data['country'].iloc[0], 'USA', "IP 123456 should map to USA")
        self.assertEqual(fraud_data['country'].iloc[2], 'Unknown', "IP 345678 should map to Unknown")

    def test_engineer_features(self):
        """Test feature engineering, including device_id-based velocity."""
        fraud_data, _, ip_to_country = load_and_check_data()
        fraud_data = merge_geolocation(fraud_data, ip_to_country)
        fraud_data = engineer_features(fraud_data)
        
        # Check engineered features
        self.assertTrue('user_transaction_count' in fraud_data.columns, "user_transaction_count should exist")
        self.assertTrue('device_transaction_count' in fraud_data.columns, "device_transaction_count should exist")
        self.assertTrue('avg_time_between_transactions' in fraud_data.columns, "avg_time_between_transactions should exist")
        self.assertTrue('hour_of_day' in fraud_data.columns, "hour_of_day should exist")
        self.assertTrue('day_of_week' in fraud_data.columns, "day_of_week should exist")
        self.assertTrue('time_since_signup' in fraud_data.columns, "time_since_signup should exist")
        
        # Check device_id-based velocity
        device1_rows = fraud_data[fraud_data['device_id'] == 'device1']
        time_diff = (pd.to_datetime(device1_rows['purchase_time'].iloc[1]) - 
                     pd.to_datetime(device1_rows['purchase_time'].iloc[0])).total_seconds() / 3600
        expected_velocity = time_diff  # Only one time difference for device1
        self.assertAlmostEqual(device1_rows['avg_time_between_transactions'].iloc[0], expected_velocity, 
                               places=2, msg="avg_time_between_transactions for device1 incorrect")
        
        # Check NaN handling
        self.assertFalse(fraud_data['avg_time_between_transactions'].isna().any(), "avg_time_between_transactions should have no NaN")
        self.assertFalse(fraud_data['time_since_signup'].isna().any(), "time_since_signup should have no NaN")
        
        # Check visualization files
        self.assertTrue(os.path.exists('plots/time_since_signup_vs_class.png'), "time_since_signup_vs_class.png should exist")
        self.assertTrue(os.path.exists('plots/hour_of_day_vs_class.png'), "hour_of_day_vs_class.png should exist")
        self.assertTrue(os.path.exists('plots/avg_time_between_transactions_vs_class.png'), 
                        "avg_time_between_transactions_vs_class.png should exist")

    def test_transform_data(self):
        """Test data transformation and saving to ./data/processed/."""
        fraud_data, creditcard_data, ip_to_country = load_and_check_data()
        fraud_data = merge_geolocation(fraud_data, ip_to_country)
        fraud_data = engineer_features(fraud_data)
        X_fraud, y_fraud, X_credit, y_credit = transform_data(fraud_data, creditcard_data)
        
        # Check no NaN values
        self.assertFalse(X_fraud.isna().any().any(), "X_fraud should have no NaN")
        self.assertFalse(X_credit.isna().any().any(), "X_credit should have no NaN")
        
        # Check categorical encoding
        self.assertTrue(any(col.startswith('source_') for col in X_fraud.columns), "source should be one-hot encoded")
        self.assertTrue(any(col.startswith('browser_') for col in X_fraud.columns), "browser should be one-hot encoded")
        self.assertTrue(any(col.startswith('country_') for col in X_fraud.columns), "country should be one-hot encoded")
        
        # Check scaling
        self.assertAlmostEqual(X_fraud['purchase_value'].mean(), 0, places=1, msg="purchase_value should be standardized (mean ~0)")
        self.assertAlmostEqual(X_fraud['purchase_value'].std(), 1, places=1, msg="purchase_value should be standardized (std ~1)")
        
        # Check saved files
        expected_files = [
            'data/processed/X_fraud.csv',
            'data/processed/y_fraud.csv',
            'data/processed/X_credit.csv',
            'data/processed/y_credit.csv'
        ]
        for file in expected_files:
            self.assertTrue(os.path.exists(file), f"{file} should exist")
        
        # Verify content of saved files
        X_fraud_saved = pd.read_csv('data/processed/X_fraud.csv')
        self.assertEqual(X_fraud.shape, X_fraud_saved.shape, "Saved X_fraud shape should match")

if __name__ == '__main__':
    unittest.main()