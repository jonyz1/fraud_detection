
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

# Create directories if they don't exist
if not os.path.exists('plots'):
    os.makedirs('plots')
if not os.path.exists('data/processed'):
    os.makedirs('data/processed')

# Set random seed for reproducibility
np.random.seed(42)

# -----------------------------------
# 1. Data Loading and Initial Checks
# -----------------------------------
def load_and_check_data():
    """Load datasets and print basic information to verify structure."""
    try:
        filepath='./data/raw/'
        fraud_data = pd.read_csv(filepath+'Fraud_Data.csv')
        creditcard_data = pd.read_csv(filepath+'creditcard.csv')
        ip_to_country = pd.read_csv(filepath+'IpAddress_to_Country.csv')
    except FileNotFoundError as e:
        print(f"Error: Dataset file not found. Ensure 'Fraud_Data.csv', 'creditcard.csv', and 'IpAddress_to_Country.csv' are in the root directory.")
        raise e
    
    print("Fraud Data Info:")
    print(fraud_data.info())
    print("\nCredit Card Data Info:")
    print(creditcard_data.info())
    print("\nIP to Country Info:")
    print(ip_to_country.info())
    
    print("\nFraud Data Missing Values:\n", fraud_data.isna().sum())
    print("\nCredit Card Data Missing Values:\n", creditcard_data.isna().sum())
    print("\nIP to Country Missing Values:\n", ip_to_country.isna().sum())
    
    return fraud_data, creditcard_data, ip_to_country

# -----------------------------------
# 2. Data Cleaning
# -----------------------------------
def clean_data(fraud_data, creditcard_data, ip_to_country):
    """Handle missing values, remove duplicates, and correct data types."""
    # Remove duplicates
    initial_rows_fraud = len(fraud_data)
    initial_rows_credit = len(creditcard_data)
    fraud_data = fraud_data.drop_duplicates()
    creditcard_data = creditcard_data.drop_duplicates()
    print(f"\nDuplicates Removed: Fraud Data ({initial_rows_fraud - len(fraud_data)} rows), Credit Card Data ({initial_rows_credit - len(creditcard_data)} rows)")
    
    # Handle missing values
    for col in ['purchase_value', 'age']:
        if fraud_data[col].isna().any():
            fraud_data[col].fillna(fraud_data[col].median(), inplace=True)
            print(f"Imputed {col} with median: {fraud_data[col].median()}")
    for col in ['source', 'browser', 'sex']:
        if fraud_data[col].isna().any():
            fraud_data[col].fillna(fraud_data[col].mode()[0], inplace=True)
            print(f"Imputed {col} with mode: {fraud_data[col].mode()[0]}")
    
    if creditcard_data['Amount'].isna().any():
        creditcard_data['Amount'].fillna(creditcard_data['Amount'].median(), inplace=True)
        print(f"Imputed Amount with median: {creditcard_data['Amount'].median()}")
    
    # Correct data types
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'], errors='coerce')
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'], errors='coerce')
    fraud_data['ip_address'] = fraud_data['ip_address'].astype(float).round().astype('Int64', errors='ignore')
    fraud_data['user_id'] = fraud_data['user_id'].astype(str)
    fraud_data['device_id'] = fraud_data['device_id'].astype(str)
    fraud_data['source'] = fraud_data['source'].astype(str)
    fraud_data['browser'] = fraud_data['browser'].astype(str)
    fraud_data['sex'] = fraud_data['sex'].astype(str)
    fraud_data['class'] = fraud_data['class'].astype(int, errors='ignore')
    
    creditcard_data['Time'] = creditcard_data['Time'].astype(float)
    creditcard_data['Amount'] = creditcard_data['Amount'].astype(float)
    creditcard_data['Class'] = creditcard_data['Class'].astype(int, errors='ignore')
    
    ip_to_country['lower_bound_ip_address'] = ip_to_country['lower_bound_ip_address'].astype(float).round().astype('Int64', errors='ignore')
    ip_to_country['upper_bound_ip_address'] = ip_to_country['upper_bound_ip_address'].astype(float).round().astype('Int64', errors='ignore')
    
    print("\nFraud Data Types After Cleaning:\n", fraud_data.dtypes)
    print("\nCredit Card Data Types After Cleaning:\n", creditcard_data.dtypes)
    
    return fraud_data, creditcard_data, ip_to_country

# -----------------------------------
# 3. Exploratory Data Analysis (EDA)
# -----------------------------------
def perform_eda(fraud_data, creditcard_data):
    """Perform univariate and bivariate analysis with visualizations and fraud pattern insights."""
    # Univariate Analysis: Fraud_Data.csv
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    sns.histplot(fraud_data['purchase_value'], bins=50, kde=True)
    plt.title('Distribution of Purchase Value')
    plt.savefig('plots/purchase_value_distribution.png')
    plt.subplot(2, 2, 2)
    sns.histplot(fraud_data['age'], bins=50, kde=True)
    plt.title('Distribution of Age')
    plt.savefig('plots/age_distribution.png')
    plt.subplot(2, 2, 3)
    sns.countplot(x='source', data=fraud_data)
    plt.title('Source Distribution')
    plt.savefig('plots/source_distribution.png')
    plt.subplot(2, 2, 4)
    sns.countplot(x='class', data=fraud_data)
    plt.title('Class Distribution (Fraud_Data)')
    plt.savefig('plots/class_distribution_fraud.png')
    plt.tight_layout()
    plt.close()
    
    # Univariate Analysis: creditcard.csv
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(creditcard_data['Amount'], bins=50, kde=True)
    plt.title('Distribution of Amount')
    plt.savefig('plots/amount_distribution.png')
    plt.subplot(1, 2, 2)
    sns.countplot(x='Class', data=creditcard_data)
    plt.title('Class Distribution (Credit Card)')
    plt.savefig('plots/class_distribution_creditcard.png')
    plt.tight_layout()
    plt.close()
    
    # Bivariate Analysis: Fraud_Data.csv
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='class', y='purchase_value', data=fraud_data)
    plt.title('Purchase Value vs. Class')
    plt.savefig('plots/purchase_value_vs_class.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='class', y='age', data=fraud_data)
    plt.title('Age vs. Class')
    plt.savefig('plots/age_vs_class.png')
    plt.close()
    
    # Bivariate Analysis: creditcard.csv
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Class', y='Amount', data=creditcard_data)
    plt.title('Amount vs. Class')
    plt.savefig('plots/amount_vs_class.png')
    plt.close()
    
    # Correlation Heatmap: creditcard.csv
    plt.figure(figsize=(12, 8))
    sns.heatmap(creditcard_data.corr(), cmap='coolwarm', annot=False)
    plt.title('Correlation Heatmap (Credit Card Data)')
    plt.savefig('plots/correlation_heatmap_creditcard.png')
    plt.close()
    
    # Fraud Pattern Insights
    print("\nFraud Pattern Insights:")
    fraud_purchase_median = fraud_data.groupby('class')['purchase_value'].median()
    fraud_age_median = fraud_data.groupby('class')['age'].median()
    credit_amount_median = creditcard_data.groupby('Class')['Amount'].median()
    print(f"- Fraud_Data.csv: Median purchase value for fraudulent (class=1): {fraud_purchase_median[1]:.2f}, non-fraudulent (class=0): {fraud_purchase_median[0]:.2f}")
    print(f"- Fraud_Data.csv: Median age for fraudulent (class=1): {fraud_age_median[1]:.2f}, non-fraudulent (class=0): {fraud_age_median[0]:.2f}")
    print(f"- Creditcard.csv: Median amount for fraudulent (Class=1): {credit_amount_median[1]:.2f}, non-fraudulent (Class=0): {credit_amount_median[0]:.2f}")
    print("- Fraudulent transactions in Fraud_Data.csv often occur at unusual hours (visualized in feature engineering).")
    print("- Severe class imbalance observed: Further analysis in class imbalance section.")

# -----------------------------------
# 4. Merge Datasets for Geolocation Analysis
# -----------------------------------
def merge_geolocation(fraud_data, ip_to_country):
    """Merge Fraud_Data.csv with IpAddress_to_Country.csv for geolocation analysis."""
    def map_ip_to_country(ip):
        try:
            ip = int(ip)
            match = ip_to_country[(ip_to_country['lower_bound_ip_address'] <= ip) & 
                                  (ip_to_country['upper_bound_ip_address'] >= ip)]
            return match['country'].iloc[0] if not match.empty else 'Unknown'
        except (ValueError, TypeError):
            return 'Unknown'
    
    fraud_data['country'] = fraud_data['ip_address'].apply(map_ip_to_country)
    print("\nGeolocation Merge Summary:")
    print(fraud_data['country'].value_counts().head())
    print(f"Unknown countries: {fraud_data['country'].eq('Unknown').sum()}")
    
    return fraud_data

# -----------------------------------
# 5. Feature Engineering
# -----------------------------------
def engineer_features(fraud_data):
    """Engineer features for fraud detection with transaction velocity based on device_id."""
    # Transaction frequency
    fraud_data['user_transaction_count'] = fraud_data.groupby('user_id')['user_id'].transform('count')
    fraud_data['device_transaction_count'] = fraud_data.groupby('device_id')['device_id'].transform('count')
    print("\nFeature Engineering: Transaction Frequency")
    print(f"Median user transaction count: {fraud_data['user_transaction_count'].median()}")
    print(f"Median device transaction count: {fraud_data['device_transaction_count'].median()}")
    
    # Transaction velocity (based on device_id)
    fraud_data = fraud_data.sort_values(['device_id', 'purchase_time'])
    fraud_data['time_diff'] = fraud_data.groupby('device_id')['purchase_time'].diff().dt.total_seconds() / 3600
    fraud_data['avg_time_between_transactions'] = fraud_data.groupby('device_id')['time_diff'].transform('mean')
    fraud_data['avg_time_between_transactions'].fillna(fraud_data['avg_time_between_transactions'].median(), inplace=True)
    print("\nFeature Engineering: Transaction Velocity")
    print(f"Median avg_time_between_transactions: {fraud_data['avg_time_between_transactions'].median():.2f} hours")
    
    # Time-based features
    fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
    fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
    fraud_data['time_since_signup'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds() / 3600
    fraud_data['time_since_signup'].fillna(fraud_data['time_since_signup'].median(), inplace=True)
    
    # Visualize time-based features vs. class
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='class', y='time_since_signup', data=fraud_data)
    plt.title('Time Since Signup vs. Class')
    plt.savefig('plots/time_since_signup_vs_class.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x='hour_of_day', hue='class', data=fraud_data)
    plt.title('Hour of Day vs. Class')
    plt.savefig('plots/hour_of_day_vs_class.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='class', y='avg_time_between_transactions', data=fraud_data)
    plt.title('Average Time Between Transactions (Device) vs. Class')
    plt.savefig('plots/avg_time_between_transactions_vs_class.png')
    plt.close()
    
    print("\nFeature Engineering: Time-Based Features")
    print(f"Median time_since_signup: {fraud_data['time_since_signup'].median():.2f} hours")
    
    # Logic for Features
    print("\nFeature Engineering Logic:")
    print("- user_transaction_count: High counts may indicate automated fraud, though limited by unique user_ids.")
    print("- device_transaction_count: Multiple transactions from one device could signal fraud.")
    print("- avg_time_between_transactions: Rapid transactions on a device suggest bot activity.")
    print("- hour_of_day, day_of_week: Fraud may occur at unusual times (e.g., late night).")
    print("- time_since_signup: Short signup-to-purchase time may indicate account takeover.")
    
    return fraud_data

# -----------------------------------
# 6. Data Transformation
# -----------------------------------
def transform_data(fraud_data, creditcard_data):
    """Encode categorical features, scale numerical features, analyze class imbalance, and save transformed datasets."""
    # Encode categorical features for Fraud_Data.csv
    categorical_cols = ['source', 'browser', 'sex', 'country']
    fraud_data_encoded = pd.get_dummies(fraud_data, columns=categorical_cols, drop_first=True)
    
    # Select features
    X_fraud = fraud_data_encoded.drop(['class', 'signup_time', 'purchase_time', 'user_id', 'device_id', 'ip_address', 'time_diff'], axis=1)
    y_fraud = fraud_data_encoded['class']
    X_credit = creditcard_data.drop('Class', axis=1)
    y_credit = creditcard_data['Class']
    
    # Impute NaN values
    num_imputer = SimpleImputer(strategy='median')
    X_fraud = pd.DataFrame(num_imputer.fit_transform(X_fraud), columns=X_fraud.columns)
    X_credit = pd.DataFrame(num_imputer.fit_transform(X_credit), columns=X_credit.columns)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_fraud = pd.DataFrame(scaler.fit_transform(X_fraud), columns=X_fraud.columns)
    X_credit = pd.DataFrame(scaler.fit_transform(X_credit), columns=X_credit.columns)
    
    # Analyze class imbalance
    print("\nClass Imbalance Analysis:")
    print("Fraud Data Class Distribution:\n", y_fraud.value_counts(normalize=True))
    print("Credit Card Data Class Distribution:\n", y_credit.value_counts(normalize=True))
    
    # Strategy for class imbalance
    print("\nClass Imbalance Strategy:")
    print("- Severe imbalance: ~2% fraud in Fraud_Data.csv, ~0.17% in creditcard.csv.")
    print("- Recommended: SMOTE to generate synthetic fraudulent samples in Task 2.")
    print("- Justification: SMOTE preserves data size, creates realistic samples, and avoids overfitting compared to random oversampling.")
    print("- Note: SMOTE deferred to Task 2 (training data only) to avoid data leakage.")
    
    # Save transformed datasets to ./data/processed/
    X_fraud.to_csv('data/processed/X_fraud.csv', index=False)
    y_fraud.to_csv('data/processed/y_fraud.csv', index=False)
    X_credit.to_csv('data/processed/X_credit.csv', index=False)
    y_credit.to_csv('data/processed/y_credit.csv', index=False)
    print("\nTransformed Datasets Saved: data/processed/X_fraud.csv, data/processed/y_fraud.csv, data/processed/X_credit.csv, data/processed/y_credit.csv")
    
    return X_fraud, y_fraud, X_credit, y_credit

# -----------------------------------
# Main Execution
# -----------------------------------
def main():
    """Execute all Task 1 steps for data analysis and preprocessing."""
    print("Starting Task 1: Data Analysis and Preprocessing")
    
    # Load data
    fraud_data, creditcard_data, ip_to_country = load_and_check_data()
    
    # Clean data
    fraud_data, creditcard_data, ip_to_country = clean_data(fraud_data, creditcard_data, ip_to_country)
    
    # Perform EDA
    perform_eda(fraud_data, creditcard_data)
    
    # Merge geolocation data
    fraud_data = merge_geolocation(fraud_data, ip_to_country)
    
    # Engineer features
    fraud_data = engineer_features(fraud_data)
    
    # Transform data and analyze class imbalance
    X_fraud, y_fraud, X_credit, y_credit = transform_data(fraud_data, creditcard_data)
    
    print("\nTask 1 Completed: Data cleaned, features engineered, and transformed datasets saved to ./data/processed/.")

if __name__ == "__main__":
    main()
