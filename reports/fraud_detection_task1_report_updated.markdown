## Task 1: Data Analysis and Preprocessing

### Data Cleaning and Preprocessing
- **Missing Values**: Checked for `NaN` values in all datasets (`Fraud_Data.csv`, `creditcard.csv`, `IpAddress_to_Country.csv`). No missing values were found initially, but imputation logic was implemented for robustness:
  - Numerical columns (`purchase_value`, `age`, `Amount`): Imputed with median to handle potential outliers.
  - Categorical columns (`source`, `browser`, `sex`): Imputed with mode.
  - Engineered features (`avg_time_between_transactions`, `time_since_signup`): Imputed `NaN` values with median to resolve `ValueError: Input X contains NaN` for SMOTE.
- **Duplicates**: Removed duplicates using `pandas.DataFrame.drop_duplicates()`. No duplicates were found.
- **Data Types**: Corrected to ensure compatibility:
  - `Fraud_Data.csv`: `signup_time` and `purchase_time` as `datetime`, `ip_address` as `int`, `class` as `int`, `user_id` and `device_id` as `str`.
  - `creditcard.csv`: `Time` and `Amount` as `float`, `Class` as `int`.
  - `IpAddress_to_Country.csv`: `lower_bound_ip_address` and `upper_bound_ip_address` as `int`.

**Code Accuracy**: The `clean_data` function ensures no `NaN` values or incorrect data types persist, with diagnostic prints to verify dataset integrity.

### Exploratory Data Analysis (EDA)
- **Univariate Analysis**:
  - `Fraud_Data.csv`:
    - `purchase_value`: Right-skewed, median ~$50, with outliers up to $500.
    - `age`: Roughly normal, centered around 20–40 years.
    - `source`: `SEO` is the most common source, followed by `Ads`.
    - `class`: ~2% fraudulent transactions, indicating severe imbalance.
  - `creditcard.csv`:
    - `Amount`: Highly skewed, most transactions <$100, outliers up to $10,000.
    - `Class`: ~0.17% fraudulent transactions, confirming extreme imbalance.
- **Bivariate Analysis**:
  - `Fraud_Data.csv`: Fraudulent transactions have higher median `purchase_value` and occur more frequently at odd hours (e.g., 2–4 AM).
  - `creditcard.csv`: Fraudulent transactions show higher `Amount` values and varying patterns in PCA features (`V1`–`V28`).
- **Visualizations**: Generated and saved in `plots/`:
  - Histograms: `purchase_value`, `age`, `Amount`.
  - Count plots: `source`, `class`, `Class`.
  - Box plot: `purchase_value` vs. `class`.
  - Correlation heatmap: `creditcard.csv`.
- **Fraud Pattern Insights**:
  - Fraudulent transactions in `Fraud_Data.csv` are associated with higher purchase values and unusual transaction times, suggesting automated or opportunistic fraud.
  - In `creditcard.csv`, high `Amount` values are a strong indicator of fraud, with some PCA features showing distinct patterns for fraudulent transactions.
  - Severe class imbalance necessitates techniques like SMOTE to improve model performance.

### Feature Engineering Logic and Implementation
- **Features Engineered**:
  - **Transaction Frequency**:
    - `user_transaction_count`: Number of transactions per `user_id`.
    - `device_transaction_count`: Number of transactions per `device_id`.
    - **Logic**: High counts may indicate bot-driven fraud or account misuse.
  - **Transaction Velocity**:
    - `avg_time_between_transactions`: Average hours between consecutive transactions per `user_id`.
    - **Logic**: Rapid transactions (low average time) suggest automated fraud.
    - Handled `NaN` values (for users with single transactions) by imputing with the median.
  - **Time-Based Features**:
    - `hour_of_day`: Hour of `purchase_time` (0–23).
    - `day_of_week`: Day of `purchase_time` (0 = Monday, 6 = Sunday).
    - `time_since_signup`: Hours between `signup_time` and `purchase_time`.
    - **Logic**: Fraudulent transactions may occur at odd hours or shortly after signup, indicating account takeover.
  - **Geolocation**:
    - `country`: Mapped from `ip_address` using `IpAddress_to_Country.csv`.
    - **Logic**: Transactions from high-risk countries or mismatches with user profiles may signal fraud.
    - Unmatched IPs assigned `'Unknown'` to avoid `NaN`.
- **Implementation**: The `engineer_features` function ensures robust feature creation, with `NaN` handling to prevent errors in SMOTE.

### Handling Class Imbalance
- **Analysis**:
  - `Fraud_Data.csv`: ~2% fraudulent (`class=1`), ~98% non-fraudulent.
  - `creditcard.csv`: ~0.17% fraudulent (`Class=1`), ~99.83% non-fraudulent.
- **Strategy**: Applied SMOTE to the training data only, using `imblearn.over_sampling.SMOTE`.
- **Implementation**:
  - Performed train-test split with `stratify` to maintain class proportions.
  - Imputed `NaN` values using `SimpleImputer` (median strategy) before SMOTE to resolve `ValueError: Input X contains NaN`.
  - Balanced training data to ~50% fraudulent, ~50% non-fraudulent.
- **Verification**: Printed class distributions before and after SMOTE to confirm balancing.

### Code Structure, Functionality, and Documentation
- **Structure**: The script (`task1_preprocessing.py`) is organized into modular functions:
  - `load_and_check_data`: Loads datasets and prints info.
  - `clean_data`: Handles missing values, duplicates, and data types.
  - `perform_eda`: Generates visualizations and fraud pattern insights.
  - `engineer_features`: Creates features with clear logic.
  - `transform_and_balance_data`: Encodes, scales, and balances data.
- **Functionality**: The `main` function orchestrates all steps, ensuring a complete workflow. Outputs (plots, class distributions) are saved or printed for verification.
- **Documentation**: Includes docstrings, in-line comments, and a comprehensive README (`README.md`) with setup and execution instructions.