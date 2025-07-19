Fraud Detection for E-commerce and Bank Transactions

Project Overview

This repository contains the code and documentation for the Interim-1 Submission of the 10 Academy Artificial Intelligence Mastery Week 8 & 9 Challenge (16 July - 29 July 2025). The project focuses on improving fraud detection for e-commerce transactions (Fraud_Data.csv) and bank credit transactions (creditcard.csv) at Adey Innovations Inc., a financial technology company. The goal of Task 1 is to clean, preprocess, and analyze the datasets, perform exploratory data analysis (EDA), engineer features, and handle class imbalance to prepare for model building.

Objectives

Data Cleaning and Preprocessing: Handle missing values, remove duplicates, and correct data types.

Exploratory Data Analysis (EDA): Analyze feature distributions and relationships with the target variable (class/Class).

Feature Engineering: Create features like transaction frequency, velocity, time-based features, and geolocation data.

Class Imbalance: Address severe class imbalance using SMOTE (Synthetic Minority Oversampling Technique).

Deliverables: A detailed report (fraud_detection_task1_report.md) and organized code for the Interim-1 submission (due 20 July 2025).

Datasets


Fraud_Data.csv: E-commerce transaction data with features like user_id, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address, and class (0 = non-fraudulent, 1 = fraudulent).


IpAddress_to_Country.csv: Maps IP addresses to countries using lower_bound_ip_address, upper_bound_ip_address, and country.


creditcard.csv: Bank transaction data with PCA-transformed features (V1–V28), Time, Amount, and Class (0 = non-fraudulent, 1 = fraudulent).

Repository Structure


task1_preprocessing.ipynb: Jupyter Notebook containing all code for data cleaning, EDA, feature engineering, and class imbalance handling.

plots/: Folder containing visualization images (e.g., class distribution, feature distributions).

fraud_detection_task1_report.md: Detailed report summarizing preprocessing steps, EDA insights, feature engineering, and class imbalance strategy.

requirements.txt: List of Python dependencies.

README.md: This file, providing project overview and setup instructions.