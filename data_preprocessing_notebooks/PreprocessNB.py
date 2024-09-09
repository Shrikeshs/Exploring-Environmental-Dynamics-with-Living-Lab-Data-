# ---------------------------------------
# Importing Libraries and Setting Up Environment
# ---------------------------------------

import boto3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
import mlflow
warnings.filterwarnings('ignore')

# Setting up AWS credentials (Please ensure these are secure and not hardcoded in production)
os.environ['AWS_ACCESS_KEY_ID'] = ''
os.environ['AWS_SECRET_ACCESS_KEY'] = ''

# ---------------------------------------
# Loading Data from S3
# ---------------------------------------

s3_bucket_name = 'leedsairqualitybucket'
s3_file_key = 'combined_file_draftfin.csv'
s3_uri = f's3://{s3_bucket_name}/{s3_file_key}'

# Read CSV file from S3 into a DataFrame
df_raw = pd.read_csv(s3_uri, storage_options={'key': os.environ['AWS_ACCESS_KEY_ID'], 'secret': os.environ['AWS_SECRET_ACCESS_KEY']})
print(df_raw.head())

# Set up MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# ---------------------------------------
# Data Preprocessing
# ---------------------------------------

# Checking the shape of the dataset
print("No of rows in dataset:", df_raw.shape[0])
print("No of columns in dataset:", df_raw.shape[1])
print("\nColumn Types:")
print(df_raw.dtypes)

# Selecting specific columns for analysis
relevant_columns = ['Timestamp', 'PM0.1', 'PM0.3', 'PM0.5', 'PM1.0', 'PM2.5', 'PM5.0',
                    'PM10.0', 'PC0.1', 'PC0.3', 'PC0.5', 'PC1.0', 'PC2.5', 'PC5.0',
                    'PC10.0', 'Temp AQ', 'Humidity AQ', 'Pressure AQ', 'VOC AQ', 
                    'No2 Gas', 'Solar Panel Power', 'CO2', 'NOX AQ']
df_selected = df_raw[relevant_columns]

# Filling missing values with the mean of each column
for col in relevant_columns:
    if col != 'Timestamp':
        df_selected[col] = df_selected[col].fillna(df_selected[col].mean())

# Dropping the Timestamp column for further analysis
df_no_timestamp = df_selected.drop(columns=['Timestamp'])

# ---------------------------------------
# Outlier Detection and Handling
# ---------------------------------------

# Calculating Q1, Q3, and IQR for outlier detection
Q1 = df_no_timestamp.quantile(0.25)
Q3 = df_no_timestamp.quantile(0.75)
IQR = Q3 - Q1

# Creating a mask for outliers
outlier_mask = (df_no_timestamp < (Q1 - 1.5 * IQR)) | (df_no_timestamp > (Q3 + 1.5 * IQR))

# Plotting boxplots to visualize outliers
plt.figure(figsize=(20, 10))
sns.boxplot(data=df_no_timestamp, orient='h')
plt.title('Boxplots of Features')
plt.show()

# Handling outliers by filling with the median
df_no_timestamp.fillna(df_no_timestamp.median(), inplace=True)

# Reattaching the 'Timestamp' column after cleaning
df_cleaned = pd.concat([df_selected['Timestamp'], df_no_timestamp], axis=1)

# Displaying the cleaned DataFrame
print(df_cleaned.head())

# ---------------------------------------
# Feature Scaling and Sequence Creation
# ---------------------------------------

from sklearn.preprocessing import MinMaxScaler

# Scaling the features using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_cleaned)
