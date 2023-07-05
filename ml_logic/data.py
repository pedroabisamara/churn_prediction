import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, LabelEncoder


data = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# ------- DATA REVIEW -------

data.columns
data.head()
data.shape
data.dtypes
data.info()
data.describe()
data.isna().sum()

data['gender'].value_counts(normalize=True)
data['SeniorCitizen'].value_counts(normalize=True)
data['Partner'].value_counts(normalize=True)
data['Dependents'].value_counts(normalize=True)
data['Churn'].value_counts(normalize=True)

# print the unique values from each column of 'data'
for col in data.columns:
    print(col, data[col].unique())

# ------- EXPLORATORY DATA ANALYSIS -------

# plot a histogram of the 'tenure' column using seaborn
sns.histplot(data['tenure'], kde=True, bins=30);

# plot a histogram of the 'MonthlyCharges' column using seaborn
sns.histplot(data['MonthlyCharges'], kde=True, bins=30);

# Explorative Data Analysis on 'Churn' column
sns.countplot(data['Churn']);

cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
plt.figure(figsize=(14, 4))

for i, col in enumerate(cols):
    ax = plt.subplot(1, len(cols), i+1)
    data[str(col)].value_counts().plot(kind='bar', color=['C0', 'C1'])
    ax.set_title(f"{col}")

# look into the relationship between cost and customer churn
sns.boxplot(x='Churn', y='MonthlyCharges', data=data)

# analyze the relationship between customer churn and a few other categorical variables
cols = ['InternetService',"TechSupport","OnlineBackup","Contract"]

plt.figure(figsize=(14,4))

for i, col in enumerate(cols):
    ax = plt.subplot(1, len(cols), i+1)
    sns.countplot(x ="Churn", hue = str(col), data = data)
    ax.set_title(f"{col}")

# identify ouutlier in the tenure column
sns.boxplot(data['tenure'])

# identify ouutlier in the MonthlyCharges column
sns.boxplot(data['MonthlyCharges'])

# ------- DATA CLEANING AND PREPROCESSING -------

# convert the 'TotalCharges' column to a numeric data type.
data['TotalCharges'] = data['TotalCharges'].apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()

# analyzing only the categorical features of 'data'
cat_features = data.select_dtypes(include='object')
cat_features.head()

# setting each categorical feature to its proper encoder
ohe_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity']
oe_features = ['OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
le_features = ['Churn']
le = LabelEncoder()
