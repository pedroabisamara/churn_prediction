import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

data.columns
data.head()
data.shape
data.dtypes
data.info()
data.describe()
data.isna().sum()

# convert the 'TotalCharges' column to a numeric data type.
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

data['gender'].value_counts(normalize=True)
data['SeniorCitizen'].value_counts(normalize=True)
data['Partner'].value_counts(normalize=True)
data['Dependents'].value_counts(normalize=True)
data['Churn'].value_counts(normalize=True)

# print the unique values from each column of 'data'
for col in data.columns:
    print(col, data[col].unique())

# plot a histogram of the 'tenure' column using seaborn
sns.distplot(data['tenure'], kde=False, bins=30);

# plot a histogram of the 'MonthlyCharges' column using seaborn
sns.distplot(data['MonthlyCharges'], kde=False, bins=30);

# plot a histogram of the 'TotalCharges' column using seaborn
sns.distplot(data['TotalCharges'], kde=False, bins=30);

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
