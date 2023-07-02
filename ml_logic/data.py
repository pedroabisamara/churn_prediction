import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

data.shape
data.dtypes
data.info()
data.describe()
data.isnull().sum()

# print the unique values from each column of 'data'
for col in data.columns:
    print(col, data[col].unique())
