

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
from sklearn import metrics

import sklearn.datasets

"""Importing The Boston House Price Dataset"""

import pandas as pd
import numpy as np

# URL of the dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston"

# Load the raw dataset
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# Combine data from alternating rows
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Create a DataFrame for features
columns = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
    "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]
house_price_dataframe = pd.DataFrame(data, columns=columns)

# Add the target column (house prices)
house_price_dataframe["Price"] = target

# Display the first few rows
print(house_price_dataframe.head())

# checking the number of rows and coloumns
house_price_dataframe.shape

# checking for missing values
house_price_dataframe.isnull().sum()

# satustucal measures of the dataset
house_price_dataframe.describe()

"""Understanding the correlation between various features in the dataset

1.  Positive Correlation
2.  Negative Correlation
"""

correlation=house_price_dataframe.corr()

#constructing the heatmap to understand the correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')

"""Splitting the data and Target"""

X=house_price_dataframe.drop(['Price'],axis=1)
Y=house_price_dataframe['Price']

print(X)
print(Y)

"""Splitting the data into Train data and test data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""Model Training

XGboost Regressor
"""

# loading the model
model=XGBRegressor()

# Training the model with X_train
model.fit(X_train,Y_train)

"""Evaluation

Prediction on Training data
"""

#accuracy for prediction on training data
training_data_prediction=model.predict(X_train)

print(training_data_prediction)

# R squared error
error_score=metrics.r2_score(Y_train,training_data_prediction)

# Mean Absolute error
score_2=metrics.mean_absolute_error(Y_train,training_data_prediction)

print("R squared error : ",error_score)
print("Mean Absolute error : ",score_2)

"""Visualizing the actual prices and predict prices"""

plt.scatter(Y_train,training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()

"""Prediction on Test data"""

test_data_prediction=model.predict(X_test)

error_score=metrics.r2_score(Y_test,test_data_prediction)
score_2=metrics.mean_absolute_error(Y_test,test_data_prediction)
print("R squared error : ",error_score)
print("Mean Absolute error : ",score_2)

