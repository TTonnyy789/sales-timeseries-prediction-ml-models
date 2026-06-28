# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:06:23 2023

@author: sharo
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, r2_score

# Assuming you have prepared your dataset in a similar way as in the previous code
product = pd.read_csv('../Products_Information.csv')
# Convert the 'date' column to datetime format and set it as the index
product['date'] = pd.to_datetime(product['date'])
product.set_index('date', inplace=True)

# Ensure 'product_type' is of categorical data type
product['product_type'] = product['product_type'].astype('category')
# Build a dictionary for the store-product grouping 

# Separate the data from the answer
start_date = '2013-01-01'
end_date = '2017-07-30'
filtered = product[(product.index >= start_date) & (product.index <= end_date)]

segmented_data = {}
# Grouping the data by store and product, and storing each group in the dictionary
for (store, product_type), group in filtered.groupby(['store_nbr', 'product_type'], observed=True):
    segmented_data[(store, product_type)] = group[['sales', 'special_offer', 'id']]

store1beauty = segmented_data[(1, 'BEAUTY')]

# Assuming X_train, y_train, X_test, y_test are your prepared datasets
train_size = int(len(store1beauty) * 0.8)
val_size = len(store1beauty) - train_size

X = store1beauty.drop(columns=['sales'])  
y = store1beauty['sales']

X_train, X_val = X.iloc[:train_size], X.iloc[val_size:]
y_train, y_val = y.iloc[:train_size], y.iloc[val_size:]

# Initialize the Gradient Boosting Regressor
gb_regressor = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)

# Train the model
gb_regressor.fit(X_train, y_train)

# Predict on validation set
predictions_gb = gb_regressor.predict(X_val)

# Evaluate the model
mse_gb = mean_squared_error(y_val, predictions_gb)
print(f"Mean Squared Error on Validation Set (Gradient Boosting): {mse_gb}")

# Defining forecasting date
future_dates = pd.date_range(start='2017-07-31', end='2017-08-15', freq='D')  # Define future dates

# Create feature data for these future dates (similar to X_train)
# This could involve extracting or generating features for these dates
# Example: Create a DataFrame with columns similar to X_train for the future dates

# Concatenate future dates and feature data into X_future DataFrame
start_date1 = '2017-07-31'
end_date1 = '2017-08-15'
ans = product[(product.index >= start_date1) & (product.index <= end_date1)]
segmented_data_ans = {}
# Grouping the data by store and product, and storing each group in the dictionary
for (store, product_type), group in ans.groupby(['store_nbr', 'product_type'], observed=True):
    segmented_data_ans[(store, product_type)] = group[['sales', 'special_offer', 'id']]

store1beauty_ans = segmented_data_ans[(1, 'BEAUTY')]

X_future = store1beauty_ans.drop(columns=['sales']) 
X_future.index = future_dates 

# Predict using the trained Gradient Boosting Regressor model
future_predictions_gb = gb_regressor.predict(X_future)
future_predictions_gb[0:15]

#%%#

#### Adding external features ####

specific_segment = segmented_data[(1, 'BEAUTY')]
specific_segment_ans = segmented_data_ans[(1, 'BEAUTY')]

# Prepare the features and target variable
X_train_EX = specific_segment[:start_date1][['special_offer', 'id', 'store_nbr']].values
Y_train_EX = specific_segment[:start_date1]['sales']

X_test_EX = specific_segment_ans[start_date1:][['special_offer', 'id', 'store_nbr']].values
Y_test_EX = specific_segment_ans[start_date1:]['sales']

# Initialize the Gradient Boosting Regressor
gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train the model
gb_regressor.fit(X_train_EX, Y_train_EX)

# Make predictions
y_predict = gb_regressor.predict(X_test_EX)

mse_gb = mean_squared_error(Y_test_EX, y_predict)
print(f"Mean Squared Error on Validation Set (Gradient Boosting): {mse_gb}")

# Plotting the forecast alongside the actual test data
plt.figure(figsize=(12, 6))
plt.plot(Y_test_EX.index, y_predict, color='blue', label='Predicted Sales')
plt.plot(Y_test_EX.index, Y_test_EX, color='red', label='Actual Sales')
plt.title("GradientBoostingRegressor Sales Forecast vs Actual Sales for Store1 BEAUTY")
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Evaluating the model's performance
r2_EXfeature_gb = r2_score(Y_test_EX, y_predict)
print("GB r2_score:", r2_EXfeature_gb)

# ['special_offer', 'id', 'store_nbr']
# GB r2_score: -0.19308528432065541
# MSE: 6.24505578511593

# ['special_offer', 'id']
# GB r2_score: -0.22703764250787883
# MSE: 6.422775160002178

# ['special_offer', 'store_nbr']
# GB r2_score: 0.053990631377524134
# MSE: 4.951767788883272

# ['id', 'store_nbr'] # Doesn't make sense
# GB r2_score: -0.37689763516376096
# MSE: 7.207198559060312



















