#%%#

### Step 1  ######################################################################
### Import essential library and load the data

# Import pandas, numpy, and etc.
import openpyxl 
import numpy as np
import pandas as pd
import seaborn as sns
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Read the data and understanding our data set
product = pd.read_csv("Products_Information.csv")


# Using info() and head() functions to briefly read the overall structure
product_info = product.info()
product_describe = product.describe()
product_rows = product.head()


# Check our data set
print(product_info)
print("------------------------------------------------")
print(" ")
print(product_describe)
print("------------------------------------------------")
print(" ")
print(product_rows)
###-------------------------------------------------------------------------------



#%%#
### Step 2  ######################################################################
### Data pre-processing, such as handle Missing Values and etc.

# Deal with missing value
missing_values = product.isnull().sum()


# Convert the 'date' column to datetime format and set it as the index
product['date'] = pd.to_datetime(product['date'])
product.set_index('date', inplace=True)


# Check the latest processing result
print(missing_values)
print("------------------------------------------------")
print(" ")
print(product.head())

# Save processed data
# product.to_csv("processed_product_data.csv")
###-------------------------------------------------------------------------------



#%%#
### Step 3  ######################################################################
### Overall EDA processing and Graph plotting

# Setting aesthetics for plots
sns.set(style="darkgrid")
sns.set_theme()
sns.set_context("talk", font_scale=0.8, rc={"lines.linewidth": 1})

# Plotting the overall special_offer over whole time period
# product['special_offer'] = product['special_offer'].astype('int')

# Plot sales over all time (supplementary)
# Aggregating sales data
overall_sales = product.groupby('date')['sales'].sum()
# Creating the plot
plt.figure(figsize=(14, 7))
plt.plot(overall_sales, label='Total Sales')
plt.title('Overall sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()
plt.show()

# Summarize the stats of sales
print(product['sales'].describe())

# Distribution of sales
plt.figure(figsize=(14, 7))
sns.histplot(product['sales'], bins=50, kde=True)
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.xlim(0, 15000) 
plt.show()

# Sales by Store (important)
plt.figure(figsize=(14, 7))
sns.boxplot(x='store_nbr', y='sales', data=product)
plt.title('Sales by Store')
plt.xlabel('Store Number')
plt.ylabel('Sales')
plt.show()

# Sales by Product
plt.figure(figsize=(14, 7))
sns.boxplot(x='product_type', y='sales', data=product)
plt.title('Sales by Product')
plt.xlabel('Product Type')
plt.ylabel('Sales') 
plt.xticks(rotation=90)
plt.show()
###-------------------------------------------------------------------------------



#%%#

### Step 4-1  ####################################################################
### Creating the randomforest regressor for the forecasting

# Encoding categorical variables
product = pd.get_dummies(product, columns=['product_type'], drop_first=True)

# Define the feature columns (including new dummy columns) and the target column
features = ['store_nbr', 'special_offer', 'id'] + [col for col in product.columns if 'product_type_' in col]
target = 'sales'

# Splitting the data
train_end_date = '2016-12-31'
validation_end_date = '2017-07-30'
test_start_date = '2017-07-31'

train_data = product[:train_end_date]
validation_data = product[train_end_date:validation_end_date]
test_data = product[test_start_date:'2017-08-15']

# Preparing the training, validation, and testing sets
X_train = train_data[features]
y_train = train_data[target]

X_validation = validation_data[features]
y_validation = validation_data[target]

X_test = test_data[features]
y_test = test_data[target]

# Initialize and train the RandomForest model
model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# You can use validation data here to tune your model parameters or for early stopping

# Predict the sales on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error on the test set
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Set:", mse)

# Plotting the predictions
plt.figure(figsize=(12,6))
plt.plot(y_test.index, y_pred, color='blue', label='Predicted Sales')
plt.plot(y_test.index, y_test, color='red', label='Actual Sales')
plt.title('Sales Forecast vs Actuals on Test Set')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
###-------------------------------------------------------------------------------



# %%
