#!/usr/bin/env python
# coding: utf-8

# ## Predicting House Prices with Linear Regression
# 
# ### Table of Contents
# 
# - [Introduction](#intro)
# - [Data Collection: Obtain a dataset with numerical features and a target variable for predictio](#data)
# - [Data Exploration and Cleaning: Explore the dataset to understand its structure, handle missing values, and ensure data qualit](#explo)
# - [Feature Selection: Identify relevant features that may contribute to the predictive model](#feat)
# - [Model Training: Implement linear regression using a machine learning library (e.g., Scikit-Learn](#mode)
# - [Model Evaluation: Evaluate the model's performance on a separate test dataset using metrics such as Mean Squared Error or R-square](#eval)
# - [Visualization: Create visualizations to illustrate the relationship between the predicted and actual values](#visua)

# ### Introduction
# 
# The objective of this project is to build a predictive model using linear regression to estimate a 
# numerical outcome based on a dataset with relevant features.Linear regression is a fundamental machine learning algorithm, and this project provides hands-on experience in developing, evaluating, and interpreting a predictive model.el.
# 

# In[42]:


# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression


# #### Data Collection.

# In[44]:


# Loading Dataset
Hou_data = pd.read_csv ('C:/Users/CHRIS/Documents/Oasis Infobyte Internship Project/Project 2 of 1/Housing.csv')
Hou_data


# #### Data Exploration and Cleaning

# In[25]:


# Checking for information in the Dataset
Hou_data.info()


# In[27]:


# Checking for Description in the Dataset
Hou_data.describe


# In[29]:


# Checking for Shape in the Dataset
Hou_data.shape


# In[31]:


# Checking for missing values in the Dataset
Hou_data.isnull().sum()


# In[33]:


# Checking for Duplicate Values in the Dataset
Hou_data.duplicated().sum()


# In[35]:


# Checking for Type data in the Dataset
Hou_data.dtypes


# #### Feature Selection

# In[55]:


# Handle categorical variables (convert strings to numerical values)
label_encoders = {}
for column in Hou_data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    Hou_data[column] = le.fit_transform(Hou_data[column])
    label_encoders[column] = le
    
# Select only numeric columns
numeric_data = Hou_data.select_dtypes(include=[float, int])

# Correlation matrix
correlation_matrix = Hou_data.corr()

# Visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Identify the target variable 
target_variable = 'price'

# Separate features (X) and target (y)
X = Hou_data.drop(columns=[target_variable])
y = Hou_data[target_variable]

# Feature selection using SelectKBest
best_features = SelectKBest(score_func=f_regression, k='all')
fit = best_features.fit(X, y)

# Create a dataframe for the scores
feature_scores = pd.DataFrame(fit.scores_, columns=['Score'])
feature_scores['Feature'] = X.columns

# Sort the features by score
feature_scores = feature_scores.sort_values(by='Score', ascending=False)

# Display the top features
print("Top Features:")
print(feature_scores)

# Visualize the top features
plt.figure(figsize=(10, 6))
sns.barplot(x='Score', y='Feature', data=feature_scores)  
plt.title('Feature Importance')
plt.show()


# #### Model Training

# In[59]:


# Handle categorical variables (convert strings to numerical values)
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for column in Hou_data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    Hou_data[column] = le.fit_transform(Hou_data[column])
    label_encoders[column] = le

# Select features (X) and target (y)
# 'Price' is the target variable.
target_variable = 'price'
X = Hou_data.drop(columns=[target_variable])
y = Hou_data[target_variable]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Visualize the relationship between actual and predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()


# #### Model Evaluation

# In[62]:


# Handle categorical variables (convert strings to numerical values)
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for column in Hou_data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    Hou_data[column] = le.fit_transform(Hou_data[column])
    label_encoders[column] = le

# Select features (X) and target (y)
# 'Price' is the target variable.
target_variable = 'price'
X = Hou_data.drop(columns=[target_variable])
y = Hou_data[target_variable]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model's performance using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Evaluate the model's performance using R-squared (R²) score
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R²) Score: {r2}")


# #### Visualization

# In[64]:


# Handle categorical variables (convert strings to numerical values)
label_encoders = {}
for column in Hou_data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    Hou_data[column] = le.fit_transform(Hou_data[column])
    label_encoders[column] = le

# Select features (X) and target (y)
# 'Price' is the target variable.
target_variable = 'price'
X = Hou_data.drop(columns=[target_variable])
y = Hou_data[target_variable]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Visualization 1: Scatter Plot of Actual vs Predicted Values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)  # Line of perfect fit
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

# Visualization 2: Residual Plot
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
sns.residplot(x=y_test, y=residuals, lowess=True, color='purple', line_kws={'color': 'red', 'lw': 2})
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Actual Values')
plt.show()

