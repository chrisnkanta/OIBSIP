#!/usr/bin/env python
# coding: utf-8

# ## Wine Quality Prediction
# 
# ### Table of Contents
# 
# - [Introduction](#intr)
# - [Classifier Models: Utilizing Random Forest, Stochastic Gradient Descent, and Support 
# Vector Classifier (SVC) for wine quality predictio](#class)
# - [Chemical Qualities: Analyzing features like density and acidity as predictors for wine quality](#chemi)
# - [Data Analysis Libraries: Employing Pandas for data manipulation and Numpy for array 
# operation](#data)
# - [Data Visualization: Using Seaborn and Matplotlib for visualizing patterns and insights in the 
# datase](#visua)...

# ### Introduction
# 
# 
# The focus is on predicting the quality of wine based on its chemical characteristics, offering a 
# real-world application of machine learning in the context of viticulture. The datase 
# encompasses diverse chemical attributes, including density and acidity, whicareas t e
# featuroffor three distinct classifier models.

# In[30]:


# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC


# #### Data Collection

# In[2]:


# Loading Dataset
wine_qty = pd.read_csv ('C:/Users/CHRIS/Documents/Oasis Infobyte Internship Project/Project 2 of 2/WineQT.csv')
wine_qty


# In[10]:


# Checking for information in the Dataset
wine_qty.info()


# In[12]:


# Checking for Description in the Dataset
wine_qty.describe


# In[16]:


# Checking for shape in the Dataset
wine_qty.shape


# In[14]:


# Checking for missing values in the Dataset
wine_qty.isnull().sum()


# In[20]:


# Checking for Duplicates in the Dataset
wine_qty.duplicated().sum()


# #### Classifier Models

# In[19]:


# Quality is the target variable
target_variable = 'quality'
X = wine_qty.drop(columns=[target_variable])
y = wine_qty[target_variable]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the models
random_forest = RandomForestClassifier(random_state=42)
sgd = SGDClassifier(random_state=42)
svc = SVC(random_state=42)

# Train and evaluate Random Forest
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
print("Random Forest Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, zero_division=0))

# Train and evaluate Stochastic Gradient Descent (SGD)
sgd.fit(X_train, y_train)
y_pred_sgd = sgd.predict(X_test)
print("\nStochastic Gradient Descent Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_sgd, zero_division=0))

# Train and evaluate Support Vector Classifier (SVC)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
print("\nSupport Vector Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc, zero_division=0))


# #### Chemical Qualities

# In[36]:


# Select the features of interest: density and acidity (pH)
features = wine_qty[['density', 'fixed acidity']]  
target = wine_qty['quality']

# Perform EDA
# Pairplot to visualize relationships
sns.pairplot(wine_qty, x_vars=['density', 'fixed acidity'], y_vars='quality', height=5, aspect=0.7, kind='scatter')
plt.show()

# Correlation between density, acidity, and quality
corr_matrix = wine_qty[['density', 'fixed acidity', 'quality']].corr()
print("\nCorrelation matrix:")
print(corr_matrix)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Train a simple Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# #### Data Analysis Libraries

# In[42]:


# Calculate mean, median, and standard deviation of 'density' and 'fixed acidity'
density_mean = wine_qty['density'].mean()
density_median = wine_qty['density'].median()
density_std = wine_qty['density'].std()

fixed_acidity_mean = wine_qty['fixed acidity'].mean()
fixed_acidity_median = wine_qty['fixed acidity'].median()
fixed_acidity_std = wine_qty['fixed acidity'].std()

print(f"\nDensity - Mean: {density_mean}, Median: {density_median}, Std: {density_std}")
print(f"Fixed Acidity - Mean: {fixed_acidity_mean}, Median: {fixed_acidity_median}, Std: {fixed_acidity_std}")

# filter data
# Select rows where wine quality is greater than or equal to 7
high_quality_wines = wine_qty[wine_qty['quality'] >= 7]
print("\nHigh quality wines (quality >= 7):")
print(high_quality_wines.head())

# Add a new column 'acidity_density_ratio' calculated as the ratio of 'fixed acidity' to 'density'
wine_qty['acidity_density_ratio'] = wine_qty['fixed acidity'] / wine_qty['density']
print("\nData with new 'acidity_density_ratio' column:")
print(wine_qty.head())


# Convert the 'density' column to a NumPy array and perform operations
density_array = wine_qty['density'].to_numpy()

# Calculate the logarithm of the 'density' values
log_density_array = np.log(density_array)
print("\nLogarithm of density values:")
print(log_density_array[:5])  # Display the first 5 log-transformed values

# Perform an element-wise addition of fixed acidity and density
acidity_density_sum = np.add(wine_qty['fixed acidity'].to_numpy(), wine_qty['density'].to_numpy())
print("\nElement-wise addition of fixed acidity and density:")
print(acidity_density_sum[:5])  # Display the first 5 results

# Calculate the mean and standard deviation of the resulting array
sum_mean = np.mean(acidity_density_sum)
sum_std = np.std(acidity_density_sum)
print(f"\nMean of acidity_density_sum: {sum_mean}, Std: {sum_std}")


# #### Data Visualization

# In[66]:


# Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = wine_qty.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title('Correlation Heatmap of Wine Dataset')
plt.show()

# Distribution Plot for Quality
plt.figure(figsize=(8, 6))
sns.countplot(x='quality', data=wine_qty)
plt.title('Distribution of Wine Quality')
plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.show()

# Distribution Plot for Alcohol Content
plt.figure(figsize=(8, 6))
sns.histplot(wine_qty['alcohol'], bins=15, kde=True, color='blue')
plt.title('Distribution of Alcohol Content')
plt.xlabel('Alcohol Content')
plt.ylabel('Density')
plt.show()

# Distribution Plot for pH Levels
plt.figure(figsize=(8, 6))
sns.histplot(wine_qty['pH'], bins=15, kde=True, color='green')
plt.title('Distribution of pH Levels')
plt.xlabel('pH')
plt.ylabel('Density')
plt.show()

# Pair Plot for Selected Features
selected_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'alcohol', 'quality']
sns.pairplot(wine_qty[selected_features], hue='quality', palette='husl')
plt.suptitle('Pair Plot of Selected Features', y=1.02)
plt.show()

