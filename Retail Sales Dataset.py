#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Data Analysis (EDA) on Retail Sales Data
# <a id = "table-of-contents"></a>
# ### Table of Content
# 
# - [Introduction](#intro)
# - [Data Loading and Cleaning: Load the retail sales dataset](#loading)
# - [Descriptive Statistics: Calculate basic statistics (mean, median, mode, standard deviation)](#desci)
# - [Time Series Analysis: Analyze sales trends over time using time series techniques](#analysis)
# - [Customer and Product Analysis: Analyze customer demographics and purchasing behavior](#customer)
# - [Visualization: Present insights through bar charts, line plots, and heatmaps](#visual)
# - [Recommendations: Provide actionable recommendations based on the EDA](#recomm)

# ### Introduction
# 
# In this project, I work with a dataset containing information about retail sales. 
# 
# The goal was to perform exploratory data analysis (EDA) to uncover patterns, trends, and insights that can help the retail business make informed decisions.

# In[7]:


# Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


# #### Data Loading and Cleaning.

# In[9]:


# loading Data
retail_sales = pd.read_csv ('C:/Users/CHRIS/Documents/Oasis Infobyte Internship Project/Project 1 of 1/retail_sales_dataset.csv')
retail_sales


# In[11]:


# Check for the information on the Dataset
retail_sales.info()


# In[13]:


# Check for missing values in the Dataset
retail_sales.isna().sum()


# In[15]:


# Duplicated Values
retail_sales.duplicated().sum()


# #### Descriptive Statistics.

# In[11]:


# Mean of the retail sales total
mean_values = retail_sales['Total Amount'].mean()
print("Mean Values:\n", mean_values)


# In[12]:


# Median of the retail sales total
median_values = retail_sales['Total Amount'].median()
print("Median Values:\n", median_values)


# In[13]:


# Mode of the retail sales total
mode_values = retail_sales['Total Amount'].mode().iloc[0]
print("Mode Values:\n", mode_values)


# In[14]:


# Standard deviation of the retail sales total
std_values = retail_sales['Total Amount'].std()
print("Standard Deviation:\n", std_values)


# #### Time Series Analysis.

# In[24]:


# Convert 'Date' column to datetime format
retail_sales['Date'] = pd.to_datetime(retail_sales['Date'])

# Aggregate total amount by date
daily_sales = retail_sales.groupby('Date')['Total Amount'].sum().reset_index()

# Display the first few rows of the aggregated data
daily_sales.head()

# Plot the daily sales data
plt.figure(figsize=(12, 6))
plt.plot(daily_sales['Date'], daily_sales['Total Amount'], marker='o', linestyle='-')
plt.title('Daily Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales Amount')
plt.grid(True)
plt.show()

# Set the 'Date' column as the index
daily_sales.set_index('Date', inplace=True)

# Decompose the time series
decomposition = seasonal_decompose(daily_sales['Total Amount'], model='additive', period=30)

# Plot the decomposition
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.show()


# #### Customer and Product Analysis.

# In[26]:


# 1. Customer Demographics

# Plot gender distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Gender', data=retail_sales)
plt.title('Gender Distribution of Customers')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.show()

# Plot age distribution
plt.figure(figsize=(10, 6))
sns.histplot(retail_sales['Age'], bins=20, kde=True)
plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Plot product category distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='Product Category', data=retail_sales)
plt.title('Product Category Distribution')
plt.xlabel('Product Category')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Calculate average purchase amount by gender
avg_purchase_by_gender = retail_sales.groupby('Gender')['Total Amount'].mean().reset_index()

# Plot average purchase amount by gender
plt.figure(figsize=(10, 6))
sns.barplot(x='Gender', y='Total Amount', data=avg_purchase_by_gender)
plt.title('Average Purchase Amount by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Purchase Amount')
plt.show()

# Calculate average purchase amount by age
avg_purchase_by_age = retail_sales.groupby('Age')['Total Amount'].mean().reset_index()

# Plot average purchase amount by age
plt.figure(figsize=(10, 6))
sns.lineplot(x='Age', y='Total Amount', data=avg_purchase_by_age, marker='o')
plt.title('Average Purchase Amount by Age')
plt.xlabel('Age')
plt.ylabel('Average Purchase Amount')
plt.show()

# Most Popular Product Categories

# Calculate the total amount spent by product category
total_amount_by_category = retail_sales.groupby('Product Category')['Total Amount'].sum().reset_index()

# Plot total amount spent by product category
plt.figure(figsize=(12, 6))
sns.barplot(x='Product Category', y='Total Amount', data=total_amount_by_category)
plt.title('Total Amount Spent by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Amount')
plt.xticks(rotation=45)
plt.show()

# Frequency of Purchases by Customer Demographics

# Frequency of purchases by gender
purchase_frequency_by_gender = retail_sales['Gender'].value_counts().reset_index()
purchase_frequency_by_gender.columns = ['Gender', 'Frequency']

# Plot frequency of purchases by gender
plt.figure(figsize=(10, 6))
sns.barplot(x='Gender', y='Frequency', data=purchase_frequency_by_gender)
plt.title('Frequency of Purchases by Gender')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.show()

# Frequency of purchases by age
purchase_frequency_by_age = retail_sales['Age'].value_counts().sort_index().reset_index()
purchase_frequency_by_age.columns = ['Age', 'Frequency']

# Plot frequency of purchases by age
plt.figure(figsize=(10, 6))
sns.barplot(x='Age', y='Frequency', data=purchase_frequency_by_age)
plt.title('Frequency of Purchases by Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# #### Visualization.

# In[48]:


# Bar Charts for Demographics and Product Categories

# Plot gender distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Gender', data=retail_sales)
plt.title('Gender Distribution of Customers')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.show()

# Plot age distribution
plt.figure(figsize=(10, 6))
sns.histplot(retail_sales['Age'], bins=20, kde=True)
plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Plot product category distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='Product Category', data=retail_sales)
plt.title('Product Category Distribution')
plt.xlabel('Product Category')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Line Plots for Purchasing Behavior

# Calculate average purchase amount by gender
avg_purchase_by_gender = retail_sales.groupby('Gender')['Total Amount'].mean().reset_index()

# Plot average purchase amount by gender
plt.figure(figsize=(10, 6))
sns.lineplot(x='Gender', y='Total Amount', data=avg_purchase_by_gender)
plt.title('Average Purchase Amount by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Purchase Amount')
plt.show()

# Calculate average purchase amount by age
avg_purchase_by_age = retail_sales.groupby('Age')['Total Amount'].mean().reset_index()

# Plot average purchase amount by age
plt.figure(figsize=(10, 6))
sns.lineplot(x='Age', y='Total Amount', data=avg_purchase_by_age, marker='o')
plt.title('Average Purchase Amount by Age')
plt.xlabel('Age')
plt.ylabel('Average Purchase Amount')
plt.show()

# Calculate the total amount spent by product category
total_amount_by_category = retail_sales.groupby('Product Category')['Total Amount'].sum().reset_index()

# Plot total amount spent by product category
plt.figure(figsize=(12, 6))
sns.lineplot(x='Product Category', y='Total Amount', data=total_amount_by_category)
plt.title('Total Amount Spent by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Amount')
plt.xticks(rotation=45)
plt.show()

# Heatmaps for Frequency of Purchases by Demographics

# Frequency of purchases by gender
purchase_frequency_by_gender = retail_sales['Gender'].value_counts().reset_index()
purchase_frequency_by_gender.columns = ['Gender', 'Frequency']

# For a heatmap, you usually need a matrix of values; you might need to use 'count' or another aggregation if 'Frequency' is categorical.
pivot_table = purchase_frequency_by_gender.pivot(index='Gender', columns='Frequency', values='Frequency')


# Plot frequency of purchases by gender
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title('Frequency of Purchases by Gender')
plt.xlabel('Frequency')
plt.ylabel('Gender')
plt.show()

# Frequency of purchases by age
purchase_frequency_by_age = retail_sales['Age'].value_counts().sort_index().reset_index()
purchase_frequency_by_age.columns = ['Age', 'Frequency']


# Create a pivot table
pivot_table = purchase_frequency_by_age.pivot(index='Age', columns='Frequency', values='Frequency')

# Plot frequency of purchases by age
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title('Frequency of Purchases by Age')
plt.xlabel('Frequency')
plt.ylabel('Age')
plt.show()


# #### Recommendations: Provide actionable recommendations based on the EDA.
# 
#  Product Performance Analysis
# 
# - Insight: Some products perform significantly better than others regarding sales volume and revenue.
# - Recommendation:
#   Increase marketing efforts and shelf space for top-performing products to boost sales further. Evaluate the performance of low-selling products.
#   Consider discontinuing or reworking them to meet customer preferences better.
# 
# Gender-Based Sales Analysis
# 
# - Insight: Sales patterns vary between different genders.
# - Recommendation:
#   Design targeted marketing campaigns that appeal specifically to different genders based on their purchasing patterns. For example, promote
#   products that are more popular with women in campaigns targeted at women.
#   Consider developing or sourcing products that cater to the preferences of the less represented gender, based on sales data.
#   Adjust store layouts or online shopping experiences to cater to gender-specific preferences, enhancing the shopping experience.
# 
# Age-Based Sales Analysis
# 
# - Insight: Sales vary across different age groups.
# - Recommendation:
#   Create targeted promotions or discounts tailored to different age groups. For example, offer discounts on products popular with younger consumers.
#   Adjust the product range to better align with the preferences of different age groups. This might include stocking more products that are popular
#   with older adults or more trendy items for younger shoppers.
