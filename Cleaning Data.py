#!/usr/bin/env python
# coding: utf-8

#  ## Cleaning Data
# 
#  ### Table of Contents
# 
#  - [Introduction](#intro)
#  - [Data Integrity: Ensuring the accuracy, consistency, and reliability of data throughout the cleaning process](#Data)
#  - [Missing Data Handling: Dealing with missing values by either imputing them or making informed decisions on how to handle gaps in the data](#missing)
#  - [Duplicate Removal: Identifying and eliminating duplicate records to maintain data uniqueness](#dupli)
#  - [Standardization: Consistent formatting and units across the dataset for accurate analysis](#stan)
#  - [Outlier Detection: Identifying and addressing outliers that may skew analysis or model performance](#outlier)

# ### Introduction
# 
# Data cleaning is the process of fixing or removing incorrect, corrupted, duplicate, or incomplete 
# data within a dataset. Messy data leads to unreliable outcomes. Cleaning data is an essentia 
# part of data analysis, and demonstrating your da-a cleaning skills is key to landing a job. He e
# are some projects to test out your d-ta cleaning skills: 

# In[1]:


# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# #### Load Data

# In[2]:


# Load Dataset
Cle_data = pd.read_csv('C:/Users/CHRIS/Documents/Oasis Infobyte Internship Project/Project 1 of 3/Project 1 of 3A/AB_NYC_2019.csv')
Cle_data


# #### Data Integrity.

# In[5]:


# Checking for the information about the Dataset
Cle_data.info()


# In[7]:


# Checking for Description in the Dataset
Cle_data.describe()


# In[9]:


# Checking for Columns in the Dataset
Cle_data.columns


# #### Missing Data Handling

# In[11]:


# Display columns with missing values
Cle_data.isna().sum()


# In[13]:


# Drop columns with a high percentage of missing values (e.g., over 50%)
threshold = len(Cle_data) * 0.5
Cle_data_threshold = Cle_data.dropna(thresh=threshold, axis=1)

# Fill missing values for numerical columns with the median
for col in Cle_data.select_dtypes(include=['float64', 'int64']).columns:
   Cle_data[col].fillna(Cle_data[col].median(), inplace=True)

# Fill missing values for categorical columns with the mode
for col in Cle_data.select_dtypes(include=['object']).columns:
   Cle_data[col].fillna(Cle_data[col].mode()[0], inplace=True)

Cle_data.isna().sum()


# #### Duplicate Removal

# In[13]:


# Checking for Duplicates in the Dataset
Cle_data.duplicated().sum()

# Remove Duplicates Values
Cle_data_duplicates = Cle_data.drop_duplicates()
Cle_data_duplicates.head()


# #### Standardization

# In[15]:


# Convert 'last_review' column to datetime
if 'last_review' in Cle_data.columns:
   Cle_data['last_review'] = pd.to_datetime(Cle_data['last_review'])

# Convert 'id' and 'host_id' to strings
Cle_data['id'] = Cle_data['id'].astype(str)
Cle_data['host_id'] = Cle_data['host_id'].astype(str)

# Convert categorical columns to category type
categorical_columns = ['neighbourhood_group', 'neighbourhood', 'room_type']
for col in categorical_columns:
    if col in Cle_data.columns:
        Cle_data[col] = Cle_data[col].astype('category')

print(Cle_data.dtypes)


# In[19]:


# Example: Standardize text in 'name' and 'host_name' columns
Cle_data['name'] = Cle_data['name'].str.lower().str.strip()
Cle_data['host_name'] = Cle_data['host_name'].str.lower().str.strip()
Cle_data.head()


# #### Outlier Detection

# In[41]:


# Visualizing a boxplot for 'price' column
sns.boxplot(x=Cle_data['price'])
plt.show()

# Z-Score method
Cle_data['z_score'] = stats.zscore(Cle_data['price'])
outliers_z = Cle_data[Cle_data['z_score'].abs() > 3]

# IQR method
Q1 = Cle_data['price'].quantile(0.25)
Q3 = Cle_data['price'].quantile(0.75)
IQR = Q3 - Q1

outliers_iqr = Cle_data[(Cle_data['price'] < (Q1 - 1.5 * IQR)) | (Cle_data['price'] > (Q3 + 1.5 * IQR))]

