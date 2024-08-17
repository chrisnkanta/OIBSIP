#!/usr/bin/env python
# coding: utf-8

# ## Unveiling the Android App Market: Analyzing Google Play Store Data
# 
# ### Table of Contents
# 
# - [Introduction](#intro)
# - [Data Preparation: Clean and correct data types for accuracy](#datap)
# - [Category Exploration: 
# Investigate app distribution across categorie](#cat)
# - [Metrics Analysis: 
# Examine app ratings, size, popularity, and pricing trend](#met)
# - [Sentiment Analysis: 
# Assess user sentiments through review](#sen)
# - [Interactive Visualization: 
# Utilize code for compelling visualization](#inte)
# - [Skill Enhancement: Integrate insights from the "Understanding Data Visualization" course](#skil)

# ### Introduction
# 
# Clean, categorize, and visualize Google Play Store data to understand app market dynamics. 
# 
# Gain in-depth insights into the Android app market by leveraging data analytics,visualization, and enhanced interpretation skills.s.

# In[13]:


# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob


# #### Data Collection

# In[14]:


# Loading Dataset
apps_data = pd.read_csv ('C:/Users/CHRIS/Documents/Oasis Infobyte Internship Project/Project 2 of 4/apps.csv')
apps_data


# #### Data Preparation

# In[3]:


# Checking for information in the Dataset
apps_data.info()


# In[4]:


# Checking for Description of the Dataset
apps_data.describe()


# In[5]:


# Checking for shape in the Dataset
apps_data.shape


# In[6]:


# Checking for missing values in the Dataset
apps_data.isnull().sum()


# In[18]:


# Checking for Duplicates in the Dataset
apps_data.duplicated().sum()


# In[7]:


# Checking for data type in the Dataset
apps_data.dtypes


# In[40]:


# Ensure the `Installs` column is in string format before using `.str.replace`
if apps_data['Installs'].dtype != 'object':
    apps_data['Installs'] = apps_data['Installs'].astype(str)

# Convert `Installs` to numeric
apps_data['Installs'] = apps_data['Installs'].str.replace('[+,]', '', regex=True).astype(int)

# Ensure the `Price` column is in string format before using `.str.replace`
if apps_data['Price'].dtype != 'object':
    apps_data['Price'] = apps_data['Price'].astype(str)

# Now safely apply the `.str.replace` method to remove the dollar sign and convert to float
apps_data['Price'] = apps_data['Price'].str.replace('$', '').astype(float)

# Convert `Last Updated` to datetime
apps_data['Last Updated'] = pd.to_datetime(apps_data['Last Updated'])

# Handle missing values
# Fill missing `Rating` and `Size` values with the mean
apps_data['Rating'] = apps_data['Rating'].fillna(apps_data['Rating'].mean())
apps_data['Size'] = apps_data['Size'].fillna(apps_data['Size'].mean())

# Fill other missing values with 'Unknown'
apps_data.fillna('Unknown', inplace=True)

# Correct data types for columns like `Reviews`
apps_data['Reviews'] = apps_data['Reviews'].astype(int)

# Display the cleaned data info and first few rows
print(apps_data.info())
print(apps_data.head())


# #### Category Exploration

# In[45]:


# Count the number of apps in each category
category_distribution = apps_data['Category'].value_counts()

# Print the investigation statement
print("To investigate the app distribution across categories, we can count the number of apps in each category. I'll calculate and display the distribution now.\n")

# Print the distribution of apps across categories
print("The distribution of apps across different categories is as follows:\n")

# Iterate through the distribution and print each category with the count
for category, count in category_distribution.items():
    print(f"{category}: {count} apps")


# #### Metrics Analysis

# In[52]:


# Convert necessary columns to appropriate data types
apps_data['Rating'] = pd.to_numeric(apps_data['Rating'], errors='coerce')
apps_data['Size'] = pd.to_numeric(apps_data['Size'], errors='coerce')
apps_data['Installs'] = apps_data['Installs'].replace('[+,]', '', regex=True).astype(int)
apps_data['Price'] = apps_data['Price'].replace(r'[\$,]', '', regex=True).astype(float)

# Examine App Ratings
ratings_distribution = apps_data['Rating'].dropna()

# Examine App Size
size_distribution = apps_data['Size'].dropna()

# Examine Popularity (Installs)
popularity_distribution = apps_data['Installs']

# Examine Pricing Trends
pricing_distribution = apps_data[apps_data['Type'] == 'Paid']['Price']

# Plotting the distributions
plt.figure(figsize=(15, 10))

# Ratings Distribution
plt.subplot(2, 2, 1)
plt.hist(ratings_distribution, bins=20, color='skyblue', edgecolor='black')
plt.title('App Ratings Distribution')
plt.xlabel('Rating')
plt.ylabel('Number of Apps')

# Size Distribution
plt.subplot(2, 2, 2)
plt.hist(size_distribution, bins=20, color='lightgreen', edgecolor='black')
plt.title('App Size Distribution')
plt.xlabel('Size (MB)')
plt.ylabel('Number of Apps')

# Popularity (Installs) Distribution
plt.subplot(2, 2, 3)
plt.hist(popularity_distribution, bins=20, color='lightcoral', edgecolor='black')
plt.title('App Popularity (Installs) Distribution')
plt.xlabel('Number of Installs')
plt.ylabel('Number of Apps')
plt.xscale('log')  # Use logarithmic scale due to the wide range of installs

# Pricing Distribution
plt.subplot(2, 2, 4)
plt.hist(pricing_distribution, bins=20, color='gold', edgecolor='black')
plt.title('App Pricing Distribution (Paid Apps)')
plt.xlabel('Price ($)')
plt.ylabel('Number of Apps')

# Adjust layout and show the plots
plt.tight_layout()
plt.show()


# #### Sentiment Analysis

# In[57]:


# Perform Sentiment Analysis
def analyze_sentiment(review):
    analysis = TextBlob(review)
    return analysis.sentiment.polarity

# Applying sentiment analysis on the 'Reviews' 
apps_data['Sentiment'] = apps_data['Reviews'].apply(lambda x: analyze_sentiment(str(x)))

# SAggregate Sentiment Results
# Calculating the average sentiment for each app
average_sentiment = apps_data.groupby('App')['Sentiment'].mean()

# Display the top apps by sentiment
top_positive_apps = average_sentiment.sort_values(ascending=False).head(10)
top_negative_apps = average_sentiment.sort_values().head(10)

# Display the results
print("Top 10 Positive Sentiment Apps:")
print(top_positive_apps)

print("\nTop 10 Negative Sentiment Apps:")
print(top_negative_apps)


# #### Interactive Visualization

# In[60]:


# Check the Category column and clean the data
if 'Category' not in apps_data.columns:
    raise ValueError("The column 'Category' does not exist in the data. Please check the column names.")

# Drop rows where 'Category' column has missing values
apps_data = apps_data.dropna(subset=['Category'])

# Count the number of apps in each category
category_counts = apps_data['Category'].value_counts()
category_counts_apps_data = category_counts.reset_index()
category_counts_apps_data.columns = ['Category', 'Number of Apps']


# Visualization: Bar plot for app distribution across categories
plt.figure(figsize=(14, 8))
sns.barplot(x='Category', y='Number of Apps', data=category_counts_apps_data, color='teal')
plt.xticks(rotation=90)
plt.xlabel('Category')
plt.ylabel('Number of Apps')
plt.title('Distribution of Apps Across Categories')
plt.tight_layout()
plt.show()

# Visualization: Pie chart for app distribution across categories
plt.figure(figsize=(10, 10))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis', len(category_counts)))
plt.title('Distribution of Apps Across Categories')
plt.show()

# Optional: Visualization of app counts by category with a countplot
plt.figure(figsize=(14, 8))
sns.barplot(x='Category', y='Number of Apps', data=category_counts_apps_data, color='teal')
plt.xlabel('Number of Apps')
plt.ylabel('Category')
plt.title('Number of Apps by Category')
plt.tight_layout()
plt.show()


# #### Skill Enhancement

# In[63]:


# Check the Category column and clean the data
if 'Category' not in apps_data.columns:
    raise ValueError("The column 'Category' does not exist in the data. Please check the column names.")

# Drop rows where 'Category' column has missing values
apps_data = apps_data.dropna(subset=['Category'])

# Count the number of apps in each category
category_counts = apps_data['Category'].value_counts()
category_counts_apps_data = category_counts.reset_index()
category_counts_apps_data.columns = ['Category', 'Number of Apps']


# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Visualization 1: Bar Plot
plt.figure(figsize=(14, 8))
sns.barplot(x='Category', y='Number of Apps', data=category_counts_apps_data, color='teal')
plt.xticks(rotation=90)
plt.xlabel('Category')
plt.ylabel('Number of Apps')
plt.title('Distribution of Apps Across Categories')
plt.tight_layout()
plt.show()

# Visualization 2: Pie Chart
plt.figure(figsize=(10, 10))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis', len(category_counts)))
plt.title('Distribution of Apps Across Categories')
plt.show()

# Visualization 3: Horizontal Bar Plot
plt.figure(figsize=(14, 8))
sns.barplot(x='Category', y='Number of Apps', data=category_counts_apps_data, color='teal')
plt.xlabel('Number of Apps')
plt.ylabel('Category')
plt.title('Number of Apps by Category')
plt.tight_layout()
plt.show()

# Visualization 4: Box Plot (if there's a numerical feature to compare across categories)
if 'Rating' in apps_data.columns:
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Category', y='Number of Apps', data=category_counts_apps_data, color='teal')
    plt.xticks(rotation=90)
    plt.xlabel('Category')
    plt.ylabel('Rating')
    plt.title('Rating Distribution by Category')
    plt.tight_layout()
    plt.show()


# In[ ]:





# In[ ]:




