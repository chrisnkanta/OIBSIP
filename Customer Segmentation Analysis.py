#!/usr/bin/env python
# coding: utf-8

# ## Customer Segmentation Analysis
# 
# ### Table of Contents
# 
# - [Introduction](#intro)
# - [Data Collection: Obtain a dataset containing customer information, purchase history, and relevant data](#data)
# - [Data Exploration and Cleaning: Explore the dataset, understand its structure, and handle any missing or inconsistent data.](#explo)
# - [Descriptive Statistics: Calculate key metrics such as average purchase value, frequency of purchases, etc.](#desc)
# - [Customer Segmentation: Utilize clustering algorithms (e.g., K-means) to segment customers based on behaviour and purchase patterns.](#cus)
# - [Visualization: Create visualizations (e.g., scatter plots, bar charts) to illustrate customer segments.](#vis)
# - [Insights and Recommendations: Analyze the characteristics of each segment and provide insight.](#insights)
# 

# ### Introduction
# 
# This data analytics project aims to perform customer segmentation analysis for an e-
# commerce company. By analyzing customer behaviuo,  purchase pattern, and  the goal is t 
# group customers into distinct segments. This segmentation can inform targeted marketi g
# strategies, improve customer satisfaction, and enhance overall business strategies.

# In[6]:


### importing the Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# #### Data Collection

# In[3]:


## Load Dataset
Cus_Seg_Analysis = pd.read_csv('C:/Users/CHRIS/Documents/Oasis Infobyte Internship Project/Project 1 of 2/ifood_df.csv')
Cus_Seg_Analysis


# #### Data Exploration and Cleaning

# In[12]:


#### Checking for the information in the Dataset
Cus_Seg_Analysis.info()


# In[14]:


#### Checking for missing values in the Dataset
Cus_Seg_Analysis.isna().sum()


# In[20]:


Cus_Seg_Analysis.columns


# In[26]:


## Duplicated Values
Cus_Seg_Analysis.duplicated().sum()


# In[30]:


Cus_Seg_Analysis.describe()


# #### Descriptive Statistics

# In[5]:


# Step 1: Clean the data (handle any negative values in 'MntRegularProds' by setting them to zero)
Cus_Seg_Analysis['MntRegularProds'] = Cus_Seg_Analysis['MntRegularProds'].apply(lambda x: max(x, 0))

# Step 2: Calculate key metrics

# 1. Average Purchase Value (Total amount spent divided by the number of purchase events)
Cus_Seg_Analysis['AveragePurchaseValue'] = Cus_Seg_Analysis['MntTotal'] / (
    Cus_Seg_Analysis['NumDealsPurchases'] + Cus_Seg_Analysis['NumWebPurchases'] + Cus_Seg_Analysis['NumCatalogPurchases'] + Cus_Seg_Analysis['NumStorePurchases']
)

# Handle cases where the purchase frequency is zero to avoid division by zero
Cus_Seg_Analysis['AveragePurchaseValue'] = Cus_Seg_Analysis['AveragePurchaseValue'].replace([float('inf'), -float('inf')], 0)

# 2. Total Purchase Value (Sum of all purchase amounts)
Cus_Seg_Analysis['TotalPurchaseValue'] = Cus_Seg_Analysis['MntTotal']

# 3. Purchase Frequency (Total number of purchases)
Cus_Seg_Analysis['PurchaseFrequency'] = (
    Cus_Seg_Analysis['NumDealsPurchases'] + Cus_Seg_Analysis['NumWebPurchases'] + Cus_Seg_Analysis['NumCatalogPurchases'] + Cus_Seg_Analysis['NumStorePurchases']
)

# 4. Average Recency (Average number of days since last purchase)
average_recency = Cus_Seg_Analysis['Recency'].mean()

# 5. Average Spending on Different Product Categories
average_spending = {
    'Wines': Cus_Seg_Analysis['MntWines'].mean(),
    'Fruits': Cus_Seg_Analysis['MntFruits'].mean(),
    'MeatProducts': Cus_Seg_Analysis['MntMeatProducts'].mean(),
    'FishProducts': Cus_Seg_Analysis['MntFishProducts'].mean(),
    'SweetProducts': Cus_Seg_Analysis['MntSweetProducts'].mean(),
    'GoldProds': Cus_Seg_Analysis['MntGoldProds'].mean(),
}

# Step 3: Summarize the calculated metrics
metrics_summary = {
    'Average Purchase Value': Cus_Seg_Analysis['AveragePurchaseValue'].mean(),
    'Total Purchase Value (Overall)': Cus_Seg_Analysis['TotalPurchaseValue'].sum(),
    'Average Purchase Frequency': Cus_Seg_Analysis['PurchaseFrequency'].mean(),
    'Average Recency': average_recency,
    'Average Spending per Category': average_spending,
}

# Print the summary of metrics
print(metrics_summary)


# #### Customer Segmentation

# In[5]:


# Select relevant features for clustering
features = [
    'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
    'NumStorePurchases', 'NumWebVisitsMonth'
]

# Handle any negative values in 'MntRegularProds' by setting them to zero
Cus_Seg_Analysis['MntRegularProds'] = Cus_Seg_Analysis['MntRegularProds'].apply(lambda x: max(x, 0))

# Normalize the data
scaler = StandardScaler()
Cus_Seg_Analysis_scaled = scaler.fit_transform(Cus_Seg_Analysis[features])

# Step 3: Apply the K-means clustering algorithm
# Determine the optimal number of clusters using the elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(Cus_Seg_Analysis_scaled)
    sse.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Based on the elbow plot, choose the optimal number of clusters
optimal_k = 3  # You can adjust this based on the elbow plot

# Apply K-means with the chosen number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
Cus_Seg_Analysis['Cluster'] = kmeans.fit_predict(Cus_Seg_Analysis_scaled)

# Step 4: Analyze the clusters
# Add the cluster labels back to the original dataframe
Cus_Seg_Analysis['Cluster'] = kmeans.labels_

# Visualize the clusters using pairplot
sns.pairplot(Cus_Seg_Analysis, hue='Cluster', vars=features, palette='viridis')
plt.show()

# Show the cluster centers
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
print(cluster_centers)


# #### Visualization

# In[9]:


# Select relevant features for clustering
features = [
    'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
    'NumStorePurchases', 'NumWebVisitsMonth'
]

# Normalize the data
scaler = StandardScaler()
Cus_Seg_Analysis_scaled = scaler.fit_transform(Cus_Seg_Analysis[features])

# Step 3: Apply the K-means clustering algorithm
# Choose the optimal number of clusters based on the elbow method
optimal_k = 3  # Adjust based on your previous analysis

# Apply K-means with the chosen number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
Cus_Seg_Analysis['Cluster'] = kmeans.fit_predict(Cus_Seg_Analysis_scaled)

# Step 4: Create visualizations

# Scatter plot of Income vs. Recency colored by cluster
plt.figure(figsize=(10, 6))
sns.scatterplot(data=Cus_Seg_Analysis, x='Income', y='Recency', hue='Cluster', palette='viridis')
plt.title('Customer Segments by Income and Recency')
plt.xlabel('Income')
plt.ylabel('Recency (Days since last purchase)')
plt.show()

# Bar chart of average spending in different categories by cluster
categories = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
cluster_means = Cus_Seg_Analysis.groupby('Cluster')[categories].mean()

cluster_means.plot(kind='bar', figsize=(12, 8), colormap='viridis')
plt.title('Average Spending by Product Category for Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Spending')
plt.legend(title='Product Categories')
plt.show()

# Pairplot to visualize distributions across clusters
sns.pairplot(Cus_Seg_Analysis, hue='Cluster', vars=['Income', 'Recency', 'MntWines', 'MntMeatProducts'], palette='viridis')
plt.suptitle('Pairplot of Selected Features by Cluster', y=1.02)
plt.show()


# #### Insights and Recommendations
# 
# Insights:
# - Highly engaged and values the service
# - These customers are price-sensitive or looking for smaller, more frequent purchases.
# 
# Recommendations:
# - Implement a loyalty program with rewards for frequent purchases. Offer personalized promotions and exclusive deals.
# - Introduce discount offers for bulk orders or bundle deals. Engage them with targeted marketing campaigns to increase the average order value.
# 
