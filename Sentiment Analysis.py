#!/usr/bin/env python
# coding: utf-8

# ##  Sentiment Analysis
# 
# ### Table of Contaents
# 
# - [Introduction](#intro)
# - [Sentiment Analysis: Analyzing text data to determine the emotional tone, whether positive, 
# negative, or neutral](#sen)
# - [Natural Language Processing (NLP): Utilizing algorithms and models to understand and 
# process human language](#natu)
# - [Machine Learning Algorithms: Implementing models for sentiment classification, such as 
# Support Vector Machines, Naive Bayes, or deep learnin  architectures](#mach)
# - [Feature Engineering: Identifying and extracting relevant features from text data to enhance 
# model performance](#feat)
# - [Data Visualization: Presenting sentiment analysis results through effective visualizations for 
# clear interpretation](#dat)

# ### Introduction
# 
# The primary goal is to develop a sentiment analysis model that can accurately classify the 
# sentiment of text data, providing valuable insights into public opinion, customer feedback, an 
# social media trends.

# In[1]:


# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import spacy
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack, csr_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')


# #### Load Dataset

# In[3]:


# Load Dataset
Sen_Analysis = pd.read_csv('C:/Users/CHRIS/Documents/Oasis Infobyte Internship Project/Project 1 of 4/Twitter_Data.csv')
Sen_Analysis.head()


# #### Clean Dataset

# In[5]:


# Check for the information in the Dataset
Sen_Analysis.info()


# In[7]:


# Check for missing values in the column
Sen_Analysis.isnull().sum()


# In[9]:


# Fill missing values with an empty string or appropriate value
Sen_Analysis['category'] = Sen_Analysis['category'].fillna('')

# Ensure all entries in 'category' are strings
Sen_Analysis['category'] = Sen_Analysis['category'].astype(str)

Sen_Analysis['category']


# In[11]:


# Check for missing values in the 'category' column
print(Sen_Analysis['category'].isnull().sum())


# In[13]:


# Check for the information in the Dataset after filling the missing column
Sen_Analysis.info()


# In[15]:


# Check for the different columns in the Dataset
Sen_Analysis.columns


# #### Sentiment Analysis

# In[17]:


# Function to get sentiment
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply the sentiment analysis to the 'category' column
Sen_Analysis['Sentiment'] = Sen_Analysis['category'].apply(get_sentiment)

# Display the first few rows with the sentiment column added
print(Sen_Analysis[['category', 'Sentiment']].head())


# In[19]:


# Convert to string and process text
text_column = 'category'
Sen_Analysis[text_column] = Sen_Analysis[text_column].astype(str).str.lower()
Sen_Analysis['tokens'] = Sen_Analysis[text_column].apply(word_tokenize)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Lemmatize the tokens
Sen_Analysis['tokens'] = Sen_Analysis['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Optionally, join tokens back into strings
Sen_Analysis['processed_text'] = Sen_Analysis['tokens'].apply(lambda x: ' '.join(x))

print(Sen_Analysis)


# #### Natural Language Processing (NLP)

# In[18]:


# Update these names based on your actual column names
text_column = 'clean_text'
label_column = 'category'

# Ensure all values in the text column are strings
Sen_Analysis[text_column] = Sen_Analysis[text_column].astype(str)

# Check for missing values in the label column
if Sen_Analysis[label_column].isnull().any():
    print("Missing values found in label column.")
    # Option 1: Drop rows with missing labels
    Sen_Analysis = Sen_Analysis.dropna(subset=[label_column])
   
# Basic Text Preprocessing Functions
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Apply preprocessing
Sen_Analysis['processed_text'] = Sen_Analysis[text_column].apply(preprocess_text)

# Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(Sen_Analysis['processed_text'])

# Labels
y = Sen_Analysis[label_column]

# Check if there are still NaN values in labels
if y.isnull().any():
    print("NaN values still found in label column.")
    # Handle this case if necessary

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Text Classification Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Classification Report:\n", metrics.classification_report(y_test, y_pred))


# #### Machine Learning Algorithms

# In[ ]:


# For classification tasks
text_column = 'clean_text'
label_column = 'category'

# Ensure all values in the text column are strings
Sen_Analysis[text_column] = Sen_Analysis[text_column].astype(str)

# Check for missing values in the label column
if Sen_Analysis[label_column].isnull().any():
    print("Missing values found in label column.")
    Sen_Analysis = Sen_Analysis.dropna(subset=[label_column])

# Basic Text Preprocessing Functions
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Apply preprocessing
Sen_Analysis['processed_text'] = Sen_Analysis[text_column].apply(preprocess_text)

# Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(Sen_Analysis['processed_text'])

# Labels
y = Sen_Analysis[label_column]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Model 1: Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)
print("Naive Bayes Model")
print("Accuracy:", metrics.accuracy_score(y_test, nb_predictions))
print("Classification Report:\n", metrics.classification_report(y_test, nb_predictions, target_names=label_encoder.classes_))

# Model 2: Support Vector Machine
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
print("Support Vector Machine Model")
print("Accuracy:", metrics.accuracy_score(y_test, svm_predictions))
print("Classification Report:\n", metrics.classification_report(y_test, svm_predictions, target_names=label_encoder.classes_))

# Model 3: Deep Learning (Neural Network)
# Prepare text data for deep learning
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(Sen_Analysis['processed_text'])
X_sequences = tokenizer.texts_to_sequences(Sen_Analysis['processed_text'])
X_padded = pad_sequences(X_sequences, maxlen=100)

# Split data for deep learning model
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X_padded, y_encoded, test_size=0.3, random_state=42)

# Build the neural network model
dl_model = Sequential()
dl_model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
dl_model.add(LSTM(128, return_sequences=True))
dl_model.add(LSTM(64))
dl_model.add(Dense(len(label_encoder.classes_), activation='softmax'))

dl_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
dl_model.fit(X_train_dl, y_train_dl, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate the deep learning model
dl_loss, dl_accuracy = dl_model.evaluate(X_test_dl, y_test_dl)
print("Deep Learning Model")
print("Accuracy:", dl_accuracy)


#  #### Feature Engineering

# In[29]:


from scipy.sparse import hstack, csr_matrix

# Strip any leading/trailing whitespace from column names
Sen_Analysis.columns = Sen_Analysis.columns.str.strip()

# Fill missing values in the text column with an empty string or drop rows with NaN values
Sen_Analysis['clean_text'].fillna('', inplace=True)  # Replace NaN with empty string
# Alternatively, you could drop rows with NaN in the text column
# df.dropna(subset=['clean_text'], inplace=True)

# Preprocess the text data
def preprocess_text(text):
    if isinstance(text, str):  # Ensure the text is a string
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^A-Za-z\s]', '', text)
        text = text.lower()
    return text

# Apply preprocessing to the text column
Sen_Analysis['cleaned_text'] = Sen_Analysis['clean_text'].apply(preprocess_text)

# Extract text features using TF-IDF
def extract_text_features(text_series):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_series)
    return tfidf_matrix

text_features = extract_text_features(Sen_Analysis['cleaned_text'])

# Feature engineering: Add tweet length and sentiment score
def get_tweet_length(text):
    return len(text)

def get_sentiment_score(text):
    return TextBlob(text).sentiment.polarity

Sen_Analysis['tweet_length'] = Sen_Analysis['clean_text'].apply(get_tweet_length)
Sen_Analysis['sentiment_score'] = Sen_Analysis['cleaned_text'].apply(get_sentiment_score)

# Combine all features into a single matrix
additional_features = Sen_Analysis[['tweet_length', 'sentiment_score']].values
additional_features_sparse = csr_matrix(additional_features)
combined_features = hstack([text_features, additional_features_sparse])

# Print the combined features matrix
print(combined_features)

# Print the shape of the combined features matrix
print(combined_features.shape)


# #### Data Visualization

# In[32]:


# Strip any leading/trailing whitespace from column names
Sen_Analysis.columns = Sen_Analysis.columns.str.strip()

# Fill missing values in the text column with an empty string or drop rows with NaN values
Sen_Analysis['clean_text'].fillna('', inplace=True)

# Preprocess the text data
def preprocess_text(text):
    if isinstance(text, str):  # Ensure the text is a string
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^A-Za-z\s]', '', text)
        text = text.lower()
    return text

# Apply preprocessing to the text column
Sen_Analysis['cleaned_text'] = Sen_Analysis['clean_text'].apply(preprocess_text)

# Perform sentiment analysis
def get_sentiment_score(text):
    return TextBlob(text).sentiment.polarity

Sen_Analysis['sentiment_score'] = Sen_Analysis['cleaned_text'].apply(get_sentiment_score)

# Display the first few rows to check the results
print(Sen_Analysis[['clean_text', 'sentiment_score']].head())

# Create a column for sentiment categories
def categorize_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

Sen_Analysis['sentiment_category'] = Sen_Analysis['sentiment_score'].apply(categorize_sentiment)

# Aggregate sentiment scores by category
sentiment_counts = Sen_Analysis['sentiment_category'].value_counts()
print(sentiment_counts)

# Set the style of the visualization
sns.set(style="whitegrid")

# Plot sentiment distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=Sen_Analysis, x='sentiment_category', palette='coolwarm')
plt.title('Distribution of Sentiment Categories')
plt.xlabel('Sentiment Category')
plt.ylabel('Number of Tweets')
plt.show()

# Plot sentiment score distribution
plt.figure(figsize=(10, 6))
sns.histplot(Sen_Analysis['sentiment_score'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# If you have other features to compare, such as tweet length vs. sentiment score
plt.figure(figsize=(10, 6))
sns.scatterplot(data=Sen_Analysis, x='tweet_length', y='sentiment_score', alpha=0.5)
plt.title('Tweet Length vs. Sentiment Score')
plt.xlabel('Tweet Length')
plt.ylabel('Sentiment Score')
plt.show()

