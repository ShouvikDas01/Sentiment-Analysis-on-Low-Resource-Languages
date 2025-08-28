"""
Created on Fri Jan  5 22:48:57 2024

EXPLORING SENTIMENT ANALYSIS IN LOW RESOURCE LANGUAGE : 
    UNVEILING LIMITATIONS IN TRANSLATION LIBRARIES.

"the script provides preparation of the Hindi Dataset , preprocessing it 
and visualising its distribution follwed by saving the pre-proccessed data
to be used on further model trainings."

@author: shouvik das
Student Id: 22196026
"""

#Importing Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
#Importing Natural Language Tool Kit Library for text pre processing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import re


#%%
#Combining the original hindi dataset to single csv file.
# Defining column names
column_names = ['experience', 'text']

# Loading the CSV files into pandas dataframes with column names
train_df = pd.read_csv(r'data/hindi/hi-train.csv', names=column_names)
test_df = pd.read_csv(r'data/hindi/hi-test.csv', names=column_names)
valid_df = pd.read_csv(r'data/hindi/hi-valid.csv', names=column_names)

# Concatenating the dataframes along the rows (axis=0)
merged_df = pd.concat([train_df, test_df, valid_df], axis=0)

# Reset index
merged_df.reset_index(drop=True, inplace=True)

# Saving merged dataset
merged_df.to_csv(r'data/hindi/hindi-data-combined.csv', index=False)


#Reading the merged CSV file.
df = pd.read_csv(r'data/hindi/hindi-data-combined.csv')
df

#%%

#Checking the info about rows and columns
df.shape


#Checking features info.
df.info()

#######################################
# Pre-Processing steps for Hindi Text #
#######################################

#Checking for null values if any
df.isna().sum()

# Downloading the Hindi stop words from nltk 
nltk.download('stopwords')

# Set of Hindi stop words
stop_words = set(stopwords.words('hindi'))

# Regular expression for stemming
stemmer = re.compile(r'(आ|िया|िए|ेई|ै|ो|ा)')

# Function to remove special characters from text
def remove_special_chars(text):
    """
    Removes special characters from Hindi text, retaining only Devanagari 
    letters and common diacritics.

    Args:
        text (str): The Hindi text to be processed.

    Returns:
        str: The text with special characters removed.
    """
    return re.sub(r"[^अ-ऋए-ऑओ-नप-रलव-हा-ृॅ-ॉॐ-॓]", " ", text)

# Function to tokenize text
def tokenize(text):
    """
    Tokenizes Hindi text into individual words.

    Args:
        text (str): The Hindi text to be tokenized.

    Returns:
        list: A list of tokens (words).
    """

    return nltk.word_tokenize(remove_special_chars(text))

# Function to filter out stop words from tokens
def filter_stopwords(tokens):
    """
    Filters out stop words from a list of tokens.

    Args:
        tokens (list): A list of tokens.

    Returns:
        list: A list of tokens without stop words.
    """
    return [token for token in tokens if token not in stop_words]

# Function for stemming
def stemming(token):
    """
    Performs stemming on a Hindi token.

   Args:
       token (str): The Hindi token to be stemmed.

   Returns:
       str: The stemmed token.
   """
    return stemmer.sub('', token)

# Function to preprocess Hindi text
def hindi_preprocess(text):
    """
    Preprocesses Hindi text by removing special characters, tokenizing,
    filtering stop words, and stemming.

    Args:
        text (str): The Hindi text to be preprocessed.

    Returns:
        list: A list of preprocessed tokens.
    """
    tokens = tokenize(text)
    tokens = filter_stopwords(tokens)
    tokens = [stemming(token) for token in tokens]
    return tokens

# Apply hindi_preprocess to the 'text' column of the DataFrame
df['text'] = df['text'].apply(hindi_preprocess)

# Display the DataFrame
print(df)


#Joining the tokens
df['text'] = df['text'].apply(lambda tokens:' '.join(tokens))
df


#Saving Pre-Processed Hindi Data #

df.to_csv(r'data/hindi/hindi-data-combined.csv',index=False)

#%%
###########################################
# Visualisation of Sentiment Destrubution #
###########################################


sns.set(style="whitegrid")
plt.figure(figsize=(5, 3))

# Countplot for sentiment distribution
sns.countplot(x='experience', data=df, palette='plasma')

plt.title('Sentiment Distribution in the Dataset')
plt.xlabel('Sentiment')
plt.ylabel('Count')

plt.show()

plt.figure(figsize=(6, 4))


# Pie chart for sentiment proportions
sentiment_counts = df['experience'].value_counts()
labels = sentiment_counts.index
colors = sns.color_palette('bright')[0:len(labels)]

plt.pie(sentiment_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Sentiment Proportions in the Dataset')
plt.show()



# Plotting histogram for the distribution of text length
plt.figure(figsize=(6, 4))

# Histogram for text lengths
df['text_length'] = df['text'].apply(len)
sns.histplot(df['text_length'], bins=30, kde=False, color='purple')

plt.title('Distribution of Text Lengths')
plt.xlabel('Text Length')
plt.ylabel('Count')

plt.show()


