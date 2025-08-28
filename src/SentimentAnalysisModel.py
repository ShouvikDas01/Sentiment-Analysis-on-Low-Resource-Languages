# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 22:48:57 2024

EXPLORING SENTIMENT ANALYSIS IN LOW RESOURCE LANGUAGE : 
    UNVEILING LIMITATIONS IN TRANSLATION LIBRARIES.

the script provides an end-to-end implementation of sentiment analysis on 
translated Hindi product reviews using multiple machine learning techniques. 
The analysis provides insights into current challenges in handling translated text.

@author: shouvik das
Student Id: 22196026
"""
#Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#Importing Natural Language Tool Kit Library for text pre processing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re


#Importing Library for text label encoding and vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


#Importing Libraries for model Building**
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, BatchNormalization, Dropout,Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

#%%
#Importing Desired Dataset (Translated Hindi/ English)
df = pd.read_csv(r'data/balanced/transformers-data-bal.csv')
df



#%% 
#######################
# Feature Engineering #
#######################

#Feature Engineering while working on Google Dataset
df.drop(columns=['original_text'], inplace=True)
df.rename(columns={'translated_text': 'text'}, inplace=True)
df
#%% 
#Feature Engineering while working on Transformers Dataset
df.drop(columns=['original_text'], inplace=True)
df.rename(columns={'translated_text': 'text'}, inplace=True)
df
#%% 
#Feature Engineering while working on English Dataset (before Balancing)
df.drop(columns=['tweet_id'], inplace=True)
df.rename(columns={'label': 'experience'}, inplace=True)
df.rename(columns={'tweet_text': 'text'}, inplace=True)
df['experience'] = df['experience'].replace({1: 'negative', 2: 'neutral', 3: 'positive'})
df
#%%

#######################
# Data Pre-Processing # 
#######################

# Performing Data Pre-processing after translation / Same for English Dataset
# Downloading NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Defining preprocessing functions

def tokenize(text):
    """
    Tokenizes the input text into words.

    Args:
      text (str): The text to be tokenized.
    
    Returns:
      list: A list of tokens (words) extracted from the text.
    """
    # Tokenizing the input text into words.
    if isinstance(text, str):
        return word_tokenize(text)
    else:
        # Converting non-string or non-bytes-like object to string
        return word_tokenize(str(text))

def remove_stop_words(tokens):
    """
    Removes stop words from the list of tokens.
    Args:
      tokens (list): A list of tokens.
    
    Returns:
      list: A list of tokens with stop words removed.
    """
    # Removing stop words from the list of tokens.
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def remove_non_alphabetic(tokens):
    """
    Removes non-alphabetic tokens from the list.

    Args:
      tokens (list): A list of tokens.
    
    Returns:
      list: A list of tokens containing only alphabetic characters.
    """
    # Removing non-alphabetic tokens from the list.
    return [word for word in tokens if word.isalpha()]

def remove_emoticons(tokens):
    """
    Removes emoticons from the list of tokens.

    Args:
      tokens (list): A list of tokens.
    
    Returns:
      list: A list of tokens with emoticons removed.
    """
    # Removing emoticons from the list of tokens.
    emoticons_pattern = r"(?:[<>:](?:[:=;]-?)?[oO\)\(\wD])|[xo\(\)]"
    return [token for token in tokens if not re.match(emoticons_pattern, token)]

def stem_words(tokens):
    """
    Applies stemming to the list of tokens.

    Args:
      tokens (list): A list of tokens.
    
    Returns:
      list: A list of stemmed tokens.
    """
    # Applying stemming to the list of tokens.
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def convert_to_lowercase(tokens):
    """
    Converts each token to lowercase.
    Args:
      tokens (list): A list of tokens.
    
    Returns:
      list: A list of tokens with all characters in lowercase.
    """
    # Converting each token to lowercase.
    return [token.lower() for token in tokens]

def preprocess_text(text):
    """
    Applies a series of text preprocessing steps to the input text.

    Args:
      text (str): The text to be preprocessed.
    
    Returns:
      list: A list of preprocessed tokens.
    """
    # Applying a series of text preprocessing steps to the input text.
    tokens = tokenize(text)
    tokens = remove_stop_words(tokens)
    tokens = remove_emoticons(tokens)
    tokens = remove_non_alphabetic(tokens)
    tokens = stem_words(tokens)
    tokens = convert_to_lowercase(tokens)
    return tokens

# Applying preprocessing to the 'text' column
df['text'] = df['text'].apply(preprocess_text)
df


#%%
#Label Encoding the data
le_model = LabelEncoder()
df['experience'] = le_model.fit_transform(df['experience'])
df


#%%
#Joining the tokens
df['text'] = df['text'].apply(lambda tokens:' '.join(tokens))
df

#%%
#Plotting word cloud
all_reviews = ' '.join(df['text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Reviews')
plt.show()

#%%
######################
# Building the model #
######################

def print_model_evaluation(model_name, y_train, y_train_pred, y_test, y_test_pred):
    """Prints a comprehensive evaluation of a machine learning model.
    Args:
      model_name (str): The name of the model being evaluated.
      y_train (numpy.ndarray): The true labels for the training set.
      y_train_pred (numpy.ndarray): The predicted labels for the training set.
      y_test (numpy.ndarray): The true labels for the test set.
      y_test_pred (numpy.ndarray): The predicted labels for the test set.
    """
    print(f"\n{' MODEL EVALUATION ':-^54}")  
    print(f"{'Model Name:':<20} {model_name}")
    print(f"{'=' * 54}")

    # Handling potential one-hot encoding and convert to categorical labels
    y_train_true = np.argmax(y_train, axis=1) if len(y_train.shape) > 1 else y_train
    y_test_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test

    # Calculating and printing accuracies with clear formatting
    print(f"\n{'Training Accuracy:':<20} {accuracy_score(y_train_true, y_train_pred):.4f}")
    print(f"{'Test Accuracy:':<20} {accuracy_score(y_test_true, y_test_pred):.4f}")

    print(f"\n{'-' * 54}")

    # Print the classification report with a clear title
    print(f"\n{'Classification Report:':<20}")
    print(classification_report(y_test_true, y_test_pred))

#%%
######################################
# Splitting and Vectorizing the data #
######################################

# Splitting the data into training and testing sets
X = df['text']
y = df['experience']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#Vectorising/Normalizing the Dataset

# Initializing the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
# Fit and transform the training data
X_train = tfidf_vectorizer.fit_transform(X_train.astype('U'))  # Converting to Unicode strings
# Transform the test data using the same vectorizer
X_test = tfidf_vectorizer.transform(X_test.astype('U'))  # Converting to Unicode strings


#%%
##################
# Running Models #
##################

#Naive Bayes
model_name = 'Naive Bayes'
model = MultinomialNB()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Print and plot model evaluation
print_model_evaluation(model_name, y_train, y_train_pred, y_test, y_test_pred)

# Plot Accuracy
plt.bar(['Training', 'Test'], [accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)], color=['blue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Accuracy Plot for Naive Bayes')
plt.show()

# Confusion Matrix for Training Set
cm_train = confusion_matrix(y_train, y_train_pred)

# Confusion Matrix for Test Set
cm_test = confusion_matrix(y_test, y_test_pred)

# Print Confusion Matrix
print("Confusion Matrix (Training Set):\n", cm_train)
print("\nConfusion Matrix (Test Set):\n", cm_test)

# Plot Confusion Matrix
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.heatmap(cm_train, annot=True, fmt='g', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix (Training Set)')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 2, 2)
sns.heatmap(cm_test, annot=True, fmt='g', cmap='Oranges', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()


#%%
#Support Vector Machine
model_name = 'SVM'
svm_model = SVC()
svm_model.fit(X_train, y_train)

y_train_pred = svm_model.predict(X_train)  
y_test_pred = svm_model.predict(X_test)

print_model_evaluation(model_name,y_train, y_train_pred, y_test, y_test_pred)

# Plot Accuracy
plt.bar(['Training', 'Test'], [accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)], color=['blue', 'orange']) 
plt.ylabel('Accuracy')
plt.title('Accuracy Plot for SVM')
plt.show()

# Confusion Matrix for Training Set
cm_train = confusion_matrix(y_train, y_train_pred)

# Confusion Matrix for Test Set
cm_test = confusion_matrix(y_test, y_test_pred)

# Print Confusion Matrix
print("Confusion Matrix (Training Set):\n", cm_train)
print("\nConfusion Matrix (Test Set):\n", cm_test)

# Plot Confusion Matrix
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.heatmap(cm_train, annot=True, fmt='g', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix (Training Set)')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 2, 2)
sns.heatmap(cm_test, annot=True, fmt='g', cmap='Oranges', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()


#%%
#Random Forest
model_name = "Random Forest"
model = RandomForestClassifier()
model.fit(X_train, y_train)  

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


print_model_evaluation(model_name, y_train, y_train_pred, y_test, y_test_pred)

# Plot accuracy scores directly
plt.bar(['Training', 'Test'], [accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)], color=['blue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Accuracy Plot for Random Forest')
plt.show()

# Confusion Matrix for Training Set
cm_train = confusion_matrix(y_train, y_train_pred)

# Confusion Matrix for Test Set
cm_test = confusion_matrix(y_test, y_test_pred)

# Print Confusion Matrix
print("Confusion Matrix (Training Set):\n", cm_train)
print("\nConfusion Matrix (Test Set):\n", cm_test)

# Plot Confusion Matrix
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.heatmap(cm_train, annot=True, fmt='g', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix (Training Set)')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 2, 2)
sns.heatmap(cm_test, annot=True, fmt='g', cmap='Oranges', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()


#%%
#Logistic Regression
model_name = "Logistic Regression"
model = LogisticRegression()
model.fit(X_train, y_train)  

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print_model_evaluation(model_name, y_train, y_train_pred, y_test, y_test_pred)

# Plot Accuracy
plt.bar(['Training', 'Test'], [accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)], color=['blue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Accuracy Plot for Logistic Regression')  
plt.show()

# Confusion Matrix for Training Set
cm_train = confusion_matrix(y_train, y_train_pred)

# Confusion Matrix for Test Set
cm_test = confusion_matrix(y_test, y_test_pred)

# Print Confusion Matrix
print("Confusion Matrix (Training Set):\n", cm_train)
print("\nConfusion Matrix (Test Set):\n", cm_test)

# Plot Confusion Matrix
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.heatmap(cm_train, annot=True, fmt='g', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix (Training Set)')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 2, 2)
sns.heatmap(cm_test, annot=True, fmt='g', cmap='Oranges', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()


#%%
#Long Short-Term Memory(LSTM)
# Assuming X_train and X_test are arrays of strings
X_train, X_test, y_train, y_test = train_test_split(df['text'].values, df['experience'].values, test_size=0.2, random_state=0)

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Pad sequences
lengths = [len(x) for x in X_train]
max_len = max(lengths)
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the LSTM model
embedding_dim = 100
lstm_units = 128
batch_size = 64
epochs = 10

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_len))
model.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
model.add(BatchNormalization())  
model.add(Dropout(0.7))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.7))
model.add(Dense(y_train.shape[1], activation='softmax'))
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Train the model
history = model.fit(X_train, y_train, epochs=epochs, verbose=1, validation_data=(X_test, y_test), batch_size=batch_size, callbacks=[early_stopping])

# Plot accuracy
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.xlabel('Epoch')
plt.xlim([0, epochs])
plt.title('Model Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xlabel('Epoch')
plt.xlim([0, epochs])
plt.title('Model Loss')
plt.legend()
plt.show()

# Predict on training and test sets
y_train_pred_prob = model.predict(X_train)
y_test_pred_prob = model.predict(X_test)

# Convert probabilities to class labels
y_train_pred = np.argmax(y_train_pred_prob, axis=1)
y_test_pred = np.argmax(y_test_pred_prob, axis=1)

# Use your print_model_evaluation function
print_model_evaluation('LSTM Model', y_train, y_train_pred, y_test, y_test_pred)

# Generate confusion matrix for training set
cm_train = confusion_matrix(np.argmax(y_train, axis=1), y_train_pred)
# Generate confusion matrix for test set
cm_test = confusion_matrix(np.argmax(y_test, axis=1), y_test_pred)

# Print Confusion Matrix
print("Confusion Matrix (Training Set):\n", cm_train)
print("\nConfusion Matrix (Test Set):\n", cm_test)

# Plot Confusion Matrix
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
plt.title('Confusion Matrix (Training Set)')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 2, 2)
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Oranges", xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()

#%%

#########################
# Hyperparameter Tuning #
#########################


# **Naive Bayes**
X = df['text']
y = df['experience']

# Split the data first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Defining the parameter grid
param_grid = {
    'tfidf__max_features': [1000, 5000, 10000, None],
    'tfidf__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'model__alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
    'model__fit_prior': [True, False],
    'tfidf__use_idf': [True, False],
    'tfidf__smooth_idf': [True, False],
    'tfidf__sublinear_tf': [True, False],
}
# Creating a pipeline with TfidfVectorizer and MultinomialNB
model = MultinomialNB()
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', model)
])
# Create StratifiedKFold instance
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# Performing GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=stratified_kfold, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print hyperparameter tuning results
print("\nHyperparameter Tuning: Naive Bayes")
print("*********************************")
print("Best Parameters:")
for key, value in grid_search.best_params_.items():
    print(f"{key:20}: {value}")

# Calculate training accuracy
y_train_pred = grid_search.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Calculate test accuracy
y_test_pred = grid_search.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training accuracy: {train_accuracy:.3f}")
print(f"Test accuracy: {test_accuracy:.3f}")

# Print classification report for test set
print("Classification Report (Test Set):\n", classification_report(y_test, y_test_pred))

# Confusion Matrix for Training Set
cm_train = confusion_matrix(y_train, y_train_pred)

# Confusion Matrix for Test Set
cm_test = confusion_matrix(y_test, y_test_pred)

# Print Confusion Matrix
print("Confusion Matrix (Training Set):\n", cm_train)
print("\nConfusion Matrix (Test Set):\n", cm_test)

# Specify class labels based on your problem
class_labels = sorted(y.unique())  # Assuming y is a pandas Series

# Plot Accuracy
plt.bar(['Training', 'Test'], [train_accuracy, test_accuracy], color=['blue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Accuracy Plot for Naive Bayes after Tuning')
plt.show()

# Plot Confusion Matrix for Training Set
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.heatmap(cm_train, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix (Training Set)')
plt.xlabel('Predicted')
plt.ylabel('True')

# Plot Confusion Matrix for Test Set
plt.subplot(1, 2, 2)
sns.heatmap(cm_test, annot=True, fmt='g', cmap='Oranges', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()


#%%
#SVM
# Create pipeline
tfidf = TfidfVectorizer()
svm = SVC()
pipeline = Pipeline([('tfidf', tfidf), ('svm', svm)])

# Hyperparameters to tune
kernels = ['linear', 'rbf']
gamma = [0.1, 1, 'auto']
C = [1, 10, 50]

# Hyperparameter grid
grid = {'tfidf__max_features': [10000, 50000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'svm__C': C,
        'svm__gamma': gamma,
        'svm__kernel': kernels}

# Grid search
pipeline = Pipeline([('tfidf', tfidf), ('svm', svm)])
# Create StratifiedKFold instance
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
cv = GridSearchCV(pipeline, grid, cv=stratified_kfold)
cv.fit(X_train, y_train)

# Best model
best_model = cv.best_estimator_

# Print hyperparameter tuning results
print("\nHyperparameter Tuning: SVM")
print("**************************")
print("Best Parameters:")
for key, value in cv.best_params_.items():
    print(f"{key:20}: {value}")

# Calculate training accuracy
y_train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Calculate test accuracy
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training accuracy: {train_accuracy:.3f}")
print(f"Test accuracy: {test_accuracy:.3f}")

# Print classification report for test set
print("\nClassification Report (Test Set):\n", classification_report(y_test, y_test_pred))

# Confusion Matrix for Training Set
cm_train = confusion_matrix(y_train, y_train_pred)

# Confusion Matrix for Test Set
cm_test = confusion_matrix(y_test, y_test_pred)

# Print Confusion Matrix
print("Confusion Matrix (Training Set):\n", cm_train)
print("\nConfusion Matrix (Test Set):\n", cm_test)

# Specify class labels based on your problem
class_labels = sorted(y.unique())  # Assuming y is a pandas Series

# Plot Accuracy
plt.bar(['Training', 'Test'], [train_accuracy, test_accuracy], color=['blue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Accuracy Plot for SVM after Tuning')
plt.show()

# Plot Confusion Matrix for Training Set
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.heatmap(cm_train, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix (Training Set)')
plt.xlabel('Predicted')
plt.ylabel('True')

# Plot Confusion Matrix for Test Set
plt.subplot(1, 2, 2)
sns.heatmap(cm_test, annot=True, fmt='g', cmap='Oranges', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()


#%%
#Logistic Regression
# Create pipeline
tfidf = TfidfVectorizer()
lr = LogisticRegression()
pipe = Pipeline([('vect', tfidf), ('model', lr)])

# Hyperparameters to tune
penalty = ['l1', 'l2']
c_values = [0.1, 1, 10]
solver = ['liblinear', 'saga']

# Create grid
grid = dict(vect__ngram_range=[(1,1), (1,2)],
            vect__max_features=[5000, 10000, None],
            model__penalty=penalty,
            model__C=c_values,
            model__solver=solver)

stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# GridSearch
cv = GridSearchCV(pipe, grid, cv=stratified_kfold, n_jobs=-1, verbose=2)

# Fit grid search
cv.fit(X_train, y_train)

# Best model
best_model = cv.best_estimator_

# Print hyperparameter tuning results
print("\nHyperparameter Tuning: Logistic Regression")
print("*********************************")
print("Best Parameters:")
for key, value in cv.best_params_.items():
    print(f"{key:20}: {value}")

# Calculate training accuracy
y_train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Calculate test accuracy
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training accuracy: {train_accuracy:.3f}")
print(f"Test accuracy: {test_accuracy:.3f}")

# Print classification report for test set
print("Classification Report (Test Set):\n", classification_report(y_test, y_test_pred))

# Confusion Matrix for Training Set
cm_train = confusion_matrix(y_train, y_train_pred)

# Confusion Matrix for Test Set
cm_test = confusion_matrix(y_test, y_test_pred)

# Print Confusion Matrix
print("Confusion Matrix (Training Set):\n", cm_train)
print("\nConfusion Matrix (Test Set):\n", cm_test)

# Specify class labels based on your problem
class_labels = sorted(y.unique())  # Assuming y is a pandas Series

# Plot Accuracy
plt.bar(['Training', 'Test'], [train_accuracy, test_accuracy], color=['blue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Accuracy Plot for Logistic Regression after tuning')
plt.show()

# Plot Confusion Matrix for Training Set
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.heatmap(cm_train, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix (Training Set)')
plt.xlabel('Predicted')
plt.ylabel('True')

# Plot Confusion Matrix for Test Set
plt.subplot(1, 2, 2)
sns.heatmap(cm_test, annot=True, fmt='g', cmap='Oranges', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()


#%%
#Random Forest 

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Vectorize text data
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)

# Define RandomForestClassifier
rf = RandomForestClassifier()

# Hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# Grid search
grid_search = GridSearchCV(rf, param_grid, cv=stratified_kfold, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_
# Print hyperparameter tuning results
print("\nHyperparameter Tuning: Random Foresr")
print("*********************************")
print("Best Parameters:")
for key, value in grid_search.best_params_.items():
    print(f"{key:20}: {value}")

# Training accuracy
y_train_pred = best_rf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Test accuracy
X_test = vectorizer.transform(X_test)
y_test_pred = best_rf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Training accuracy: {train_accuracy:.3f}')
print(f'Test accuracy: {test_accuracy:.3f}')

# Print classification report for test set
print("Classification Report (Test Set):\n", classification_report(y_test, y_test_pred))

# Confusion Matrix for Training Set
cm_train = confusion_matrix(y_train, y_train_pred)

# Confusion Matrix for Test Set
cm_test = confusion_matrix(y_test, y_test_pred)

# Print Confusion Matrix
print("Confusion Matrix (Training Set):\n", cm_train)
print("\nConfusion Matrix (Test Set):\n", cm_test)

# Specify class labels based on your problem
class_labels = sorted(y.unique())  # Assuming y is a pandas Series

# Plot Accuracy
plt.bar(['Training', 'Test'], [train_accuracy, test_accuracy], color=['blue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Accuracy Plot for Random Forest')
plt.show()

# Plot Confusion Matrix for Training Set
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.heatmap(cm_train, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix (Training Set)')
plt.xlabel('Predicted')
plt.ylabel('True')

# Plot Confusion Matrix for Test Set
plt.subplot(1, 2, 2)
sns.heatmap(cm_test, annot=True, fmt='g', cmap='Oranges', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()


#%%
#LSTM

#Splitting the dataset 
X_train, X_test, y_train, y_test = train_test_split(df['text'].values, df['experience'].values, test_size=0.2, random_state=0)

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Pad sequences
max_len = max(len(x) for x in X_train)
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the search parameters
param_grid = {
    'embedding_dim': [50, 100],
    'lstm_units': [64, 128],
    'batch_size': [32, 64],
    'epochs': [10, 20],
    'dropout_rate': [0.7]
}

# Creating a custom estimator class
class MyKerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, embedding_dim=50, lstm_units=64, batch_size=32, epochs=10, dropout_rate=0.5):
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout_rate = dropout_rate

    def fit(self, X, y, **fit_params):
        callbacks = fit_params.pop('callbacks', [])  # Remove 'callbacks' from fit_params
        model = Sequential()
        model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=self.embedding_dim, input_length=max_len))
        model.add(Bidirectional(LSTM(self.lstm_units, return_sequences=True)))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        callbacks += [early_stopping]

        # Save training history for plotting
        history = model.fit(X, y, epochs=self.epochs, verbose=1, validation_data=(X_test, y_test), batch_size=self.batch_size, callbacks=callbacks, **fit_params)

        self.model = model
        self.history = history  # Save history for plotting

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

    def score(self, X, y):
        y_pred = to_categorical(self.predict(X), num_classes=y.shape[1])
        return accuracy_score(y, y_pred)

    def get_params(self, deep=True):
        return {
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'dropout_rate': self.dropout_rate
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

# Adding Kfolds   
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# Create the grid search object
grid_search = GridSearchCV(estimator=MyKerasClassifier(), param_grid=param_grid, scoring='accuracy', cv=stratified_kfold, n_jobs=-1, verbose=2)
# Fit the grid search object
grid_search.fit(X_train, y_train, callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

# Get the best parameters
best_params = grid_search.best_params_
print('Hyperparameter tuning: LSTM')
print('*****************************')
print('Best Parameters:', best_params)

# Retrain the model with the best parameters
best_model = MyKerasClassifier(**best_params)
best_model.fit(X_train, y_train)

# Evaluate on training set
train_accuracy = best_model.score(X_train, y_train)
print("Training Accuracy:", train_accuracy)

# Evaluate on test set
test_accuracy = best_model.score(X_test, y_test)
print("Testing Accuracy:", test_accuracy)

# Make predictions
y_pred = best_model.predict(X_test)

# Print confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion Matrix:")
print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))

# Print classification report
from sklearn.metrics import classification_report
print("Classification Report:")
print(classification_report(np.argmax(y_test, axis=1), y_pred))

# Plot training history
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(best_model.history.history['accuracy'])
plt.plot(best_model.history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(best_model.history.history['loss'])
plt.plot(best_model.history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()


#%%
#######################################
# Visualisation after getting Results #
#######################################

#Same Code used for before tuning as well.(just change the values to before tuning accuracy)

#Plotting Comparison graph after tuning
datasets = ['Transformers', 'Google', 'English']
models = ['Naive Bayes', 'SVM', 'Random Forest', 'Logistic Regression', 'LSTM']

# Training accuracies
train_acc = [[76.5, 94, 81.5, 67.7, 84.1],  
             [86.5, 90.9, 99.9, 99.9, 88.8],
             [97.2, 99.8, 99.8, 99.5, 93.4]]
             
# Plot training accuracy
# fig, ax = plt.subplots() 

x = np.arange(len(models))
width = 0.2 

# Training accuracy plot
fig, ax = plt.subplots(figsize=(10, 5)) 

colors = ['#6FA836', '#FCB044', '#3F88C5']

ax.bar(x - width, train_acc[0], width, color=colors[0], label=datasets[0]+' Train')
ax.bar(x, train_acc[1], width, color=colors[1], label=datasets[1]+' Train')
ax.bar(x + width, train_acc[2], width, color=colors[2], label=datasets[2]+' Train')

   
ax.set_ylabel('Accuracy')
ax.set_title('Training Accuracy after Tuning')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

plt.tight_layout()
plt.show()


# Testing accuracies
test_acc = [[57.6, 60.9, 58.8, 60.7, 56.5],
            [67.5, 68.5, 65.6, 68.7, 61.0],
            [75.4, 76.4, 72.3, 75.5, 69.7]]
            
# Plot test accuracy
# fig, ax = plt.subplots()

x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 5))

colors = ['#CC3F44', '#FF9234', '#1985AD']

ax.bar(x - width, test_acc[0], width, color=colors[0], label=datasets[0]+' Test')
ax.bar(x, test_acc[1], width, color=colors[1], label=datasets[1]+' Test') 
ax.bar(x + width, test_acc[2], width, color=colors[2], label=datasets[2]+' Test')

ax.set_ylabel('Accuracy')
ax.set_title('Testing Accuracy after Tuning')  
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

plt.tight_layout() 
plt.show()

#%%
#Plotting Overall Accuracy Graph
# Dataset names
datasets = ['Transformers','Google', 'English']

# Model names
models = ['Naive Bayes', 'SVM', 'Random Forest', 'Logistic Regression', 'LSTM']

# Accuracy before tuning 
train_acc_before = np.array([[76.1, 93.2, 99.8, 80.4, 71.5], #Transformers
                             [78.3, 95.3, 99.9, 84.8, 83.1], #Google
                             [74.6, 97.4, 99.8, 88.3, 94]])  #English
                    
test_acc_before = np.array([[56.5, 59.4, 58.7, 60.0, 54.4],
                            [65.2, 67.2, 65.7, 68.1, 63.2],
                            [64.5, 74.6, 72.8, 73.7, 71]])
                   
# Accuracy after tuning                 
train_acc_after = np.array([[76.5, 94, 81.5, 67.7, 84.1],
                            [86.5, 90.9, 99.9, 99.9, 88.8],
                            [97.2, 99.8, 99.8, 99.5, 93.4]])
                   
test_acc_after = np.array([[57.6, 60.9, 58.8, 60.7, 56.5],
                           [67.5, 68.5, 65.6, 68.7, 61.0],
                           [75.4, 76.4, 72.3, 75.5, 69.7]])

# Plotting
bar_width = 0.2
bar_positions_train_before = np.arange(len(models))
bar_positions_train_after = bar_positions_train_before + bar_width
bar_positions_test_before = bar_positions_train_after + bar_width
bar_positions_test_after = bar_positions_test_before + bar_width

colors_before = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a']
colors_after = ['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6']

for i, dataset in enumerate(datasets):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for j, model in enumerate(models):
        ax.bar(bar_positions_train_before[j], train_acc_before[i, j], bar_width, label=f'{model} - Train Before', color=colors_before[j], alpha=0.7)
        ax.bar(bar_positions_train_after[j], train_acc_after[i, j], bar_width, label=f'{model} - Train After', color=colors_after[j], alpha=0.7)

        ax.bar(bar_positions_test_before[j], test_acc_before[i, j], bar_width, label=f'{model} - Test Before', color=colors_before[j], alpha=0.5)
        ax.bar(bar_positions_test_after[j], test_acc_after[i, j], bar_width, label=f'{model} - Test After', color=colors_after[j], alpha=0.5)

        # Display numbers on top of bars
        ax.text(bar_positions_train_before[j], train_acc_before[i, j] + 1, f'{train_acc_before[i, j]:.1f}', ha='center')
        ax.text(bar_positions_train_after[j], train_acc_after[i, j] + 1, f'{train_acc_after[i, j]:.1f}', ha='center')
        ax.text(bar_positions_test_before[j], test_acc_before[i, j] + 1, f'{test_acc_before[i, j]:.1f}', ha='center')
        ax.text(bar_positions_test_after[j], test_acc_after[i, j] + 1, f'{test_acc_after[i, j]:.1f}', ha='center')

    ax.set_xticks(bar_positions_test_before)
    ax.set_xticklabels(models)
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Accuracy Comparison for {dataset} Dataset - Before and After Tuning')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

