# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 22:48:57 2024

EXPLORING SENTIMENT ANALYSIS IN LOW RESOURCE LANGUAGE : 
    UNVEILING LIMITATIONS IN TRANSLATION LIBRARIES.

"This script allows to balance dataset by dropping number of samples to level 
samples."

@author: shouvik das
Student ID: 22196026
"""



import pandas as pd
df = pd.read_csv(r'data/augmented/covidaug.csv')
df['experience'] = df['experience'].replace({1: 'negative', 2: 'neutral', 3: 'positive'})
df


# In[2]:


# Display class distribution
class_distribution = df['experience'].value_counts()
print("Class Distribution of English Data:")
print(class_distribution)


# In[3]:



df2 = pd.read_csv(r'data/translated/transformers-data.csv')
df2


#%%


# Display class distribution
class_distribution = df2['experience'].value_counts()
print("Class Distribution:")
print(class_distribution)


#%%
#Balancing Transformers Data

# Assuming 'df' is your DataFrame with 'experience' and 'text' columns

# Count of positive (experience label 3) samples to drop
samples_to_drop = 385

# Identify the positive samples
data_samples = df2[df2['experience'] == 'positive']

# Randomly drop the specified number of positive samples
df2 = df2.drop(data_samples.sample(samples_to_drop).index)

# Display class distribution after dropping positive experiences
class_distribution_after_drop = df2['experience'].value_counts()
print("Class Distribution after dropping positive experiences:")
print(class_distribution_after_drop)


# In[6]:


df2.to_csv(r'data/balanced/transformers-data-bal.csv',index=False)


#%%
#Balancing English Data

samples_to_drop = 551

# Identify the positive samples
data_samples = df[df['experience'] == 'negative']

# Randomly drop the specified number of positive samples
df = df.drop(data_samples.sample(samples_to_drop).index)

# Display class distribution after dropping positive experiences
class_distribution_after_drop = df['experience'].value_counts()
print("Class Distribution after dropping negative experiences:")
print(class_distribution_after_drop)



# Count of positive (experience label 3) samples to drop
samples_to_drop = 1456

# Identify the positive samples
data_samples = df[df['experience'] == 'neutral']

# Randomly drop the specified number of positive samples
df = df.drop(data_samples.sample(samples_to_drop).index)

# Display class distribution after dropping positive experiences
class_distribution_after_drop = df['experience'].value_counts()
print("Class Distribution after dropping neutral experiences:")
print(class_distribution_after_drop)


# In[9]:


df.to_csv(r'data/balanced/eng-data-bal.csv',index=False)


#%%
#Balancing Google Dataset


df1 = pd.read_csv(r'data/translated/google-data.csv')
df1


samples_to_drop = 385

# Identify the positive samples
data_samples = df1[df1['experience'] == 'positive']

# Randomly drop the specified number of positive samples
df1 = df1.drop(data_samples.sample(samples_to_drop).index)

# Display class distribution after dropping positive experiences
class_distribution_after_drop = df1['experience'].value_counts()
print("Class Distribution after dropping positive experiences:")
print(class_distribution_after_drop)


df1.to_csv(r'data/balanced/google-data-bal.csv',index=False)


#%%
#To View Data after balance


a = pd.read_csv(r'data/balanced/google-data-bal.csv')
a


# In[14]:


b = pd.read_csv(r'data/balanced/transformers-data-bal.csv')
b


# In[8]:


import pandas as pd
c = pd.read_csv(r'data/balanced/eng-data-bal.csv')
c


#%%
#To check class distribution

# Display class distribution
class_distribution = c['experience'].value_counts() ##can cahnge datafram accordingly to check
print("Class Distribution:")
print(class_distribution)






