# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 22:48:57 2024

EXPLORING SENTIMENT ANALYSIS IN LOW RESOURCE LANGUAGE : 
    UNVEILING LIMITATIONS IN TRANSLATION LIBRARIES.

This script translates the hindi dataset into english using google API library and 
uses async function to asynchronously save the translated data.

@author: shouvik das
Student ID: 22196026
"""
#Google Translation Library
from googletrans import Translator
import pandas as pd
import nest_asyncio
import asyncio
from tqdm import tqdm
import csv


#Initialising Tranforers Library for translating
# Applying nest_asyncio to handle asyncio in Jupyter environments
nest_asyncio.apply()

# Asynchronous function to translate and write rows
async def translate_and_write(row, writer):
    """
    Translates a single row of data and writes it to a CSV file.

    Args:
        row (pandas.Series): A row of data containing the text to be translated and its sentiment.
        writer (csv.writer): A CSV writer object for writing the translated data.
    """
    # Create a Translator object
    translator = Translator()
    
    # Extracting original text and sentiment from the DataFrame row
    original_text = row['text']
    sentiment = row['experience']  
    
    # Translating the original text from Hindi to English
    translated_text = translator.translate(original_text, src='hi', dest='en').text
    
    # Writing the original text, translated text, and sentiment to the CSV file
    writer.writerow([original_text, translated_text, sentiment])

# Asynchronous main function
async def main():
    """
    Translates a dataset of Hindi product reviews into English using Google Translate API and asynchronous execution.
    Saves the translated data along with original text and experience ratings to a CSV file.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(r'data/hindi/hindi-data-combined.csv')
    
    # Open a new CSV file for writing translated data
    with open(r'data/translated/google-data.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f) 
        writer.writerow(['original_text', 'translated_text', 'experience'])

        # Initialize tqdm progress bar
        progress = tqdm(total=len(df))
    
        # List to store asynchronous tasks
        tasks = []

        # Iterate through DataFrame rows
        for index, row in df.iterrows():
            # Append the translation task to the tasks list
            tasks.append(translate_and_write(row, writer))
            
            # Update progress bar
            progress.update(1)

        # Execute all asynchronous tasks concurrently
        await asyncio.gather(*tasks)
        
        # Closing the progress bar
        progress.close()

# Run the event loop with asyncio.run
if __name__ == '__main__':
    asyncio.run(main())
