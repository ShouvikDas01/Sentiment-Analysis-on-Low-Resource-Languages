# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 22:48:57 2024

EXPLORING SENTIMENT ANALYSIS IN LOW RESOURCE LANGUAGE : 
    UNVEILING LIMITATIONS IN TRANSLATION LIBRARIES.

"This script translates the hindi dataset into english using transformers library and 
uses async function to asynchronously save the translated data."

@author: shouvik das
Student ID: 22196026
"""
#HuggingFace's Translation Library "Transformers"
from transformers import MarianMTModel, MarianTokenizer
from concurrent.futures import ThreadPoolExecutor
import nest_asyncio
import asyncio
import pandas as pd
from tqdm import tqdm
import csv


df = pd.read_csv(r'data/hindi/hindi-data-combined.csv')
df

nest_asyncio.apply()

#Initialising pre-trained model
model_name = "Helsinki-NLP/opus-mt-hi-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Defining translation function
def translate_text(text):
    """
    Translates a given text using the loaded MarianMT translation model.
    Args:
        text (str): The text to be translated.

    Returns:
        str: The translated text.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128) 
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# Defining translation coroutine
async def translate_and_save_batch(batch, output_file):
    """
    Translates a batch of text data in parallel and saves the results to a CSV file.
    Args:
        batch (pd.DataFrame): A batch of data containing text to be translated.
        output_file (str): The path to the output CSV file.
    """
    original_texts = batch["text"].tolist()
    translated_texts = list(tqdm(ThreadPoolExecutor().map(translate_text, original_texts), total=len(original_texts)))

    # Saving translated sentences to CSV file
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        for original_text, translated_text, experience in zip(original_texts, translated_texts, batch["experience"].tolist()):
            csv_writer.writerow([original_text, translated_text, experience])

# Translating asynchronously and save to CSV file using parallel processing
async def main():
    """
    Translates a dataset of Hindi product reviews into English using asynchronous 
    execution and parallel processing.
    Saves the translated data along with original text and experience ratings to a CSV file.
    """
    output_file = r'data/translated/transformers-data.csv'
    header = ["text", "Translated", "experience"]
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)

    batch_size = 8  #setting up batch size 
    for i in range(0, len(df), batch_size):
        batch_rows = df.iloc[i:i + batch_size]
        await translate_and_save_batch(batch_rows, output_file)

# Run the event loop
if __name__ == "__main__":
    asyncio.run(main())  