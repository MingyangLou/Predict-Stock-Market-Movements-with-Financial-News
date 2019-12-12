#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:20:28 2019

@author: l.kate
"""
import pandas as pd
import numpy as np
import string
import re

# S&P 500 company name list
'''
After future cleaning, this compnay name list is used as the key to 
identify whether a news headline is directly related to this company. 
I chose the first word of each company's name.
'''
component = pd.read_csv('SP500_components.csv')['Head']
namelist = component.tolist()
# Clean the name list to match the news text
def clean_name(text):
    
    # Convert words to lower case
    text = text.lower()
    # Remove '.com'
    text = re.sub(r'\.com','',text)
    # Remove symbols
    text = re.sub(r'[_"\-;%()|.,+&=*%.,!?/:#@\[\]\']', '', text)
    # Remove extra spances
    text = re.sub(r' ','',text)
    return text

clean_list = []
for company in namelist:
    clean_list.append(clean_name(company))
print(clean_list)
# Add certain companies
l = ['google','aol']
clean_list = set(clean_list+l)
clean_company_list = pd.DataFrame(clean_list,columns=['company'])
clean_company_list.to_csv('company_list.csv')

news = pd.read_csv("financial_news_preprocessed.csv")

def clean_text(text):
    '''Remove unwanted characters and format the text to create fewer nulls word embeddings'''
    
    # Convert words to lower case
    text = text.lower()
    # Remove punctuations
    #text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[_"\-;%()|.,+&=*%.,!?/:#@\[\]\']', '', text)
   
    # Remove extra blanks
    text = re.sub(r'  ',' ',text)
    # Remove useless suffixes
    text = re.sub(r' plc', '', text)

    # Cheack punctuation and symbols again
    #text = text.translate(str.maketrans('', '', string.punctuation))
    # Replace contractions with their longer forms 
    
    return text

news['cleanheadline']=news.headline.map(lambda x: clean_text(x))
news['mark'] = news.cleanheadline.map(lambda x: 1 if any(item in set(clean_list) for item in x.split()) else 0)
filtered = news[news.mark==1][['datetime','cleanheadline']]
filtered.to_csv('news_filtered_1210.csv')
