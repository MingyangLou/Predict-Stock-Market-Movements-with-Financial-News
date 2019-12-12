#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:58:06 2019

@author: l.kate
"""
# Load pakages
import pandas as pd
import numpy as np
import string
from datetime import datetime
import tensorflow as tf
import os
import re
import json
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score as acc
import matplotlib.pyplot as plt

# Preprocess functions in keras
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
# Models in keras
from keras.models import Sequential
from keras import initializers
from keras.layers import Dropout, Activation,Embedding, Convolution1D, MaxPooling1D, Input, Dense, BatchNormalization, Flatten, Reshape, Concatenate
from keras.layers.recurrent import LSTM, GRU
from keras.layers import concatenate
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras import regularizers

# Load files
sp500 = pd.read_csv('SP_500_label.csv')
#news = pd.read_csv("news_filtered_1210.csv")[['datetime','cleanheadline']]
news = pd.read_csv("financial_news_preprocessed.csv")[['datetime','headline']]
tech_indicators = pd.read_csv('tech_indicators_update.csv')

# Clean data
#news.datetime = news.datetime.map(lambda x:x[:10])
news['datetime'] = news['datetime'].apply(lambda x:datetime.strptime(x,'%m/%d/%y'))
news = news.sort_values('datetime')
news = news.reset_index(drop = True)
#news.rename(columns={"cleanheadline": "headline"},inplace = True)
# Remove the extra dates that are in label

sp500['Date'] = sp500['Date'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
sp500 = sp500[sp500.Date.isin(news.datetime)]
sp500 = sp500.sort_values('Date')
sp500 = sp500.reset_index(drop = True)
sp500['year'] = sp500.Date.map(lambda x: x.year)

tech_indicators['Date'] = tech_indicators['Date'].apply(lambda x:datetime.strptime(x,'%m/%d/%y'))
tech_indicators = tech_indicators[tech_indicators.Date.isin(news.datetime)]
tech_indicators = tech_indicators.reset_index(drop = True)
# Create a list of the closing prices and their corresponding daily headlines from the news
label = []
headlines = []
for row in sp500.iterrows():
    daily_headlines = []
    date = row[1]['Date']
    label.append(row[1]['label'])
    for row_ in news[news.datetime==date].iterrows():
        daily_headlines.append(row_[1]['headline'])
    
    # Track progress
    headlines.append(daily_headlines)
    if len(label) % 500 == 0:
        print(len(label))
        
# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python        
with open('contractions.txt') as json_file:
    contractions = json.load(json_file)

def clean_text(text):
    '''Remove unwanted characters and format the text to create fewer nulls word embeddings'''
    
    # Convert words to lower case
    text = text.lower()
    # Remove punctuations
    text = re.sub(r'[_"\-;%()|.,+&=*%.,!?/:#@\[\]]', ' ', text)
    # Replace contractions with their longer forms 
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    # Cheack punctuation and symbols again
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)

    return text

# Clean the headlines
clean_headlines = []

for daily_headlines in headlines:
    clean_daily_headlines = ''
    for headline in daily_headlines:
        clean_daily_headlines = clean_daily_headlines + clean_text(headline)
    clean_headlines.append(clean_daily_headlines)


t = Tokenizer()
t.fit_on_texts(clean_headlines)
# Take a look at our tokens
print(list(t.word_index.items())[-1])
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(clean_headlines)
# pad documents to a max length of 4 words
max_length = 200
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# load the whole embedding into memory
embeddings_index = {}
with open('glove.840B.300d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_dim = 300
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in t.word_index.items():
	if word in embeddings_index:
		embedding_matrix[i] = embeddings_index[word]

'''
Split the data into training set and test set
cutoff year: 2012 (train:2006-2011 [1309 observations], test:2012-2013)
Params:
    x_train/x_test: financial news headlines
    tech_train/tech_test: technical indicators
    y_train/y_test: labels
'''
x_train, x_test = padded_docs[:1309],padded_docs[1309:]
y_train, y_test = label[:1309],label[1309:]


tech_indicators1 = tech_indicators[tech_indicators.columns[~tech_indicators.columns.isin(['Date'])]]
tech_train = tech_indicators1.iloc[:1309,]
tech_test = tech_indicators1.iloc[1309:,]
tech_test = tech_test.reset_index(drop = True)

x_train = np.array(x_train)
x_test = np.array(x_test)
tech_train = np.array(tech_train)
tech_test = np.array(tech_test)
# Reshape tech_train and tech_test to fit the LSTM layer 
tech_train = np.reshape(tech_train,(tech_train.shape[0],1,tech_train.shape[1]))
tech_test = np.reshape(tech_test,(tech_test.shape[0],1,tech_test.shape[1]))

y_train = np.array(y_train)
y_test = np.array(y_test)
print(len(x_train)) #1309
print(len(x_test))  #472
print(len(tech_train)) #1309
print(len(tech_test))  #472
print(x_train.shape,tech_train.shape)
# Number of tech indicators
tech_num = len(tech_indicators1.columns)

filter_length1 = 3
filter_length2 = 5
dropout = 0.5
learning_rate = 0.01
weights = initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=2)
nb_filter = 64
rnn_output_size = 128
rnn_output_size_tech =50
hidden_dims = 128
wider = True
deeper = True

if wider == True:
    nb_filter *= 2
    rnn_output_size *= 2
    hidden_dims *= 2

def build_model():
    #model1 = Sequential()
    inputlayer1 = Input(shape=(max_length,))
    model1 = Embedding(vocab_size, 
                         embedding_dim,
                         weights=[embedding_matrix], 
                         input_length=max_length)(inputlayer1)
    model1 = Dropout(dropout)(model1)
    
    model1 = Convolution1D(filters = nb_filter, 
                             kernel_size = filter_length1, 
                             padding = 'same',
                            activation = 'relu')(model1)
    model1 = Dropout(dropout)(model1)
    
    if deeper == True:
        model1 = Convolution1D(filters = nb_filter, 
                                 kernel_size = filter_length1, 
                                 padding = 'same',
                                activation = 'relu')(model1)
        model1 = Dropout(dropout)(model1)
    
    model1 = MaxPooling1D(pool_size=2, strides=1,padding='valid')(model1)
    model1 = LSTM(rnn_output_size, 
                   activation=None,
                   kernel_initializer=weights,
                   dropout = dropout)(model1)
   
    model1 = Dense(1,kernel_initializer = weights)(model1)
    
    ####
    inputlayer2 = Input(shape = (1,tech_num)) 
    model2 = LSTM(rnn_output_size_tech, 
                   activation=None,
                   kernel_initializer=weights,
                   dropout = dropout)(inputlayer2)
    model2 = Dropout(dropout)(model2)
    
    
    model2 = Dense(1, kernel_initializer = weights)(model2)
   
    ####

    #model = Sequential()
    model = concatenate([model1,model2])
    model = Dense(hidden_dims, kernel_initializer=weights)(model)
    model = Dropout(dropout)(model)
    
    if deeper == True:
        model = Dense(hidden_dims//2, kernel_initializer=weights)(model)
        model = Dropout(dropout)(model)

    
    model = Dense(1, kernel_initializer = weights,)(model)
    model = Activation('relu')(model)
    model = Dense(1, kernel_initializer = weights)(model)
    model = Activation('softmax')(model)
    model_combined = Model(inputs = [inputlayer1,inputlayer2],outputs = model)
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model_combined.compile(loss='binary_crossentropy',
                  optimizer=sgd,metrics=['accuracy'])
    return model_combined
# Use grid search to help find a better model
for deeper in [False]:
    for wider in [True,False]:
        for learning_rate in [0.01]:
            for dropout in [0.3, 0.5]:
                model = build_model()
                print()
                print("Current model: Deeper={}, Wider={}, LR={}, Dropout={}".format(
                    deeper,wider,learning_rate,dropout))
                print()
                save_best_weights = 'question_pairs_weights_deeper={}_wider={}_lr={}_dropout={}.h5'.format(
                    deeper,wider,learning_rate,dropout)

                callbacks = [ModelCheckpoint(save_best_weights, monitor='val_loss', save_best_only=True),
                             EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto'),
                             ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=3)]

                history = model.fit([x_train,tech_train],
                                    y_train,
                                    batch_size=128,
                                    epochs=100,
                                    validation_split=0.15,
                                    verbose=True,
                                    shuffle=True,
                                    callbacks = callbacks)

# Make predictions with the best weights
deeper=False
wider=False
dropout=0.3
learning_Rate = 0.01
# Need to rebuild model in case it is different from the model that was trained most recently.
model = build_model()

model.load_weights('./question_pairs_weights_deeper={}_wider={}_lr={}_dropout={}.h5'.format(
                    deeper,wider,learning_rate,dropout))
predictions = model.predict([x_test,tech_test], verbose = True)
(eval_loss, eval_accuracy) = model.evaluate( [x_test,tech_test], y_test)
