#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 21:51:13 2019

@author: l.kate
"""

import pandas as pd
import numpy as np
import string
from datetime import datetime
import tensorflow as tf
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score as acc
import matplotlib.pyplot as plt

from keras.models import Sequential
#from tensorflow.python.keras import Sequential
from keras import initializers
from keras.layers import Dropout, Activation,Embedding, Convolution1D, MaxPooling1D, Input, Dense, BatchNormalization, Flatten, Reshape, Concatenate
#from tensorflow.python.layers import Dropout, Activation,Embedding, Convolution1D, MaxPooling1D, Input, Dense, BatchNormalization, Flatten, Reshape, Concatenate
from keras.layers.recurrent import LSTM, GRU
from keras.layers import concatenate
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras import regularizers

# Load files
#cd '/Users/l.kate/Downloads/242Project'
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
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}
#nltk.download('stopwords')
def clean_text(text):
    '''Remove unwanted characters and format the text to create fewer nulls word embeddings'''
    
    # Convert words to lower case
    text = text.lower()
    # Remove punctuations
    #text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[_"\-;%()|.,+&=*%.,!?/:#@\[\]]', ' ', text)
    # Remove line breaks
    text = re.sub(r'\n','',text)
    # Remove 'reuters'
    text = re.sub(r' reuters ',' ',text)
    # Remove extra blanks
    text = re.sub(r'  ',' ',text)
    # Remove useless suffixes
    text = re.sub(r' plc', '', text)

    # Cheack punctuation and symbols again
    text = text.translate(str.maketrans('', '', string.punctuation))
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
    
    # Remove stopwords
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)

    # stem words
    #text = ps.stem(text)

    return text

# Clean the headlines
clean_headlines = []

for daily_headlines in headlines:
    clean_daily_headlines = []
    for headline in daily_headlines:
        clean_daily_headlines.append(clean_text(headline))
    clean_headlines.append(clean_daily_headlines)

# Find the number of times each word was used and the size of the vocabulary
word_counts = {}

for date in clean_headlines:
    for headline in date:
        for word in headline.split():
            if word not in word_counts:
                word_counts[word] = 1
            else:
                word_counts[word] += 1
            
print("Size of Vocabulary:", len(word_counts))

# Load GloVe's embeddings
embeddings_index = {}
with open('glove.840B.300d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

print('Word embeddings:', len(embeddings_index))

# Find the number of words that are missing from GloVe, and are used more than our threshold.
missing_words = 0
threshold = 10

for word, count in word_counts.items():
    if count > threshold:
        if word not in embeddings_index:
            missing_words += 1
            
missing_ratio = round(missing_words/len(word_counts),4)*100
            
print("Number of words missing from GloVe:", missing_words)
print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))
#Number of words missing from GloVe: 224
#Percent of words that are missing from vocabulary: 0.83%
#Number of words missing from GloVe: 6421
#Percent of words that are missing from vocabulary: 4.55%
# Limit the vocab that we will use to words that appear ≥ threshold or are in GloVe

#dictionary to convert words to integers
vocab_to_int = {} 

value = 0
for word, count in word_counts.items():
    if count >= threshold or word in embeddings_index:
        vocab_to_int[word] = value
        value += 1

# Special tokens that will be added to our vocab
codes = ["<UNK>","<PAD>"]   

# Add codes to vocab
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)

# Dictionary to convert integers to words
int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word

usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

print("Total Number of Unique Words:", len(word_counts))
print("Number of Words we will use:", len(vocab_to_int))
print("Percent of Words we will use: {}%".format(usage_ratio))
#Total Number of Unique Words: 27002
#Number of Words we will use: 21711
#Percent of Words we will use: 80.41%

#Total Number of Unique Words: 141274
#Number of Words we will use: 92953
#Percent of Words we will use: 65.8%

# Need to use 300 for embedding dimensions to match GloVe's vectors.
embedding_dim = 300

nb_words = len(vocab_to_int)
# Create matrix with default values of zero
word_embedding_matrix = np.zeros((nb_words, embedding_dim))
for word, i in vocab_to_int.items():
    if word in embeddings_index:
        word_embedding_matrix[i] = embeddings_index[word]
    else:
        # If word not in GloVe, create a random embedding for it
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        embeddings_index[word] = new_embedding
        word_embedding_matrix[i] = new_embedding

# Check if value matches len(vocab_to_int)
print(len(word_embedding_matrix))

# Change the text from words to integers
# If word is not in vocab, replace it with <UNK> (unknown)
word_count = 0
unk_count = 0

int_headlines = []

for date in clean_headlines:
    int_daily_headlines = []
    for headline in date:
        int_headline = []
        for word in headline.split():
            word_count += 1
            if word in vocab_to_int:
                int_headline.append(vocab_to_int[word])
            else:
                int_headline.append(vocab_to_int["<UNK>"])
                unk_count += 1
        int_daily_headlines.append(int_headline)
    int_headlines.append(int_daily_headlines)

unk_percent = round(unk_count/word_count,4)*100

print("Total number of words in headlines:", word_count)
print("Total number of UNKs in headlines:", unk_count)
print("Percent of words that are UNK: {}%".format(unk_percent))

# Find the length of headlines
lengths = []
for date in int_headlines:
    for headline in date:
        lengths.append(len(headline))

# Create a dataframe so that the values can be inspected
lengths = pd.DataFrame(lengths, columns=['counts'])
lengths.describe()

# Limit the length of a day's news to 200 words, and the length of any headline to 16 words.
# These values are chosen to not have an excessively long training time and 
# balance the number of headlines used and the number of words from each headline.
#max_headline_length = 16
#max_daily_length = 200
max_headline_length = 16
max_daily_length = 200
pad_headlines = []

for date in int_headlines:
    pad_daily_headlines = []
    for headline in date:
        # Add headline if it is less than max length
        if len(headline) <= max_headline_length:
            for word in headline:
                pad_daily_headlines.append(word)
        # Limit headline if it is more than max length  
        else:
            headline = headline[:max_headline_length]
            for word in headline:
                pad_daily_headlines.append(word)
    
    # Pad daily_headlines if they are less than max length
    if len(pad_daily_headlines) < max_daily_length:
        for i in range(max_daily_length-len(pad_daily_headlines)):
            pad = vocab_to_int["<PAD>"]
            pad_daily_headlines.append(pad)
    # Limit daily_headlines if they are more than max length
    else:
        pad_daily_headlines = pad_daily_headlines[:max_daily_length]
    pad_headlines.append(pad_daily_headlines)

# training set: test set = 85:15
#x_train, x_test, y_train, y_test = train_test_split(pad_headlines, label, test_size = 0.15, random_state = 2)
x_train, x_test = pad_headlines[:1309],pad_headlines[1309:]
y_train, y_test = label[:1309],label[1309:]


tech_indicators1 = tech_indicators[tech_indicators.columns[~tech_indicators.columns.isin(['Date'])]]
tech_train = tech_indicators1.iloc[:1309,]
tech_test = tech_indicators1.iloc[1309:,]
tech_test = tech_test.reset_index(drop = True)
#x_train = pd.concat([x_train,tech_train[['MA_Cross','ROC_5','RSI']]],axis =1)
#x_test = pd.concat([x_test,tech_test[['MA_Cross','ROC_5','RSI']]],axis =1)

x_train = np.array(x_train)
x_test = np.array(x_test)
tech_train = np.array(tech_train)
tech_test = np.array(tech_test)
tech_train = np.reshape(tech_train,(tech_train.shape[0],1,tech_train.shape[1]))
tech_test = np.reshape(tech_test,(tech_test.shape[0],1,tech_test.shape[1]))

y_train = np.array(y_train)
y_test = np.array(y_test)
print(len(x_train)) #1309
print(len(x_test))  #472
print(len(tech_train)) #1309
print(len(tech_test))  #472
print(x_train.shape,tech_train.shape)
tech_num = len(tech_indicators1.columns)
filter_length1 = 3
filter_length2 = 5
dropout = 0.5
learning_rate = 0.001
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

'''
def build_model():
    
    model1 = Sequential()
    
    model1.add(Embedding(nb_words, 
                         embedding_dim,
                         weights=[word_embedding_matrix], 
                         input_length=max_daily_length))
    model1.add(Dropout(dropout))
    
    model1.add(Convolution1D(filters = nb_filter, 
                             kernel_size = filter_length1, 
                             padding = 'same',
                            activation = 'relu'))
    model1.add(Dropout(dropout))
    
    if deeper == True:
        model1.add(Convolution1D(filters = nb_filter, 
                                 kernel_size = filter_length1, 
                                 padding = 'same',
                                activation = 'relu'))
        model1.add(Dropout(dropout))
    
    model1.add(LSTM(rnn_output_size, 
                   activation=None,
                   kernel_initializer=weights,
                   dropout = dropout))
    ####

    model2 = Sequential()
    
    model2.add(Embedding(nb_words, 
                         embedding_dim,
                         weights=[word_embedding_matrix], 
                         input_length=max_daily_length))
    model2.add(Dropout(dropout))
    
    
    model2.add(Convolution1D(filters = nb_filter, 
                             kernel_size = filter_length2, 
                             padding = 'same',
                             activation = 'relu'))
    model2.add(Dropout(dropout))
    
    if deeper == True:
        model2.add(Convolution1D(filters = nb_filter, 
                                 kernel_size = filter_length2, 
                                 padding = 'same',
                                 activation = 'relu'))
        model2.add(Dropout(dropout))
    
    model2.add(LSTM(rnn_output_size, 
                    activation=None,
                    kernel_initializer=weights,
                    dropout = dropout))
    ####

    model = Sequential()
    
    #model.add(Merge([model1, model2], mode='concat'))
    #model.add(Concatenate()[model1,model2])
    merged = Concatenate()[model1,model2]
    #model.add(merged)
    #model.add(Dense(hidden_dims, kernel_initializer=weights))
    Dense(hidden_dims, kernel_initializer=weights)(merged)
    model.add(Dropout(dropout))
    
    if deeper == True:
        model.add(Dense(hidden_dims//2, kernel_initializer=weights))
        model.add(Dropout(dropout))

    model.add(Dense(1, 
                    kernel_initializer = weights,
                    name='output'))

    model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=learning_rate,clipvalue=1.0))
    return model
'''
def build_model():
    #model1 = Sequential()
    inputlayer = Input(shape=(max_daily_length+3,))
    model1 = Embedding(nb_words, 
                         embedding_dim,
                         weights=[word_embedding_matrix], 
                         input_length=max_daily_length+3)(inputlayer)
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
    
    model1 = LSTM(rnn_output_size, 
                   activation=None,
                   kernel_initializer=weights,
                   dropout = dropout)(model1)
    #model1 = Dense(1, 
    #                kernel_initializer = weights,
    #                name='output')(model1)
    model1 = Dense(1,kernel_initializer = weights)(model1)
    
    ####
    
    #model2 = Sequential()
    model2 = Embedding(nb_words, 
                         embedding_dim,
                         weights=[word_embedding_matrix], 
                         input_length=max_daily_length+3)(inputlayer)
    model2 = Dropout(dropout)(model2)
    
    
    model2 = Convolution1D(filters = nb_filter, 
                             kernel_size = filter_length2, 
                             padding = 'same',
                             activation = 'relu')(model2)
    model2 = Dropout(dropout)(model2)
    
    if deeper == True:
        model2 = Convolution1D(filters = nb_filter, 
                                 kernel_size = filter_length2, 
                                 padding = 'same',
                                 activation = 'relu')(model2)
        model2 = Dropout(dropout)(model2)
    
    model2 = LSTM(rnn_output_size, 
                    activation=None,
                    kernel_initializer=weights,
                    dropout = dropout)(model2)
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
    model_combined = Model(inputs = inputlayer,outputs = model)
    model_combined.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=learning_rate,clipvalue=1.0),metrics=['accuracy'])
    return model_combined

def build_model1():
    #model1 = Sequential()
    inputlayer1 = Input(shape=(max_daily_length,))
    model1 = Embedding(nb_words, 
                         embedding_dim,
                         weights=[word_embedding_matrix], 
                         input_length=max_daily_length)(inputlayer1)
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
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model_combined.compile(loss='binary_crossentropy',
                  optimizer=sgd,metrics=['accuracy'])
    return model_combined
# Use grid search to help find a better model
for deeper in [False]:
    for wider in [True,False]:
        for learning_rate in [0.001]:
            for dropout in [0.3, 0.5]:
                model = build_model1()
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
learning_Rate = 0.001
# Need to rebuild model in case it is different from the model that was trained most recently.
model = build_model1()

model.load_weights('./question_pairs_weights_deeper={}_wider={}_lr={}_dropout={}.h5'.format(
                    deeper,wider,learning_rate,dropout))
predictions = model.predict([x_test,tech_test], verbose = True)
(eval_loss, eval_accuracy) = model.evaluate( [x_test,tech_test], y_test)                
