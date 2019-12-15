# -*- coding: utf-8 -*-

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
# NLP
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
# Preprocess functions in keras
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
# Models in keras
from keras.models import Sequential
from keras import initializers
from keras.layers import Dropout, Activation,Embedding,CuDNNGRU,Bidirectional,Convolution1D, MaxPooling1D, Input, Dense, BatchNormalization
from keras.layers.recurrent import LSTM, GRU
from keras.layers import concatenate, Conv1D, CuDNNLSTM
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD, RMSprop
from keras import regularizers
from keras.layers.core import Reshape, Flatten
from keras.utils import to_categorical
#Word2Vec
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors

sp500 = pd.read_csv('SP_500_label.csv')
Corpus = pd.read_csv('corpus.csv')[['datetime','text_final']]
Corpus_date = pd.read_csv('corpus_date.csv')

tech_indicators = pd.read_csv('tech_indicators_update.csv')
tech_indicators['Date'] = tech_indicators['Date'].apply(lambda x:datetime.strptime(x,'%m/%d/%y'))
tech_indicators = tech_indicators[tech_indicators.Date.isin(sp500.Date)]
tech_indicators = tech_indicators.reset_index(drop = True)

tech_indicators1 = tech_indicators[tech_indicators.columns[~tech_indicators.columns.isin(['Date'])]]
tech_train = tech_indicators1.iloc[:1309,]
tech_test = tech_indicators1.iloc[1309:,]
tech_test = tech_test.reset_index(drop = True)
tech_train = np.array(tech_train)
tech_test = np.array(tech_test)

t = Tokenizer()
t.fit_on_texts(Corpus_date['docs'])

# Take a look at our tokens
print(list(t.word_index.items())[-1])
print(len(list(t.word_index.items())))

vocab_size = len(t.word_index) + 1

encoded_docs = t.texts_to_sequences(Corpus_date['docs'])
# pad documents to a max length
max_length = max(len(x) for x in encoded_docs)
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

embeddings_index = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

embedding_dim = 300
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in t.word_index.items():
    if word in embeddings_index:
        embedding_matrix[i] = embeddings_index[word]
    else:   
        embedding_matrix[i] = np.random.normal(0,np.sqrt(0.25),embedding_dim)
    if i%5000 ==0:
        print(i)

x_train, x_test = padded_docs[:1309],padded_docs[1309:]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
label = sp500['label']
y_ohe = ohe.fit_transform(label.values.reshape(-1, 1))
# type: np.ndarray
y_train, y_test = y_ohe[:1309],y_ohe[1309:]

x_train = np.array(x_train)
x_test = np.array(x_test)

file_path = "best_model.hdf5"
check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                              save_best_only = True, mode = "min")
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)
tech_num = len(tech_indicators1.columns)

def build_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):
    inp = Input(shape = (max_length,))
    x = Embedding(vocab_size, embedding_dim, weights = [embedding_matrix], trainable = False)(inp)
    x1 = SpatialDropout1D(dr)(x)

    x_gru = Bidirectional(CuDNNGRU(units, return_sequences = True))(x1)
    x1 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool1_gru = GlobalAveragePooling1D()(x1)
    max_pool1_gru = GlobalMaxPooling1D()(x1)
    
    x3 = Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool3_gru = GlobalAveragePooling1D()(x3)
    max_pool3_gru = GlobalMaxPooling1D()(x3)
    
    x_lstm = Bidirectional(CuDNNLSTM(units, return_sequences = True))(x1)
    x1 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool1_lstm = GlobalAveragePooling1D()(x1)
    max_pool1_lstm = GlobalMaxPooling1D()(x1)
    
    x3 = Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool3_lstm = GlobalAveragePooling1D()(x3)
    max_pool3_lstm = GlobalMaxPooling1D()(x3)
    
    
    x = concatenate([avg_pool1_gru, max_pool1_gru, avg_pool3_gru, max_pool3_gru,
                    avg_pool1_lstm, max_pool1_lstm, avg_pool3_lstm, max_pool3_lstm])
    x = BatchNormalization()(x)
    x = Dropout(0.2)(Dense(128,activation='relu') (x))
    x = BatchNormalization()(x)
    x = Dropout(0.2)(Dense(100,activation='relu') (x))
    x = Dense(2, activation = "softmax")(x)

    model = Model(inputs = inp, outputs = x)
    #sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    history = model.fit(x_train, y_train, batch_size = 128, epochs = 15, validation_split=0.1, 
                        verbose = 1, callbacks = [check_point, early_stop])
    model = load_model(file_path)
    return model

model = build_model(lr = 1e-4, lr_d = 0, units = 128, dr = 0.5)
pred = model.predict(x_test, batch_size = 1024)

y_true = label[1309:].tolist()

(eval_loss, eval_accuracy) = model.evaluate(x_test, y_test)
eval_accuracy

y_pred = np.round(np.argmax(pred, axis=1)).astype(int)
y_pred = pd.DataFrame(y_pred,columns=['NN'])
y_pred.head()

y_pred.to_csv('Neural_Networks_pred.csv')
