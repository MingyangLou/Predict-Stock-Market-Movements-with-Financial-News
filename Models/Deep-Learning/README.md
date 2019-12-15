## Text Processing
`corpus.py`
- Transform headlines to corpus
  * To lower case
  * Tokenize
  * Remove stop words & symbols
- Run once, store the result to a csv file
## Model Implementation
`CNN_BILSTM.py`
### CNN + BILSTM

The advantage of using CNN is that it can capture local information through convolution filters. Bidirectional LSTM helps the model to get not only past information, but also future information.

For word embedding, we used the Word2Vec model, which was pretrained from 100 billion words from Google News using the continuous bag-of-words architecture. Words not included in the Word2Vec were assigned random vectors. The convolutional layer took the result of embedding layer as input, and its output was pooled to a smaller dimension and was then fed into the BI-LSTM layer.

Code adapted from https://www.kaggle.com/parth05rohilla/bi-lstm-and-cnn-model-top-10/.

## Setup
The code is based on Keras 2.2.5 and as backend I recommend Tensorflow 1.14.0. I cannot ensure that the code works with different versions for Keras / Tensorflow or with different backends for Keras. The BILSTM layer requires GPU, I ran the model on Google Colab.
