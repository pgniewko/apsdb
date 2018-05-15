#! /usr/bin/env python

import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard
from gensim.models.word2vec import Word2Vec

from nnutils import load_data
from nnutils import transform_Y
from nnutils import tokenize_text


def create_model_cnn_1( top_words, embedding_vecor_length, max_length, number_of_classes, embedding_matrix):
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, weights=[embedding_matrix], input_length=max_length, trainable=False) )

    # Convolutional model (3x conv, flatten, 2x dense)
    model.add(Convolution1D(64, 3, padding='same'))
    model.add(Convolution1D(32, 3, padding='same'))
    model.add(Convolution1D(16, 3, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(180,activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(number_of_classes, activation='softmax'))
 
    return model

def create_model_lstm_1( top_words, embedding_vecor_length, max_length, number_of_classes, lembedding_matrix, stm_out = 196):
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, weights=[embedding_matrix], input_length=max_length, trainable=False ))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(number_of_classes,activation='softmax'))
    return model


## LOAD Word2Vec model
opath = './model/word2vec/'
SIZE=100
WINDOW=5
MIN_COUNT=10
ITER=1
model = Word2Vec.load(opath+'Abstracts_Word2Vec_SIZE_%d_WINDOW_%d_MIN_COUNT_%d_ITER_%d.bin' %( SIZE,WINDOW,MIN_COUNT,ITER )  )
##


# Using keras to load the dataset with the top_words
sample_size = 5000
journals_ = ['PRA','PRB','PRC','PRD','PRE']

print("LOAD DATA")
(X_train,y_train),(X_test,y_test) = load_data(sample_size=sample_size, \
                                              feature_='abstract', \
                                              journals=journals_)


MAX_NB_WORDS=10000
MAX_SEQUENCE_LENGTH=200

print("TOKENIZE DATA")
emb, X_train, X_test, MAX_NB_WORDS = tokenize_text(X_train, X_test, model, \
                                   MAX_NB_WORDS=MAX_NB_WORDS, \
                                   MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,\
                                   EMBEDDING_DIM=SIZE)

y_train = transform_Y(y_train) 
y_test  = transform_Y(y_test)

# Define a network architecture
top_words = MAX_NB_WORDS
embedding_vecor_length = SIZE
max_length = MAX_SEQUENCE_LENGTH
number_of_classes = len(journals_)
print("CREATE DEEP NN MODEL")
model = create_model_cnn_1( top_words, embedding_vecor_length, max_length, number_of_classes, emb)
print(model.summary())

# Log to tensorboard
tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=25, callbacks=[tensorBoardCallback], batch_size=64)

# Evaluation on the test set
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
