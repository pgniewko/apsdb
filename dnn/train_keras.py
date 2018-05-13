#! /usr/bin/env python

import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard

from nnutils import load_data

# Using keras to load the dataset with the top_words
top_words = 10000
sample_size = 10
journals_ = ['PRA','PRB','PRC','PRD','PRE']
number_of_classes = len( journals_ )
(X_train, y_train), (X_test, y_test) = load_data(SAMPLE_SIZE=sample_size, \
                                                     top_words_=top_words, \
                                                     feature_='abstract', \
                                                     yrange=[1990,2010], \
                                                     journals=journals_ )

print X_train
print y_train
sys.exit(1)
# Pad the sequence to the same length
max_length = 200
X_train = sequence.pad_sequences(X_train, maxlen=max_length, padding='post')
X_test = sequence.pad_sequences(X_test, maxlen=max_length, padding='post')

# Using embedding from Keras
embedding_vecor_length = 300
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_length))

# Convolutional model (3x conv, flatten, 2x dense)
model.add(Convolution1D(64, 3, padding='same'))
model.add(Convolution1D(32, 3, padding='same'))
model.add(Convolution1D(16, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(180,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(number_of_classes, activation='softmax'))

# Log to tensorboard
tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3, callbacks=[tensorBoardCallback], batch_size=64)

# Evaluation on the test set
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
