#! /usr/bin/env python

import sys
import numpy
import itertools
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard
from gensim.models.word2vec import Word2Vec

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import confusion_matrix

from nnutils import load_data
from nnutils import transform_Y
from nnutils import tokenize_text

def get_rocs(model, X_test, y_test, number_of_classes):
    n_classes = number_of_classes 
    
    y_score = model.predict(X_test)
    y_pred_classes = y_score.argmax(axis=-1)  #probas_to_classes(y_score)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return (fpr,tpr,roc_auc)


def plot_rocs(mymodel, X_test, y_test, number_of_classes):
    lw=2
    n_classes = number_of_classes 

    # Get ROCs data
    fpr,tpr,roc_auc = get_rocs(mymodel, X_test, y_test, number_of_classes) 

    # Plot all ROC curves
    f, ax1 = plt.subplots(1, figsize=(7,7))
    
    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','forestgreen','salmon','darkorchid'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=15)
    plt.ylabel('True Positive Rate',fontsize=15)
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize=12)
    plt.xlabel('Predicted label',fontsize=12)

    plt.show()


def create_model_simple(top_words,embedding_vecor_length,max_length,number_of_classes,embedding_matrix):
    model = Sequential()
    model.add(Embedding(top_words,embedding_vecor_length,weights=[embedding_matrix],input_length=max_length,trainable=False))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(number_of_classes, activation='softmax'))
    return model


def create_model_cnn(top_words,embedding_vecor_length,max_length,number_of_classes,embedding_matrix):
    model = Sequential()
    model.add(Embedding(top_words,embedding_vecor_length,weights=[embedding_matrix],input_length=max_length,trainable=False))

    # Convolutional model (3x conv, flatten, 2x dense)
    model.add(Convolution1D(64, 5, activation='relu', padding='same'))
    model.add( MaxPooling1D(5) ) 
    model.add(Convolution1D(32, 5, activation='relu', padding='same'))
    model.add( MaxPooling1D(5) ) 
    model.add(Convolution1D(16, 5, activation='relu', padding='same'))
    model.add( MaxPooling1D(5) ) 
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(180,activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(number_of_classes, activation='softmax'))
    return model


def create_model_lstm(top_words,embedding_vecor_length,max_length,number_of_classes,embedding_matrix,lstm_out=196):
    model = Sequential()
    model.add(Embedding(top_words,embedding_vecor_length,weights=[embedding_matrix],input_length=max_length,trainable=False))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(number_of_classes,activation='softmax'))
    return model


def create_model_lstm_cnn(top_words,embedding_vecor_length,max_length,number_of_classes,embedding_matrix,lstm_out=196):
    model = Sequential()
    model.add(Embedding(top_words,embedding_vecor_length,weights=[embedding_matrix],input_length=max_length,trainable=False))
    model.add(Dropout(0.2))
    model.add(Convolution1D(32, 5, activation='relu',padding='same'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(number_of_classes,activation='softmax'))
    return model


def train_model(model_, X_train, y_train, epochs_=10, callbacks_=None, batch_size_=64):
    model_.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_.fit(X_train, y_train, epochs=epochs_, callbacks=callbacks_, batch_size=batch_size_)
    return model_


def test_model(model_, X_test, y_test):
    # Evaluation on the test set
    scores = model_.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))


def save_model(model_,opath='./model/keras/',fn='model.h5'):
    model_.save(opath + fn)


if __name__ == "__main__":
    ## LOAD Word2Vec model
    opath = './model/word2vec/'
    SIZE=200
    WINDOW=5
    MIN_COUNT=5
    ITER=5
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

    y_train_classes = y_train
    y_test_classes  = y_test 
    y_train = transform_Y(y_train) 
    y_test  = transform_Y(y_test)

    # Define a network architecture
    top_words = MAX_NB_WORDS
    embedding_vecor_length = SIZE
    max_length = MAX_SEQUENCE_LENGTH
    number_of_classes = len(journals_)
    print("CREATE DEEP NN MODEL")
    tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
    
    model_fnn      = create_model_simple( top_words, embedding_vecor_length, max_length, number_of_classes, emb)
    model_cnn      = create_model_cnn( top_words, embedding_vecor_length, max_length, number_of_classes, emb)
    model_lstm     = create_model_lstm( top_words, embedding_vecor_length, max_length, number_of_classes, emb)
    model_lstm_cnn = create_model_lstm_cnn( top_words, embedding_vecor_length, max_length, number_of_classes, emb)
    models = [model_fnn, model_cnn, model_lstm, model_lstm_cnn]
    f, ax1 = plt.subplots(1, figsize=(7,7))

    labels = ['FF','CNN-1D', 'LSTM','CNN-1D-LSTM']
    colors = ['aqua', 'darkorange', 'cornflowerblue','forestgreen','salmon','darkorchid'] 
    model_idx = 0
    for model in models:
        print(model.summary())
        # Log to tensorboard
        print("TRAINING MODEL: " + labels[model_idx])
        model = train_model(model, X_train, y_train, epochs_=10, callbacks_=[tensorBoardCallback])
        print("SAVING MODEL: "+ labels[model_idx])
        save_model(model, fn=labels[model_idx]+'.h5')
        print("EVALUATE MODEL: " + labels[model_idx])
        test_model(model, X_test, y_test)
        fpr,tpr,roc_auc = get_rocs(model, X_test, y_test, number_of_classes) 
  
        plt.plot(fpr["macro"], tpr["macro"],
            label=labels[model_idx]+' :average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
        color=colors[model_idx], linestyle=':', linewidth=4)
        model_idx += 1

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title('ROC for different NN architectures')
    plt.legend(loc="lower right")
    plt.show()


    plot_rocs(model_lstm_cnn, X_test, y_test, number_of_classes)

    y_score = model.predict(X_test)
    y_pred_classes = y_score.argmax(axis=-1)  #probas_to_classes(y_score)

    cnf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
    np.set_printoptions(precision=2)
    plot_confusion_matrix(cnf_matrix, journals_,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues)


