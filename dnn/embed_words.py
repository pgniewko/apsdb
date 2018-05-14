#! /usr/bin/env python

import sys
import numpy as np
import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def read_sentences(fname):
    fin = open(fname, 'rU')
    sentences = []
    for line in fin:
        pairs = line.rstrip('\n').split()
        sentences.append( pairs )

    return sentences

if __name__ == "__main__":

    SENTENCES_FILE='./data/tokenized_abstracts.dat'
    #SENTENCES_FILE='./data/tokenized_titles.dat'
    opath = './model/word2vec/'
  

    SIZES=[100,200,300]
    WINDOWS=[5,10,15]
    MIN_COUNTS=[2,5,10]
    
    SIZES=[300]
    WINDOWS=[10]
    MIN_COUNTS=[5]
   
    print("READING SENTENCES ...")
    sentences = read_sentences( SENTENCES_FILE )
    print("Number of sentences=%d" %( len(sentences) ) )

    for SIZE in SIZES:
      for WINDOW in WINDOWS:
        for MIN_COUNT in MIN_COUNTS:
            print("EMBEDDING WORDS: size=%d, window=%d, min_count=%d" % (SIZE, WINDOW, MIN_COUNT) )
            model = Word2Vec(sentences, min_count=MIN_COUNT, size=SIZE, window=WINDOW,iter=1,sg=1)
            print("EMBEDDING DONE !")

            print model
            print model.most_similar('atom')
    
            model.save(opath+'Abstracts_Word2Vec_SIZE_'+str(SIZE)+'_WINDOW_'+str(WINDOW)+"_MIN_COUNT_"+str(MIN_COUNT)+".bin")
  
    print("ALL EMBEDDINGS ARE CREATED")

    model = Word2Vec.load(opath+'Abstracts_Word2Vec_SIZE_300_WINDOW_10_MIN_COUNT_5.bin')
    X = model[model.wv.vocab]
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    plt.rcParams['figure.figsize'] = [7, 7]
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    plt.title('Word2Vec embedding (size=%d;window=%d,min_count=%d). t-SNE' %(300,10,5) )
    plt.xlabel(r'$\rm X_1$', fontsize=30)
    plt.ylabel(r'$\rm X_2$', fontsize=30)
    plt.show()


    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    # create a scatter plot of the projection
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.title('Word2Vec embedding (size=%d;window=%d,min_count=%d). PCA' %(300,10,5) )
    plt.xlabel(r'$\rm X_1$', fontsize=30)
    plt.ylabel(r'$\rm X_2$', fontsize=30)
    plt.show()
