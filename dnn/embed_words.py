#! /usr/bin/env python

import sys
import numpy as np
import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from nnutils import read_sentences


if __name__ == "__main__":

    SENTENCES_FILE='./data/tokenized_abstracts.dat'
#    SENTENCES_FILE='./data/tokenized_titles.dat'
    opath = './model/word2vec/'
  

    SIZES=[100,200,300]
    WINDOWS=[5,10,15]
    MIN_COUNTS=[2,5,10]
    
#    SIZES=[300]
#    WINDOWS=[10]
#    MIN_COUNTS=[5]
   
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

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,7))
    model = Word2Vec.load(opath+'Abstracts_Word2Vec_SIZE_300_WINDOW_10_MIN_COUNT_5.bin')
    X = model[model.wv.vocab]
    

    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    ax1.scatter(X_tsne[:, 0], X_tsne[:, 1])
    ax1.set_title('t-SNE (size=%d;window=%d,min_count=%d' %(300,10,5) )
    ax1.set_xlabel(r'$\rm X_1$', fontsize=30)
    ax1.set_ylabel(r'$\rm X_2$', fontsize=30)


    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    ax2.scatter(X_pca[:, 0], X_pca[:, 1])
    ax2.set_title('PCA (size=%d;window=%d,min_count=%d)' %(300,10,5) )
    ax2.set_xlabel(r'$\rm X_1$', fontsize=30)
    ax2.set_ylabel(r'$\rm X_2$', fontsize=30)

    f.text(0.95, 0.05, '(c) 2018, P.G.',fontsize=10, color='gray', ha='right', va='bottom', alpha=0.5)

    plt.show()
