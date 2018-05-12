#! /usr/bin/env python

import sys
import numpy as np
from nnutils import get_all_data
from nnutils import get_sentences
from nnutils import get_words
from nnutils import clean_text
from nnutils import number_of_unique_words

if __name__ == "__main__":
    opath = './results/'

    fo_1 = open(opath+'titles.dat', 'w')
    fo_2 = open(opath+'titles.dat', 'w')
   

    data_titles    = get_all_data(feature_='title',    journals=['PRA','PRB','PRC','PRD','PRE'])
    data_abstracts = get_all_data(feature_='abstract', journals=['PRA','PRB','PRC','PRD','PRE'])

    titles_words = []
    abstracts_words = []
    for index, row in data_titles.iterrows():
        y_ = row['year']
        j_ = row['journal']
        tit_ = row['title']
        sents_ = get_sentences(tit_)
        all_words = []
        for sent_ in sents_:
            sent_  = sent_.encode('utf-8')
            words_ = get_words(sent_)
            for w_ in words_:
                all_words.append(w_)

        s = str(y_) + " " + str(j_) + " " + str( len(sents_) ) + " " + str( len(all_words) ) + "\n"
        fo_1.write(s)

        for w_ in all_words:
            titles_words.append(w_) 

    fo_1.close()
    

    for index, row in data_abstracts.iterrows():
        y_ = row['year']
        j_ = row['journal']
        abs_ = row['abstract']
        sents_ = get_sentences(abs_)
        all_words = []
        for sent_ in sents_:
            sent_  = sent_.encode('utf-8')
            words_ = get_words(sent_)
            for w_ in words_:
                all_words.append(w_)
       
        s = str(y_) + " " + str(j_) + " " + str( len(sents_) ) + " " + str( len(all_words) ) + "\n"
        fo_2.write(s)
        
        for w_ in all_words:
            abstracts_words.append(w_) 

    fo_2.close()

    num1, dict1 = number_of_unique_words(titles_words)
    num2, dict2 = number_of_unique_words(abstracts_words)
