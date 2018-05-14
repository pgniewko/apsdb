#! /usr/bin/env python

import sys
import numpy as np
from nnutils import get_all_data
from nnutils import get_sentences
from nnutils import get_words
from nnutils import clean_text
from nnutils import number_of_unique_words

if __name__ == "__main__":
    BUFF_SIZE=0 

    opath1 = './results/'
    opath2 = './data/'

    fo_1 = open(opath1+'titles.dat', 'w', BUFF_SIZE)
    fo_2 = open(opath1+'abstracts.dat', 'w', BUFF_SIZE)
    
    fo_3 = open(opath2+'tokenized_titles.dat', 'w', BUFF_SIZE)
    fo_4 = open(opath2+'tokenized_abstracts.dat', 'w', BUFF_SIZE)
   
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
                fo_3.write(w_ + " ")
            fo_3.write('\n')
      
        s = str(y_) + " " + str(j_) + " " + str( len(sents_) ) + " " + str( len(all_words) ) + "\n"
        fo_1.write(s)

        for w_ in all_words:
            titles_words.append(w_) 

    fo_1.close()
    fo_3.close()    

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
                fo_4.write(w_ + " ")
            fo_4.write('\n')
          
        s = str(y_) + " " + str(j_) + " " + str( len(sents_) ) + " " + str( len(all_words) ) + "\n"
        fo_2.write(s)
        
        for w_ in all_words:
            abstracts_words.append(w_) 

    fo_2.close()
    fo_4.close()

