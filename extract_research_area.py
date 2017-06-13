#! /usr/bin/env python

import os
import json
import matplotlib.pylab as plt
import numpy as np
from utils import get_classification_jsonfile
from utils import get_disciplines
from utils import get_concepts
from utils import get_year_jsonfile, get_journal_short_json
from utils import get_all_affiliations

def browse_papers(path_, dict_):
    
    for root, dirs, files in os.walk(path_):
        for name in files:
            if name.endswith(( ".json" )):
                jfile = root + "/" + name
                year_ =  get_year_jsonfile(jfile)
                journal_ = get_journal_short_json(jfile)
                concepts = get_classification_jsonfile(jfile)

                affiliations_ = get_all_affiliations(jfile)
#                print len(affiliations_), affiliations_

                if len(concepts) > 0:
                    print year_, journal_
                    print get_disciplines(concepts)
                    print get_concepts(concepts)


    return dict_


if __name__ == "__main__":
    
    pubs_data = {}

    database_path = '../data/aps-dataset-metadata-abstracts-2016'
#    database_path = '../data/aps-dataset-metadata-abstracts-2016/PRE'
#    database_path = '../data_test'

    pubs_data = browse_papers(database_path, pubs_data)
#    f, ax = plt.subplots(figsize=(10,10))
#
#   
#    top_journals = get_top_N(pubs_data, 6)
#
#
#    tot_per_year = total_per_year(pubs_data)
#    NN = len(top_journals)
#    ctable, jmap = cumulative_freqs(tot_per_year, NN, pubs_data, top_journals)
#
#    plt.subplot(1,2,1)
#    plt.xlabel('Year', fontsize=25)
#    plt.ylabel('Frequency', fontsize=25)
#    m,n = ctable.shape
#    
#    for k_ in pubs_data.keys():
#        if k_ in top_journals:
#            k_idx = jmap[k_]
#            x = []
#            y = []
#            for ii in range(m):
#                x.append(ctable[ii][0])
#                y.append(ctable[ii][k_idx])
#            plt.plot(x, y, '-', lw=2, label=k_)
#    
#
#    plt.title('TOP %d JOURNALS' %( len(top_journals) ) ) 
#    plt.legend(numpoints=1, loc=0)
#   
#    plt.subplot(1,2,2)
#    plt.xlabel('Year', fontsize=25)
#    plt.ylabel('Total # of papers', fontsize=25)
#
#    plt.semilogy(ctable.T[0], ctable.T[1] ,'o-',lw=2,color='blue')
#
#   
#    plt.show()
#
#    
