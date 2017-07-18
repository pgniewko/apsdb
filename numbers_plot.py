#! /usr/bin/env python

import os
import json
import matplotlib.pylab as plt
import numpy as np
from utils import get_year_jsonfile, get_journal_short_json
from utils import get_top_N, create_time_record


def browse_papers(path_, dict_):
    
    for root, dirs, files in os.walk(path_):
        for name in files:
            if name.endswith(( ".json" )):
                jfile = root + "/" + name

                year = get_year_jsonfile(jfile)
                j_id = get_journal_short_json(jfile)
                if j_id in dict_:
                    if year in dict_[j_id]:
                        dict_[j_id][year] += 1
                    else:
                        dict_[j_id][year] = 1
                else:
                    dict_[j_id] = {}
                    dict_[j_id][year] = 1


    return dict_


if __name__ == "__main__":
    
    pubs_data = {}

    database_path = '../data/aps-dataset-metadata-abstracts-2016'
#    database_path = '../data/aps-dataset-metadata-abstracts-2016/PRE'
    pubs_data = browse_papers(database_path, pubs_data)
    f, ax = plt.subplots(figsize=(8,8))

    ax.set_xlabel('Year', fontsize=20)
    ax.set_ylabel('# of PAPERS', fontsize=20)
   
    top_journals = get_top_N(pubs_data, 6)

    for k_ in pubs_data.keys():
        if k_ in top_journals:
            y, c = create_time_record(k_, pubs_data)
            ax.plot(y, c,'o',label=k_)

    plt.title('TOP %d JOURNALS' %( len(top_journals) ) ) 
    plt.legend(numpoints=1, loc=0)

    f.text(0.95, 0.05, '(c) 2017, P.G.',fontsize=10, color='gray', ha='right', va='bottom', alpha=0.5)
    
    plt.show()

    

