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


def total_per_year(dict_):
    total_papers = {}
    for journal_ in dict_.keys():
        for year_ in dict_[journal_].keys():
            if year_ not in total_papers.keys():
                total_papers[year_] = 0
 
            total_papers[year_] += dict_[journal_][year_]


    return total_papers


def cumulative_freqs(tot_per_year_, N_, pubs_data_, top_journals_):
    min_y = 10000
    max_y = 0
    for journal_ in pubs_data_.keys():
        for years_ in pubs_data_[journal_].keys():
            min_y = min(min_y, int(years_) )
            max_y = max(max_y, int(years_) )

    years_span = max_y - min_y + 1
    cumulative_table = np.zeros( ( years_span, N_ + 2) )

    for y_ in range( years_span ):
        cumulative_table[y_][0] = min_y + y_
        cumulative_table[y_][1] = tot_per_year[str(min_y + y_)]

    journal_idx = 2

    journal_to_idx_map = {}

    for k_ in pubs_data.keys():
        if k_ in top_journals_:
            journal_to_idx_map[k_] = journal_idx
            years_, counts = create_time_record(k_, pubs_data_)
            for i_, y_ in enumerate(years_):
                year_total = tot_per_year_[y_]
                journal_total = counts[i_]

                frac = float(journal_total) / float(year_total)
                if journal_idx == 2:
                    cumulative_table[int(y_)-min_y][journal_idx] = frac
                else:
                    cumulative_table[int(y_)-min_y][journal_idx] = frac # +  cumulative_table[int(y_)-min_y][journal_idx-1]

            journal_idx += 1


    return cumulative_table, journal_to_idx_map


if __name__ == "__main__":
    
    pubs_data = {}

    database_path = '../data/aps-dataset-metadata-abstracts-2016'
#    database_path = '../data/aps-dataset-metadata-abstracts-2016/PRE'
#    database_path = '../data_test'
    pubs_data = browse_papers(database_path, pubs_data)
    f, ax = plt.subplots(figsize=(8,8))

   
    top_journals = get_top_N(pubs_data, 6)


    tot_per_year = total_per_year(pubs_data)
    NN = len(top_journals)
    ctable, jmap = cumulative_freqs(tot_per_year, NN, pubs_data, top_journals)

    plt.subplot(1, 2, 1)
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    m,n = ctable.shape
    
    for k_ in pubs_data.keys():
        if k_ in top_journals:
            k_idx = jmap[k_]
            x = []
            y = []
            for ii in range(m):
                x.append(ctable[ii][0])
                y.append(ctable[ii][k_idx])
            plt.plot(x, y, '-', lw=2, label=k_)
    

    plt.legend(numpoints=1, loc=0)
   
    plt.subplot(1, 2, 2)
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Total # of papers', fontsize=20)

    plt.semilogy(ctable.T[0], ctable.T[1] ,'o-',lw=2,color='blue')

    f.text(0.95, 0.05, '(c) 2017, P.G.',fontsize=10, color='gray', ha='right', va='bottom', alpha=0.5)
    plt.show()

    

