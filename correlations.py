#! /usr/bin/env python

import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pylab as plt
from scipy.stats.stats import pearsonr
from dcor import distance_stats

if __name__ == "__main__":
    
    database_path = '../db_text/papers_data.txt'

    journals = ['PRA','PRB','PRC','PRD','PRE','PRL']
    #journals = ['PRA','PRAPPLIED']

    

    df = pd.read_table(database_path, delim_whitespace=True, \
    names=( 'year','month','day','journal','issue','volume','doi','coauts','affs','countries','titlen','numpages','cits','refs'), \
    comment='#', header=None )

    cats = ['year','month','day','issue','volume','coauts','affs','countries','titlen','numpages','cits','refs']
    j_colors = {'PRA':'blue','PRB':'green','PRC':'red','PRD':'orange','PRE':'black','PRL':'magenta','PRAPPLIED':'red'}

    fout = open('../results/correlations.dat','w')
    
    nsamples = 5
    fig, ax1 = plt.subplots(1, 1, sharey=True, figsize=(7,7) )
    for j_ in journals:
        pearson_corr = []
        dist_corr = []
        for i,el_i in enumerate(cats):
            for j,el_j in enumerate(cats):
                if j > i:
                    df_j = df[df['journal'] == j_]
                    new_data = df_j[ [el_i,el_j] ]#._get_numeric_data()
                    new_data = new_data.dropna()

                    r = 0.0
                    corrxy = 0.0
                    DIVISOR = 0
                    for i in range(nsamples):
                        try:
                            sample_size = min(1000, len(new_data[el_i]))
                            new_data_sample = new_data.sample( sample_size )
                            data_i = pd.to_numeric( new_data_sample[el_i] )
                            data_j = pd.to_numeric( new_data_sample[el_j] )
                            DIVISOR += 1
                        except ValueError:
                            continue

                        ri,pi = pearsonr(data_i,data_j)
                        covxyi, corrxyi, varxi, varyi = distance_stats(data_i, data_j)
                   
                        r += ri
                        corrxy += corrxyi

                    r      /= DIVISOR
                    corrxy /= DIVISOR
                    pearson_corr.append(r)
                    dist_corr.append(corrxy)
                    s = "%10s  %10s  %10s  % f  % f \n" % ( j_, el_i, el_j, r, corrxy)
                    fout.write(s)

        ax1.plot(dist_corr, pearson_corr,'o', color=j_colors[j_],label=j_ )


    fout.close()

    plt.xlim(0,1)
    plt.ylim(-1,1)

    plt.xlabel('Distance Correlation Coefficient',fontsize=15)
    plt.ylabel('Pearson Correlation Coefficient',fontsize=15)

    plt.legend(numpoints=1, loc=0, fontsize=15)
    fig.text(0.95, 0.05, '(c) 2018, P.G.',fontsize=10, color='gray', ha='right', va='bottom', alpha=0.5)
    plt.show()


