#! /usr/bin/env python

import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pylab as plt
from scipy.stats.stats import pearsonr
from dcor import distance_stats

if __name__ == "__main__":
    
    database_path = '../db_text/papers_data.txt'
    journals = ['PRA','PRB','PRC','PRD','PRE','PRL']

    df = pd.read_table(database_path, delim_whitespace=True, \
    names=( 'year','month','day','journal','issue','volume','doi','coauts','affs','countries','titlen','numpages','cits','refs'), \
    comment='#', header=None )

    j_colors = {'PRA':'blue','PRB':'green','PRC':'red','PRD':'orange','PRE':'black','PRL':'magenta','PRAPPLIED':'red'}

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,7))
    ax1.set_xlabel('Year',fontsize=20)
    ax1.set_ylabel('Total citations (in a given year)',fontsize=20)
    ax2.set_xlabel('Year',fontsize=20)
    ax2.set_ylabel('Number of citations per paper',fontsize=20)

    for j_ in journals:
        years = []
        cits  = []
        years_normalized = []
        cits_normalized  = []
        num_paps = []
 
        df_j = df[df['journal'] == j_]
        min_year = np.min( df_j['year'] )
        max_year = np.max( df_j['year'] )
        for y_ in range(min_year, max_year + 1,1):
            df_y = df_j[ df_j['year'] == y_ ]
            sum_cits = np.sum( df_y['cits'] )

            years.append( y_ )
            cits.append( sum_cits )
            num_paps.append( len(df_y) )
 

        for i, year_i in enumerate(years):
            if year_i > 2010:
                continue

            sum_paps = 0
            for j, year_j in enumerate(years):
                if year_j >= year_i:
                    sum_paps += num_paps[j]

            years_normalized.append(year_i)
            cits_normalized.append( float( cits[i] ) / float( sum_paps ) )


        ax1.plot(years, cits, 'o-', color=j_colors[j_], label=j_ )
        ax2.plot(years_normalized, cits_normalized, 'o-', color=j_colors[j_], label=j_ )
        

    plt.legend(numpoints=1, loc=0, fontsize=15)
    f.text(0.95, 0.05, '(c) 2018, P.G.',fontsize=10, color='gray', ha='right', va='bottom', alpha=0.5)
    plt.show()


