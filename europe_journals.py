#! /usr/bin/env python

import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

if __name__ == "__main__":

    journals_ = ['PRL','PRE','PRA','PRB','PRD']
    country_ = ['United.Kingdom','France','Italy', 'Germany','Russian.Federation','Spain','Switzerland','Poland','Sweden']
    colors_ = {'PRL':'orange','PRE':'blue','PRA':'green','PRB':'pink','PRD':'yellow'}
    
    df = pd.read_table(sys.argv[1], delim_whitespace=True, names=( 'doi', 'journal', 'year', 'country'), comment='#', header=None )
    dat_ = df.groupby(['journal','country']).size().sort_values(ascending=False)

    index_ = 1
    x = np.arange(5)

    fig, ax = plt.subplots(1, 9, figsize=(7,7)) 
   
    for c_ in country_:
        t_ = {}
        sum_ = 0
        for j_ in journals_:
            t_[j_] = dat_[j_][c_]
            sum_ += dat_[j_][c_]

        df_jc = pd.DataFrame( dict(journal=t_.keys(), counts=t_.values(), color=t_.keys() ) ) # t_.items(), columns=['journal','counts'] )
        plt.subplot(3,3, index_)
        plt.ylim([0,25000])

        plt.bar(x, df_jc['counts'], color=df_jc['color'].apply(lambda x: colors_[x]))
        ax[index_-1].yaxis.set_major_formatter( ScalarFormatter(useMathText=True) ) 
        plt.title(c_.replace(".",' ')+"\n(#" + str(sum_) + ")")
        plt.xticks(x, t_.keys(), rotation='horizontal') 
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
        index_ += 1

    fig.tight_layout()
    fig.text(0.95, 0.0, '(c) 2017, P.G.',fontsize=10, color='gray', ha='right', va='bottom', alpha=0.5)
    plt.show()
