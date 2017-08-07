#! /usr/bin/env python

import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pylab as plt

if __name__ == "__main__":

    df = pd.read_table(sys.argv[1], delim_whitespace=True, names=( 'doi', 'journal', 'year', 'nauthors'), comment='#', header=None )


    dat_ = df.groupby(['journal','year'])['nauthors'].mean()
    means  = df.groupby(['year'])['nauthors'].mean()
    max_  = df.groupby(['year'])['nauthors'].max()
    errors = df.groupby(['year'])['nauthors'].std()
    


    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14,7) )
   
    ax3 = ax1.twinx()
    means.plot(yerr=errors, ax=ax1, label='All APS journals',color='blue')
    max_.plot(ax=ax3,color='green',linewidth=1.5, label='Max number of authors')
    ax1.set_xlabel('Year', fontsize=20)
    ax1.set_ylabel('Number of authors', fontsize=20, color='blue')
    ax1.grid(linestyle='--', linewidth=1)
    ax1.tick_params('y',colors='blue')
    ax3.set_ylabel('Max number of authors', color='green', fontsize=20)
    ax3.tick_params('y', colors='green')
    ax1.legend(numpoints=1, loc=2, fontsize=15)
    ax3.legend(numpoints=1, loc=4, fontsize=15)

    

    means_prl = dat_['PRL']
    means_pr  = dat_['PR']
    means_pra = dat_['PRA']
    means_prb = dat_['PRB']
    means_prd = dat_['PRD']
    means_pre = dat_['PRE']
    
    ax4 = ax2.twinx()
    means_prl.plot(ax=ax2,label="PRL")
    means_pr.plot(ax=ax2, label="PR")
    means_pra.plot(ax=ax2, label="PRA")
    means_prb.plot(ax=ax2, label="PRB")
    means_prd.plot(ax=ax2, label="PRD")
    means_pre.plot(ax=ax2, label="PRE")
    
    
    ax2.set_xlabel('Year', fontsize=20)
    ax4.set_ylabel('Number of authors', color='blue',fontsize=20)
    ax2.grid(linestyle='--', linewidth=1)
    ax2.legend(numpoints=1, loc=0, fontsize=15)
    ax4.axes.get_yaxis().set_ticks([])
    fig.text(0.95, 0.05, '(c) 2017, P.G.',fontsize=10, color='gray', ha='right', va='bottom', alpha=0.5)

    plt.show()


