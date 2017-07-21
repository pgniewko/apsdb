#! /usr/bin/env python

import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pylab as plt

if __name__ == "__main__":

    df = pd.read_table(sys.argv[1], delim_whitespace=True, names=( 'doi', 'journal', 'year', 'country'), comment='#', header=None )

#    print df.head()

#    print df.columns


#    print df.describe()
 
    dat_ = df.groupby(['year','country']).size()

    df['COUNTER'] =1       #initially, set that counter to 1.
    group_data = df.groupby(['year','country'])['COUNTER'].sum() #sum function
    print group_data

#    print dat_.keys().shape
#    print dir(dat_)
#    print dat_['year']
#    plt.plot( dat_ )
#    plt.show()

    print dat_['2016']

    sys.exit(1)
    df.index.name = 'ID'

    print df.columns
    print df.__doc__
    print df.index
    print df.iloc[12]['year']


    df = df.cumsum('year')
    plt.figure()
    df.plot()
    plt.show()


    

