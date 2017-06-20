#! /usr/bin/env python

import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pylab as plt

if __name__ == "__main__":

    df = pd.read_table(sys.argv[1], delim_whitespace=True, names=( 'doi', 'journal', 'year', 'country'), comment='#', header=None )
    df.index.name = 'ID'

    print df.columns
    print df.__doc__
    print df.index
    print df.iloc[12]['year']


    df = df.cumsum('year')
    plt.figure()
    df.plot()
    plt.show()


    

