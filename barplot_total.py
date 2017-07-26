#! /usr/bin/env python

import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pylab as plt
import geopandas as gpd
import geograpy
#import countryinfo.countryinfo as countryinfo


corrections = { \
"Moldova, Republic of":'Moldova',\
"Russian Federation":"Russia",\
"Bolivia, Plurinational State of":"Bolivia",\
"Republic of Ireland":"Ireland",\
"Iran, Islamic Republic of":"Iran",\
"Czechia":"Czech Rep.",\
"Czech Republic":"Czech Rep.",\
"Korea, Democratic People's Republic of":"Korea",\
"Korea, Republic of":"Korea",\
"British Indian Ocean Territory":"India",\
"Viet Nam":"Vietnam",\
"Lao People's Democratic Republic":"Lao PDR",\
"Bosnia and Herzegovina":"Bosnia and Herz.",\
"Macedonia, Republic of":"Macedonia",\
"Venezuela, Bolivarian Republic of":"Venezuela",\
"Dominican Republic":"Dominican Rep.",\
"Taiwan, Province of China":"China",\
"Tanzania, United Republic of":"Tanzania",\
"Saint Martin (French part)":"France",\
"French Southern Territories":"France",\
"Congo, The Democratic Republic of the":"Congo",\
"Saint Pierre and Miquelon":"France",\
"Saint Helena, Ascension and Tristan da Cunha":"United Kingdom"
}


def create_countries_map(f1):
    map_1 = {}
    fi = open(f1, 'rU')
    
    for line in fi:
        pairs = line.split()
        doi_ = pairs[0]
        country_ = pairs[3]

        if doi_ in map_1:
            map_1[doi_].append( country_ )
        else:
            map_1[doi_] = [country_]

    return map_1


def create_authors_map(f2):
    map_2 = {}
    fi = open(f1, 'rU')
    
    for line in fi:
        pairs = line.split()
        doi_ = pairs[0]
        journal_ = pairs[1]
        year_ = int( pairs[2] )
        n_authors_ = pairs[3]

        if doi_ in map_1:
            map_2[doi_] = [ journal_, year_, n_authors_ ]

    return map_2

#def combine_two_maps(map_1, map_2):
    


if __name__ == "__main__":

    colors = ['orange','blue','green','yellow','pink','magenta']
    colors_=[]
    TOP_X = 15
    YEAR=1989
    for i in range(TOP_X):
        colors_.append( colors[i%6] )

    x = np.arange(TOP_X)

    db1 = sys.argv[1]
    db2 = sys.argv[2]

    df = pd.read_table(sys.argv[1], delim_whitespace=True, names=( 'doi', 'journal', 'year', 'country'), comment='#', header=None )
    dat_ = df[df['year']> YEAR].groupby(['country']).size().sort_values(ascending=False)
   
    print dat_
    t_ = {}
    c_l = []
    n_l = []
    for c_ in dat_[0:TOP_X].keys():
        t_[c_] = dat_[c_]
        cc_ = c_
        if c_ in corrections:
            cc_ = corrections[c_]
        c_l.append( c_ )
        n_l.append( dat_[c_])

    fig, ax = plt.subplots(figsize=(7,7)) 
    width=0.75

    dat_ = pd.DataFrame( dict( country=dat_.keys(), counts=dat_.values ) )
    
    plt.barh(x, dat_['counts'][range(TOP_X-1,-1,-1)], width, color=colors_)
    for i, v in enumerate(dat_['counts'][range(TOP_X-1,-1,-1)] ):
        ax.text(v + 500, i-0.1, str(v), color='black')

    plt.yticks(x, [ corrections[a.replace('.',' ')] if a.replace('.',' ') in corrections else a.replace('.',' ') for a in  dat_['country'][range(TOP_X-1,-1,-1)]] , rotation='horizontal')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
 
    plt.title("Number of papers for TOP %d countries after %d " %(TOP_X, YEAR) )
    plt.xlim([0,1.2*dat_['counts'][range(TOP_X-1,-1,-1)].max()])

    plt.xlabel("Number of papers",fontsize=15)
    fig.text(0.95, 0.0, '(c) 2017, P.G.',fontsize=10, color='gray', ha='right', va='bottom', alpha=0.5)
    plt.show()
