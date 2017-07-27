#! /usr/bin/env python

import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pylab as plt
import geopandas as gpd
import geograpy
import matplotlib.cm as cm
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
    fi = open(f2, 'rU')
    
    for line in fi:
        pairs = line.split()
        doi_ = pairs[0]
        journal_ = pairs[1]
        year_ = int( pairs[2] )
        n_authors_ = pairs[3]

        if not doi_ in map_2:
            map_2[doi_] = [ journal_, year_, n_authors_ ]

    return map_2


def combine_two_maps(map_1, map_2):

    map_3 = {}

    d_l = []
    j_l = []
    y_l = []
    n_l = []
    c_l = []
    for k1_ in map_1.keys():
        try:
            j_, y_, n_ = map_2[k1_]
            c_ = map_1[k1_]
        except KeyError:
            print k1_, " record not found."
            sys.exit(1)
            continue

        d_l.append( k1_)
        j_l.append( j_ )
        y_l.append( y_ )
        n_l.append( n_ )
        c_l.append( c_ )


    map_3 = pd.DataFrame( dict( doi=d_l, journal=j_l, year=y_l, nauthors=n_l, countries=c_l ) )
    return map_3

def P_A(country_, dat_, YEAR=1900):

    d_ = dat_[dat_['year']> YEAR]
 
    counter = 0.0
    for cl in d_['countries']:
        if country_ in cl:
            counter += 1.0

    p_a = counter / len( d_['countries'])

    return p_a


def P_AB(ca_, cb_,  dat_, YEAR=1900):

#    if JOURNAL != "ALL":
#        dat_journal = dat_year[ dat_year['journal'] == JOURNAL ]
#    else:
#        dat_journal = dat_year
#
#    if NAUTHORS == -1:
#        dat_nauthors = dat_journal
#    else:
#        dat_nauthors = dat_journal[ dat_journal ]
        
    
    d_ = dat_[dat_['year']> YEAR]
 
    counter = 0.0
    for cl in d_['countries']:
        if ca_ in cl and cb_ in cl :
            counter += 1.0

    p_a = counter / len( d_['countries'])

    return p_a




if __name__ == "__main__":

    colors = ['orange','blue','green','yellow','pink','magenta']
    colors_=[]
    TOP_X = 15
    TOP_X = 5
    YEAR=1989
    for i in np.arange(TOP_X):
        c = cm.summer(i/float(TOP_X),1)
        colors_.append( c )

    x = np.arange(TOP_X)

    db1 = sys.argv[1]
    db2 = sys.argv[2]

    df = pd.read_table(sys.argv[1], delim_whitespace=True, names=( 'doi', 'journal', 'year', 'country'), comment='#', header=None )
    dat_ = df[df['year']> YEAR].groupby(['country']).size().sort_values(ascending=False)
   
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

    plt.yticks(x, [ corrections[a.replace('.',' ')] if a.replace('.',' ') in corrections else a.replace('.',' ') for a in  dat_['country'][range(TOP_X-1,-1,-1)]] , rotation=45)
    
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
 
    plt.title("Number of papers for TOP %d countries (%d-2016) " %(TOP_X, YEAR) , fontsize=14)

    plt.xlim([0,1.2*dat_['counts'][range(TOP_X-1,-1,-1)].max()])

    plt.xlabel("Number of papers",fontsize=15)
    fig.text(0.95, 0.0, '(c) 2017, P.G.',fontsize=10, color='gray', ha='right', va='bottom', alpha=0.5)
    plt.subplots_adjust(left=0.16)
 #   plt.show()


#    print x
#    print top_countries_list


    m1_ = create_countries_map( db1 )
    m2_ = create_authors_map( db2 )
    m3_ = combine_two_maps(m1_, m2_)

#    print m3_
#    print type(m3_)
#    print m3_

    
    p_france = P_A('France', m3_)
    p_germany = P_A('Germany', m3_)
    p_italy = P_A('Italy', m3_)


    print p_france
    print p_germany

    p_f_g = P_AB('France','Germany', m3_)
    print p_f_g

    top_countries_list = dat_['country'][0:TOP_X]
    print top_countries_list

    bayes_probs = np.zeros([TOP_X, TOP_X])

    for i_, tc_i in enumerate( top_countries_list) :
        for j_, tc_j in enumerate( top_countries_list):
            if i_ != j_:
                p_j  = P_A(tc_j, m3_)
                p_i_and_j = P_AB(tc_i, tc_j, m3_) 
                p_i_cond_j = p_i_and_j / p_j
                bayes_probs[i_][j_] = p_i_cond_j
        
    print bayes_probs
