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

def combine_two_maps(map_1, map_2):
    


if __name__ == "__main__":


    db1 = sys.argv[1]
    db2 = sys.argv[2]

    map_countries = 
