#! /usr/bin/env python

import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pylab as plt
import geopandas as gpd
import geograpy

import countryinfo.countryinfo as countryinfo

#def replace_char(ch1, ch2):
    

if __name__ == "__main__":

    df = pd.read_table(sys.argv[1], delim_whitespace=True, names=( 'doi', 'journal', 'year', 'country'), comment='#', header=None )
     

    dat_ = df.groupby(['country']).size().sort_values(ascending=False)
  
    as_list = dat_.index.tolist()
    
    for ix, it_ in enumerate(as_list):
        as_list[ix] = it_.replace('.',' ')
        
#        places = geograpy.get_place_context(text=unicode(as_list[ix]))
        
#        print as_list[ix], places.countries

    dat_.index = as_list

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))

#    print cities['name']
  
    print 'Warsaw' in cities['name'].tolist()
    countries = countryinfo.countries

    t1=[]
    t2=[]
    t3=[]
    for c in countries:
        if c['name'] in  dat_.index:
            try:
                ix = cities['name'].tolist().index(c['capital'])
                print ix, c['name'], c['capital']
            #coordinates = cities['geometry'][ix]
            except ValueError:
                print c['capital'], " was not found."
                continue

            #print c['name'], c['capital'], coordinates
            t1.append( c['name'] )
            t2.append( c['capital']  )
            

    d = {'Name':t1,'Capital':t2}
    df_capitals = pd.DataFrame(data=d)



