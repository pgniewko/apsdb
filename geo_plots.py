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
"Dominican Republic":"Dominican Rep."
}

if __name__ == "__main__":

    df = pd.read_table(sys.argv[1], delim_whitespace=True, names=( 'doi', 'journal', 'year', 'country'), comment='#', header=None )
    dat_ = df.groupby(['country']).size().sort_values(ascending=False)
 
    as_list = dat_.index.tolist()
    for ix, it_ in enumerate(as_list):
        as_list[ix] = it_.replace('.',' ')
    dat_.index = as_list

    
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


    cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))

    countries = countryinfo.countries

    t1=[]
    t2=[]
    t3=[]
    t4=[]     
    t5=[]
    t6=[]
    t7=[]
    t8=[]

    for cc_ in world['name']:
        print cc_

    world['papers'] = 0.0 * world.gdp_md_est
#    for c in countries:
    for c in dat_.index:
        try:
            cc = corrections[c]
        except:
            cc = c

        if cc in world['name'].tolist():
            try:
                #ix = cities['name'].tolist().index(c['capital'])
                #ix_w = world['name'].tolist().index( c['name'] )
                ix_w = world['name'].tolist().index( cc )
            except ValueError:
                print c, cc, " was not Found in world name "
                continue
           
            #print ix, ix_w, c['name'], world['continent'][ix_w]
#            t1.append( c['name'] )
            t1.append( c )
            t2.append( "" ) #c['capital']  )
            t3.append( "" ) # cities['geometry'][ix] )
#            t4.append( dat_.values[ dat_.index.tolist().index( c['name'] ) ] )
            t4.append( dat_.values[ dat_.index.tolist().index( c ) ] )

            t5.append( world['geometry'][ix_w] )
            t6.append( world['continent'][ix_w] )
            t7.append( world['pop_est'][ix_w] )
            t8.append( world['gdp_md_est'][ix_w] )
            world['papers'][ix_w] += ( dat_.values[ dat_.index.tolist().index( c ) ] )**1.0
        else:
            print "Not processed:", cc, cc in world['name'].tolist()

    world['papers_per_capita'] = world.papers / world.pop_est
    world['papers_per_gdp'] = world.papers / world.gdp_md_est

    d = {'name':t1,'capital':t2, 'point':t3, 'counts':t4, 'geometry':t5, 'continent':t6, 'pop_est': t7, 'gdp_md_est': t8}
    df_mydata = gpd.GeoDataFrame(data=d)

    # Remove Antarctica so the map looks nicer:
    world = world[(world.name != "Antarctica") & (world.name != "Fr. S. Antarctic Lands") & (world.name != "Greenland")]
    world = world.to_crs({'init': 'epsg:3395'}) # world.to_crs(epsg=3395) would also work
    base = world.plot(column='papers_per_capita', cmap='rainbow')
 
    plt.show()
