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

if __name__ == "__main__":

    TOTAL_NUMBER = 0
    REGISTERED_NUMBER = 0
    df = pd.read_table(sys.argv[1], delim_whitespace=True, names=( 'doi', 'journal', 'year', 'country'), comment='#', header=None )
    dat_ = df.groupby(['country']).size().sort_values(ascending=False)
 
    TOTAL_NUMBER = dat_.sum()

    as_list = dat_.index.tolist()
    for ix, it_ in enumerate(as_list):
        as_list[ix] = it_.replace('.',' ')
    dat_.index = as_list

    
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    
    # Extra field to store the number of papers
    world['papers'] = 0.0 * world.gdp_md_est


    cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))

#    countries = countryinfo.countries

    t1=[]
    t2=[]
    t3=[]
    t4=[]     
    t5=[]
    t6=[]

    for c in dat_.index:
        try:
            cc = corrections[c]
        except:
            cc = c

        if cc in world['name'].tolist():
            try:
                ix_w = world['name'].tolist().index( cc )
            except ValueError:
                print cc, " [",c,"]", " was not Found in world name."
                continue
           
            t1.append( c )
            t2.append( dat_.values[ dat_.index.tolist().index( c ) ] )

            t3.append( world['geometry'][ix_w] )
            t4.append( world['continent'][ix_w] )
            t5.append( world['pop_est'][ix_w] )
            t6.append( world['gdp_md_est'][ix_w] )
            world['papers'][ix_w] += dat_.values[ dat_.index.tolist().index( c ) ]
            REGISTERED_NUMBER += dat_.values[ dat_.index.tolist().index( c ) ]
        else:
            REGISTERED_NUMBER += dat_.values[ dat_.index.tolist().index( c ) ]
            print "Not processed (not in geopandas):", cc, ".", "Number of citations missed= ", dat_.values[ dat_.index.tolist().index( c ) ]


    print "TOTAL_NUMBER= ", TOTAL_NUMBER, " REGISTERED_NMBER= ", REGISTERED_NUMBER


    world['papers_per_capita'] = 1000.0 * world.papers / world.pop_est
    world['papers_per_gdp'] = 1000.0 * world.papers / world.gdp_md_est

    d = {'name':t1, 'counts':t2, 'geometry':t3, 'continent':t4, 'pop_est': t5, 'gdp_md_est': t6}
    # My custom data frame
    df_mydata = gpd.GeoDataFrame(data=d)

    # Remove Antarctica so the map looks nicer:
    world = world[(world.name != "Antarctica") & (world.name != "Fr. S. Antarctic Lands") & (world.name != "Greenland")]

    fig, ax = plt.subplots(1, figsize=(14,7))
    vmin, vmax = 0, np.array(world['papers_per_capita']).max()
    base = world.plot(ax=ax, column='papers_per_capita', cmap='rainbow', vmin=vmin, vmax=vmax)

#    
    plt.title('Number of APS papers per 1000 citizens', fontsize=20)
    plt.xlabel('Latitude [$^\circ$]', fontsize=15)
    plt.ylabel('Longitude [$^\circ$]', fontsize=15)
#

    cax = fig.add_axes([0.92, 0.12, 0.03, 0.7])
    sm = plt.cm.ScalarMappable(cmap='rainbow', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    fig.colorbar(sm, cax=cax)

    plt.show()
   
##
    world['papers'] /= 1000.0
    fig, ax = plt.subplots(1, figsize=(7,7))
    vmin, vmax = 0, np.array( world['papers'][world['continent'] == "Europe"]  ).max()
    base = world.plot(ax=ax, column='papers', cmap='rainbow', vmin=vmin, vmax=vmax)

#    

    plt.title('Number of APS papers in Europe [in thousands]', fontsize=20)
    plt.xlabel('Latitude [$^\circ$]', fontsize=15)
    plt.ylabel('Longitude [$^\circ$]', fontsize=15)
#

    cax = fig.add_axes([0.92, 0.12, 0.03, 0.7])
    sm = plt.cm.ScalarMappable(cmap='rainbow', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    fig.colorbar(sm, cax=cax)


    plt.show()
