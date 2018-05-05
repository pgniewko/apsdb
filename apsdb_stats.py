#! /usr/bin/env python

import sys
import os
import pprint
import pymongo
import matplotlib.pylab as plt

if __name__ == "__main__":
    
    client = pymongo.MongoClient()
    db = client['apsdb']
    aps_db = db['aps-articles-basic']
    mongo_db_num = aps_db.count()

    # PRINT OUT THE SIZE OF THE DATABASE
    print("Size of the database at this point %i" %(mongo_db_num) )

    # PRINT OUT THE FIRST PUBLISHED PAPER
    print ("First published paper:")
    pprint.pprint( aps_db.find_one(sort=[('year', 1)]) )
    print("###########################################")

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,7))
    
    ax1.set_xlabel('Year',fontsize=20)
    ax1.set_ylabel('Max citations',fontsize=20)
    ax2.set_xlabel('Year',fontsize=20)
    ax2.set_ylabel('Fraction',fontsize=20)

    # PRINT THE NUMBER OF PAPERS IN EACH JOURNAL
    #js_ = ["PR","PRL","PRA","PRB","PRC","PRD","PRE","PRFLUIDS","PRX","RMP"]
    js_ = ["PR","PRL","PRA","PRB","PRC","PRD","PRE"]
    for j_ in js_:
        paps_   = aps_db.find( {"journal":j_} )
        tot_num = paps_.count()
        tot_frac= float(tot_num) / float(mongo_db_num)
        start_y = paps_.sort( 'year',1 )[0]['year']
        end_y   = paps_.sort( 'year',-1 )[0]['year']
        print( j_ +" " + str(tot_num) + " " + str(start_y) + " "+ str(end_y) + " " + str(tot_frac))
       
    for j_ in js_:
        paps_ = aps_db.find( {"journal":j_} )
        start_y = paps_.sort( 'year',1 )[0]['year']
        end_y   = paps_.sort( 'year',-1 )[0]['year']
        years_l = []
        most_cited_l = []
        zero_cit_f_l = []
        for y_ in range( start_y, end_y+1, 1):
            if y_ > 2016:
                continue

            paps_2 = aps_db.find( {"journal":j_,'year':y_} )
            paps_zer = aps_db.find( {"journal":j_,'year':y_,'citations':0} )
            most_cited = start_y = paps_2.sort( 'citations',-1 )[0]['citations']
            
            frac_ = float( paps_zer.count() ) / float( paps_2.count() )
            years_l.append(y_)
            most_cited_l.append(most_cited)
            zero_cit_f_l.append(frac_)
        
       
        ax1.plot(years_l, most_cited_l, 'o-', label=j_)
        ax2.plot(years_l, zero_cit_f_l, 'o-', label=j_)
      
    
    f.text(0.95, 0.05, '(c) 2018, P.G.',fontsize=10, color='gray', ha='right', va='bottom', alpha=0.5)
    ax2.legend(loc=0) 
    plt.show()


