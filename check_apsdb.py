#! /usr/bin/env python

import os
import pprint
import pymongo

def browse_papers(path_):

    FILE_COUNTER = 0
    for root, dirs, files in os.walk(path_):
        for name in files:
            if name.endswith(( ".json" )):
                FILE_COUNTER += 1

    return FILE_COUNTER
                

if __name__ == "__main__":
    
    client = pymongo.MongoClient()
    db = client['apsdb']
    aps_db = db.apsdb
    mongo_db_num = aps_db.count()

    # FIND MOST CITED ARTICLE:
    print("Paper most cited by other APS papers")
    pprint.pprint( aps_db.find_one(sort=[('citations', -1)]) )

    # FIND MOST CITING PAPER:
    print("Paper citing most number of other APS papers")
    pprint.pprint( aps_db.find_one(sort=[('num_references', -1)]) )
   
    # PRINT ALL PAPERS WITH NUMER OF CITATION LARGER THAN 500:
    print("CITATION > 500")
    for entry in aps_db.find({'citations': {"$gt": 500}}).sort('year'):
        pprint.pprint(entry)
    
    db_path = '../data/aps-dataset-metadata-abstracts-2016'
    entries_num = browse_papers(db_path)

    if mongo_db_num == entries_num :
        print("Number of entries in apd-database (=%i) is the same as in mongodb (=%i)" % (entries_num, mongo_db_num))
    else:
        print("Number of entries in apd-database (=%i) differs from the number in mongodb (=%i)" % (entries_num, mongo_db_num))
