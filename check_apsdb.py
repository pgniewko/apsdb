#! /usr/bin/env python

import os
import pprint
from pymongo import MongoClient

def browse_papers(path_):

    FILE_COUNTER = 0
    for root, dirs, files in os.walk(path_):
        for name in files:
            if name.endswith(( ".json" )):
                FILE_COUNTER += 1

    return FILE_COUNTER
                

if __name__ == "__main__":
    client = MongoClient()
    db = client['apsdb']
    aps_db = db.apsdb
    mongo_db_num = aps_db.count()

   
#    for entry in aps_db.find({'citations': 0}):
#        pprint.pprint(entry)

    
#    for entry in aps_db.find({'num_pages': "N/A"}):
#        pprint.pprint(entry)
    
    db_path = '../data/aps-dataset-metadata-abstracts-2016'
    entries_num = browse_papers(db_path)

    if mongo_db_num == entries_num :
        print("Number of entries in apd-database (=%i) is the same as in mongodb (=%i)" % (entries_num, mongo_db_num))
    else:
        print("Number of entries in apd-database (=%i) differs from the number in mongodb (=%i)" % (entries_num, mongo_db_num))
