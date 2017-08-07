#! /usr/bin/env python

import os
import json
import geograpy

from utils import get_year_jsonfile, get_journal_short_json
from utils import get_doi
from utils import get_coauthors_jsonfile
from utils import get_all_affiliations

from utils import failed_tries

def browse_aps(path_, database_file):

#    header = ['#','doi', 'journal', 'year', 'country']

    list_ = []
    fo = open(database_file, 'w')
#    fo.write( " ".join(header) + "\n" )
    for root, dirs, files in os.walk(path_):
        for name in files:
            if name.endswith(( ".json" )):
                jfile = root + "/" + name

                coauthors_number_ = len( get_coauthors_jsonfile(jfile) )
                year_ = get_year_jsonfile(jfile) 
                journal_ = get_journal_short_json(jfile)
                doi_ = get_doi(jfile)


                if coauthors_number_ > 0:
                    record = [doi_, journal_, year_, str( coauthors_number_) ]
                    fo.write( " ".join(record) + "\n" )
                        
    fo.close()

    return True

if __name__ == "__main__":
    
    database_path = '../data/aps-dataset-metadata-abstracts-2016'
    o_path = "../db_text/aps_doi_j_y_coauthors.txt"

    pubs_data = browse_aps(database_path, o_path)


