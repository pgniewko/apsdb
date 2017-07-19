#! /usr/bin/env python

import os
import json
import geograpy

from utils import get_year_jsonfile, get_journal_short_json
from utils import get_doi
from utils import get_all_affiliations


def extract_country(affiliation):
    places = geograpy.get_place_context(text=affiliation)
    try:
        country_ = places.country_mentions[0][0]
        return country_
    except IndexError, e:
        return ""


def browse_aps(path_, database_file):

#    header = ['#','doi', 'journal', 'year', 'country']

    list_ = []
    fo = open(database_file, 'w')
#    fo.write( " ".join(header) + "\n" )
    for root, dirs, files in os.walk(path_):
        for name in files:
            if name.endswith(( ".json" )):
                jfile = root + "/" + name

                affiliations_ = get_all_affiliations(jfile)
                year_ = get_year_jsonfile(jfile) 
                journal_ = get_journal_short_json(jfile)
                doi_ = get_doi(jfile)


                countries_list = []
                for aff_ in affiliations_:
                    if len(aff_) > 0:
                        try:
                            country_ = extract_country(aff_)
                            if len(country_) > 0:
                                countries_list.append(country_)

                        except Exception, e:
                            continue

                countries_list = list( set( countries_list) )

                if len(countries_list) > 0:
                    for c_ in countries_list:
                        c_parts = c_.split()
                        record = [doi_, journal_, year_, ".".join(c_parts) ]
                        fo.write( " ".join(record) + "\n" )
                        

    fo.close()
    return True

if __name__ == "__main__":
    
    database_path = '../data/aps-dataset-metadata-abstracts-2016'
    o_path = "../mongodb_data/aps_doi_j_y_country.txt"

    pubs_data = browse_aps(database_path, o_path)


