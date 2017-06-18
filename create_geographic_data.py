#! /usr/bin/env python

import os
import json
import matplotlib.pylab as plt
import numpy as np
import geograpy

from utils import get_classification_jsonfile
from utils import get_disciplines
from utils import get_concepts
from utils import get_year_jsonfile, get_journal_short_json
from utils import get_all_affiliations

def extract_country(affiliation):
    places = geograpy.get_place_context(text=affiliation)
#    print affiliation
#    print "extracted coutnries: ", places.country_mentions
    try:
        country_ = places.country_mentions[0][0]
        return country_
    except IndexError, e:
        return ""
        
#    print country_
#    print places.region_mentions
#    print places.city_mentions



def browse_papers(path_, dict_):
    
    for root, dirs, files in os.walk(path_):
        for name in files:
            if name.endswith(( ".json" )):
                jfile = root + "/" + name
                year_ =  get_year_jsonfile(jfile)
                journal_ = get_journal_short_json(jfile)
                concepts = get_classification_jsonfile(jfile)

                affiliations_ = get_all_affiliations(jfile)

#                print jfile, len(affiliations_), affiliations_

                for aff_ in affiliations_:
                    country = extract_country(aff_)
                    print aff_, "||", country, "||", len(country)


                print "_____________________"

    return dict_


if __name__ == "__main__":
    
    pubs_data = {}

    database_path = '../data/aps-dataset-metadata-abstracts-2016'
#    database_path = '../data/aps-dataset-metadata-abstracts-2016/PRE'
    database_path = '../data_test'

    pubs_data = browse_papers(database_path, pubs_data)


