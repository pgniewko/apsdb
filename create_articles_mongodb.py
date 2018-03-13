#! /usr/bin/env python

import os
#import json
import numpy as np

import pymongo
from pymongo import MongoClient
import matplotlib.pylab as plt

from utils import get_date_jsonfile
from utils import get_journal_short_json
from utils import get_coauthors_jsonfile
from utils import get_doi
from utils import get_issue_volume
from utils import get_all_affiliations
from utils import get_all_countries
from utils import get_number_of_pages
from utils import get_title
from utils import parse_csv_file


def browse_papers(path_, csv_file):

    dict_1, dict_2 = parse_csv_file(csv_file)
    client = MongoClient()
    db = client['apsdb']

    for root, dirs, files in os.walk(path_):
        for name in files:
            if name.endswith(( ".json" )):
                jfile = root + "/" + name

                year,month,day = get_date_jsonfile(jfile)
                journal = get_journal_short_json(jfile)
                issue,volume = get_issue_volume(jfile)
                doi = get_doi(jfile)
                num_pages = get_number_of_pages(jfile)
                coauthors = get_coauthors_jsonfile(jfile)
                affiliations = get_all_affiliations(jfile)
                countries = get_all_countries(jfile)
                title = get_title(jfile)


                aps_paper = {'year':year, 'month':month, 'day':day}
                aps_paper['journal'] = journal
                aps_paper['issue'] = issue
                aps_paper['volume'] = volume
                aps_paper['doi'] = doi
                aps_paper['num_authors'] = len(coauthors)
                aps_paper['num_affs'] = len(affiliations)
                aps_paper['num_countries'] = len(countries)
                aps_paper['num_pages'] = num_pages
                
                if doi in dict_1.keys():
                    aps_paper['citations'] = len( dict_1[doi] )
                else:
                    aps_paper['citations'] = 0
                
                if doi in dict_2.keys():
                    aps_paper['cited_articles'] = len( dict_2[doi] )
                else:
                    aps_paper['cited_articles'] = 0
                    


                aps = db.apsdb
                aps.insert_one(aps_paper)


    print aps.find({"citations": 10})
                

if __name__ == "__main__":
    
    database_path = '../data/aps-dataset-metadata-abstracts-2016'
    database_path = '../data/aps-dataset-metadata-abstracts-2016/PRL/47/'
    citations_path = '../data/aps-dataset-citations-2016/aps-dataset-citations-2016.csv'
    
    browse_papers(database_path, citations_path)




