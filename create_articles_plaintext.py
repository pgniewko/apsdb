#! /usr/bin/env python

import os
import json
import numpy as np

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


def browse_papers(path_, csv_file, fout):
    fo = open(fout, 'w')
    print("Processing citations ...")
    dict_1, dict_2 = parse_csv_file(csv_file)

    print("Processing files ...")

    tmp_list = []
    for root, dirs, files in os.walk(path_):
        for name in files:
            if name.endswith(( ".json" )):
                jfile = root + "/" + name
                data = json.load( open(jfile) )

                year,month,day = get_date_jsonfile(jfile,data)
                journal = get_journal_short_json(jfile,data)
                issue,volume = get_issue_volume(jfile,data)
                doi = get_doi(jfile,data)
                num_pages = get_number_of_pages(jfile,data)
                coauthors = get_coauthors_jsonfile(jfile,data)
                affiliations = get_all_affiliations(jfile,data)
                countries = get_all_countries(jfile,data)
                title = get_title(jfile,data)

                str_out = ""
                str_out += str(year)    + " "
                str_out += str(month)   + " "
                str_out += str(day)     + " " 
                str_out += str(journal) + " "
                str_out += str(issue)   + " "
                str_out += str(volume)  + " "
                str_out += str(doi)     + " " 
                str_out += str( len(coauthors) ) + " "
                str_out += str( len(affiliations) ) + " "
                str_out += str( len(countries) ) + " " 
                str_out += str( len(title) ) + " "
                str_out += str( num_pages ) + " "

                
                if doi in dict_1.keys():
                    str_out += str( len( dict_1[doi] ) ) + " "
                else:
                    str_out += str( 0 ) + " "
                
                if doi in dict_2.keys():
                    str_out += str( len( dict_2[doi] ) ) + " "
                else:
                    str_out += str( 0 ) + " "


                fo.write(str_out + "\n")

    fo.close()


if __name__ == "__main__":
    
    database_path  = '../data/aps-dataset-metadata-abstracts-2016'
    citations_path = '../data/aps-dataset-citations-2016/aps-dataset-citations-2016.csv'
    ofile = "../db_text/papers_data.txt" 

    print("PlainText build in progress ...")


    browse_papers(database_path, citations_path, ofile)    
    


