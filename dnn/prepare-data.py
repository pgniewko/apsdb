#! /usr/bin/env python


import os
import sys
import json
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import parse_csv_file

from utils import get_journal_short_json
from utils import get_clean_title
from utils import get_date_jsonfile
from utils import get_clean_abstract
from utils import get_doi

def browse_papers(path_, csv_file, ofiles):
    BUFF_SIZE=0
    of1,of2,of3,of4,of5 = ofiles
   
    f1 = open(of1, 'w', BUFF_SIZE)
    f2 = open(of2, 'w', BUFF_SIZE)
    f3 = open(of3, 'w', BUFF_SIZE)
    f4 = open(of4, 'w', BUFF_SIZE)
    f5 = open(of5, 'w', BUFF_SIZE)

    print("Processing citations ...")
    dict_1, dict_2 = parse_csv_file(csv_file)

    print("Processing files ...")

    tmp_list = []
    for root, dirs, files in os.walk(path_):
        for name in files:
            if name.endswith(( ".json" )):
                jfile = root + "/" + name
                data = json.load( open(jfile) )

                try:
                    year,_,_ = get_date_jsonfile(jfile,data)
                    journal = get_journal_short_json(jfile,data)
                    doi = get_doi(jfile,data)
                    title = get_clean_title(jfile,data)

                    if doi in dict_1.keys():
                        cits = len(dict_1[doi])
                    else:
                        cits = 0

                    abstract = get_clean_abstract(jfile,data)
                    abstract = abstract.replace('\n', ' ').replace('\r', '')

                    _=str(journal)
                    _=str(year)
                    _=str(cits)
                    _=abstract.encode('utf-8')
                    _=title.encode('utf-8')

                    f1.write(str(journal) + "\n")
                    f2.write(str(year) + "\n")
                    f3.write(str(cits) + "\n")
                    f4.write( abstract.encode('utf-8') + "\n")
                    f5.write( title.encode('utf-8') + "\n")

                except KeyError as e:
                    print 'KeyError', e
                    pass
                except IOError as e:
                    print 'KeyError', e
                    pass
                except UnicodeEncodeError as e:
                    print 'KeyError', e
                    pass
         

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    return


if __name__ == "__main__":

    database_path = '../../data/aps-dataset-metadata-abstracts-2016'
    citations_path = '../../data/aps-dataset-citations-2016/aps-dataset-citations-2016.csv'
    ofile1 = "./data/journals_data.txt"
    ofile2 = "./data/year_data.txt"
    ofile3 = "./data/citations_data.txt"
    ofile4 = "./data/abstract_data.txt"
    ofile5 = "./data/title_data.txt"

    olist = [ofile1,ofile2,ofile3,ofile4,ofile5]
    browse_papers(database_path, citations_path, olist)
