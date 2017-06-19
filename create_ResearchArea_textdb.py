#! /usr/bin/env python

import os
import json
import matplotlib.pylab as plt
import numpy as np
from utils import get_classification_jsonfile
from utils import get_disciplines
from utils import get_concepts
from utils import get_year_jsonfile
from utils import get_journal_short_json

def browse_papers(path_, dict_):
    
    for root, dirs, files in os.walk(path_):
        for name in files:
            if name.endswith(( ".json" )):
                jfile = root + "/" + name
                year_ =  get_year_jsonfile(jfile)
                journal_ = get_journal_short_json(jfile)
                concepts = get_classification_jsonfile(jfile)


                if len(concepts) > 0:
                    print year_, journal_
                    print get_disciplines(concepts)
                    print get_concepts(concepts)


    return dict_


if __name__ == "__main__":
    
    pubs_data = {}

    database_path = '../data/aps-dataset-metadata-abstracts-2016'
#    database_path = '../data/aps-dataset-metadata-abstracts-2016/PRE'
#    database_path = '../data_test'

    pubs_data = browse_papers(database_path, pubs_data)




