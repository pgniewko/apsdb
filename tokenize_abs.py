#! /usr/bin/env python

import os
import json
import matplotlib.pylab as plt
import numpy as np

from utils import get_abstract


def tokenize_abstract(jf_):
    abst_, format_ = get_abstract(jf_)
    return []

def browse_papers(path_):
    
    for root, dirs, files in os.walk(path_):
        for name in files:
            if name.endswith(( ".json" )):
                jfile = root + "/" + name
              
                tokenize_abstract(jfile)
 
    return []


if __name__ == "__main__":
    

    database_path = '../data_test'

    pubs_data = browse_papers(database_path)
