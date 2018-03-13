#! /usr/bin/env python

import numpy as np
import matplotlib.pylab as plt


from utils import parse_csv_file


def build_network(csv_file):

    dict_1, dict_2 = parse_csv_file(csv_file)
    return  



if __name__ == "__main__":
    
    citations_path = '../data/aps-dataset-citations-2016/aps-dataset-citations-2016.csv'
    
    build_network(itations_path)




