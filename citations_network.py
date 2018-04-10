#! /usr/bin/env python

from utils import parse_csv_file

def build_graph(csv_file):
    dict_1, dict_2 = parse_csv_file(csv_file)
    


if __name__ == "__main__":
    
    citations_path = '../data/aps-dataset-citations-2016/aps-dataset-citations-2016.csv'
    
    build_graph(citations_path)

    
    

