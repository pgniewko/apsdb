#! /usr/bin/env python

from utils import parse_csv_file
import networkx as nx

def build_graph(csv_file, xmin=60):
    print("Parsing citation file")
    dict_1, dict_2 = parse_csv_file(csv_file)
    
    print("Selecting papers for a graph. xmin=%i" %(xmin) )
    edges_dict = {}
    key_idx = 1
    for key in dict_1.keys():
        if len( dict_1[key] ) >= xmin:
            edges_dict[key] = key_idx
            key_idx += 1

    
    number_of_nodes = len( edges_dict.keys() )
    
    print("Building a graph")
    G = nx.Graph()
    G.add_nodes_from( range(1,number_of_nodes+1) )

    for key in edges_dict:
        key_citations = dict_1[key]
        key_idx = edges_dict[key]
        for citation in key_citations:
            if citation in edges_dict.keys():
                citation_idx = edges_dict[citation]
                G.add_edge(key_idx, citation_idx, weight=1.0)

    return G


if __name__ == "__main__":
    
    citations_path = '../data/aps-dataset-citations-2016/aps-dataset-citations-2016.csv'
#    citations_path = '../data/aps-dataset-citations-2016/aps-dataset-citations-2016_MOCK.csv'

    ograph = '../results/citations_graph.gexf'

    G = build_graph(citations_path, xmin=60)
    nx.info(G)
    nx.write_gexf(G, ograph)
    
    

