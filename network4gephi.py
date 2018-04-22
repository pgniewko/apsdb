#! /usr/bin/env python

import os
from utils import parse_csv_file
import networkx as nx

def browse_papers(path_, csv_file, xmin=60):
    print("Processing citations ...")
    dict_1, dict_2 = parse_csv_file(csv_file)

    print("Processing files ...")

    papers_list = {}
    for root, dirs, files in os.walk(path_):
        for name in files:
            if name.endswith(( ".json" )):
                jfile = root + "/" + name
                data = json.load( open(jfile) )

                year,month,day = get_date_jsonfile(jfile,data)
                journal = get_journal_short_json(jfile,data)
                issue,volume = get_issue_volume(jfile,data)
                coauthors = get_coauthors_jsonfile(jfile,data)
                title = get_title(jfile,data)
                doi_ = get_doi(jfile,data)
                
                if doi in dict_1.keys():
                    cits_ = len( dict_1[doi] )
                else:
                    cits_ = 0
                
                if doi in dict_2.keys():
                    refs_ = len( dict_2[doi] )
                else:
                    refs_ = 0

                if cits_ >= xmin:
                    papers_list[doi_] = [ str(title),str(journal),str(year),str(volume),str(issue),str(cits_),str(refs_) ]


    print("Database processed ...")
    return papers_list


def build_graph(csv_file, xmin=60, papersdata=None):
    print("Parsing citation file")
    dict_1, dict_2 = parse_csv_file(csv_file)
    
    print("Selecting papers for a graph. xmin=%i" %(xmin) )
    G = nx.Graph()
    edges_dict = {}
    key_idx = 1
    
#    G.add_nodes_from( range(1,number_of_nodes+1) )
    for key in dict_1.keys():
        if len( dict_1[key] ) >= xmin:
            edges_dict[key] = key_idx
            
            if papersdata != None:
                pl_ = papersdata[key]
                G.add_node(key_idx, title=pl_[0],journal=pl_[1],year=pl_[2],\
                volume=pl_[3],issue=pl_[4],no_cits=pl_[5],no_refs=pl_[6])
            
            else:
                G.add_node(key_idx)

            key_idx += 1

            if len( dict_1[key] ) != int(cits_):
                print "ERROR:", key, len(dict_1[key]), int(cits_)

    
#    number_of_nodes = len( edges_dict.keys() )
    
    print("Building a graph")

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
    database_path = '../data/aps-dataset-metadata-abstracts-2016'
    database_path = '../data/aps-dataset-metadata-abstracts-2016/PRA'

    ograph = '../results/citations_graph_xmin60.gexf'

    papers_data = browse_papers(database_path, citations_path, xmin=60)

    G = build_graph(citations_path, xmin=60, papersdata=papers_data)
    nx.info(G)
    nx.write_gexf(G, ograph)
    
