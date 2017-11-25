#! /usr/bin/env python3

import sys
import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

from scholar.scholar import ScholarQuerier
from utils import get_number_citations

def get_article_cits(author_id, t_):
    pubs = schlib.get_publications(author_id, cstart = 0, pagesize = 1000, flush = True, new_code=True)

    titles = pubs[0]
    pubids = pubs[1]
    paper_id = ""
    for i in range( len( titles ) ):
        title_ = titles[i]
        if t_ in title_:
            paper_id =  pubids[i]

    if len(paper_id) == 0:
        return ("", "")

    pap_cits = schlib.get_article_cite_history(author_id, paper_id)
    year = pap_cits[0]
    cits = pap_cits[1]

    coll_ = []
    for ix in range( len( year ) ):
        coll_.append( (year[ix], cits[ix]) )  

    return coll_
    


if __name__ == "__main__":
    schlib = importr('scholar')
    author_id = "XdCxAuYAAAAJ"

    my_titles = ["Statistical contact potentials in protein coarse-grained modeling: from pair to multi-body potentials",\
    "BioShell-Threading: versatile Monte Carlo package for protein 3D threading",\
    "Coarse-grained modeling of mucus barrier properties",\
    "Multibody coarse‐grained potentials for native structure recognition and quality assessment of protein models",\
    "Optimization of profile-to-profile alignment parameters for one-dimensional threading"] 

    for my_paper in my_titles:
        print (my_paper ) 
        cits_collection = get_article_cits(author_id, my_paper)
         
        print ( cits_collection )
        print ( get_number_citations( my_paper ) )
 





