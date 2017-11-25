#! /usr/bin/env python3

import sys
import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects


def get_article_id(author_id, t_):
    pubs = schlib.get_publications(author_id, cstart = 0, pagesize = 1000, flush = False)

    titles = pubs[0].levels
    pubids = pubs[7].levels

    for i, title_ in enumerate(titles):
        if t_ in title_:
            return pubids[i]



if __name__ == "__main__":
    schlib = importr('scholar')
    author_id = "XdCxAuYAAAAJ"
    paper_id = get_article_id(author_id, "BioShell-Threading: versatile Monte Carlo package for protein 3D threading") 
    
    pap_cit = schlib.get_article_cite_history(author_id, paper_id)

    print( pap_cit )
 
    print ( type ( pap_cit ) )





