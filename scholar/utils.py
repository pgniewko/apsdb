#! /usr/bin/env python

from scholar import *

def get_number_citations(doi_):
    
    querier  = ScholarQuerier()
    query    = SearchScholarQuery()
    query.set_phrase(doi_)
    querier.send_query(query)

    if len(querier.articles) > 0:
        try: 
           items = sorted(list(querier.articles[0].attrs.values()),key=lambda item: item[2])
        except IndexError:
            return -1
    else:
        return -1

    cits_ = -1

    for item in items:
        if item[1] == 'Citations':
            cits_ = item[0]

    return cits_

if __name__ == "__main__":
    DOI = "10.1103/PhysRevLett.56.2700"
    print "DOI: ", DOI, " # CITATIONS: ", get_number_citations(DOI)