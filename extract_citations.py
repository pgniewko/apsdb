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

    print ( author_id, paper_id)
    pap_cits = schlib.get_article_cite_history(author_id, paper_id)
    year = pap_cits[0]
    cits = pap_cits[1]

    coll_ = []
    for ix in range( len( year ) ):
        coll_.append( (year[ix], cits[ix]) )  

    return coll_
   
if __name__ == "__main__":
    # TODO:
    # 1. Make sure that I can continously request google-scholar site
    #    Probably I may require contacting someone in GS
    # 2. One the content of the GS search is obtained try to extract an author id
    #    If there is no author with google scholar account then proceed with just citations
    #    Otherwise aquaire author_id and find a citation record. 
    #   

    schlib = importr('scholar')
    author_id = "XdCxAuYAAAAJ"

    my_titles = ["Statistical contact potentials in protein coarse-grained modeling: from pair to multi-body potentials",\
    "BioShell-Threading: versatile Monte Carlo package for protein 3D threading",\
    "Coarse-grained modeling of mucus barrier properties",\
    "Multibody coarse‐grained potentials for native structure recognition and quality assessment of protein models",\
    "Optimization of profile-to-profile alignment parameters for one-dimensional threading"] 
    
    my_titles = ["Statistical contact potentials in protein coarse-grained modeling: from pair to multi-body potentials"]

    for my_paper in my_titles:
        print (my_paper ) 
        cits_collection = get_article_cits(author_id, my_paper)
         
        print ( cits_collection )
        print ( get_number_citations( my_paper ) )
        cits, url_ =  get_number_citations( my_paper )

        from urllib.request import urlopen, Request, urlopen
        from bs4 import SoupStrainer, BeautifulSoup
        import re
        from html.entities import name2codepoint
        HEADERS = {'User-Agent': 'Mozilla/5.0'}
        header = HEADERS
        request = Request(url_, headers=header)
        html_ = urlopen(request).read()
        html_ = html_.decode('utf8')

        refre = re.compile(r'href="/citations*=en\"')
        
        reflist = refre.findall(html_)
        reflist = [re.sub('&(%s);' % '|'.join(name2codepoint), lambda m:
                      chr(name2codepoint[m.group(1)]), s) for s in reflist]
