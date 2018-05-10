
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

CITATION_FILE="./data/citations_data.txt"
YEAR_FILE="./data/year_data.txt"
JOURNAL_FILE="./data/journals_data.txt"

TITLES_FILE="./data/title_data.txt"
ABSTRACTS_FILE="./data/abstract_data.txt"


def read_abstracts(fin):
    ab_list = []
    fo = open(fin,'rU')
    for line in fo:
        ab_list.append( line )

    return pd.DataFrame( ab_list )

def get_data(SAMPLE_SIZE=100,feature_='abstract',yrange=[1990,2010],journals=['PRA','PRB']):
    
    if feature_ == 'abstract':
        df_f = read_abstracts( ABSTRACTS_FILE )
        #df_f = pd.read_table(ABSTRACTS_FILE, names=None, delim_whitespace=False, comment='#', header=None ) 
        df_f.columns = ['abstract']
    else:    
        df_f = pd.read_table(TITLES_FILE, names=None, delim_whitespace=False, comment='#', header=None )
        df_f.columns = ['title']
    
    df_y = pd.read_table(YEAR_FILE, names=None, delim_whitespace=True, comment='#', header=None )
    df_y.columns = ['year']
    idcs_y = (df_y['year'] > yrange[0]) & (df_y['year'] < yrange[1])
    
    df_j = pd.read_table(JOURNAL_FILE, delim_whitespace=True, comment='#', header=None ) 
    df_j.columns = ['journal']
    idcs_j = (df_j['journal'].isin( journals)  )

    # COMBINE INDICES
    idcs = idcs_y & idcs_j
    
    df_j = df_j[idcs]
    df_y = df_y[idcs]
    df_f = df_f[idcs] 
    min_cases = df_j.groupby(['journal']).size().min()
    

    if min_cases < 2*SAMPLE_SIZE:
        print ("SAMPLE_SIZE=%d TO LARGE. MAX NUMBER OF SAMPLES CAN BE %d" % ( SAMPLE_SIZE, min_cases / 2 ) )
        SAMPLE_SIZE = min_cases / 2


    df_combined = pd.concat([df_j, df_y, df_f], axis=1)
   
    TRAIN_LIST = []
    TEST_LIST  = []
    for journal_x in journals:
        df_jx = df_combined[ df_combined['journal'] == journal_x]  
        train_x, test_x = train_test_split(df_jx, test_size=SAMPLE_SIZE, train_size=SAMPLE_SIZE,random_state=42)
        TRAIN_LIST.append( train_x)
        TEST_LIST.append( test_x)

    train_data = pd.concat(TRAIN_LIST, axis=0)
    test_data  = pd.concat(TEST_LIST, axis=0)

    return ( train_data, test_data )




