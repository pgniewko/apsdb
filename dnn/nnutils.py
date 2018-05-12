import re
import sys
import nltk
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer #, LancasterStemmer, RegexpStemmer, SnowballStemmer

default_stemmer = PorterStemmer()
default_stopwords = stopwords.words('english') # or any other list of your chose

## HARD-CODED FILE NAMES
CITATION_FILE="./data/citations_data.txt"
YEAR_FILE="./data/year_data.txt"
JOURNAL_FILE="./data/journals_data.txt"
TITLES_FILE="./data/title_data.txt"
ABSTRACTS_FILE="./data/abstract_data.txt"
##


def read_abstracts(fin):
    ab_list = []
    fo = open(fin,'rU')
    for line in fo:
        ab_list.append( line )

    return pd.DataFrame( ab_list )


def get_all_data(feature_='abstract',yrange=[1990,2010],journals=['PRA','PRB']):
    if feature_ == 'abstract':
        df_f = read_abstracts( ABSTRACTS_FILE )
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
    df_combined = pd.concat([df_j, df_y, df_f], axis=1)

    return df_combined


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


def clean_text(text, stem=False):
    text = text.decode('utf-8')

    def tokenize_text(text):
        return [w for s in sent_tokenize(text) for w in word_tokenize(s)]

    def remove_special_characters(text, characters=string.punctuation.replace('-', '')):
        tokens = tokenize_text(text)
        pattern = re.compile('[{}]'.format(re.escape(characters)))
        return ' '.join(filter(None, [pattern.sub('', t) for t in tokens]))

    def stem_text(text, stemmer=default_stemmer):
        tokens = tokenize_text(text)
        return ' '.join([stemmer.stem(t) for t in tokens])

    def remove_stopwords(text, stop_words=default_stopwords):
        tokens = [w for w in tokenize_text(text) if w not in stop_words]
        return ' '.join(tokens)
    
    def remove_nonalphabetic(text):
        return re.sub('[^A-Za-z]', ' ', text)

    def remove_nonalphabetic_w_replace(text):
        text = text.replace('+', ' ').replace('.', ' ').replace(',', ' ').replace(':', ' ').replace('\'', '')
        text = text.replace('-', ' ')
    # remove digits with regex
        text = re.sub("(^|\W)\d+($|\W)", " ", text)
        return text


    text = remove_nonalphabetic(text)
    text = text.strip(' ') #strip whitespaes
    text = text.lower() #lowercase
    if stem:
        text = stem_text(text) #stemming
    
    text = remove_special_characters(text) #remove punctuation and symbols
    text = remove_special_characters(text) #remove punctuation and symbols
    text = remove_special_characters(text) #remove punctuation and symbols
    text = remove_stopwords(text) #remove stopwords
    #text.strip(' ') # strip white spaces again?

    return text


def get_sentences(text):
    text = text.decode('utf-8')
    sentences = sent_tokenize(text)
    return sentences


def get_words(sentence):
    sentence = clean_text(sentence, stem=True)
    words = word_tokenize(sentence) 
    return words


def number_of_unique_words(sents):
    my_dict = []
    for sentence in sents:
        for word in sentence:
            if word in my_dict.keys():
                my_dict[word] += 1
            else:
                my_dict[word]  = 0

    return len( my_dict.keys() ), my_dict



