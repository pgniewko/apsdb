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
from keras.utils.np_utils import to_categorical

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

    def remove_short_words(text):
        tokens = [word for word in tokenize_text(text) if len(word) > 1]
        return ' '.join(tokens)

    text = remove_nonalphabetic(text)
    text = text.strip(' ') #strip whitespaes
    text = text.lower() #lowercase
    if stem:
        text = stem_text(text) #stemming
    
    #text = remove_special_characters(text) #remove punctuation and symbols
    #text = remove_special_characters(text) #remove punctuation and symbols
    #text = remove_special_characters(text) #remove punctuation and symbols
    text = remove_stopwords(text) #remove stopwords
    text = remove_short_words(text)
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


def number_of_unique_words(words):
    my_dict = {}
    for word in words:
        if word in my_dict.keys():
            my_dict[word] += 1
        else:
            my_dict[word]  = 1

    return len( my_dict.keys() ), my_dict


def text2words(mytext):
    all_words = []
    
    sents_ = get_sentences(mytext)
    for sent_ in sents_:
        sent_  = sent_.encode('utf-8')
        words_ = get_words(sent_)
        for w_ in words_:
            all_words.append(w_)

    return all_words


def words2int(words, dict_):
    decoded_words = []
    for w_ in words:
        if w_ in dict_.keys():
            decoded_words.append( dict_[key] )

    return np.array(decoded_words)


def load_data(sample_size=100, top_words_=10000, feature_='abstract', yrange=[1990,2010], journals=['PRA','PRB']):
    
    TRAIN, TEST = get_data(SAMPLE_SIZE=sample_size, feature_=feature_, yrange=yrange, journals=journals)
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    opath = './results/' 
    if feature_ == 'titles':
        dict_stats = pickle.load( open( opath + 'titles.pickle' ) ) 
    elif feature_ == 'abstracts':
        dict_stats = pickle.load( open( opath + 'abstracts.pickle' ) )

    journals_dict = {}
    top_words_dict = {}
    jc = 0
    for j_ in journals:    
        if j_ not in journals_dict.keys():
            journals_dict[j_] = jc
            jc += 1

    wc = 1
    for key, value in sorted(dict_stats.iteritems(), key=lambda (k,v): (v,k)):
        if wc < top_words_:
            top_words_dict[key] = wc
            wc += 1
    
    for index, row in TRAIN.iterrows():
        y_ = row['year']
        j_ = row['journal']
        feat_ = row[feature_]
        feat_words = text2words(feat_)
        w_ints = words2int(feat_words, top_words_dict)
        X_train.append( w_ints )
        y_train.append( journals_dict[j_] )

    y_train = to_categorical(y_train, len(journals))
      
    for index, row in TEST.iterrows():
        y_ = row['year']
        j_ = row['journal']
        feat_ = row[feature_]
        feat_words = text2words(feat_)
        w_ints = words2int(feat_words, top_words_dict)
        X_test.append( w_ints )
        y_test.append( journals_dict[j_] )
    
    y_test = to_categorical(y_test, len(journals))

    
    return (X_train, y_train) , (X_test, y_test)

