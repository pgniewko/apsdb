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
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


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


def read_sentences(fname):
     fin = open(fname, 'rU')
     sentences = []
     for line in fin:
         pairs = line.rstrip('\n').split()
         sentences.append( pairs )

     return sentences


def text2words(mytext):
    all_words = []
    
    sentences = get_sentences(mytext)
    for sentence in sentences:
        sentence  = sentence.encode('utf-8')
        words = get_words(sentence)
        all_words += words

    return all_words


def load_data(sample_size=1000, feature_='abstract', yrange=[1990,2010], journals=['PRA','PRB']):
    
    TRAIN, TEST = get_data(SAMPLE_SIZE=sample_size, feature_=feature_, yrange=yrange, journals=journals)
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    journals_dict = {}
    jcounter = 0
    for key in journals:
        if key not in journals_dict:
            journals_dict[key] = jcounter
            jcounter += 1

    for index, row in TRAIN.iterrows():
        y_ = row['year']
        j_ = row['journal']
        text = row[feature_]
        X_train.append( ' '.join( text2words(text)) )
        y_train.append( journals_dict[j_] )

    for index, row in TEST.iterrows():
        y_ = row['year']
        j_ = row['journal']
        text = row[feature_]
        X_test.append( ' '.join(text2words(text)) )
        y_test.append( journals_dict[j_] )
    
    return (X_train, y_train) , (X_test, y_test)


def tokenize_text(train_l, test_l, word2vec, MAX_NB_WORDS=10000,MAX_SEQUENCE_LENGTH=200, EMBEDDING_DIM=300):
    
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(train_l + test_l)
    
    sequences_train = tokenizer.texts_to_sequences(train_l)
    sequences_test  = tokenizer.texts_to_sequences(test_l)
   
    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))

    data_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
    data_test  = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

    print('Preparing embedding matrix')

    nb_words = min(MAX_NB_WORDS, len(word_index))+1
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue 

        if word in word2vec.wv.vocab:
            embedding_matrix[i] = word2vec.wv.word_vec(word)

    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    return embedding_matrix, data_train, data_test, nb_words


def transform_Y(Y):
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    return dummy_y   




