#! /usr/bin/env python

import numpy as np
import json
from scholar.scholar import *

failed_tries = 0

def get_year_jsonfile(json_file_):
    with open(json_file_) as data_file:    
        data = json.load(data_file)
    return data['date'].split('-')[0]


def get_journal_short_json(json_file_):
    with open(json_file_) as data_file:    
        data = json.load(data_file)
    return data['journal']['id']

def get_doi(json_file_):
    with open(json_file_) as data_file:    
        data = json.load(data_file)
    return data['id']

def get_abstract(json_file_):
    with open(json_file_) as data_file:    
        data = json.load(data_file)
    return (data['abstract']['value'], data['abstract']['format'])


def get_coauthors_jsonfile(json_file_):
    with open(json_file_) as data_file:    
        data = json.load(data_file)
    try:
        return data['authors']
    except KeyError:
        global failed_tries
        failed_tries += 1
        return []


def create_time_record(jname_, dict_):
    years_ = []
    counts_= []
    if jname_ in dict_:
        jdict = dict_[jname_]
        for key_ in jdict.keys():
            years_.append( key_  )
            counts_.append( dict_[jname_][key_]  )

    else:
        return [],[]

    return years_, counts_

def get_top_N(dict_, N):
    total_count = {}
    for key in dict_.keys():
        if key not in total_count:
             total_count[key] = 0

        for year in dict_[key]:
            total_count[key] += dict_[key][year]

    summary_list = []
    for key in total_count:
        summary_list.append( total_count[key] )

    
    summary_list.sort()
    summary_list = summary_list[-min(N, len(summary_list) ):]

    min_count = np.array(summary_list).min()
    final_list = []
    for key in total_count:
        if total_count[key] >= min_count:
            final_list.append(key)

    return final_list


def get_classification_jsonfile(json_file_):
    with open(json_file_) as data_file:    
        data = json.load(data_file)

    try:
        return data['classificationSchemes']
    except KeyError:
        return []


def get_disciplines(concepts_):
    disciplines_ = []
    for i in range( len(concepts_['physh']['disciplines']) ):
        disciplines_.append(concepts_['physh']['disciplines'][i]['label'])

    return disciplines_

def get_concepts(concepts_):
    conc_ = []
    for i in range( len(concepts_['physh']['concepts']) ):
        conc_.append( [concepts_['physh']['concepts'][i]['facet']['label'], concepts_['physh']['concepts'][i]['label']] )


    return conc_

def get_all_affiliations(json_file_):

    with open(json_file_) as data_file:    
        data = json.load(data_file)

    try:
        data_ = data['affiliations']
    except KeyError:
        return []

    aff_ = []
    for ai in data_:    
        aff_.append(ai['name'])

    return aff_


def get_number_citations(doi_):
#    ScholarConf.LOG_LEVEL = 4    
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
        
        if item[1] == 'Versions list':
            cit_link_ = item[0]

    return cits_, cit_link_
