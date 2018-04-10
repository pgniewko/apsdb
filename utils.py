#! /usr/bin/env python

import numpy as np
import json
import csv
import geograpy

failed_tries = 0

def get_date_jsonfile(json_file_):
    with open(json_file_) as data_file:    
        data = json.load(data_file)
    y,m,d = data['date'].split('-')
    return (int(y),int(m),int(d))

def get_journal_short_json(json_file_):
    with open(json_file_) as data_file:    
        data = json.load(data_file)
    return data['journal']['id']

def get_doi(json_file_):
    with open(json_file_) as data_file:    
        data = json.load(data_file)
    return data['id']

def get_issue_volume(json_file_):
    with open(json_file_) as data_file:
        data = json.load(data_file)
    return (int(data['issue']['number']),int(data['volume']['number']))

def get_number_of_pages(json_file_):
    with open(json_file_) as data_file:
        data = json.load(data_file)
    try:
        return int(data['numPages'])
    except KeyError:
        return "N/A"
  

def get_abstract(json_file_):
    with open(json_file_) as data_file:    
        data = json.load(data_file)
    return (data['abstract']['value'], data['abstract']['format'])

def get_title(json_file_):
    with open(json_file_) as data_file:    
        data = json.load(data_file)
    return data['title']['value']

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

def extract_country(affiliation):
     places = geograpy.get_place_context(text=affiliation)
     try:
         country_ = places.country_mentions[0][0]
         return country_
     except IndexError, e:
         return ""

def get_all_countries(json_file_):
    affiliations_ = get_all_affiliations(json_file_)

    countries_list = []
    for aff_ in affiliations_:
        if len(aff_) > 0:
            try:
               country_ = extract_country(aff_)
               if len(country_) > 0:
                   countries_list.append(country_)

            except Exception, e:
               continue

    countries_list = list( set( countries_list) )
 
    return countries_list


def parse_csv_file(csv_file):
    aps_dict_1 = {} # doi's citations
    aps_dict_2 = {} # what doi cites
    with open(csv_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[1] in aps_dict_1:
                aps_dict_1[row[1]].append(row[0])
            else:
                aps_dict_1[row[1]] = [row[0]]

            if row[0] in aps_dict_2:
                aps_dict_2[row[0]].append(row[1])
            else:
                aps_dict_2[row[0]] = [row[1]]

    return aps_dict_1, aps_dict_2


