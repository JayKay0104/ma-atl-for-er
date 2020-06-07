#!/usr/bin/env python
# coding: utf-8

#%%

#import pandas as pd
#import numpy as np
import Levenshtein as lev
import py_stringmatching as sm
import re
from datetime import datetime
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
#import nltk
from nltk.corpus import stopwords
sw = set(stopwords.words('english'))
from nltk.stem import PorterStemmer 
pstemmer = PorterStemmer()
import logging
logger = logging.getLogger(__name__)


#%%

# remove text within parentheses from string
def remove_paren(string):
    # ensure input is string by converting it to string and lower case it
    string = str(string).lower()
    # first exclude text within parentheses and ensure no whitespace at the beginning or end
    string = re.sub(r'\(.*\)','',string).strip()
    # before returning the string without parentheses also ensure no double whitespaces are in the string
    return re.sub(r'  ',' ',string).strip()
        
def remove_sw(string,output_list=False,stem=False):
    string_lst = re.sub('[^A-Za-z0-9\s\t\n]+', '', str(string).lower().strip()).split()
    if(stem):
        if(output_list):
            return [pstemmer.stem(w) for w in string_lst if not w in sw]
        else:
            return ' '.join([pstemmer.stem(w) for w in string_lst if not w in sw]).strip()
    else:
        if(output_list):
            return [w for w in string_lst if not w in sw]
        else:
            return ' '.join([w for w in string_lst if not w in sw]).strip()

#%% String Similarity Measures

# this function is quite slow because it needs to tokenize the words in 3-grams first
def jac_q3_sim(str1,str2):
    try:
        # not needed as we already casted all to string and
        # lower cased and stripped all values before handing it over
        #str1 = str(str1).lower().strip()
        #str2 = str(str2).lower().strip()
        # assign a sim score of -1 when one of them is null
        if (str1 == 'nan' or str2 == 'nan' or str1 == '' or str2 == ''):
            return -1
        else:
            q3_tok = sm.QgramTokenizer(qval=3,return_set=True)
            jac = sm.Jaccard()
            return jac.get_raw_score(q3_tok.tokenize(str1),q3_tok.tokenize(str2))
    except:
        logger.warning('Issue with Jaccard_q3_Sim, hence -1 assigned')
        return -1


#%%

# TOO SLOW!!!
#def jac_an_sim(str1,str2):
#    try:
#        # lower case the two input strings
#        str1 = str(str1).lower().strip()
#        str2 = str(str2).lower().strip()
#        # assign a sim score of -1 when one of them is null
#        if (str1 == '' or str2 == '' or str1 == 'nan' or str2 == 'nan'):
#            return -1
#        else:
#            an_tok = sm.AlphanumericTokenizer(return_set=True)
#            jac = sm.Jaccard()
#            return jac.get_raw_score(an_tok.tokenize(str1),an_tok.tokenize(str2))
#    except:
#        logger.warning('No Strings at least one of the two str1: {} and str2: {} or could not be tokenized, hence -1 assigend'.format(str1,str2))
#        return -1


#%%
        
def jac_an_sim(str1, str2):
    try:
        # not needed as we already casted all to string and
        # lower cased and stripped all values before handing it over
        #str1 = str(str1).lower().strip()
        #str2 = str(str2).lower().strip()
        # create (basically tokens sperated by whitespace) token sets (need to be set for Jaccard)
        a = set(str1.split())
        b = set(str2.split())
        # calculate intersection of both sets
        c = a.intersection(b)
        if (str1 == 'nan' or str2 == 'nan' or str1 == '' or str2 == ''):
            return -1.0
        else:
            return float(len(c)) / float(len(a) + len(b) - len(c))
    except:
        logger.warning('Issue with Jaccard_an_Sim, hence -1 assigned')
        return -1
    
#%%
def relaxed_jaccard_sim(str1, str2):
    try:
        # not needed as we already casted all to string and
        # lower cased and stripped all values before handing it over
        #str1 = str(str1).lower().strip()
        #str2 = str(str2).lower().strip()
        if (str1 == 'nan' or str2 == 'nan' or str1 == '' or str2 == ''):
            return -1
        a = set(str1.split())
        b = set(str2.split())
        c = []
        for a_ in a:
            for b_ in b:
                if lev_sim(a_, b_) > 0.7:
                    c.append(a_)
    
        intersection = len(c)
        min_length = min(len(a), len(b))
        if intersection > min_length:
            intersection = min_length
        return float(intersection) / float(len(a) + len(b) - intersection)
    except:
        logger.warning('Issue with Relaxed Jaccard Sim, hence -1 assigned')
        return -1

#%%
    
def containment_sim(str1, str2):
    try:
        # not needed as we already casted all to string and
        # lower cased and stripped all values before handing it over
        # a = set(str(str1).lower().split())
        # b = set(str(str2).lower().split())
        a = set(str1.split())
        b = set(str2.split())
        c = a.intersection(b)
        if (str1 == 'nan' or str2 == 'nan' or str1 == '' or str2 == ''):
            return -1
        elif (len(a) == 0 or len(b) == 0):
            return -1
        else:
            return float(len(c)) / float(min(len(a), len(b)))
    except:
        logger.warning('Issue with Containment Sim, hence -1 assigned')
        return -1

#%%

def lev_sim(str1,str2):
    try:
        # not needed as we already casted all to string and
        # lower cased and stripped all values before handing it over
        # str1 = str(str1).lower().strip()
        # str2 = str(str2).lower().strip()
        # assign a sim score of -1 when one of them is null
        if (str1 == 'nan' or str2 == 'nan' or str1 == '' or str2 == ''):
            return -1
        else:
            max_len = max(len(str1), len(str2))
            return 1 - (lev.distance(str1,str2) / max_len)
    except:
        logger.warning('Issue with Levenshtein, hence -1 assigned')
        return -1


#%%

def exact_sim(str1,str2):
    try:
        # not needed as we already casted all to string and
        # lower cased and stripped all values before handing it over
        # str1 = str(str1).lower().strip()
        # str2 = str(str2).lower().strip()
        # assign a sim score of -1 when one of them is null
        if (str1 == 'nan' or str2 == 'nan' or str1 == '' or str2 == ''):
            return -1
        elif (str1 == str2):
            return 1
        else:
            return 0
    except:
        logger.warning('Issue with exact_sim, hence -1 assigend')
        return -1        


#%% Numeric Similarity or Distance Measures 

def num_sim(nr1,nr2):
    try: 
        nr1 = int(nr1)
        nr2 = int(nr2)
        # if at least one is missing (in the dataset -1 was used to encode NaN) then return a sim score of -1 
        if ((nr1==-1) or (nr2==-1)): 
            return -1
        # if they both are equal than 1
        elif(nr1==nr2): 
            return 1
        # if they differ in only 10 pages then 0.5
        elif ((abs(nr1-nr2))<10):
            return 0.5
        else: 
            return 0
    # when the input could not be encoded in a integer than a similarity score of -1 is returned
    except:
        logger.warning('No Numbers at least one of the two nr1: {} and nr2: {}, hence -1 assigend'.format(nr1,nr2))
        return -1


#%%

def num_abs_diff(nr1,nr2):
    try: 
        nr1 = int(nr1)
        nr2 = int(nr2)
        # if at least one is missing (in the dataset -1 was used to encode NaN) then return a sim score of -1 
        if ((nr1==-1) or (nr2==-1)): 
            return -1
        return abs(nr1-nr2)
    # when the input could not be encoded in a integer than a similarity score of -1 is returned
    except:
        logger.warning('No Numbers at least one of the two nr1: {} and nr2: {}, hence -1 assigend'.format(nr1,nr2))
        return -1


#%% Date Similarity or Distance Measure

# function from https://stackoverflow.com/questions/40417606/python-library-to-extract-date-format-from-given-string/40423535
# slightly adapted though

# not a complete list in general but includes all identified formats in the datasets
date_formats = ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '(%d %b %Y)', '%d-%m-%Y', '%m-%d-%Y', '%m-%d-%y', '%Y-%m-%d', '%d-%m-%y',
                '%b. %d, %Y', '%Y/%m/%d', '%Y', '%B %d, %Y' , '%d/%m/%Y', '%m/%d/%Y', 
                '%m/%d/%y', '%Y.%m.%d', '%d.%m.%Y', '%m.%d.%Y', '%B %d %Y', '%B %Y', '%c', '%x', '%y', '%-y']

def onlyAfterDigit(match):
    return match.group(1)

def get_date_format(string):
    # pre-process: delete ordinals st, nd, rd, th from string
    string = re.sub('(\d)(st|nd|rd|th)',onlyAfterDigit,str(string).strip())
    out = 'NO DATE'
    for fmt in date_formats:
        try:
            datetime.strptime(string, fmt)
            out = fmt
            break
        except ValueError:
            continue
    return out

def alignDTFormat(string):
    # pre-process: delete ordinals st, nd, rd, th from string
    string = re.sub('(\d)(st|nd|rd|th)',onlyAfterDigit,str(string).strip())
    # function to align the date format to '%Y-%m-%d' string (string is returned and not datetime object)
    try:
        currentFormat = get_date_format(string)
        return datetime.strftime(datetime.strptime(string,currentFormat),'%Y-%m-%d')
    except:
        return ''

def days_sim(str1, str2):
    try:
        str1 = str(str1).strip()
        str2 = str(str2).strip()
        # assign a sim score of -1 when one of them is null
        if (str1 == '' or str2 == '' or str1 == 'nan' or str2 == 'nan'):
            return -1
        else:
            # convert the strings to datetime first.
            d1 = datetime.strptime(str1, get_date_format(str1))
            d2 = datetime.strptime(str2, get_date_format(str2))
            # return a sim score of 1 when the two dates differ in less than 365 days
            return 1 if (abs((d2 - d1).days)<365) else 0
    except:
        logger.warning('No Date at least one of the two str1: {} and str2: {}, hence -1 assigend'.format(str1,str2))
        return -1

def years_sim(str1, str2):
    try:
        str1 = str(str1).strip()
        str2 = str(str2).strip()
        # assign a sim score of -1 when one of them is null
        if (str1 == '' or str2 == '' or str1 == 'nan' or str2 == 'nan'):
            return -1
        else:
            # convert the strings to datetime first.
            d1 = datetime.strptime(str1, get_date_format(str1))
            d2 = datetime.strptime(str2, get_date_format(str2))
            # return a sim score of 1 when the two dates differ in less than two years
            return 1 if (abs(((d2 - d1).days)/365)<2) else 0
    except:
        logger.warning('No Date at least one of the two str1: {} and str2: {}, hence -1 assigend'.format(str1,str2))
        return -1        
    
def days_diff(str1,str2):
    try:
        str1 = str(str1).strip()
        str2 = str(str2).strip()
        # assign a sim score of -1 when one of them is null
        if (str1 == '' or str2 == '' or str1 == 'nan' or str2 == 'nan'):
            return -1
        else:
            # convert the strings to datetime first.
            d1 = datetime.strptime(str1, get_date_format(str1))
            d2 = datetime.strptime(str2, get_date_format(str2))
            # return a sim score of 1 when the two dates differ in less than 365 days
            return abs((d2 - d1).days)
    except:
        logger.warning('No Date at least one of the two str1: {} and str2: {}, hence -1 assigend'.format(str1,str2))
        return -1        
    
#%%

# commented out all_missing as this feature could be misleading
#def all_missing(str1,str2):
#    try:
#        # lower case the two input strings
#        str1 = str(str1).lower().strip()
#        str2 = str(str2).lower().strip()
#        # assign a sim score of 1 when both of them are null
#        if ((str1 == '' and str2 == '') or (str1 == 'nan' and str2 == 'nan')):
#            return 1
#        else:
#            return 0
#    except:
#        logger.warning('No String at least one of the two str1: {} and str2: {}, hence -1 assigend'.format(str1,str2))
#        return -1

#%%  

def calculate_cosine_tfidf(df):   
    if(len(df.columns) != 2):
        raise ValueError('The function calculate_tfidf needs the dataframe with two columns\
                         where each contain the data of the long string attribute from one source')
    # commented out because pre-processing not needed. Was already applied before handing it over, to make it faster.
#    df['concat'] = df.apply(lambda row: "{} {}".format(re.sub('[^A-Za-z0-9\s\t\n]+', '', str(row[0]).lower().strip()),
#                                                       re.sub('[^A-Za-z0-9\s\t\n]+', '', str(row[1]).lower().strip())),axis=1)
    
    df['concat'] = df.apply(lambda row: "{} {}".format(row[0],row[1]),axis=1)
    
    clf = TfidfVectorizer()
    clf.fit(df['concat'])
    
    tfidf_series1 = clf.transform(df.iloc[:,0]).todense()
    tfidf_series2 = clf.transform(df.iloc[:,1]).todense()
    
    cosine_tfidf_sim = [1 - cosine(tfidf_series1[x],tfidf_series2[x]) for x in range(df.shape[0])]
    return cosine_tfidf_sim
