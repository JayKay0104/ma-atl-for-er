#!/usr/bin/env python
# coding: utf-8

#%%

import Levenshtein as lev
import py_stringmatching as sm
import re
from datetime import datetime
#import nltk
from nltk.corpus import stopwords
sw = set(stopwords.words('english'))
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

# remove stopwords from string
def remove_sw(string,output_list=False):
    string = str(string).lower().split()
    if(output_list):
        return [w for w in string if not w in sw]
    else:
        return ' '.join([w for w in string if not w in sw]).strip()

#%% String Similarity Measures

def jac_q3_sim(str1,str2):
    try:
        # lower case the two input strings
        str1 = str(str1).lower().strip()
        str2 = str(str2).lower().strip()
        # assign a sim score of -1 when one of them is null
        if (str1 == '' or str2 == '' or str1 == 'nan' or str2 == 'nan'):
            return -1
        else:
            q3_tok = sm.QgramTokenizer(qval=3,return_set=True)
            jac = sm.Jaccard()
            return jac.get_raw_score(q3_tok.tokenize(str1),q3_tok.tokenize(str2))
    except:
        logger.warning('No Strings at least one of the two str1: {} and str2: {} or could not be tokenized, hence -1 assigend'.format(str1,str2))
        return -1


#%%

def jac_an_sim(str1,str2):
    try:
        # lower case the two input strings
        str1 = str(str1).lower().strip()
        str2 = str(str2).lower().strip()
        # assign a sim score of -1 when one of them is null
        if (str1 == '' or str2 == '' or str1 == 'nan' or str2 == 'nan'):
            return -1
        else:
            an_tok = sm.AlphanumericTokenizer(return_set=True)
            jac = sm.Jaccard()
            return jac.get_raw_score(an_tok.tokenize(str1),an_tok.tokenize(str2))
    except:
        logger.warning('No Strings at least one of the two str1: {} and str2: {} or could not be tokenized, hence -1 assigend'.format(str1,str2))
        return -1


#%%

def lev_sim(str1,str2):
    try:
        # lower case the two input strings
        str1 = str(str1).lower().strip()
        str2 = str(str2).lower().strip()
        # assign a sim score of -1 when one of them is null
        if (str1 == '' or str2 == '' or str1 == 'nan' or str2 == 'nan'):
            return -1
        else:
            max_len = max(len(str1), len(str2))
            return 1 - (lev.distance(str1,str2) / max_len)
    except:
        logger.warning('No Strings at least one of the two str1: {} and str2: {}, hence -1 assigend'.format(str1,str2))
        return -1


#%%

def exact_sim(str1,str2):
    try:
        # lower case the two input strings
        str1 = str(str1).lower().strip()
        str2 = str(str2).lower().strip()
        # assign a sim score of -1 when one of them is null
        if (str1 == '' or str2 == '' or str1 == 'nan' or str2 == 'nan'):
            return -1
        elif (str1 == str2):
            return 1
        else:
            return 0
    except:
        logger.warning('No Strings at least one of the two str1: {} and str2: {}, hence -1 assigend'.format(str1,str2))
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

def all_missing(str1,str2):
    try:
        # lower case the two input strings
        str1 = str(str1).lower().strip()
        str2 = str(str2).lower().strip()
        # assign a sim score of 1 when both of them are null
        if ((str1 == '' and str2 == '') or (str1 == 'nan' and str2 == 'nan')):
            return 1
        else:
            return 0
    except:
        logger.warning('No String at least one of the two str1: {} and str2: {}, hence -1 assigend'.format(str1,str2))
        return -1

