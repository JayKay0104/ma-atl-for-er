#!/usr/bin/env python
# coding: utf-8

#%%
import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
import glob
import itertools
import re
import py_stringsimjoin as ssj
import py_stringmatching as sm
from collections import Counter
import sim_measures as sim
import logging
import random
import numpy.matlib
from itertools import groupby
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings('ignore')


#%%

def removeElements(lst, k, less_than_k=False): 
    counted = Counter(lst) 
    if(not less_than_k):  
        return [el for el in lst if counted[el] >= k]
    else:
        return [el for el in lst if counted[el] < k]


#%%

def readDataInDictionary(path_to_directory = '../datasets/', pattern_of_filename = '(.*)', sep=';'):
    """
    Function to read in the datasests. The datasets need to be stored as csv files!
    
    Parameters:
    
    path_to_directory: Specify the path from the current directory to the directory where the datasets are stored
    Default: \'../datasets/\'
    
    pattern_of_filename: Specify the pattern of the filenames of the datasets. Only the datasets where the filenames
    matches the pattern get read in. The regex need to be enclosed in parentheses. 
    Exp: Files that only have letters in there filename match this pattern \'([a-zA-Z]*)\'
    Default: \'(.*)\' (reads in all csv files contained in the directory)
    
    sep: Specify the delimiter of the csv files. All csv files need to have the same delimiter
    Default: ';'
    """
    pattern_name = '{}{}.csv'.format(path_to_directory,pattern_of_filename)
    logger.debug('pattern_name: {}'.format(pattern_name))
    file_list = glob.glob('{}*.csv'.format(path_to_directory))
    logger.debug('file_list: {}'.format(file_list))
    res = [re.findall(pattern_name,x)[0] for x in file_list if bool(re.match(pattern_name,x))]
    file_list = [x for x in file_list if bool(re.match(pattern_name,x))]
    logger.debug('file_list: {}'.format(file_list))
    
    dfs = {}
    for i in range(len(file_list)):
        dfs.update({res[i]:pd.read_csv(file_list[i],sep=sep,low_memory=False)})
        logger.info('{} is read in and is stored in the dictionary with they key [\'{}\']'.format(file_list[i],res[i]))
    return dfs


#%%

def getDataTypes(df):
    """
    Get the data types for each column and return it in a dictionary like column:datatype!
    
    Datetime formats that can be detected are:
    date_formats = ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '(%d %b %Y)', '%d-%m-%Y', '%m-%d-%Y', 
                '%b. %d, %Y', '%Y/%m/%d', '%Y', '%B %d, %Y' , '%d/%m/%Y', '%m/%d/%Y', 
                '%m/%d/%y', '%Y.%m.%d', '%d.%m.%Y', '%m.%d.%Y', '%B %d %Y', '%B %Y', '%c', '%x', '\'%Y']
    """
    logger.info('Start detecting datatypes for all columns of dataframe:')
    types_dict = {}
    for column in df:
        if('_id' in column): continue
        values_in_column = df[column].dropna()
        length = len(values_in_column)
        k=0
        types_list=list(set(values_in_column.apply(type).tolist()))
    
        if len(types_list) == 0: 
            # No type was detected, hence String is assigned
            types_dict[column] = 'str'
        elif len(types_list) >1: 
            # More than one type was detected, hence String is assigned
            types_dict[column] = 'str'
        else:
            #check for date because python type function does not detect date formats within strings
            if str in types_list:
                if(values_in_column.nunique()<=2):
                    types_dict[column] = 'custom'
                    logger.info('Datatype for Column {} detected: {}'.format(column, types_dict[column]))
                    continue
                for value in values_in_column:
                    if ((k<=(length/2)) and (sim.get_date_format(value)!='NO DATE')):
                        k += 1
                logger.debug('k is equal to: {}'.format(k))
                if(k>=(length/2)):
                    # if most of the time date was detected date is assigned as data type for this column
                    types_dict[column] = 'date'
                else:
                    # if more string than date then string is assigned
                    types_dict[column] = 'str'
            else:
                # if not string or date and only one data type was detected, integer is assigend as data_type of column
                types_dict[column] = 'num'
        logger.info('Datatype for Column {} detected: {}'.format(column, types_dict[column]))
    return types_dict

#%%

def checkAlignedDataTypes(types_dict_list):
    """
    This function takes as input a list of dictionaries. Each dictionary should contain as keys the column of the dataset 
    and as corresponding value the data type for that column. The function then checks whether the common attributes among
    the datasets share the same data type, which is a requirement in order to create potential correspondences and labeled
    feature vectors.
    If the data types are the same across all common attributes, the function returns True and False otherwise.
    """
    lst = []
    aligned_types_lst = []
    for types_dict in types_dict_list:
        keys = ['_'.join(s.split('_')[1:]) for s in list(types_dict.keys())]
        values = types_dict.values()
        lst.append(tuple(zip(keys,values)))
        aligned_types_lst.append(dict(zip(keys,values)))
    lst = [item for sublist in lst for item in sublist]
    
    wrong_format = removeElements(lst,2,less_than_k=True)
    logger.debug('wrong format: '.format(wrong_format))
    
    if(len(wrong_format)>0):
        for i in range(len(wrong_format)):
            key = wrong_format[i][0]
            logger.info('Different Format for {}'.format(key))
            key_value_lst = [types_dict[key] for types_dict in aligned_types_lst]
            #key_value_lst = []
            #for types_dict in aligned_types_lst:
            #    key_value_lst.append(types_dict[key])
            key_value_set = set(key_value_lst)
            logger.info('Types: {}'.format(key_value_set))
        logger.info('No Dictionary with Data Type per Column can be returned. Align schema and format of correpsonding columns first!')
        return False
    else:
        logger.info('All Datasets share the same Data Type across common attributes!')
        return True


#%%

def getAlignedDataTypeSchema(types_dict_list, lst_of_ids_to_be_removed=[]):
    """
    This function takes as input a list of dictionaries that contain as keys the column of each dataset and as 
    corresponding value the data type for that column (output of function getDataType(df). 
    Besides that, one can optionally pass as second argument a list of column names that act as ids among all datasets 
    and hence can be removed from the dictionary.
    The output is the final type_per_column dictionary that is required in order to create the labeled feature
    vector with createLabeledFeatureVector()
    """
    ids_to_be_removed = set(lst_of_ids_to_be_removed)
    lst = []
    aligned_types_lst = []
    for types_dict in types_dict_list:
        keys = ['_'.join(s.split('_')[1:]) for s in list(types_dict.keys())]
        values = types_dict.values()
        lst.append(tuple(zip(keys,values)))
        aligned_types_lst.append(dict(zip(keys,values)))
    lst = [item for sublist in lst for item in sublist]
    
    wrong_format = removeElements(lst,2,less_than_k=True)
    logger.debug('wrong format: '.format(wrong_format))
    
    if(len(wrong_format)>0):
        for i in range(len(wrong_format)):
            key = wrong_format[i][0]
            logger.info('Different Format for {}'.format(key))
            key_value_lst = []
            for types_dict in aligned_types_lst:
                key_value_lst.append(types_dict[key])
            key_value_set = set(key_value_lst)
            logger.info('Types: {}'.format(key_value_set))
        logger.info('No Dictionary with Data Type per Column can be returned. Align schema and format of correpsonding columns first!')
        return None
    else:
        final_data_type_dict = dict(set(removeElements(lst,2)))
        if(len(ids_to_be_removed)>0):
            for item in ids_to_be_removed:
                final_data_type_dict.pop(item,None)
        return final_data_type_dict


#%%

def returnAlignedDataTypeSchema(df_dict,lst_of_ids_to_be_removed=[]):
    dt_dfs_lst = []
    for df in df_dict:
        dt_dfs_lst.append(getDataTypes(df_dict[df]))
    
    if(checkAlignedDataTypes(dt_dfs_lst)):
        type_per_column = getAlignedDataTypeSchema(dt_dfs_lst,lst_of_ids_to_be_removed)
        logger.info('type_per_column dictionary returned')
        return type_per_column
    else:
        logger.info('type_per_column dictionarynot returned but list')
        return dt_dfs_lst


#%%
    
def returnTrueMatches(df_dict = {}, merge_on = 'isbn', prefix=True):
    """
    Merges each dataframe contained in df_dict with each other (so every binary combination) and 
    returns the merged dataframes in a dictionary
    
    Parameter:
    df_dict: Dictionary containing min two dataframes that can be merged with each other based on merge_on column
    merge_on: Column to merge on
    prefix: Specify if the merge_on column has the dataframe name (key in dictionary) as prefix; Default: True
    """
    for key in df_dict:
        if(not any(merge_on in s for s in df_dict[key].columns.values.tolist())):
            logger.info('At least one dataframe in df_dict does not contain the merge_on column: {}. Align schema first!'.format(merge_on))
            return None
    matches_dict = {}
    if(prefix):
        # ensure the the merge column has the same type for each df (here string is used).
        for key in df_dict:
            df_dict[key]['{}_{}'.format(key,merge_on)] = df_dict[key]['{}_{}'.format(key,merge_on)].apply(lambda s: str(s))
        for i in itertools.combinations(df_dict, 2):
            matches_dict.update({'matches_{}_{}'.format(i[0],i[1]):pd.merge(df_dict[i[0]],df_dict[i[1]],left_on='{}_{}'.format(i[0],merge_on),right_on='{}_{}'.format(i[1],merge_on))})
            logger.info('{} and {} got merged in {}.\tAmount of True Matches: {}'.format(i[0],i[1],'matches_{}_{}'.format(i[0],i[1]),matches_dict['matches_{}_{}'.format(i[0],i[1])].shape[0]))
        return matches_dict
    else:
        # ensure the the merge column has the same type for each df (here string is used).
        for key in df_dict:
            df_dict[key][merge_on] = df_dict[key][merge_on].apply(lambda s: str(s))
        for i in itertools.combinations(df_dict, 2):
            matches_dict.update({'matches_{}_{}'.format(i[0],i[1]):pd.merge(df_dict[i[0]],df_dict[i[1]],on=merge_on)})
            logger.info('{} and {} got merged in {}.\tAmount of True Matches: {}'.format(i[0],i[1],'matches_{}_{}'.format(i[0],i[1]),matches_dict['matches_{}_{}'.format(i[0],i[1])].shape[0]))
        return matches_dict


#%%

def createPotentialCorr(df1, df2, ids = (), filtering_attributes = (), measure='jaccard', 
                        tokenization=sm.QgramTokenizer(qval=3,return_set=True), threshold=0.3):
    """
    Perform filtering on df1 and df2 and create potential correspondences. Here the best performing filtering startegy
    should be used, which should have been evaluated upfront.
    
    Parameter:
    df1, df2: the two dataframes where the potential correspondences shall get created from
    ids: Tuple of the two ids from df1 and df2: Exp: (df1_id,df2_id)
    filtering_attributes: the two attributes that shall be used for filtering: Exp: ('ban_author_lower','half_author_lower')
    measure: Either 'jaccard' or 'levenshtein'
    tokenization: Has to be of type Tokenizer from py_stringmatching
    threshold: range (0,1) for jaccard and [1,inf) for levenshtein
    """
    # first some pre-processing to ensure all values are lowercased and stripped
    df1[filtering_attributes[0]] = df1[filtering_attributes[0]].apply(lambda s: str(s).lower().strip())
    df2[filtering_attributes[1]] = df2[filtering_attributes[1]].apply(lambda s: str(s).lower().strip())
    logger.info('Filtering Attribute normalized (lowercased and stripped)')
        
    if((measure=='jaccard' or measure=='jac') and (0<threshold<0.9)):
        out = ssj.jaccard_join(df1,df2,*ids,*filtering_attributes,
                                tokenization, threshold, 
                                l_out_attrs=df1.columns.drop(ids[0]).values.tolist(), 
                                r_out_attrs=df2.columns.drop(ids[1]).values.tolist(),
                                n_jobs=-3)
        logger.info('Finished: Potential correspondences created using filtering based on Jaccard Similarity!')
        return out
    elif((measure=='levenshtein' or measure=='edit_distance' or measure=='lev') and threshold>=1):
        out = ssj.edit_distance_join(df1,df2,*ids,*filtering_attributes,
                                threshold, 
                                l_out_attrs=df1.columns.drop(ids[0]).values.tolist(), 
                                r_out_attrs=df2.columns.drop(ids[1]).values.tolist(),
                                n_jobs=-3)
        logger.info('Finished: Potential correspondences created using filtering based on Levenshtein Similarity!')
        return out
    else:
        logger.error('Either measure {} not supported (only jaccard and levenshtein) or threshold {} not meaningful.'.format(measure,threshold))
        return None

#%%

def returnPotentialCorr(df_dict, filt_df_dict):
    """
    Creates potential correspondences for each dataset combination of the dataframes from df_dict 
    using filtering (string similarity joins) with the parameters defined in filt_df_dict and stores them
    in dataframes that are returned as a dictionary where the keys are in the form 'corr_df1_df2'.
    
    Parameters:
    df_dict: Dictionary containing min two dataframes that follow the same structure/schema. The names
    of the dataframes are the keys in the dictionary and the dataframe itself are the values
    
    filt_df_dict: Dictionary containing the filtering parameters for each dataset combination. Here
    it is important that when the df_dict contains dataframes in the following order:
    df1,df2,df3 
    then filt_df_dict has to contain the combinations as keys like this: 
    df1_df2, df1_df3, df2_df3.
    Different orders like df2_df1 or df3_df2 will cause problems!
    Exp.
    filt_df_dict = {'df1_df2':{'ids':('df1_id','df2_id'),
                               'filtering_attributes':('df1_author','df2_firstauthor'),
                               'measure':'jaccard',
                               'tokenizer':sm.QgramTokenizer(qval=3,return_set=True),
                               'threshold':0.3},
                    'df1_df3':{'ids':('df1_id','df3_id'),
                               'filtering_attributes':('df1_author','df3_firstauthor'),
                               'measure':'jaccard',
                               'tokenizer':sm.QgramTokenizer(qval=3,return_set=True),
                               'threshold':0.3},
                    'df2_df3':{'ids':('df2_id','df3_id'),
                               'filtering_attributes':('df2_firsttwoauthors','df3_firsttwoauthors'),
                               'measure':'levenshtein',
                               'threshold':5},
    
    """
    corr_dict = {}
    logger.info('Start returning potential correspondences for the dataset combinations')
    for i in itertools.combinations(df_dict, 2):
        logger.debug(i)
        ids = filt_df_dict['{}_{}'.format(i[0],i[1])]['ids']
        logger.debug('IDs: {}'.format(ids))
        filt_attr = filt_df_dict['{}_{}'.format(i[0],i[1])]['filtering_attributes']
        logger.debug('Filtering attributes: {}'.format(filt_attr))
        mea = filt_df_dict['{}_{}'.format(i[0],i[1])]['measure']
        logger.debug('Measure: {}'.format(mea))
        thr = filt_df_dict['{}_{}'.format(i[0],i[1])]['threshold']
        logger.debug('Threshold: {}'.format(thr))
        if((mea!='levenshtein' and mea!='edit_distance' and mea!='lev')):
            tok = filt_df_dict['{}_{}'.format(i[0],i[1])]['tokenizer']
            logger.debug('Tokenizer: {}'.format(tok))
            corr_dict.update({'corr_{}_{}'.format(i[0],i[1]):createPotentialCorr(df_dict[i[0]],df_dict[i[1]],ids,filt_attr,mea,tok,thr)})
            logger.debug('For {} and {} potential correspondences are stored in {}.\tAmount of pot. Corr. after filtering: {}'.format(i[0],i[1],'corr_{}_{}'.format(i[0],i[1]),corr_dict['corr_{}_{}'.format(i[0],i[1])].shape[0]))
        else:
            corr_dict.update({'corr_{}_{}'.format(i[0],i[1]):createPotentialCorr(df_dict[i[0]],df_dict[i[1]],ids,filt_attr,measure=mea,threshold=thr)})
            logger.debug('For {} and {} potential correspondences are stored in {}.\tAmount of pot. Corr. after filtering: {}'.format(i[0],i[1],'corr_{}_{}'.format(i[0],i[1]),corr_dict['corr_{}_{}'.format(i[0],i[1])].shape[0]))
    logger.info('Finished! Potential Correspondences created')
    return corr_dict


#%%

def returnLabeledFeatureVectors(corr_dict, type_per_column, columns_to_be_dropped=[], identifier='', unique_identifier='', no_prefix=False):
    """
    Creates features of the potential correspondences for each dataset combination stored in corr_dict
    using the pre-defined similarity measures for the corresponding data type of each attribute/column.
    The datatypes for each column have to be stored in the type_per_column dicitionary.
    
    Parameters:
    corr_dict: Dictionary containing min one dataframe containing pot. corr. of two datasest. 
    The names of the pot. corr. dataframes are the keys in the dictionary and the dataframe itself 
    are the values
    
    type_per_column: Dictionary specifying the data type of each column, so that the functions knows
    which similarity measures need to be performed in order to create the feature vectors. 
    This dictionary can be retrieved with the function returnAlignedDataTypeSchema(...)
    
    """
    feature_dict = {}
    for corr in corr_dict:
        feature_dict.update({'feature_{}_{}'.format(corr.split('_')[1],corr.split('_')[2]):createLabeledFeatureVector(corr_dict[corr], type_per_column, columns_to_be_dropped, identifier, unique_identifier,no_prefix)})
        logger.debug('For the pot. corr. of {} and {} features are created and stored in feature_{}_{}.'.format(corr.split('_')[1],corr.split('_')[2],corr.split('_')[1],corr.split('_')[2]))
    logger.info('\nFinished! All labeled feature vectors are created for all dataset combinations')
    return feature_dict


#%%

def createLabeledFeatureVector(corr, type_per_column, columns_to_be_dropped=[], identifier='', unique_identifier='', no_prefix=False):
    
    """
    This function creates the labeled feature vectors for the pot. corr. provided with the argument corr.
    
    Parameters:
    corr: Dataframe that contains the potential correspondences of two datasources (as retrieved by
          performing filtering using the createPotentialCorr() function).
    type_per_column: Dictionary specifying the data type of each column, so that the functions knows
                     which similarity measures need to be performed in order to create the feature vectors. 
                     This dictionary can be retrieved with the function returnAlignedDataTypeSchema(...)
    columns_to_be_dropped: A list of column names that shall be ignored, like ['_id','_sim_score']
    identifier: Name of the identifiers. Only the last part of the name. Exp: if the names of the
                IDs are l_ban_id and r_wor_id, only 'id' has to be provided here!
    unique_identifier: Unique identifier in order to create the labels. For the topic books this is
                       'isbn'.
    """
    
    logger.info('Start Function')
    if(type_per_column==0):
        logger.error('No Type per Column Dictionary defined!')
        return None
    if(len(unique_identifier)==0):
        logger.error('No Unique Identifier defined!')
        return None
    feature_vector = pd.DataFrame()
    corr_columns = corr.columns.drop(columns_to_be_dropped)
    
    if(no_prefix):
        corr_l_name = corr_columns.str.split('_')[0][0] #to get the name of the orignial data source
        corr_r_name = corr_columns.str.split('_')[-1][0] #to get the name of the orignial data source
        
        # the column names of the unique identifiers across both data sources
        l_uid = '{}_{}'.format(corr_l_name,unique_identifier)
        r_uid = '{}_{}'.format(corr_r_name,unique_identifier)
    
        # the column names of the identifiers
        l_id = '{}_{}'.format(corr_l_name,identifier)
        r_id = '{}_{}'.format(corr_r_name,identifier)
    
        # write ids and the label in the feature_vector dataframe
        feature_vector['l_id'] = corr.apply(lambda row: row[l_id],axis=1)
        feature_vector['r_id'] = corr.apply(lambda row: row[r_id],axis=1)
        feature_vector['label'] = corr.apply(lambda row: 1 if str(row[l_uid])==str(row[r_uid]) else 0,axis=1)
        logger.info('Correspondences labelled!')
    
        ids = set([identifier,unique_identifier])
        # retrieve the header in order to identify the common attributes
        corr_header = [corr_columns.str.split('_')[i][1:] for i in range(len(corr_columns))]
        corr_header = ['_'.join(x) for x in corr_header if len(corr_header)>1]
        common_attributes = list(set(removeElements(corr_header,2))- ids)
        logger.info('Common attributes identified!')
    else:
        corr_l_name = corr_columns.str.split('_')[0][1] #to get the name of the orignial data source
        corr_r_name = corr_columns.str.split('_')[1][1] #to get the name of the orignial data source
    
        # the column names of the unique identifiers across both data sources
        l_uid = 'l_{}_{}'.format(corr_l_name,unique_identifier)
        r_uid = 'r_{}_{}'.format(corr_r_name,unique_identifier)
    
        # the column names of the identifiers
        l_id = 'l_{}_{}'.format(corr_l_name,identifier)
        r_id = 'r_{}_{}'.format(corr_r_name,identifier)
    
        # write ids and the label in the feature_vector dataframe
        feature_vector['l_id'] = corr.apply(lambda row: row[l_id],axis=1)
        feature_vector['r_id'] = corr.apply(lambda row: row[r_id],axis=1)
        feature_vector['label'] = corr.apply(lambda row: 1 if str(row[l_uid])==str(row[r_uid]) else 0,axis=1)
        logger.info('Correspondences labelled!')
    
        ids = set([identifier,unique_identifier])
        # retrieve the header in order to identify the common attributes
        corr_header = [corr_columns.str.split('_')[i][2:] for i in range(len(corr_columns))]
        corr_header = ['_'.join(x) for x in corr_header if len(corr_header)>1]
        common_attributes = list(set(removeElements(corr_header,2))- ids)
        logger.info('Common attributes identified!')
    
    # iterate through common_attributes in order to calculate the sim scores using the similarity measures
    # dedicated for the specific types
    for attr in common_attributes:
        logger.info(corr.columns[corr.columns.str.endswith('{}_{}'.format(corr_l_name,attr))][0])
        ser1 = corr[corr.columns[corr.columns.str.endswith('{}_{}'.format(corr_l_name,attr))]]
        logger.info(corr.columns[corr.columns.str.endswith('{}_{}'.format(corr_r_name,attr))][0])
        ser2 = corr[corr.columns[corr.columns.str.endswith('{}_{}'.format(corr_r_name,attr))]]
        feature_vector = pd.concat([feature_vector,calcSimSeries(ser1,ser2,attr,type_per_column[attr])], axis=1)
    logger.info('\nFinished! Labeled Feature Vectors created for {} and {}'.format(corr_l_name,corr_r_name))
    return feature_vector


#%%

def createLabeledFeatureVectorFast(corr, type_per_column, corr_l_name='', corr_r_name='', columns_to_be_dropped=['_id','_sim_score'], ids={'id','isbn'}):
    logger.info('Start Function')
    if(type_per_column==0):
        logger.error('No Type per Column Dictionary defined!')
        return None
    feature_vector = pd.DataFrame()
    corr_columns = corr.columns.drop(columns_to_be_dropped)

    # write ids and the label in the feature_vector dataframe
    feature_vector['l_id'] = corr['l_{}_id'.format(corr_l_name)]
    feature_vector['r_id'] = corr['r_{}_id'.format(corr_r_name)]
    feature_vector['l_{}_isbn'.format(corr_l_name)] = corr.apply(lambda row: str(row['l_{}_isbn'.format(corr_l_name)]),axis=1)
    feature_vector['r_{}_isbn'.format(corr_r_name)] = corr.apply(lambda row: str(row['r_{}_isbn'.format(corr_r_name)]),axis=1)
    feature_vector['label'] = corr.apply(lambda row: 1 if row['l_{}_isbn'.format(corr_l_name)]==row['r_{}_isbn'.format(corr_r_name)] else 0,axis=1)

    # retrieve the header in order to identify the common attributes
    corr_header = [corr_columns.str.split('_')[i][2:] for i in range(len(corr_columns))]
    corr_header = ['_'.join(x) for x in corr_header if len(corr_header)>1]
    common_attributes = list(set(removeElements(corr_header,2))- ids)

    # iterate through common_attributes in order to calculate the sim scores using the similarity measures
    # dedicated for the specific types
    for attr in common_attributes:
        logger.info(corr.columns[corr.columns.str.endswith('{}_{}'.format(corr_l_name,attr))][0])
        ser1 = corr[corr.columns[corr.columns.str.endswith('{}_{}'.format(corr_l_name,attr))]]
        logger.info(corr.columns[corr.columns.str.endswith('{}_{}'.format(corr_r_name,attr))][0])
        ser2 = corr[corr.columns[corr.columns.str.endswith('{}_{}'.format(corr_r_name,attr))]]
        feature_vector = pd.concat([feature_vector,calcSimSeries(ser1,ser2,attr,type_per_column[attr])], axis=1)
    return feature_vector


#%%

def calcSimSeries(series1, series2, attrName, data_type):
    """
    Calculate the implemented similarity measures (in module sim_measures) that are selected for the data_type
    between the two pd.Series
    """
    df = pd.concat([series1,series2], axis=1, sort=False,ignore_index=True)
    if(data_type=='str'):
        ser_lev_sim = df.apply(lambda row: sim.lev_sim(row[0],row[1]),axis=1)
        logger.debug('Levenshtein Similarity measured for {}'.format(attrName))
        #q3_tok = sm.QgramTokenizer(qval=3,return_set=True)
        #q3_tok_row0 = df[0].apply(lambda s: q3_tok.tokenize(s.lower().strip()) if (s.lower().strip() != '' or s.lower().strip() != 'nan' else []) )
        #q3_tok_row1 = df[1].apply(lambda s: q3_tok.tokenize(s.lower().strip()) if (s.lower().strip() != '' or s.lower().strip() != 'nan' else [])
        #df_temp = pd.DataFrame([q3_tok_row0,q3_tok_row1])
        #ser_jac_w_sim = df_temp.apply(lambda row: jac_sim(row[0],row[1]),axis=1)
        ser_jac_q3_sim = df.apply(lambda row: sim.jac_q3_sim(row[0],row[1]),axis=1)
        logger.debug('Jaccard Similarity with trigram tokenization measured for {}'.format(attrName))
        ser_jac_an_sim = df.apply(lambda row: sim.jac_an_sim(row[0],row[1]),axis=1)
        logger.debug('Jaccard Similarity with alphanumeric tokenization measured for {}'.format(attrName))
        ser_exact_sim = df.apply(lambda row: sim.exact_sim(row[0],row[1]),axis=1)
        logger.debug('Exact Similarity measured for {}'.format(attrName))
        ser_all_missing = df.apply(lambda row: sim.all_missing(row[0],row[1]),axis=1)
        logger.debug('All missing measured for {}'.format(attrName))
        logger.debug('Similarity measures calculated for {}'.format(attrName))
        df = pd.concat([ser_lev_sim,ser_jac_q3_sim,ser_jac_an_sim,ser_exact_sim,ser_all_missing],axis=1,sort=False,ignore_index=True)
        df.rename({0:attrName+'_lev_sim', 1:attrName+'_jac_q3_sim', 2:attrName+'_jac_an_sim', 3:attrName+'_exact_sim', 4:attrName+'_all_missing'},axis=1,inplace=True)
        return df
    elif(data_type=='num'):
        ser_num_abs_diff = df.apply(lambda row: sim.num_abs_diff(row[0],row[1]),axis=1)
        logger.debug('Absolute Difference measured for {}'.format(attrName))
        ser_num_sim = df.apply(lambda row: sim.num_sim(row[0],row[1]),axis=1)
        logger.debug('Numeric Similarity measured for {}'.format(attrName))
        ser_all_missing = df.apply(lambda row: sim.all_missing(row[0],row[1]),axis=1)
        logger.debug('All missing measured for {}'.format(attrName))
        logger.debug('Similarity measures calculated for {}'.format(attrName))
        df = pd.concat([ser_num_abs_diff,ser_num_sim,ser_all_missing],axis=1,sort=False,ignore_index=True)
        df.rename({0:attrName+'_num_abs_diff', 1:attrName+'_num_sim', 2:attrName+'_all_missing'},axis=1,inplace=True)
        return df
    elif(data_type=='date'):
        df[0] = df[0].apply(lambda s: sim.alignDTFormat(s))
        df[1] = df[1].apply(lambda s: sim.alignDTFormat(s))
        ser_days_sim = df.apply(lambda row: sim.days_sim(row[0],row[1]),axis=1)
        logger.debug('Days Similarity measured for {}'.format(attrName))
        ser_years_sim = df.apply(lambda row: sim.years_sim(row[0],row[1]),axis=1)
        logger.debug('Years Similarity measured for {}'.format(attrName))
        ser_days_diff = df.apply(lambda row: sim.days_diff(row[0],row[1]),axis=1)
        logger.debug('Days Difference measured for {}'.format(attrName))
        ser_all_missing = df.apply(lambda row: sim.all_missing(row[0],row[1]),axis=1)
        logger.debug('All missing measured for {}'.format(attrName))
        logger.debug('Similarity measures calculated for {}'.format(attrName))
        df = pd.concat([ser_days_sim,ser_years_sim,ser_days_diff,ser_all_missing],axis=1,sort=False,ignore_index=True)
        df.rename({0:attrName+'_days_sim', 1:attrName+'_years_sim', 2:attrName+'_days_diff', 3:attrName+'_all_missing'},axis=1,inplace=True)
        return df
    elif(data_type=='custom'):
        ser_exact_sim = df.apply(lambda row: sim.exact_sim(row[0],row[1]),axis=1)
        logger.debug('Exact Similarity measured for {}'.format(attrName))
        ser_all_missing = df.apply(lambda row: sim.all_missing(row[0],row[1]),axis=1)
        logger.debug('All missing measured for {}'.format(attrName))
        logger.debug('Similarity measures calculated for {}'.format(attrName))
        df = pd.concat([ser_exact_sim,ser_all_missing],axis=1,sort=False,ignore_index=True)
        df.rename({0:attrName+'_exact_sim', 1:attrName+'_all_missing'},axis=1,inplace=True)
        return df
    else:
        logger.error('No Similarity Measure for {} DataType defined.'.format(data_type))
        return None


#%%

def calcSim(l_value, r_value, sim, data_type='str'):
    """
    [DEPRECATED] Calc the similariy between two values by providing the similarity measure (sim) as implemented in module 
    sim_measures and the corresponding data_type. This function is not used, though.
    """
    if(data_type=='str'):
        if(sim=='lev_sim'):
            return sim.lev_sim(l_value, r_value)
        elif(sim=='jac_q3_sim'):
            return sim.jac_q3_sim(l_value, r_value)
        elif(sim=='jac_w_sim'):
            return sim.jac_w_sim(l_value, r_value)
        elif(sim=='exact_sim'):
            return sim.exact_sim(l_value, r_value)
        elif(sim=='nw_sim'):
            return sim.nw_sim(l_value, r_value)
        elif(sim=='all_missing'):
            return sim.all_missing(l_value, r_value)
        else:
            logger.error('Similarity Measure for String not defined.')
    elif(data_type=='num'):
        if(sim=='num_abs_diff'):
            return sim.num_abs_diff(l_value, r_value)
        elif(sim=='num_sim'):
            return sim.num_sim(l_value, r_value)
        elif(sim=='all_missing'):
            return sim.all_missing(l_value, r_value)
        else:
            logger.error('Similarity Measure for Numeric Values not defined.')
    elif(data_type=='date'):
        l_value = sim.alignDTFormat(l_value)
        r_value = sim.alignDTFormat(l_value)
        if(sim=='days_sim'):
            return sim.days_sim(l_value, r_value)
        elif(sim=='years_sim'):
            return sim.years_sim(l_value, r_value)
        elif(sim=='days_diff'):
            return sim.days_sim(l_value, r_value)
        else:
            logger.error('Similarity Measure for Date Values not defined.')
    elif(data_type=='custom'):
        l_value = sim.alignDTFormat(l_value)
        r_value = sim.alignDTFormat(l_value)
        if(sim=='exact_sim'):
            return sim.exact_sim(l_value, r_value)
        elif(sim=='all_missing'):
            return sim.years_sim(l_value, r_value)
        elif(sim=='days_diff'):
            return sim.all_missing(l_value, r_value)
        else:
            logger.error('Custom Similarity Measure not defined.')


#%%
    
def rescaleFeatureVectors(feature_vector, col_to_be_dropped=['l_id', 'r_id', 'label'], col_to_be_rescaled_endswith='diff'):
    """
    This function rescales all features that are not already a similarity score (range 0 to 1)
    but a distance or difference. Important the similarity scores calculated with Levenshtein
    are already a true similarity score (range 0 to 1) and not a distance anymore. Hence,
    they do not get rescaled here. Only the columns that end with the name as specified by col_to_be_rescaled_endswith
    are rescaled
    This function does not return anything but changes the dataframe feature_vector inplace provided as argument.
    """
    logger.info('Rescaling features that end with {}'.format(col_to_be_rescaled_endswith))
    feature_vector.replace(-1, np.nan, inplace=True)
    for column in feature_vector.columns.drop(col_to_be_dropped):
        if (column.endswith(col_to_be_rescaled_endswith)):
            feature_vector[column] -= feature_vector[column].min()
            feature_vector[column] /= feature_vector[column].max()
            feature_vector[column] = 1 - feature_vector[column]
            feature_vector.rename(columns={column: column.replace(col_to_be_rescaled_endswith, '{}_sim'.format(col_to_be_rescaled_endswith))}, inplace=True)
    feature_vector.replace(np.nan,-1, inplace=True)
    logger.info('All features from the frature_vector dataframes that end with {} are now rescaled inplace'.format(col_to_be_rescaled_endswith))
        
#%%
    
def rescaleFeatureVectorsInDict(feature_dict, col_to_be_dropped=['l_id', 'r_id', 'label'], col_to_be_rescaled_endswith='diff'):
    """
    This function rescales all features that are stored in the feature_vectors dataframes within the dictionary
    feature_dict and that are not already a similarity score (range 0 to 1).
    but a distance or difference. Important the similarity scores calculated with Levenshtein
    are already a true similarity score (range 0 to 1) and not a distance anymore. Hence,
    they do not get rescaled here. Only the columns that end with the name as specified by col_to_be_rescaled_endswith
    are rescaled
    This function does not return anything but changes the dataframe feature_vector inplace provided as argument.
    """
    logger.info('Rescaling feature dataframes within the dictionary that end with {}'.format(col_to_be_rescaled_endswith))
    
    for feature_df in feature_dict:
        feature_vector = feature_dict[feature_df]
        feature_vector.replace(-1, np.nan, inplace=True)
        for column in feature_vector.columns.drop(col_to_be_dropped):
            if (column.endswith(col_to_be_rescaled_endswith)):
                logger.debug('Column that ends with diff found. Hence will be rescaled!')
                feature_vector[column] -= feature_vector[column].min()
                feature_vector[column] /= feature_vector[column].max()
                feature_vector[column] = 1 - feature_vector[column]
                feature_vector.rename(columns={column: column.replace(col_to_be_rescaled_endswith, '{}_sim'.format(col_to_be_rescaled_endswith))}, inplace=True)
        feature_vector.replace(np.nan,-1, inplace=True)
    logger.info('All features from the frature_vector dataframes within the dictionary that end with {} are now rescaled inplace'.format(col_to_be_rescaled_endswith))
#%%
            
def returnWeightedSumOfSimScores(feature_vector, columns_to_be_ignored=['l_id', 'r_id', 'label'], weight=None):
    """
    Returns the weighted sum of sim scores, from the sim scores (all need to be in range 0 to 1) of the provided
    dataframe.
    
    Paramters:
    feature_vector: Dataframe where the sim scores are stored
    columns_to_be_ignored: Columns that should not be summed up as they do not store a similarity score
        Default: ['l_id', 'r_id', 'label']
    weight: the method that shall be used for weighting the similarity scores. Either None or 'density'
        Default: None
        Source: The idea with using density in order to weight the features for the sim scores aggregation comes
                from Anna Primpeli's paper: "Unsupervised Bootstrapping of Active Learning for Entity Resolution", 2019
    """
    columns = [x for x in feature_vector.drop(columns=columns_to_be_ignored).columns.tolist() if 'sim' in x]
    rel_columns = feature_vector.drop(columns=columns_to_be_ignored)[columns]
    #logger.debug('Relevant columns are: {}'.format(rel_columns.columns))
    rel_columns = rel_columns.replace(-1,np.nan)
    if (weight is None):
        rel_columns_sum = rel_columns.sum(axis=1,skipna=True)   # calculate sum of values for each row (pot. corr.)
        rel_columns_mean = rel_columns_sum/len(rel_columns.columns) # calculate mean
        
    # idea with using density in order to weight the features for the sim scores aggregation
    # from Anna Primpeli's paper: "Unsupervised Bootstrapping of Active Learning for Entity Resolution", 2019
    if (weight == 'density'):
        column_weights = []
        for column in rel_columns:
            nan_values = rel_columns[column].isna().sum()   # get amount of missing values in column
            ratio = float(nan_values)/float(len(rel_columns[column]))
            column_weights.append(1.0-ratio)
            
        #logger.debug(column_weights)
        weighted_columns = rel_columns*column_weights
        #logger.debug(weighted_columns.iloc[0])
            
        rel_columns_sum = weighted_columns.sum(axis=1, skipna=True)
        rel_columns_mean = rel_columns_sum/len(rel_columns.columns)
      
    #rescale 
    sum_weighted_similarity = np.interp(rel_columns_mean, (rel_columns_mean.min(), rel_columns_mean.max()), (0, +1))
    #feature_vector['sum_weighted_sim'] = sum_weighted_similarity
    return sum_weighted_similarity    
    
#%%
            
def attachWeightedSumOfSimScoresToFeatureVector(feature_dict, resulting_column_name='sum_weighted_sim', columns_to_be_ignored=['l_id', 'r_id', 'label'], weight='density'):
    """
    Returns the weighted sum of sim scores, from the sim scores (all need to be in range 0 to 1) of the provided
    dataframe.
    
    Paramters:
    feature_dict: Dictionary where all the Dataframes with the feature vectors containing the sim scores are stored
    columns_to_be_ignored: Columns that should not be summed up as they do not store a similarity score
        Default: ['l_id', 'r_id', 'label']
    weight: the method that shall be used for weighting the similarity scores. Either None or 'density'
        Default: None
        Source: The idea with using density in order to weight the features for the sim scores aggregation comes
                from Anna Primpeli's paper: "Unsupervised Bootstrapping of Active Learning for Entity Resolution", 2019
                
    This function does not return anything but calls the function returnWeightedSumOfSimScores() on each feature vector df
    contained in the dictionary provided as argument and adds the resulted sum of the (weighted) sim scores to each feature
    vector dataframe as column with the name of the argument resulting_column_name
    """
    for feature_df in feature_dict:
        feature_dict[feature_df][resulting_column_name] = returnWeightedSumOfSimScores(feature_dict[feature_df], columns_to_be_ignored, weight)


#%%

def returnSortedDataFrame(feature_vector, label='label', ids=['l_id','r_id'], final_id='id', agg_sim='sum_weighted_sim'):
    """
    Sorts the sum of the (weighted) similarity scores and returns a dataframe containing label, ids, sim, and idx.
    
    Parameters:
    feature_vector: Dataframe where the data is stored
    label: String with the column name where the labels are stored
        Default: 'label'
    ids: When in the dataframe currently no ID for the pot. corr. are stored but rather two single ids for
         each datasource then the two column names can be handed in a list as this argument and the two IDs
         get combined to a single ID (which will be stored in the column with the name of the String from
         final_id argument). If the dataframe already has an ID that identifies pot. corr. an empty list
         can be handed to ids and the final_id argument needs to specify the correct name of the ID
        Default: ['l_id','r_id']
    final_id: Name of the single ID which is the combination of the IDs from the two datasources. If the
              dataframe already has this final_id and does not need to be created based on the two single 
              IDs then the ids argument needs to be set to an empty list and final_id needs to have the 
              name of the column where the ID is stored. Otherwise final_id is just the name of the column
              where the newly created ID will be stored.
    agg_sim: Name of the column where the sum of the (weighted) similarity scores is stored.
    """
    if(len(ids)==2):    # in the dataframe an ID for each source is stored, hence we combine them to one
        feature_vector[final_id] = feature_vector.apply(lambda row: '{}_{}'.format(row[ids[0]],row[ids[1]]),axis=1)
    
    sorted_data = list(zip(feature_vector[label], feature_vector[final_id], feature_vector[agg_sim], np.arange(feature_vector[final_id].size)))
    random.Random(1).shuffle(sorted_data)
    sorted_data.sort(key = lambda t: t[2])
    sorted_label,sorted_ids,sorted_sim,sorted_idx = zip(*sorted_data)
    df_sorted = pd.DataFrame({'label':sorted_label,'ids':sorted_ids,'sim':sorted_sim,'idx':sorted_idx})
    return df_sorted


#%%
    
def returnDictWithSortedDataFrames(feature_dict, label='label', ids=['l_id','r_id'], final_id='id', agg_sim='sum_weighted_sim'):
    """
    For each dataframe within the feature_dict argument this function sorts the sum of the (weighted)
    similarity scores and returns a dataframe containing label, ids, sim, and idx. All resulting dataframes
    are stored in a dictionary which is returned by this function!
    
    Parameters:
    feature_vector: Dataframe where the data is stored
    label: String with the column name where the labels are stored
        Default: 'label'
    ids: When in the dataframe currently no ID for the pot. corr. are stored but rather two single ids for
         each datasource then the two column names can be handed in a list as this argument and the two IDs
         get combined to a single ID (which will be stored in the column with the name of the String from
         final_id argument). If the dataframe already has an ID that identifies pot. corr. an empty list
         can be handed to ids and the final_id argument needs to specify the correct name of the ID
        Default: ['l_id','r_id']
    final_id: Name of the single ID which is the combination of the IDs from the two datasources. If the
              dataframe already has this final_id and does not need to be created based on the two single 
              IDs then the ids argument needs to be set to an empty list and final_id needs to have the 
              name of the column where the ID is stored. Otherwise final_id is just the name of the column
              where the newly created ID will be stored.
    agg_sim: Name of the column where the sum of the (weighted) similarity scores is stored.
    """
    logger.info('Start calculating the sorted dataframes for each feature dataframe within feature_dict')
    df_sorted_dict = {}
    for feature_df in feature_dict:
        df_sorted_dict.update({feature_df:returnSortedDataFrame(feature_dict[feature_df], label, ids, final_id, agg_sim)})
    logger.info('Finished calculating the sorted dataframes for each feature dataframe within feature_dict')
    return df_sorted_dict
    
#%%

#from Anna Primpeli's paper: "Unsupervised Bootstrapping of Active Learning for Entity Resolution", 2019
#code from https://dataplatform.cloud.ibm.com/analytics/notebooks/54d79c2a-f155-40ec-93ec-ed05b58afa39/view?access_token=6d8ec910cf2a1b3901c721fcb94638563cd646fe14400fecbb76cea6aaae2fb1 (accessed:12.09.2019)
def elbow_threshold(similarities, labels):
    """
    This code is taken from Anna Primpeli's code at https://github.com/aprimpeli/UnsupervisedBootAL/blob/master/code/similarityutils.py 
    and she has it from https://dataplatform.cloud.ibm.com/analytics/notebooks/54d79c2a-f155-40ec-93ec-ed05b58afa39/view?access_token=6d8ec910cf2a1b3901c721fcb94638563cd646fe14400fecbb76cea6aaae2fb1 (accessed:12.09.2019)
    The function returns the elbow_threshold
    """
    sim_list = list(similarities)
    nPoints = len(sim_list)
    allCoord = np.vstack((range(nPoints), sim_list)).T
    
    firstPoint = allCoord[0]
    # get vector between first and last point - this is the line
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    idxOfBestPoint = np.argmax(distToLine)
    

    #print "Knee of the curve is at index =",idxOfBestPoint
    #print "Knee value =", similarities[idxOfBestPoint]
       
    return similarities[idxOfBestPoint],idxOfBestPoint

#%%

def returnAggSimScoresPerAttribute(feature_vector, columns_to_be_ignored=['id', 'label', 'sum_weighted_sim', 'sum_sim']):
    feature_vector = feature_vector.copy()
    columns = feature_vector.drop(columns=columns_to_be_ignored).columns.tolist()
    columns = [x for x in columns if 'sim' in x]
    columns = sorted(columns)
    res = [list(i) for j, i in groupby(columns, lambda s: s.partition('_')[0])]
    rel_columns = feature_vector[columns]
    rel_columns = rel_columns.replace(-1,np.nan)
    final_columns = []
    for lst in res:
        col_sum_sim = '{}_sum_sim'.format(lst[0].partition('_')[0])
        final_columns.append(col_sum_sim)
        feature_vector[col_sum_sim] = rel_columns[lst].mean(axis=1,skipna=True)
    #feature_vector.replace(np.nan,-1)
    return feature_vector[columns_to_be_ignored+final_columns].replace(np.nan,-1)

#%%

def createCandSet(non_matches,true_matches):
    # get the bins of the true matches to get the amount of instances per bin later
    bins_true = pd.cut(true_matches['sim'],10,labels=False)
    # shuffle the non-matches datset and assign it to df
    df = non_matches.sample(frac=1, axis=0, random_state=42).reset_index(drop=True)
    # calculate the bins for df and attach new column 'bin' to it
    df['bin'] = pd.cut(df['sim'],10,labels=False)
    # create an empty DataFrame where the sampled non-matches get stored
    sample_df = pd.DataFrame()
    for i in range(10):
        k = bins_true.value_counts()[i]
        #print('i {} with amount true matches {}'.format(i,k))
        #print('i {} with amount non_matches {}'.format(i,df[df['bin']==i].shape[0]))
        if(k>=df[df['bin']==i].shape[0]):
            sample = df[df['bin']==i]
            sample_df = sample_df.append(sample,ignore_index=True)
            df.drop(sample.index,inplace=True)
        else:
            sample = df[df['bin']==i].sample(n=k)
            sample_df = sample_df.append(sample,ignore_index=True)
            df.drop(sample.index,inplace=True)

    k = true_matches.shape[0]-sample_df.shape[0]
    if(k>0):
        sample = df.sample(n=k,random_state=42)
        sample_df = sample_df.append(sample,ignore_index=True)
        df.drop(sample.index,inplace=True)
    return sample_df