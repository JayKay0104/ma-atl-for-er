# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 21:59:03 2020

@author: jonas
"""

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


#%%

def removeElements(lst, k, less_than_k=False): 
    counted = Counter(lst) 
    if(not less_than_k):  
        return [el for el in lst if counted[el] >= k]
    else:
        return [el for el in lst if counted[el] < k]


#%%
        
def returnLabeledFeatureVectorsForSampling(corr_dict, type_per_column, columns_to_be_dropped=[], identifier='', unique_identifier='', no_prefix=False):
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
        feature_dict.update({'feature_{}_{}'.format(corr.split('_')[1],corr.split('_')[2]):createLabeledFeatureVectorForSampling(corr_dict[corr], type_per_column, columns_to_be_dropped, identifier, unique_identifier,no_prefix)})
        logger.debug('For the pot. corr. of {} and {} features are created and stored in feature_{}_{}.'.format(corr.split('_')[1],corr.split('_')[2],corr.split('_')[1],corr.split('_')[2]))
    logger.info('\nFinished! All labeled feature vectors are created for all dataset combinations')
    return feature_dict


##%%
#
#def createLabeledFeatureVectorForSamplingFast(corr, type_per_column, corr_l_name='', corr_r_name='', columns_to_be_dropped=['_id','_sim_score'], ids={'id','isbn'}):
#    logger.info('Start Function')
#    if(type_per_column==0):
#        logger.error('No Type per Column Dictionary defined!')
#        return None
#    feature_vector = pd.DataFrame()
#    corr_columns = corr.columns.drop(columns_to_be_dropped)
#
#    # write ids and the label in the feature_vector dataframe
#    feature_vector['l_id'] = corr['l_{}_id'.format(corr_l_name)]
#    feature_vector['r_id'] = corr['r_{}_id'.format(corr_r_name)]
#    feature_vector['l_{}_isbn'.format(corr_l_name)] = corr.apply(lambda row: str(row['l_{}_isbn'.format(corr_l_name)]),axis=1)
#    feature_vector['r_{}_isbn'.format(corr_r_name)] = corr.apply(lambda row: str(row['r_{}_isbn'.format(corr_r_name)]),axis=1)
#    feature_vector['label'] = corr.apply(lambda row: 1 if row['l_{}_isbn'.format(corr_l_name)]==row['r_{}_isbn'.format(corr_r_name)] else 0,axis=1)
#
#    # retrieve the header in order to identify the common attributes
#    corr_header = [corr_columns.str.split('_')[i][2:] for i in range(len(corr_columns))]
#    corr_header = ['_'.join(x) for x in corr_header if len(corr_header)>1]
#    common_attributes = list(set(removeElements(corr_header,2))- ids)
#
#    # iterate through common_attributes in order to calculate the sim scores using the similarity measures
#    # dedicated for the specific types
#    for attr in common_attributes:
#        logger.info(corr.columns[corr.columns.str.endswith('{}_{}'.format(corr_l_name,attr))][0])
#        ser1 = corr[corr.columns[corr.columns.str.endswith('{}_{}'.format(corr_l_name,attr))]]
#        logger.info(corr.columns[corr.columns.str.endswith('{}_{}'.format(corr_r_name,attr))][0])
#        ser2 = corr[corr.columns[corr.columns.str.endswith('{}_{}'.format(corr_r_name,attr))]]
#        feature_vector = pd.concat([feature_vector,calcSimSeriesForSampling(ser1,ser2,attr,type_per_column[attr])], axis=1)
#    return feature_vector

def createLabeledFeatureVectorForSampling(corr, type_per_column, columns_to_be_dropped=[], identifier='', unique_identifier='', no_prefix=False):
    
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
    if('_id' in columns_to_be_dropped):
        columns_to_be_dropped.remove('_id')
    cols = [c for c in corr.columns for substring in columns_to_be_dropped if substring in c.lower()]
    if('_id' in corr.columns):
        cols.append('_id')
    corr_columns = corr.columns.drop(cols)
    
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
        feature_vector = pd.concat([feature_vector,calcSimSeriesForSampling(ser1,ser2,attr,type_per_column[attr])], axis=1)
    logger.info('\nFinished! Labeled Feature Vectors created for {} and {}'.format(corr_l_name,corr_r_name))
    return feature_vector

#%%

def calcSimSeriesForSampling(series1, series2, attrName, data_type):
    """
    Calculate the implemented similarity measures (in module sim_measures) that are selected for the data_type
    between the two pd.Series
    """
    df = pd.concat([series1,series2], axis=1, sort=False,ignore_index=True)
    if(data_type=='long_str'):
        ### PREPROCESSING ###
        # remove english stopwords and apply porter stemmer for long strings
        # use function remove_sw(string,output_list=False,stem=True) from sim_measures for that
        # it also applies re.sub('[^A-Za-z0-9\s\t\n]+', '', str(string).lower().strip()).split() to the string
        df.iloc[:,0] = df.iloc[:,0].apply(lambda s: sim.remove_sw(s,output_list=False,stem=True))
        df.iloc[:,1] = df.iloc[:,1].apply(lambda s: sim.remove_sw(s,output_list=False,stem=True))
        # here we do not calculate the cosine similarity because we still have to many 
        # pot. correspondences and the memory consumption is too high
        # when we have sampled the candidate set we will calculate the cosince similarity
        # but in order to create the candidate set we will calculate all other similarity measures
        ### COSINE ###
        # create a mask that is true if at least one of the two colums contain NaN value or empty string
        # mask = ((df.iloc[:,0]=='nan') | (df.iloc[:,1]=='nan') | (df.iloc[:,0]=='') | (df.iloc[:,1]==''))
        #ser_cosine_tfidf_sim = pd.Series(sim.calculate_cosine_tfidf(df))
        #ser_cosine_tfidf_sim.where(~mask,other=-1,inplace=True)
        #logger.debug('Cosine similarity with TfIdf-Weighting measured for {} as it is a long_str'.format(attrName))
        ### LEVENSHTEIN ###
        ser_lev_sim = df.apply(lambda row: sim.lev_sim(row[0],row[1]),axis=1)
        logger.debug('Levenshtein Similarity measured for {}'.format(attrName))
        ### JACCARD 3-GRAM ###
        ser_jac_q3_sim = df.apply(lambda row: sim.jac_q3_sim(row[0],row[1]),axis=1)
        logger.debug('Jaccard Similarity with trigram tokenization measured for {}'.format(attrName))
        ### JACCARD WS/AN ###
        ser_jac_an_sim = df.apply(lambda row: sim.jac_an_sim(row[0],row[1]),axis=1)
        # as we apply re.sub('[^A-Za-z0-9\s\t\n]+', '', str(string).lower().strip()).split() whitespace tok. is is basically like alphanumeric tok.
        logger.debug('Jaccard Similarity with whitespace tokenization measured for {}'.format(attrName))
        ### RELAXED JACCARD ###
        ser_rel_jac_sim = df.apply(lambda row: sim.relaxed_jaccard_sim(row[0],row[1]),axis=1)
        logger.debug('Relaxed Jaccard Similarity with whitespace tokenization and Inner Levenshtein measured for {}'.format(attrName))
        ### CONTAINMENT ###
        ser_containment_sim = df.apply(lambda row: sim.containment_sim(row[0],row[1]),axis=1)
        logger.debug('Relaxed Jaccard Similarity with whitespace tokenization and Inner Levenshtein measured for {}'.format(attrName))
        ### EXACT SIM ###
        ser_exact_sim = df.apply(lambda row: sim.exact_sim(row[0],row[1]),axis=1)
        logger.debug('Exact Similarity measured for {}'.format(attrName))
        #commented out all_missing as could be misleading (-1 already assigned for missing 
        #values no valuable info if both missing as everything -1 then)
        ### ALL MISSING ###
#        ser_all_missing = df.apply(lambda row: sim.all_missing(row[0],row[1]),axis=1)
#        logger.debug('All missing measured for {}'.format(attrName))
        # finished calculating similarity measures for this attribute
        logger.debug('Similarity measures calculated for {}'.format(attrName))
        # create DataFrame with results
        #if(ser_cosine_tfidf_sim.shape[0] != ser_lev_sim.shape[0]):
        #    raise ValueError('Something wrong with cosine tfidf calculation. Wrong shape!')
        df = pd.concat([ser_lev_sim,ser_jac_q3_sim,ser_jac_an_sim,ser_rel_jac_sim,ser_containment_sim,ser_exact_sim],axis=1,sort=False,ignore_index=True)
        # rename. did it in two steps as it is 
        df.rename({0:attrName+'_lev_sim', 1:attrName+'_jac_q3_sim', 2:attrName+'_jac_an_sim', 3:attrName+'_rel_jac_an_sim', 4:attrName+'_containment_sim', 5:attrName+'_exact_sim'},axis=1,inplace=True)
        return df
    elif(data_type=='str'):
        # for string no stop words are removed and also no stemming is applied
        # we only ensure the strings are lower cased and stripped and remove signs other than letters, digits, and whitespaces
        # we do it here for all values so we do not need to perform it everytime when calculating the sim measures
        df.iloc[:,0] = df.iloc[:,0].apply(lambda s: re.sub('[^A-Za-z0-9\s]+', '', str(s).lower().strip()))
        df.iloc[:,1] = df.iloc[:,1].apply(lambda s: re.sub('[^A-Za-z0-9\s]+', '', str(s).lower().strip()))
        ### LEVENSHTEIN ###
        ser_lev_sim = df.apply(lambda row: sim.lev_sim(row[0],row[1]),axis=1)
        logger.debug('Levenshtein Similarity measured for {}'.format(attrName))
        ### JACCARD 3-GRAM ###
        ser_jac_q3_sim = df.apply(lambda row: sim.jac_q3_sim(row[0],row[1]),axis=1)
        logger.debug('Jaccard Similarity with trigram tokenization measured for {}'.format(attrName))
        ### JACCARD WS/AN ###
        ser_jac_an_sim = df.apply(lambda row: sim.jac_an_sim(row[0],row[1]),axis=1)
        # as we apply re.sub('[^A-Za-z0-9\s\t\n]+', '', str(string).lower().strip()).split() whitespace tok. is is basically like alphanumeric tok.
        logger.debug('Jaccard Similarity with whitespace tokenization measured for {}'.format(attrName))
        ### RELAXED JACCARD ###
        ser_rel_jac_sim = df.apply(lambda row: sim.relaxed_jaccard_sim(row[0],row[1]),axis=1)
        logger.debug('Relaxed Jaccard Similarity with whitespace tokenization and Inner Levenshtein measured for {}'.format(attrName))
        ### CONTAINMENT ###
        ser_containment_sim = df.apply(lambda row: sim.containment_sim(row[0],row[1]),axis=1)
        logger.debug('Relaxed Jaccard Similarity with whitespace tokenization and Inner Levenshtein measured for {}'.format(attrName))
        ### EXACT SIM ###
        ser_exact_sim = df.apply(lambda row: sim.exact_sim(row[0],row[1]),axis=1)
        logger.debug('Exact Similarity measured for {}'.format(attrName))
        #commented out all_missing as could be misleading (-1 already assigned for missing 
        #values no valuable info if both missing as everything -1 then)
        ### ALL MISSING ###
#        ser_all_missing = df.apply(lambda row: sim.all_missing(row[0],row[1]),axis=1)
#        logger.debug('All missing measured for {}'.format(attrName))
        # finished calculating similarity measures for this attribute
        logger.debug('Similarity measures calculated for {}'.format(attrName))
        # create DataFrame with results
        df = pd.concat([ser_lev_sim,ser_jac_q3_sim,ser_jac_an_sim,ser_rel_jac_sim,ser_containment_sim,ser_exact_sim],axis=1,sort=False,ignore_index=True)
        # rename. did it in two steps as it is 
        df.rename({0:attrName+'_lev_sim', 1:attrName+'_jac_q3_sim', 2:attrName+'_jac_an_sim', 3:attrName+'_rel_jac_an_sim', 4:attrName+'_containment_sim', 5:attrName+'_exact_sim'},axis=1,inplace=True)
        return df
    elif(data_type=='num'):
        ser_num_abs_diff = df.apply(lambda row: sim.num_abs_diff(row[0],row[1]),axis=1)
        logger.debug('Absolute Difference measured for {}'.format(attrName))
#        ser_num_sim = df.apply(lambda row: sim.num_sim(row[0],row[1]),axis=1)
#        logger.debug('Numeric Similarity measured for {}'.format(attrName))
        # all_missing commented out because it does not provide valuable info
#        ser_all_missing = df.apply(lambda row: sim.all_missing(row[0],row[1]),axis=1)
#        logger.debug('All missing measured for {}'.format(attrName))
        logger.debug('Similarity measures calculated for {}'.format(attrName))
        df = pd.DataFrame({attrName+'_num_abs_diff':ser_num_abs_diff})
#        df = pd.concat([ser_num_abs_diff,ser_num_sim],axis=1,sort=False,ignore_index=True)
#        df.rename({0:attrName+'_num_abs_diff', 1:attrName+'_num_sim'},axis=1,inplace=True)
        return df
    elif(data_type=='date'):
        df[0] = df[0].apply(lambda s: sim.alignDTFormat(s))
        df[1] = df[1].apply(lambda s: sim.alignDTFormat(s))
#        ser_days_sim = df.apply(lambda row: sim.days_sim(row[0],row[1]),axis=1)
#        logger.debug('Days Similarity measured for {}'.format(attrName))
#        ser_years_sim = df.apply(lambda row: sim.years_sim(row[0],row[1]),axis=1)
#        logger.debug('Years Similarity measured for {}'.format(attrName))
        ser_years_diff = df.apply(lambda row: sim.years_diff(row[0],row[1]),axis=1)
        logger.debug('Years Difference measured for {}'.format(attrName))
        ser_months_diff = df.apply(lambda row: sim.months_diff(row[0],row[1]),axis=1)
        logger.debug('Months Difference measured for {}'.format(attrName))
        ser_days_diff = df.apply(lambda row: sim.days_diff(row[0],row[1]),axis=1)
        logger.debug('Days Difference measured for {}'.format(attrName))
        # all_missing commented out because it does not provide valuable info
#        ser_all_missing = df.apply(lambda row: sim.all_missing(row[0],row[1]),axis=1)
#        logger.debug('All missing measured for {}'.format(attrName))
        logger.debug('Similarity measures calculated for {}'.format(attrName))
        df = pd.concat([ser_days_diff,ser_months_diff,ser_years_diff],axis=1,sort=False,ignore_index=True)
        df.rename({0:attrName+'_days_diff', 1:attrName+'_months_diff', 2:attrName+'_years_diff'},axis=1,inplace=True)
#        df = pd.concat([ser_days_sim,ser_years_sim,ser_days_diff],axis=1,sort=False,ignore_index=True)
#        df.rename({0:attrName+'_days_sim', 1:attrName+'_years_sim', 2:attrName+'_days_diff'},axis=1,inplace=True)
        return df
    # custom is for binary attributes (maybe not relevant)
    elif(data_type=='custom'):
        ser_exact_sim = df.apply(lambda row: sim.exact_sim(row[0],row[1]),axis=1)
        logger.debug('Exact Similarity measured for {}'.format(attrName))
        # all_missing commented out because it does not provide valuable info
#        ser_all_missing = df.apply(lambda row: sim.all_missing(row[0],row[1]),axis=1)
#        logger.debug('All missing measured for {}'.format(attrName))
        logger.debug('Similarity measures calculated for {}'.format(attrName))
        df = pd.DataFrame({attrName+'_exact_sim':ser_exact_sim})
#        df = pd.concat([ser_exact_sim],axis=1,sort=False,ignore_index=True)
#        df.rename({0:attrName+'_exact_sim'},axis=1,inplace=True)
        return df
    else:
        logger.error('No Similarity Measure for {} DataType defined.'.format(data_type))
        return None
    
#%%
    
def calcWeightedSumOfSimScores(feature_vector):
    
    columns = [x for x in feature_vector.columns if 'sim' in x]
    if('sum_weighted_sim' in columns):
        columns.remove('sum_weighted_sim')
        logger.debug('sum_weighted_sim was in columns and hence got removed. Only the non-aggregated similarity scores containing sim as substring\
              are consider')
    #print(columns)
    rel_columns = feature_vector[columns]
    rel_columns = rel_columns.replace(-1,np.nan)
    
    column_weights = []
    for column in rel_columns:
        nan_values = rel_columns[column].isna().sum()   # get amount of missing values in column
        ratio = float(nan_values)/float(len(rel_columns[column]))
        column_weights.append(1.0-ratio)
    
    #print(column_weights)
    #logger.debug(column_weights)
    weighted_columns = rel_columns*column_weights
    #logger.debug(weighted_columns.iloc[0])
    
    rel_columns_sum = weighted_columns.sum(axis=1, skipna=True)
    rel_columns_mean = rel_columns_sum/len(rel_columns.columns)
          
    #rescale
    sum_weighted_similarity = np.interp(rel_columns_mean, (rel_columns_mean.min(), rel_columns_mean.max()), (0, +1))
    
    return sum_weighted_similarity


#%%
    
def sampleFromNegCorr(non_matches,true_matches,agg_sim='sim'):
    # get the bins of the true matches to get the amount of instances per bin later
    bins_true = pd.cut(true_matches[agg_sim],10,labels=False)
    # shuffle the non-matches datset and assign it to df
    df = non_matches.sample(frac=1, axis=0, random_state=42).reset_index(drop=True)
    # calculate the bins for df and attach new column 'bin' to it
    df['bin'] = pd.cut(df[agg_sim],10,labels=False)
    # create an empty DataFrame where the sampled non-matches get stored
    sample_df = pd.DataFrame()
    for i in range(10):
        try:
            k = bins_true.value_counts()[i]
        except KeyError:
            k = 0
        #print('i {} with amount true matches {}'.format(i,k))
        #print('i {} with amount non_matches {}'.format(i,df[df['bin']==i].shape[0]))
        if(k==0): continue
        elif(k>=df[df['bin']==i].shape[0]):
            sample = df[df['bin']==i]
            sample_df = sample_df.append(sample,ignore_index=True)
            df.drop(sample.index,inplace=True)
        else:
            sample = df[df['bin']==i].sample(n=k)
            sample_df = sample_df.append(sample,ignore_index=True)
            df.drop(sample.index,inplace=True)

    k = true_matches.shape[0]-sample_df.shape[0]
    if(k>0):
        try:
            sample = df.sample(n=k,random_state=42)
        except:
            sample = df.sample(frac=1,random_state=42)
        sample_df = sample_df.append(sample,ignore_index=True)
        df.drop(sample.index,inplace=True)
    return sample_df

#%%
    
def checkForSufficientNegCorr(feature_dict,k=500):
    
    to_be_deleted = []
    for key in feature_dict:
        if(k > feature_dict[key][feature_dict[key]['label']==0].shape[0]):
            print('Less than {} negative correspondences for {}. Hence will delete it!'.format(k,key))
            to_be_deleted.append(key)
    for key in to_be_deleted:
        del feature_dict[key]

#%%



def createCanddSet(feature_dict, matches_feature_dict, df_dict, type_dict_column):
    
    agg_sim = 'sum_weighted_scores'
    final_id = 'ids'
    ids=['l_id','r_id']
    candset_dict = {}
    for key in feature_dict:
        # key is in format: feature_ds1_ds2
        ds1_name = key.split('_')[1]
        ds2_name = key.split('_')[2]
        feature_dict[key][agg_sim] = calcWeightedSumOfSimScores(feature_dict[key])
        feature_dict[key][final_id] = feature_dict[key].apply(lambda row: '{}_{}'.format(row[ids[0]],row[ids[1]]),axis=1)
        matches_feature_dict[key][agg_sim] = calcWeightedSumOfSimScores(matches_feature_dict[key])
        matches_feature_dict[key][final_id] = matches_feature_dict[key].apply(lambda row: '{}_{}'.format(row[ids[0]],row[ids[1]]),axis=1)
        
        sorted_df_non_matches = feature_dict[key][feature_dict[key]['label']==0][[final_id,agg_sim]].sort_values(by=agg_sim,axis=0,ascending=True)
        sorted_df_matches = matches_feature_dict[key][[final_id,agg_sim]].sort_values(by=agg_sim,axis=0,ascending=True)
        
        ids_sampled_neg_corr = sampleFromNegCorr(sorted_df_non_matches,sorted_df_matches,agg_sim)
        neg_sample = pd.merge(ids_sampled_neg_corr[final_id],feature_dict[key],on=final_id)
        final_sample = neg_sample.append(matches_feature_dict[key],ignore_index=True,verify_integrity=True)
        
        if('long_str' in type_dict_column.values()):
            ds1_long_string_attr = ['{}_{}'.format(ds1_name,key) for key in type_dict_column if type_dict_column[key]=='long_str']
            ds2_long_string_attr = ['{}_{}'.format(ds2_name,key) for key in type_dict_column if type_dict_column[key]=='long_str']
            ds1_ls_attr = pd.merge(ids_sampled_neg_corr[['l_id','r_id']], df_dict[ds1_name],left_on='l_id',right_on='{}_{}'.format(ds1_name,'id'))
            final_ls_attr = pd.merge(ds1_ls_attr, df_dict[ds2_name],left_on='r_id',right_on='{}_{}'.format(ds2_name,'id'))
            
            attr_zipped = zip(ds1_long_string_attr,ds2_long_string_attr)
            for att_combo in attr_zipped:
                final_ls_attr[att_combo[0]] = final_ls_attr[att_combo[0]].apply(lambda s: sim.remove_sw(s,output_list=False,stem=True))
                final_ls_attr[att_combo[1]] = final_ls_attr[att_combo[1]].apply(lambda s: sim.remove_sw(s,output_list=False,stem=True))
                # create a mask that is true if at least one of the two colums contain NaN value or empty string
                mask = ((final_ls_attr[att_combo[0]]=='nan') | (final_ls_attr[att_combo[1]]=='nan') | (final_ls_attr[att_combo[0]]=='') | (final_ls_attr[att_combo[1]]==''))
                ser_cosine_tfidf_sim = pd.Series(sim.calculate_cosine_tfidf(final_ls_attr[att_combo[0],att_combo[1]]),name=att_combo[0].split('_')[1:]+'_cosine_tfidf_sim')
                ser_cosine_tfidf_sim.where(~mask,other=-1,inplace=True)
                final_sample = pd.concat(final_sample,ser_cosine_tfidf_sim)
        
        print(final_sample.columns)
        print(key)
        candset_key = key.replace('feature_','')
        print(candset_key)
        candset_dict.update({candset_key:final_sample})
        
    return candset_dict

#%%
    
def getCandsetsWithOrgAttribute(candset_dict,dataset_dict):
    candsets_attr = {}
    for cand in candset_dict:
        temp = candset_dict[cand][['ids','label']].copy()
        temp['l_id'] = temp['ids'].apply(lambda s: '_'.join(s.split('_')[:2]))
        temp['r_id'] = temp['ids'].apply(lambda s: '_'.join(s.split('_')[2:]))
        l_key = cand.split('_')[0]
        r_key = cand.split('_')[1]
        l_temp_attr = pd.merge(temp,dataset_dict[l_key],left_on='l_id',right_on=l_key+'_id')
        r_l_temp_attr = pd.merge(l_temp_attr,dataset_dict[r_key],left_on='r_id',right_on=r_key+'_id')
        cols = list(r_l_temp_attr.columns.drop(['ids','label','l_id','r_id']))
        cols.sort(key=(lambda x: '_'.join(x.split('_')[1:])))
        final_cols = ['ids','label']+cols
        candsets_attr.update({cand:r_l_temp_attr[final_cols]})
    return candsets_attr

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
        feature_vector = pd.concat([feature_vector,sim.calcSimSeries(ser1,ser2,attr,type_per_column[attr])], axis=1)
    logger.info('\nFinished! Labeled Feature Vectors created for {} and {}'.format(corr_l_name,corr_r_name))
    return feature_vector