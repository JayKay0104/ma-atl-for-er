# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 08:51:30 2020

@author: jonas
"""
#%%
#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_color_codes()
#import random
import numpy.matlib
import itertools
#import math
#import json
#import copy
#import glob
#import re
#from IPython.display import display
#import logging

from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
# for calulating the DomainRelatedness
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
import da_weighting as da
import warnings
import sys
sys.path.append('../')
import support_utils as sup
#from sklearn.model_selection import cross_val_predict

#%% Transfer Learning Experiments
    
def getF1SourceTargetFixedAvg(source,target,target_training,target_test,estimator,rel_columns,da_weighting,target_train_size,n=10):
    """
    Perform Transfer Learning Experiment with naive transfer of matching rule trained on training set from source
    and evaluated on the test set from the target using the specified 
    estimator and the specified features (rel_columns). Besides that, from the target training a random sample of
    size (target_train_size) is sampled and used to train the same estimator and tested on the target test set.
    The results are averaged over n random samples.
    
    Source: all labeled source instances
    Target: all target instances (not necessarily labeled)
    Target_training: training set of the target
    Target_test: test set / validation set of the target
    Estimator: sklearn Estimator that shall be used
    Rel_columns: List of features that shall be used for training
    Da_weighting: (optional) Domain Adaptation Technique to use. Either 'no_weighting', 'nn', or 'lr_predict_proba'
    Target_train_size: Amount of target features that shall be used for training a classifier
                       for comparison
    n: specifies on how many random samples the experiments shall be performed and averaged.
     Default: 10
    """
    X_source = source[rel_columns]
    y_source = source['label']
    #amount_of_training_source_instances = X_source.shape[0]
    f1_target = []
    
    X_target = target[rel_columns]
#    y_target = target_training['label'].copy()
    
    #create same train test split as was created for the target benchmarks
    #X_target_train = target_training[rel_columns].drop(columns='label')
    X_target_test = target_test[rel_columns]
    #y_target_train = target_training['label']
    y_target_test = target_test['label']
    
    if(da_weighting is None):  
        # train a classifier using all source instances
        clf_source = estimator.fit(X_source, y_source)
        predicted_source_on_target = clf_source.predict(X_target_test)
        f1_source = f1_score(y_target_test,predicted_source_on_target)
    else:
        sample_weight = da.getSampleWeightsOfDomainAdaptation(X_source, X_target, da_weighting)
        clf_source = estimator.fit(X_source, y_source, sample_weight)
        predicted_source_on_target = clf_source.predict(X_target_test)
        f1_source = f1_score(y_target_test,predicted_source_on_target)
    
    # append y_target_train back to X_target_train
    # Note the train test split (stratified,size=0.33,random_state=42),
    # was only done in order to retrieve the same test set as was used 
    # and will be used for all experiments!
    #training = X_target_train.copy()
    #training['label'] = y_target_train.copy()
    training = target_training[rel_columns].copy()
    training['label'] = target_training['label'].copy()
    
    for i in range(n):
        # create stratified sample of the target training instances
        # here no random_state is set for sample as we want to randomly sample different target instances 
        # and then average the results of n runs
        train_sample_target = training.groupby('label').apply(lambda x: x.sample(n=int(target_train_size/2))).droplevel(level=0)
        train_target = train_sample_target[rel_columns].copy()
        y_train_target = train_sample_target['label'].copy()
        # train a the estimator using the just created training set of size target_train_size for the target
        clf_target = estimator.fit(train_target, y_train_target)
        predicted_target = clf_target.predict(X_target_test)
        f1_target.append(f1_score(y_target_test,predicted_target))

    # average the results of the target and output them. the source results do not need to be averaged
    # as they are calculated outside the loop as nothing changes there.
    return (round(f1_source,3),round((sum(f1_target)/n),3))

#%%
 
def performSingleTLExp(source,target,source_target_name,target_train,target_test,estimators,features,da_weighting=None,n=10,switch_roles=True):
    """
    Backup function to perform single experiment.
    """
    x_instances = [10,14,20,24,28,32,38,44,50,60,70,80,90,100,120,140,160,180,200,300,500]
    
    d = {}
    
    l = len(estimators.keys())
    sup.printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    if(switch_roles):
        target_source_name = '{}_{}'.format(source_target_name.split('_')[1],source_target_name.split('_')[0])
        for i, clf in enumerate(estimators):
            a_transfer_results = []
            a_target_results = []
            b_transfer_results = []
            b_target_results = []
            
            for x in x_instances:
                # all features (also non-dense ones)
                # transfer from a to b 
                res = getF1SourceTargetFixedAvg(source,target,target_train,target_test,estimators[clf],features,da_weighting,x,n)
                a_transfer_results.append(res[0])
                a_target_results.append(res[1])
                # transfer from b to a
                res = getF1SourceTargetFixedAvg(target,source,target_train,target_test,estimators[clf],features,da_weighting,x,n)
                b_transfer_results.append(res[0])
                b_target_results.append(res[1])
            
            a_transfer_res = sum(a_transfer_results)/len(x_instances)
            a_target_max = max(a_target_results)
            b_transfer_res = sum(b_transfer_results)/len(x_instances)
            b_target_max = max(b_target_results)
            
            try:
                idx = np.argwhere(np.diff(np.sign(np.array(a_transfer_results) - np.array(a_target_results)))).flatten()[0]
                a_x_target_instances = x_instances[idx]
            except Exception:
                a_x_target_instances = np.nan
            try:
                idx = np.argwhere(np.diff(np.sign(np.array(b_transfer_results) - np.array(b_target_results)))).flatten()[0]
                b_x_target_instances = x_instances[idx]
            except Exception:
                b_x_target_instances = np.nan
            if(source_target_name not in d):
                if(da_weighting is None):
                    d.update({source_target_name:{'no_weighting':{clf:{'transfer_avg_result':a_transfer_res,
                                                                       'target_max_result':a_target_max,
                                                                       'x_target_exceed':a_x_target_instances,
                                                                       'y_transfer_results':a_transfer_results,
                                                                       'y_target_results':a_target_results,
                                                                       'n_runs':n}}}})
                else:
                    d.update({source_target_name:{da_weighting:{clf:{'transfer_avg_result':a_transfer_res,
                                                                       'target_max_result':a_target_max,
                                                                       'x_target_exceed':a_x_target_instances,
                                                                       'y_transfer_results':a_transfer_results,
                                                                       'y_target_results':a_target_results,
                                                                       'n_runs':n}}}})
            else:
                if(da_weighting is None):
                    d[source_target_name]['no_weighting'].update({clf:{'transfer_avg_result':a_transfer_res,
                                                                       'target_max_result':a_target_max,
                                                                       'x_target_exceed':a_x_target_instances,
                                                                       'y_transfer_results':a_transfer_results,
                                                                       'y_target_results':a_target_results,
                                                                       'n_runs':n}})
                else:
                    d[source_target_name][da_weighting].update({clf:{'transfer_avg_result':a_transfer_res,
                                                                       'target_max_result':a_target_max,
                                                                       'x_target_exceed':a_x_target_instances,
                                                                       'y_transfer_results':a_transfer_results,
                                                                       'y_target_results':a_target_results,
                                                                       'n_runs':n}})
            if(target_source_name not in d):
                if(da_weighting is None):
                    d.update({target_source_name:{'no_weighting':{clf:{'transfer_avg_result':b_transfer_res,
                                                                       'target_max_result':b_target_max,
                                                                       'x_target_exceed':b_x_target_instances,
                                                                       'y_transfer_results':b_transfer_results,
                                                                       'y_target_results':b_target_results,
                                                                       'n_runs':n}}}})
                else:
                    d.update({target_source_name:{da_weighting:{clf:{'transfer_avg_result':b_transfer_res,
                                                                       'target_max_result':b_target_max,
                                                                       'x_target_exceed':b_x_target_instances,
                                                                       'y_transfer_results':b_transfer_results,
                                                                       'y_target_results':b_target_results,
                                                                       'n_runs':n}}}})
            else:
                if(da_weighting is None):
                    d[target_source_name]['no_weighting'].update({clf:{'transfer_avg_result':b_transfer_res,
                                                                       'target_max_result':b_target_max,
                                                                       'x_target_exceed':b_x_target_instances,
                                                                       'y_transfer_results':b_transfer_results,
                                                                       'y_target_results':b_target_results,
                                                                       'n_runs':n}})
                else:
                    d[target_source_name][da_weighting].update({clf:{'transfer_avg_result':b_transfer_res,
                                                                       'target_max_result':b_target_max,
                                                                       'x_target_exceed':b_x_target_instances,
                                                                       'y_transfer_results':b_transfer_results,
                                                                       'y_target_results':b_target_results,
                                                                       'n_runs':n}})
            # Update Progress Bar
            sup.printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    else:
        for i, clf in enumerate(estimators):
            a_transfer_results = []
            a_target_results = []

            for x in x_instances:
                # perform transfer learning experiments with source as source and target as target
                res = getF1SourceTargetFixedAvg(source,target,target_train,target_test,estimators[clf],features,da_weighting,x,n)
                a_transfer_results.append(res[0])
                a_target_results.append(res[1])

            a_transfer_res = sum(a_transfer_results)/len(x_instances)
            a_target_max = max(a_target_results)

            try:
                idx = np.argwhere(np.diff(np.sign(np.array(a_transfer_results) - np.array(a_target_results)))).flatten()[0]
                a_x_target_instances = x_instances[idx]
            except Exception:
                a_x_target_instances = np.nan
            if(source_target_name not in d):
                if(da_weighting is None):
                    d.update({source_target_name:{'no_weighting':{clf:{'transfer_avg_result':a_transfer_res,
                                                                       'target_max_result':a_target_max,
                                                                       'x_target_exceed':a_x_target_instances,
                                                                       'y_transfer_results':a_transfer_results,
                                                                       'y_target_results':a_target_results,
                                                                       'n_runs':n}}}})
                else:
                    d.update({source_target_name:{da_weighting:{clf:{'transfer_avg_result':a_transfer_res,
                                                                       'target_max_result':a_target_max,
                                                                       'x_target_exceed':a_x_target_instances,
                                                                       'y_transfer_results':a_transfer_results,
                                                                       'y_target_results':a_target_results,
                                                                       'n_runs':n}}}})
            else:
                if(da_weighting is None):
                    d[source_target_name]['no_weighting'].update({clf:{'transfer_avg_result':a_transfer_res,
                                                                       'target_max_result':a_target_max,
                                                                       'x_target_exceed':a_x_target_instances,
                                                                       'y_transfer_results':a_transfer_results,
                                                                       'y_target_results':a_target_results,
                                                                       'n_runs':n}})
                else:
                    d[source_target_name][da_weighting].update({clf:{'transfer_avg_result':a_transfer_res,
                                                                       'target_max_result':a_target_max,
                                                                       'x_target_exceed':a_x_target_instances,
                                                                       'y_transfer_results':a_transfer_results,
                                                                       'y_target_results':a_target_results,
                                                                       'n_runs':n}})
            # Update Progress Bar
            sup.printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    return d

#%%
    
def performTLFromDict(candsets,candsets_train,candsets_test,estimators,all_features,dense_features_dict=None,da_weighting=None,n=10):
    """
    ***IMPORTANT*** -> Very time consuming. Hence, results should be saved to hard disk with saveTLResultsToJSON() function, so that the experiments not necessarily need to be repeated.
    Perform Transfer Learning Experiment for each combination of source-target pairs in candsets dictionary
    with naive transfer of matching rule trained on source instances and evaluated on all target 
    instances - target_train_size for all estimators specified in estimators and for all_features as well 
    as only dense features per combination. The results are averaged over n runs.
    
    @parameters
    candsets: Dictionary containing all candidate sets (pot. correspondences)
    candsets_train: Dictionary containing all training sets (pot. correspondences)
    candsets_test: Dictionary containing all test sets (pot. correspondences)
    estimators: Dicitionary with sklearn Estimators that shall be used for the TL Experiment. Dictionary should be of form {'logreg':LogisticRegression(),'logregcv':LogisticRegressionCV(),...}
    All_features: List of with all features
    Dense_features_dict: Dictionary with list of onle the dense feature for each combination. Exp: When source ban_half and target wor_half then the dense features across
    ban, half and wor need to be saved in a list which is the value of for dense_features_dict['ban_half_wor']. It is important that the key is compound of ban, half, wor in alphabetical order seperated by '_'
    n: specifies on how many random samples the experiments shall be performed and averaged. 100 will explode computing time!!! Default: 10
    """
    x_instances = [10,14,20,24,28,32,38,44,50,60,70,80,90,100,120,140,160,180,200,300,500]

    d = {}
    
    combinations = []
    for combo in itertools.combinations(candsets, 2):
        if((combo[0].split('_')[0] in combo[1].split('_')) or (combo[0].split('_')[1] in combo[1].split('_'))):
            combinations.append(combo)
    #print(combinations)
    
    l = len(combinations)
    sup.printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    for i, combo in enumerate(combinations):
        for clf in estimators:
            a_transfer_results = []
            a_target_results = []
            b_transfer_results = []
            b_target_results = []
            a_transfer_results_dense = []
            a_target_results_dense = []
            b_transfer_results_dense = []
            b_target_results_dense = []
            for x in x_instances:
                # all features (also non-dense ones)
                # transfer from a to b 
                res = getF1SourceTargetFixedAvg(candsets[combo[0]],candsets[combo[1]],candsets_train[combo[1]],candsets_test[combo[1]],estimators[clf],all_features,da_weighting,x,n)
                a_transfer_results.append(res[0])
                a_target_results.append(res[1])
                # transfer from b to a
                res = getF1SourceTargetFixedAvg(candsets[combo[1]],candsets[combo[0]],candsets_train[combo[0]],candsets_test[combo[0]],estimators[clf],all_features,da_weighting,x,n)
                b_transfer_results.append(res[0])
                b_target_results.append(res[1])
                
                if(dense_features_dict is not None):
                    # only the dense features
                    dense_feature_key = '_'.join(sorted(set(combo[0].split('_')+combo[1].split('_'))))
                    # transfer from a to b 
                    res = getF1SourceTargetFixedAvg(candsets[combo[0]],
                                                        candsets[combo[1]],
                                                        candsets_train[combo[1]],
                                                        candsets_test[combo[1]],
                                                        estimators[clf],
                                                        dense_features_dict[dense_feature_key],da_weighting,x,n)
                    a_transfer_results_dense.append(res[0])
                    a_target_results_dense.append(res[1])
                    # transfer from b to a
                    res = getF1SourceTargetFixedAvg(candsets[combo[1]],
                                                        candsets[combo[0]],
                                                        candsets_train[combo[0]],
                                                        candsets_test[combo[0]],
                                                        estimators[clf],
                                                        dense_features_dict[dense_feature_key],da_weighting,x,n)
                    b_transfer_results_dense.append(res[0])
                    b_target_results_dense.append(res[1])
            
            # all features
            a_transfer_res = sum(a_transfer_results)/len(x_instances)
            a_target_max = max(a_target_results)
            b_transfer_res = sum(b_transfer_results)/len(x_instances)
            b_target_max = max(b_target_results)
            try:
                idx = np.argwhere(np.diff(np.sign(np.array(a_transfer_results) - np.array(a_target_results)))).flatten()[0]
                a_x_target_instances = x_instances[idx]
            except Exception:
                a_x_target_instances = np.nan
            try:
                idx = np.argwhere(np.diff(np.sign(np.array(b_transfer_results) - np.array(b_target_results)))).flatten()[0]
                b_x_target_instances = x_instances[idx]
            except Exception:
                b_x_target_instances = np.nan
            if(combo not in d):
                if(da_weighting is None):
                    d.update({combo:{'all':{'no_weighting':{clf:{'transfer_avg_result':a_transfer_res,
                                                                       'target_max_result':a_target_max,
                                                                       'x_target_exceed':a_x_target_instances,
                                                                       'y_transfer_results':a_transfer_results,
                                                                       'y_target_results':a_target_results,
                                                                       'n_runs':n}}}}})
                else:
                    d.update({combo:{'all':{da_weighting:{clf:{'transfer_avg_result':a_transfer_res,
                                                                       'target_max_result':a_target_max,
                                                                       'x_target_exceed':a_x_target_instances,
                                                                       'y_transfer_results':a_transfer_results,
                                                                       'y_target_results':a_target_results,
                                                                       'n_runs':n}}}}})
            else:
                if(da_weighting is None):
                    d[combo]['all']['no_weighting'].update({clf:{'transfer_avg_result':a_transfer_res,
                                                                       'target_max_result':a_target_max,
                                                                       'x_target_exceed':a_x_target_instances,
                                                                       'y_transfer_results':a_transfer_results,
                                                                       'y_target_results':a_target_results,
                                                                       'n_runs':n}})
                else:
                    d[combo]['all'][da_weighting].update({clf:{'transfer_avg_result':a_transfer_res,
                                                                 'target_max_result':a_target_max,
                                                                       'x_target_exceed':a_x_target_instances,
                                                                       'y_transfer_results':a_transfer_results,
                                                                       'y_target_results':a_target_results,
                                                                       'n_runs':n}})                
            if(combo[::-1] not in d):
                if(da_weighting is None):
                    d.update({combo[::-1]:{'all':{'no_weighting':{clf:{'transfer_avg_result':b_transfer_res,
                                                                       'target_max_result':b_target_max,
                                                                       'x_target_exceed':b_x_target_instances,
                                                                       'y_transfer_results':b_transfer_results,
                                                                       'y_target_results':b_target_results,
                                                                       'n_runs':n}}}}})
                else:
                    d.update({combo[::-1]:{'all':{da_weighting:{clf:{'transfer_avg_result':b_transfer_res,
                                                                       'target_max_result':b_target_max,
                                                                       'x_target_exceed':b_x_target_instances,
                                                                       'y_transfer_results':b_transfer_results,
                                                                       'y_target_results':b_target_results,
                                                                       'n_runs':n}}}}})
            else:
                if(da_weighting is None):
                    d[combo[::-1]]['all']['no_weighting'].update({clf:{'transfer_avg_result':b_transfer_res,
                                                                       'target_max_result':b_target_max,
                                                                       'x_target_exceed':b_x_target_instances,
                                                                       'y_transfer_results':b_transfer_results,
                                                                       'y_target_results':b_target_results,
                                                                       'n_runs':n}})
                else:
                    d[combo[::-1]]['all'][da_weighting].update({clf:{'transfer_avg_result':b_transfer_res,
                                                                       'target_max_result':b_target_max,
                                                                       'x_target_exceed':b_x_target_instances,
                                                                       'y_transfer_results':b_transfer_results,
                                                                       'y_target_results':b_target_results,
                                                                       'n_runs':n}})
            if(dense_features_dict is not None):        
                # dense features
                a_transfer_res_dense = sum(a_transfer_results_dense)/len(x_instances)
                a_target_max_dense = max(a_target_results_dense)
                b_transfer_res_dense = sum(b_transfer_results_dense)/len(x_instances)
                b_target_max_dense = max(b_target_results_dense)
                try:
                    idx = np.argwhere(np.diff(np.sign(np.array(a_transfer_results_dense) - np.array(a_target_results_dense)))).flatten()[0]
                    a_x_target_instances_dense = x_instances[idx]
                except Exception:
                    a_x_target_instances_dense = np.nan
                try:
                    idx = np.argwhere(np.diff(np.sign(np.array(b_transfer_results_dense) - np.array(b_target_results_dense)))).flatten()[0]
                    b_x_target_instances_dense = x_instances[idx]
                except Exception:
                    b_x_target_instances_dense = np.nan
                if('dense' not in d[combo]):
                    if(da_weighting is None):
                        d[combo].update({'dense':{'no_weighting':{clf:{'transfer_avg_result':a_transfer_res_dense,
                                                   'target_max_result':a_target_max_dense,
                                                   'x_target_exceed':a_x_target_instances_dense,
                                                   'y_transfer_results':a_transfer_results_dense,
                                                   'y_target_results':a_target_results_dense,
                                                   'n_runs':n}}}})
                    else:
                        d[combo].update({'dense':{da_weighting:{clf:{'transfer_avg_result':a_transfer_res_dense,
                                                   'target_max_result':a_target_max_dense,
                                                   'x_target_exceed':a_x_target_instances_dense,
                                                   'y_transfer_results':a_transfer_results_dense,
                                                   'y_target_results':a_target_results_dense,
                                                   'n_runs':n}}}})
                else:
                    if(da_weighting is None):
                        d[combo]['dense']['no_weighting'].update({clf:{'transfer_avg_result':a_transfer_res_dense,
                                                   'target_max_result':a_target_max_dense,
                                                   'x_target_exceed':a_x_target_instances_dense,
                                                   'y_transfer_results':a_transfer_results_dense,
                                                   'y_target_results':a_target_results_dense,
                                                   'n_runs':n}})
                    else:
                        d[combo]['dense'][da_weighting].update({clf:{'transfer_avg_result':a_transfer_res_dense,
                                                   'target_max_result':a_target_max_dense,
                                                   'x_target_exceed':a_x_target_instances_dense,
                                                   'y_transfer_results':a_transfer_results_dense,
                                                   'y_target_results':a_target_results_dense,
                                                   'n_runs':n}})                
                if('dense' not in d[combo[::-1]]):
                    if(da_weighting is None):
                        d[combo[::-1]].update({'dense':{'no_weighting':{clf:{'transfer_avg_result':b_transfer_res_dense,
                                                         'target_max_result':b_target_max_dense,
                                                         'x_target_exceed':b_x_target_instances_dense,
                                                         'y_transfer_results':b_transfer_results_dense,
                                                         'y_target_results':b_target_results_dense,
                                                         'n_runs':n}}}})
                    else:
                        d[combo[::-1]].update({'dense':{da_weighting:{clf:{'transfer_avg_result':b_transfer_res_dense,
                                                         'target_max_result':b_target_max_dense,
                                                         'x_target_exceed':b_x_target_instances_dense,
                                                         'y_transfer_results':b_transfer_results_dense,
                                                         'y_target_results':b_target_results_dense,
                                                         'n_runs':n}}}})
                else:
                    if(da_weighting is None):
                        d[combo[::-1]]['dense']['no_weighting'].update({clf:{'transfer_avg_result':b_transfer_res_dense,
                                                         'target_max_result':b_target_max_dense,
                                                         'x_target_exceed':b_x_target_instances_dense,
                                                         'y_transfer_results':b_transfer_results_dense,
                                                         'y_target_results':b_target_results_dense,
                                                         'n_runs':n}})
                    else:
                        d[combo[::-1]]['dense'][da_weighting].update({clf:{'transfer_avg_result':b_transfer_res_dense,
                                                         'target_max_result':b_target_max_dense,
                                                         'x_target_exceed':b_x_target_instances_dense,
                                                         'y_transfer_results':b_transfer_results_dense,
                                                         'y_target_results':b_target_results_dense,
                                                         'n_runs':n}})
        # Update Progress Bar
        sup.printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    return d

#%%
def splittingDF(df, stratified=True):
    if(stratified):
        df_target = df.copy()
        df_source = df.groupby('label').apply(lambda x: x.sample(frac=0.5, random_state=1)).droplevel(level=0)
        df_target.drop(index=df_source.index,inplace=True) 
    else:
        df_target = df.copy()
        df_source = df.sample(frac=0.5, random_state=1)
        df_target.drop(index=df_source.index,inplace=True) 
    return df_source,df_target


#%%

def calcDomainRelatednessCV(source, target, relevant_columns, cv=5, metric='phi'):
    """
    Calculate the domain relatedness. This function adds a new column 'source', which indicates whether an instance
    is coming from the source or from the target dataset (used as label then) and concatenates source and target instances
    with each other. A LogisticRegressionCV estimator is then used to train a model which predicts from where a certain
    instance is coming from. Results of cross validation with scoring function either f1 (range (0,1)) or phi-coefficient 
    (range (-1,1)) is indicating whether the two domains are related or not. A bad score below 0.5 for f1 and below 0.2 
    for phi indicates that the two domains are rather related with each other.
    
    This approach is similar to https://blog.bigml.com/2014/01/03/simple-machine-learning-to-detect-covariate-shift/
    """
    source = source[relevant_columns].copy()
    target = target[relevant_columns].copy()
    
    # add new column 'source' to source and target dataset
    source['source'] = source.iloc[:,0].apply(lambda x: 0)
    target['source'] = target.iloc[:,0].apply(lambda x: 1)
    # concatenate source and target
    train = source.append(target,ignore_index=True,verify_integrity=True)
    # store the label 'source' in new object
    train_y = train['source'].copy()
    # remove the label 'source' from the features
    train.drop(columns='source',inplace=True)
    # create a LogisticRegressionCV with cv=5 and solver='liblinear'
    clf = LogisticRegressionCV(cv=5, solver='liblinear',class_weight='balanced')
    if(metric=='f1'):
        # output the mean of the cross_validated f1-scores
        return np.mean(cross_val_score(clf, train, train_y, cv=cv, scoring='f1'))
    else:
        # there seems to be a "bug" in matthews_corrcoef https://github.com/scikit-learn/scikit-learn/issues/1937
        # it is only a warning which gets raised, hence I suppress it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mat_corr = make_scorer(matthews_corrcoef)
            res = np.mean(cross_val_score(clf, train, train_y, cv=cv, scoring=mat_corr))
        # output the mean of the cross_validated phi-scores
        return res
    
def calcDomainRelatednessCVinDict(candsets, all_features, dense_features_dict, cv=5, metric='phi'):
    d = {}
    
    combinations = []
    for combo in itertools.combinations(candsets, 2):
        if((combo[0].split('_')[0] in combo[1].split('_')) or (combo[0].split('_')[1] in combo[1].split('_'))):
            combinations.append(combo)
    #print(combinations)
    
    l = len(combinations)
    sup.printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    for i,combo in enumerate(combinations):
        d.update({combo:{'all':calcDomainRelatednessCV(candsets[combo[0]],candsets[combo[1]],all_features,cv,metric)}})
        # only the dense features
        dense_feature_key = '_'.join(sorted(set(combo[0].split('_')+combo[1].split('_'))))
        d[combo].update({'dense':calcDomainRelatednessCV(candsets[combo[0]],candsets[combo[1]],dense_features_dict[dense_feature_key],cv,metric)})
    
        # Update Progress Bar
        sup.printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
        
    return d