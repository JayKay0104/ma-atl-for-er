# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:09:47 2020

@author: jonas
"""

#import logging
#LOGGER = logging.getLogger(__name__)
import numpy as np
import al_learningalgos as la
import al_committees as com
from libact.base.dataset import Dataset
from libact.labelers import IdealLabeler
#from libact.query_strategies import UncertaintySampling #, QueryByCommittee, RandomSampling
from sklearn.model_selection import train_test_split#, cross_val_predict
from sklearn.metrics import f1_score, precision_recall_fscore_support
import time

#import scipy.stats as st
#from scipy.spatial.distance import cdist
#import sklearn as sk
#from sklearn.svm import LinearSVC
#from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, \
#    RidgeClassifier, RidgeClassifierCV

#from cvxopt import matrix, solvers
import la_ext as le
import da_weighting as da
#import itertools
import sys
sys.path.append('../')
import support_utils as sup

#%%  

def active_weighted_transfer_learning(candsets,candsets_train,candsets_test,source_name,target_name,feature,estimator_name,
                                      query_strategy,quota,weighting=None,disagreement='vote',n=5):
    """
    query_strategy: 
        Possible strategies are: 
            Baselines: 'uncertainty', 'random'
            Heterogeneous Committees: 'lr_lscv_rf_dt', 'lr_lsvc_dt_xgb', 'lr_lsvc_dt_gpc', 'lr_svc_dt_xgb_rf' ,'lr_svc_rf_dt', 'lr_svc_dt_gpc', 'lr_svc_dt_xgb',
            Homogeneous Committees: 'homogeneous_committee' (it will then take the specified committee for the model used)
    """
    
    training_accuracy_scores = []
    training_f1_scores = []
    test_accuracy_scores =[]
    test_f1_scores = []
    runtimes = []
    #n_labeled = 0
    
    X_source = candsets[source_name][feature].to_numpy()
    y_source = candsets[source_name]['label'].to_numpy()
    X_target = candsets[target_name][feature].to_numpy()
    #y_target = candsets[target_name]['label'].to_numpy()
    # the source instances are all labeled and used as initial training set
    # hence, n_labeled == the size of of source instances
    n_labeled = y_source.shape[0]  
    
    # check if domain adaptation is desired
    if(weighting is None):
        print('No Unsupervised Domain Adaptation performed')
        sample_weight = None
    else:
        print('Unsupervised Domain Adaptation: Calculate sample_weight for the source instances using'.format(weighting))
        sample_weight = da.getSampleWeightsOfDomainAdaptation(X_source, X_target, weighting)
    
    # doing the train_test_split with fixed size (33% testing) and fixed random_state (42) as well as stratified
    # in order to be able to compare results and ensure reproducibility
    X_target_train = candsets_train[target_name][feature]
    y_target_train = candsets_train[target_name]['label']
    
    X_target_test = candsets_test[target_name][feature]
    y_target_test = candsets_train[target_name]['label']
    
    print('Train_test_split: random_state = 42, stratified ; LR solver: liblinear')
    # create libact DataSet Object containting the validation set
    test_ds = Dataset(X=X_target_test,y=y_target_test)
    
    print('Starting ATL Experiments (WITH transfer!) source {} and target {}'.format(source_name,target_name))
    for i in range(n):
        print('{}. Run of {}'.format(i+1,n))
        
        train_ds, fully_labeled_trn_ds = initializeAWTLPool(X_source, y_source, X_target_train, 
                                                           y_target_train, n_labeled, sample_weight)
        
        # if quota -1 it means it is not a fixed amount
        # create the quota which is the amount of all instances 
        # in the training pool minus the amount of already labeled ones
        if(quota == -1): 
            quota = train_ds.len_unlabeled()
        
        # cerate the IdealLabeler with the full labeled training set
        lbr = IdealLabeler(fully_labeled_trn_ds)


        model = la.getLearningModel(estimator_name)
        
        qs = com.getQueryStrategy(query_strategy, train_ds, disagreement, estimator_name)
            
        train_acc, train_f1, test_acc, test_f1, model_, runt = run_atl(train_ds,test_ds,lbr,model,qs,quota,n_labeled)

        training_accuracy_scores.append(train_acc)
        training_f1_scores.append(train_f1)
        test_accuracy_scores.append(test_acc)
        test_f1_scores.append(test_f1)
        runtimes.append(runt)
    
    runt = np.mean(runtimes)
    
    key = '{}_{}'.format(source_name,target_name)
    if(weighting is None):
        # append weighting strategy to query_strategy name to be able to distinguish 
        d = {key:{estimator_name:{query_strategy:{'no_weighting':{'quota':quota,'n_runs':n,'n_init_labeled':n_labeled,
                                                                  'model_params':model_.get_params(),'avg_runtime':runt,
                                                                  'training_accuracy_scores':training_accuracy_scores,
                                                                  'training_f1_scores':training_f1_scores,
                                                                  'test_accuracy_scores':test_accuracy_scores,
                                                                  'test_f1_scores':test_f1_scores}}}}}
    else:
        d = {key:{estimator_name:{query_strategy:{weighting:{'quota':quota,'n_runs':n,'n_init_labeled':n_labeled,
                                                             'model_params':model_.get_params(),'avg_runtime':runt,
                                                             'training_accuracy_scores':training_accuracy_scores,
                                                             'training_f1_scores':training_f1_scores,
                                                             'test_accuracy_scores':test_accuracy_scores,
                                                             'test_f1_scores':test_f1_scores,
                                                             'sample_weights':sample_weight}}}}}
    return d

#%%
    
def initializeAWTLPool(X_source, y_source, X_target_train, y_target_train, n_labeled, sample_weights):
    
    # create training set by appending the train sample from target to all instances from source
    X_train = np.vstack([X_source,X_target_train])
    y_train = np.append(y_source,y_target_train)
    
    # train_ds is the whole X_train but only the n_labeled (all source) instances are labeled
    train_ds = le.AWTLDataset(X=X_train,y=np.concatenate([y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]), da_weights=sample_weights)

    # here we have the fully labeled training set 
    fully_labeled_trn_ds = Dataset(X=X_train,y=y_train)

    return train_ds, fully_labeled_trn_ds

#%%
def run_atl(train_ds,test_ds,lbr,model,qs,quota,n_init_labeled):
    
    start_time = time.time()
    
    E_in, E_in_f1, E_out, E_out_f1 = [], [], [], []
    #E_out_P, E_out_R = [], []
    
    labels = []
    
    l = quota
    sup.printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
       
    for i in range(quota):
        # QBC
        ask_id = qs.make_query()
        # Labeler for QBC on train_ds and Random on train_ds2
        lb = lbr.label(train_ds.data[ask_id][0])         
        # QBC
        train_ds.update(ask_id, lb)
        labels.append(lb)
                
        model.train(train_ds)
        
        X_train_current,y_train_current,sample_weight = train_ds.format_sklearn() 
        E_in = np.append(E_in, model.score(train_ds))
        E_in_f1 = np.append(E_in_f1, f1_score(y_train_current, model.predict(X_train_current), pos_label=1, average='binary', sample_weight=sample_weight))
        
        X_test,y_test = test_ds.format_sklearn()
        E_out = np.append(E_out, model.score(test_ds))
        prec, recall, f1score, support = precision_recall_fscore_support(y_test, model.predict(X_test), average='binary')
        
        E_out_f1 = np.append(E_out_f1, f1score)
        #E_out_P = np.append(E_out_P, prec)
        #E_out_R = np.append(E_out_R, recall)
                    
        # Update Progress Bar
        sup.printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
        
    
    runt = time.time() - start_time
    print('Runtime: {:.2f} seconds'.format(runt))
    
    return E_in, E_in_f1, E_out, E_out_f1, model, runt


#%%
    
def atl_different_weighted_settings_combos(candsets,candsets_train,candsets_test,combinations,weighting,dense_features_dict,estimator_names,query_strategies,
                                           quota,disagreement='vote',n=5,switch_roles=False):
    
    d = {}
    
    if(switch_roles):
        for combo in combinations:
            source_name = combo[0]
            target_name = combo[1]
            dense_feature_key = '_'.join(sorted(set(source_name.split('_')+target_name.split('_'))))
            feature = dense_features_dict[dense_feature_key]
            # key not needed
            #key = '{}_{}'.format(source_name,target_name)
        
            print('Start with ATL using different settings for source {} and target {}'.format(source_name,target_name))
        
            temp = atl_different_weighted_settings_single(candsets,candsets_train,candsets_test,source_name,target_name,feature,estimator_names,query_strategies,
                                                          quota,weighting,disagreement,n)
            #if(key in d):
            #    d[key].update(temp[key])
            #else:
            d.update(temp)
            # switch roles: now the previous target serves as source
            source_name = combo[1]
            target_name = combo[0]
            #dense feature key is the same
            #dense_feature_key = '_'.join(sorted(set(source_name.split('_')+target_name.split('_'))))
            #feature = dense_features_dict[dense_feature_key]
            #key = '{}_{}'.format(source_name,target_name)
        
            print('Start with ATL using different settings for source {} and target {}'.format(source_name,target_name))
        
            temp = atl_different_weighted_settings_single(candsets,candsets_train,candsets_test,source_name,target_name,feature,estimator_names,query_strategies,
                                                          quota,weighting,disagreement,n)
            #if(key in d):
            #    d[key].update(temp[key])
            #else:
            d.update(temp)
    else:
        for combo in combinations:
            source_name = combo[0]
            target_name = combo[1]
            dense_feature_key = '_'.join(sorted(set(source_name.split('_')+target_name.split('_'))))
            feature = dense_features_dict[dense_feature_key]
            #key = '{}_{}'.format(source_name,target_name)
            
            print('Start with ATL using different settings for source {} and target {}'.format(source_name,target_name))
            
            temp = atl_different_weighted_settings_single(candsets,candsets_train,candsets_test,source_name,target_name,feature,estimator_names,query_strategies,
                                                          quota,weighting,disagreement,n)
            #if(key in d):
            #    d[key].update(temp[key])
            #else:
            d.update(temp)
    
    return d

#%%
    
def atl_different_weighted_settings_single(candsets,candsets_train,candsets_test,source_name,target_name,dense_features,estimators,query_strategies,
                                           quota,weighting=None,disagreement='vote',n=5):
    """
    Run Active Transfer Learning with different settings as specified in estimators and query_strategies!
    """
    d = {}
    #dense_feature_key = '_'.join(sorted(set(source_name.split('_')+target_name.split('_'))))
    #feature = dense_features_dict[dense_feature_key]
    key = '{}_{}'.format(source_name,target_name)
    for est in estimators:
        print('Start with Estimator: {}'.format(est))
        for qs in query_strategies:
            print('Start with Query Strategy: {}'.format(qs))
            for weight in weighting:
                print('Start with Weighting Strategy: {}'.format(weight))
                temp = active_weighted_transfer_learning(candsets,candsets_train,candsets_test,source_name,target_name,dense_features,est,qs,
                                                         quota,weight,disagreement,n)
                if(key in d):
                    if(est in d[key]):
                        if(qs in d[key][est]):
                            d[key][est][qs].update(temp[key][est][qs])
                        else:
                            d[key][est].update(temp[key][est])
                    else:
                        d[key].update(temp[key])
                else:
                    d.update(temp)
    
    return d
