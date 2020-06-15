# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:09:47 2020

@author: jonas
"""

import logging
LOGGER = logging.getLogger(__name__)
import numpy as np
import numpy.matlib
import al_learningalgos as la
import al_committees as com
from libact.base.dataset import Dataset
from libact.labelers import IdealLabeler
from libact.query_strategies import UncertaintySampling #, RandomSampling, QueryByCommittee
#from sklearn.model_selection import train_test_split #, cross_val_predict
from sklearn.metrics import f1_score, precision_recall_fscore_support
import time
#import itertools
#import da_weighting as da
import copy
from collections import Counter

#import scipy.stats as st
#from scipy.spatial.distance import cdist
#import sklearn as sk
#from sklearn.svm import LinearSVC
#from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, \
#    RidgeClassifier, RidgeClassifierCV

#from cvxopt import matrix, solvers
import la_ext as le

import sys
sys.path.append('../')
import support_utils as sup
import data_explore_utils as dex
        

#%%  

def active_unsuper_bootstrapped_learning(candsets,candsets_train,candsets_test,target_name,feature,estimator_name,
                                         query_strategy,quota,warm_start,reweight='score_based',disagreement='vote',n=5):
    """
    query_strategy: 
        Possible strategies are: 
            Baselines: 'uncertainty', 'random'
            Heterogeneous Committees: 'lr_lrcv_rf_dt', 'lr_lsvc_dt_gpc'
            Homogeneous Committees: 'homogeneous_committee' (it will then take the specified committee for the model used)
    """
    
    #training_accuracy_scores = []
    training_f1_scores = []
    #test_accuracy_scores =[]
    test_f1_scores = []
    #runtimes = []
    pool_correctness = []
    #n_labeled = 0
    
    X_target_train = candsets_train[target_name][feature]
    y_target_train = candsets_train[target_name]['label']
    X_target_test = candsets_test[target_name][feature]
    y_target_test = candsets_test[target_name]['label']
    # the source instances are all labeled and used as initial training set
    # hence, n_labeled == the size of of source instances
    n_labeled = 2
    
    # create libact DataSet Object containting the validation set
    test_ds = Dataset(X=X_target_test,y=y_target_test)
    
    print('Starting AL Experiments with Unsupervised Bootstrapping for target {}'.format(target_name))
    for i in range(n):
        print('{}. Run of {}'.format(i+1,n))
        
        X_train = X_target_train.copy()
        y_train = y_target_train.copy()
        train_ds, fully_labeled_trn_ds, labeled_weight = initializeUnsupervisedPool(X_train,y_train,warm_start,reweight)
        
        # if quota -1 it means it is not a fixed amount
        # create the quota which is the amount of all instances 
        # in the training pool minus the amount of already labeled ones
        if(quota == -1): 
            quota = train_ds.len_unlabeled()
        
        # cerate the IdealLabeler with the full labeled training set
        lbr = IdealLabeler(fully_labeled_trn_ds)


        model = getLearningModel(estimator_name, warm_start)
        
        qs = getQueryStrategy(query_strategy, train_ds, disagreement, estimator_name)

            
        #train_acc, train_f1, test_acc, test_f1, model_, runt, share_of_corrected_labels = run_weighted_atl(train_ds,test_ds,lbr,
        #                                                                                                   model,qs,quota,n_labeled)
        train_f1, test_f1, unsupervised_correctness_, model_, pool_ids_ = run_noisy_al(train_ds, test_ds, lbr, model, qs, 
                                                                                       quota,reweight, labeled_weight, warm_start=False)

        #training_accuracy_scores.append(train_acc)
        training_f1_scores.append(train_f1)
        #test_accuracy_scores.append(test_acc)
        test_f1_scores.append(test_f1)
        #runtimes.append(runt)
        pool_correctness.append(unsupervised_correctness_)
    
    #runt = np.mean(runtimes)
    
    key = target_name
    d = {key:{estimator_name:{query_strategy:{'quota':quota,'n_runs':n,'n_init_labeled':n_labeled,
                                              'pool_correctness':pool_correctness,
                                              'model_params':model_.get_params(),
                                              'training_f1_scores':training_f1_scores,
                                              'test_f1_scores':test_f1_scores}}}}
    return d

#%%
    
def initializeUnsupervisedPool(X_target_train, y_target_train, warm_start, reweight):
    print('Initialize Dataset Object')
    
    agg_sim = 'sum_weighted_scores'
    X_target_train[agg_sim] = dex.calcWeightedSumOfSimScores(X_target_train)
    X_target_train['label'] = y_target_train.copy()
    X_target_train.sort_values(by=agg_sim,axis=0,ascending=True,inplace=True)
    #sorted_dataset = list(zip(y_target_train, X_target_train[ids], X_target_train[agg_sim], np.arange(X_target_train[ids].shape[0])))
    #sorted_dataset = zip(self.labels, self.ids, sum_weighted_similarity, np.arange(self.ids.size))
    #random.Random(1).shuffle(sorted_dataset)
    #sorted_dataset.sort(key = lambda t: t[2])
    #sorted_dataset = map(lambda x:x[2],sorted_dataset)
    
    #getElbowThrElementsFromSortedDataset(sorted_dataset)
    
    elbow_th, index = dex.elbow_threshold(np.array(X_target_train[agg_sim]))
    print('Elbow_Threshold: {} and Index: {}'.format(elbow_th,index))
    X_target_train['noisy_label'] = X_target_train[agg_sim].apply(lambda x: 1 if (x>=elbow_th) else 0)
    
    pool_labels = np.asarray([None]*X_target_train['label'].shape[0])
       
    pool_unsupervised_labels = np.asarray(X_target_train['noisy_label'])
    #pool_unsupervised_labels = X_target_train['noisy_label']
    #pool_unsupervised_weights= np.asarray([0.0]*X_target_train['label'].shape[0])

    if reweight == 'score_based':
        bootstrapping_weights = []
    
        for w in X_target_train[agg_sim]:
            #normalize weights so that they are between 0-1
            nom = abs(w-elbow_th) 
            if w<elbow_th: den = elbow_th - np.min(X_target_train[agg_sim])
            else: den = np.max(X_target_train[agg_sim]) - elbow_th
            score = nom/den
            if score < 0.0: score=0.0
            
            weight = round(score,1)
            bootstrapping_weights.append(weight)
                
    
        pool_unsupervised_weights = np.asarray(bootstrapping_weights)
    
        #add most certain positive and most certain negative in the pool just for the initialization of QBC strategy: to fix
        combined = list(zip(pool_unsupervised_labels,pool_unsupervised_weights, np.arange(0,len(pool_unsupervised_weights))))
        pos  = filter(lambda x: x[0]==1, combined)
        neg = filter(lambda x: x[0]==0, combined)
        pos_weight = list(map(lambda x: x[1], pos))
        neg_weight = list(map(lambda x: x[1], neg))
        sure_positive =filter(lambda x,pos_w =max(pos_weight): (x[0]==1 and x[1]==pos_w) ,combined)
        sure_negative =filter(lambda x,neg_w =max(neg_weight): (x[0]==0 and x[1]==neg_w) ,combined)
    
        sure_positive_indices = list(map(lambda x:x[2],sure_positive))
        sure_negative_indices = list(map(lambda x:x[2],sure_negative))
        
        #take only one
        sure_negative_indices_sample = sure_negative_indices[0]
        sure_positive_indices_sample = sure_positive_indices[0]
        
        pool_labels[sure_positive_indices_sample]=1
        pool_labels[sure_negative_indices_sample]=0
        pool_unsupervised_labels[sure_positive_indices] = 1 
        pool_unsupervised_labels[sure_negative_indices_sample] = 0
        pool_unsupervised_weights[sure_positive_indices_sample] = max(pos_weight)
        pool_unsupervised_weights[sure_negative_indices_sample] = max(neg_weight)
       
        #fix weight for labeled data points
        weights_sum = np.sum(pool_unsupervised_weights)
        
        if warm_start:
            labeled_weight = 1
        else: 
            labeled_weight = weights_sum/10
            
        ## new block above
        train_ds =  le.UnsupervisedPoolDataset(X_target_train.drop(columns=['noisy_label','label',agg_sim]), pool_labels, pool_unsupervised_labels, pool_unsupervised_weights, reweight)

    else: 
        train_ds =  le.UnsupervisedPoolDataset(X_target_train.drop(columns=['noisy_label','label',agg_sim]), pool_labels, pool_unsupervised_labels, reweight)
    

#    return pool,labeled_weight
    # here we have the fully labeled training set 
    fully_labeled_trn_ds = Dataset(X=X_target_train.drop(columns=['noisy_label','label',agg_sim]),y=X_target_train['label'])
    
    return train_ds, fully_labeled_trn_ds, labeled_weight

#%%
#
#def calcWeightedSumOfSimScores(feature_vector):
#    
#    columns = feature_vector.columns[feature_vector.columns.str.endswith('sim')].tolist()
#    if('sum_weighted_sim' in columns):
#        columns.remove('sum_weighted_sim')
#        print('sum_weighted_sim was in columns and hence got removed. Only the non-aggregated\
#              similarity scores containing sim as substring are consider')
#    print(columns)
#    rel_columns = feature_vector[columns]
#    rel_columns = rel_columns.replace(-1,np.nan)
#    
#    column_weights = []
#    for column in rel_columns:
#        nan_values = rel_columns[column].isna().sum()   # get amount of missing values in column
#        ratio = float(nan_values)/float(len(rel_columns[column]))
#        column_weights.append(1.0-ratio)
#    
#    print(column_weights)
#    #logger.debug(column_weights)
#    weighted_columns = rel_columns*column_weights
#    #logger.debug(weighted_columns.iloc[0])
#    
#    rel_columns_sum = weighted_columns.sum(axis=1, skipna=True)
#    rel_columns_mean = rel_columns_sum/len(rel_columns.columns)
#          
#    #rescale
#    sum_weighted_similarity = np.interp(rel_columns_mean, (rel_columns_mean.min(), rel_columns_mean.max()), (0, +1))
#    
#    return sum_weighted_similarity

#%%
    
#def elbow_threshold(sorted_similarities):
#    """
#    This code is taken from Anna Primpeli's code at https://github.com/aprimpeli/UnsupervisedBootAL/blob/master/code/similarityutils.py 
#    and she has it from https://dataplatform.cloud.ibm.com/analytics/notebooks/54d79c2a-f155-40ec-93ec-ed05b58afa39/view?access_token=6d8ec910cf2a1b3901c721fcb94638563cd646fe14400fecbb76cea6aaae2fb1 (accessed:12.09.2019)
#    The function returns the elbow_threshold
#    """
#    sim_list = list(sorted_similarities)
#    nPoints = len(sim_list)
#    allCoord = np.vstack((range(nPoints), sim_list)).T
#    
#    firstPoint = allCoord[0]
#    # get vector between first and last point - this is the line
#    lineVec = allCoord[-1] - allCoord[0]
#    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
#    vecFromFirst = allCoord - firstPoint
#    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
#    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
#    vecToLine = vecFromFirst - vecFromFirstParallel
#    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
#    idxOfBestPoint = np.argmax(distToLine)
#       
#    return sorted_similarities[idxOfBestPoint],idxOfBestPoint
#%%
    
def getQueryStrategy(query_strategy, train_ds, disagreement, estimator_name=None):
    print('Initialize Query Strategy')
    if query_strategy == 'uncertainty':
        qs = UncertaintySampling(train_ds, method='lc', model=la.LogisticRegression_())
    elif query_strategy == 'random':
        qs = le.RandomSampling_(train_ds)
    elif query_strategy == 'lr_lsvc_rf_dt':
        if disagreement == 'kl_divergence':
            raise ValueError('when using kl_divergence lsvc cannot be in the committee as linearSVC does not provide predict_proba().\
                             Use svc instead or change disagreement to vote!')
        qs = le.QueryByCommitteeUBoot_(train_ds, models=[la.RandomForest_(),la.DecisionTree_(),
                                                   la.LogisticRegression_(solver='liblinear',max_iter=1000),
                                                   la.LinearSVC_()],
                    disagreement=disagreement)
    # committee with probabilistic models (SVC with prob=True used here instead of LinearSVC)
    elif query_strategy == 'lr_svc_rf_dt':
        qs = le.QueryByCommitteeUBoot_(train_ds, models=[la.RandomForest_(),la.DecisionTree_(),
                                                la.LogisticRegression_(solver='liblinear',max_iter=1000),
                                                la.SVC_(kernel='linear',probability=True)],
                    disagreement=disagreement)
    elif query_strategy == 'lr_svc_dt_xgb':
        qs = le.QueryByCommitteeUBoot_(train_ds, models=[la.LogisticRegression_(solver='liblinear',max_iter=1000),
                                                la.SVC_(kernel='linear',probability=True),la.DecisionTree_(),
                                                la.XGBClassifier_(objective="binary:logistic")],
            disagreement=disagreement)
    # committee of five 
    elif query_strategy == 'lr_svc_dt_xgb_rf':
        qs = le.QueryByCommitteeUBoot_(train_ds, models=[la.LogisticRegression_(solver='liblinear',max_iter=1000),
                                                la.SVC_(kernel='linear',probability=True),la.DecisionTree_(),
                                                la.XGBClassifier_(objective="binary:logistic"),la.RandomForest_()],
            disagreement=disagreement)
    elif query_strategy == 'lr_lsvc_dt_gpc':
        if disagreement == 'kl_divergence':
            raise ValueError('when using kl_divergence lsvc cannot be in the committee as linearSVC does not provide predict_proba().\
                             Use svc instead or change disagreement to vote!')
        qs = le.QueryByCommitteeUBoot_(train_ds, models=[la.LogisticRegression_(solver='liblinear',max_iter=1000),
                                                    la.LinearSVC_(),la.DecisionTree_(),la.GaussianProcess_()],
                    disagreement=disagreement)
    elif query_strategy == 'lr_lsvc_dt_xgb':
        if disagreement == 'kl_divergence':
            raise ValueError('when using kl_divergence lsvc cannot be in the committee as linearSVC does not provide predict_proba().\
                             Use svc instead or change disagreement to vote!')
        qs = le.QueryByCommitteeUBoot_(train_ds, models=[la.LogisticRegression_(solver='liblinear',max_iter=1000),
                                                    la.LinearSVC_(),la.DecisionTree_(),
                                                    la.XGBClassifier_(objective="binary:logistic")],
            disagreement=disagreement)
    elif query_strategy == 'homogeneous_committee':
        committee = com.CommitteeModels(estimator_name)
        qs = le.QueryByCommitteeUBoot_(train_ds,models= committee.committee['models'])
    else:
        print("Query strategy not defined!")
    return qs

#%%
    
def getLearningModel(estimator_name, warm_start=True):
    print('Initialize Learning Model')
    if estimator_name == 'rf': 
        model = la.RandomForest_(random_state=42,warm_start=warm_start, n_estimators=10)
    elif estimator_name =='lr': 
        model = la.LogisticRegression_(random_state=42,solver='liblinear', max_iter=1000)
    elif estimator_name == 'dt':
        model = la.DecisionTree_(random_state=42)
    elif estimator_name == 'lsvc':
        model = la.LinearSVC_(random_state=42)
    elif estimator_name == 'svc':
        model = la.SVC_(random_state=42,kernel='linear',probability=True)
    elif estimator_name == 'xgb':
        model = la.XGBClassifier_(random_state=42,objective="binary:logistic")
    elif estimator_name == 'gpc':
        model = la.GaussianProcess_(random_state=42)
    elif estimator_name == 'lrcv':
        model = la.LogisticRegressionCV_(random_state=42,cv=5, solver='liblinear', max_iter=1000)
    else:
        print('Unknown model type!')
    return model

#%%
    
def run_noisy_al(train_ds, test_ds, lbr, model, qs, quota,reweight=None, labeled_weight=2.0, warm_start=False):
#
#    if 'model_validation' in qs_name:
#        model_validation = True
#    if 'relabeling' in qs_name:
#        relabeling = True
    
    start_time = time.time()
    pool_ids = []
    
    E_in_f1, E_out_f1, E_out_P, E_out_R, unsupervised_correctness = [], [], [], [], []
    #E_in_f1_regressor, E_out_f1_regressor = [], []  
    labels = []
    correctedlabels = []
    confidence_queries = []
    initial_weights = copy.deepcopy(train_ds.get_sample_weights())
    initial_weights_queries = []
    print("Labeled weight:", labeled_weight)
    print("Warm start:", model.model.warm_start)

    l = quota
    sup.printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    for i in range(quota): 
        # Standard usage of libact objects
        ask_id = qs.make_query()
        query_conf = train_ds._sample_weights[ask_id]
        confidence_queries.append(query_conf)
        initial_weights_queries.append(initial_weights[ask_id])
        X, _ = zip(*train_ds.data)
        
        
        lb = lbr.label(X[ask_id])
        train_ds.update(ask_id, lb)
        
        corrected = 0
        if train_ds._y_unsupervised[ask_id] != lb :
            corrected = 1
            train_ds.update_unsupervised(ask_id, lb)
                 
        correctedlabels.append(corrected)
        labels.append(lb)

        # update weights of labeled pairs
        if reweight!= None:
            weight = labeled_weight
            train_ds.update_single_weight(ask_id, weight)
            

        if model.model.warm_start and 1 in train_ds._y and 0 in train_ds._y and i>0 :
            model.model.n_estimators +=2
            model.train(train_ds)
        
        else: 
            
            model.train(Dataset(train_ds._X,train_ds._y_unsupervised), sample_weight=train_ds.get_sample_weights())
                
        train_ds_x, train_ds_y = train_ds.get_unsupervised_labeled_entries()
        testx, testy = test_ds.format_sklearn()
        
        E_in_f1 = np.append(E_in_f1, f1_score(train_ds_y, model.predict(train_ds_x), pos_label=1, average='binary'))
        prec, recall, fscore, support = precision_recall_fscore_support(testy, model.predict(testx), average='binary')
        
        E_out_f1 = np.append(E_out_f1, fscore)
        E_out_P = np.append(E_out_P, prec)
        E_out_R = np.append(E_out_R, recall)
        
       
        #correctness of unsupervised pool
        correct_elements = len([i for i, j in zip(lbr.y, train_ds._y_unsupervised) if i == j])
       
        correctness = float(correct_elements)/float(len(lbr.y))
        unsupervised_correctness = np.append(unsupervised_correctness, correctness)
        
        # Update Progress Bar
        sup.printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)  
          
        if i == quota-1:
            print("F1: % 2.3f " % E_out_f1[-1])
            print("Initial weights of queried pairs:", Counter(initial_weights_queries))


    runt = time.time() - start_time
    print('Runtime: {:.2f} seconds'.format(runt))
    print('Correctness of unsupervised Pool: % 2.3f' % unsupervised_correctness[-1])
    
    #print("Labels")
    #display(Counter(labels).keys())
    #display(Counter(labels).values())
    
    return  E_in_f1, E_out_f1, unsupervised_correctness, model, pool_ids


#%%
    
def al_unsuper_bootstrapped_different_settings_single(candsets,candsets_train,candsets_test,target_name,feature,estimators,query_strategies,quota,warm_start,
                                                      reweight='score_based',disagreement='vote',n=5):
    """
    Run Active Learning with different settings as specified in estimators and query_strategies!
    """
    d = {}
    for est in estimators:
        print('Start with Estimator: {}'.format(est))
        for qs in query_strategies:
            print('Start with Query Strategy: {}'.format(qs))
            temp = active_unsuper_bootstrapped_learning(candsets,candsets_train,candsets_test,target_name,feature,est,qs,quota,
                                                        warm_start,reweight,disagreement,n)
            if(target_name in d.keys()):
                if(est in d[target_name].keys()):
                    d[target_name][est].update(temp[target_name][est])
                else:
                    d[target_name].update(temp[target_name])
            else:
                d.update(temp)
    
    return d

#%%
    
def active_learning_different_settings_all(candsets,candsets_train,candsets_test,feature,estimators,query_strategies,quota,
                                       warm_start,reweight='score_based',disagreement='vote',n=5):
    
    al_results = {}
    for candset in candsets:
        print('Start with AL using different settings for {}'.format(candset))
        target_name = candset
        temp = al_unsuper_bootstrapped_different_settings_single(candsets,candsets_train,candsets_test,target_name,
                                                                 feature,estimators,query_strategies,quota,
                                                                 warm_start,reweight,disagreement,n)
        if(target_name in al_results):
            al_results[target_name].update(temp[target_name])
        else:
            al_results.update(temp)
    
    return al_results

