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

from sklearn.metrics import f1_score, precision_recall_fscore_support
import time

import la_ext as le
import da_weighting as da
import itertools
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
    
    training_accuracy_scores, training_f1_scores, test_accuracy_scores, test_f1_scores, test_precision, test_recall = [],[],[],[],[],[]
    model_pred_prob_start, model_feature_import_start, model_depth_tree_start = [],[],[]
    model_pred_prob_end, model_feature_import_end, model_depth_tree_end = [],[],[]
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
        print('Unsupervised Domain Adaptation: Calculate sample_weight for the source instances using {}'.format(weighting))
        sample_weight = da.getSampleWeightsOfDomainAdaptation(X_source, X_target, weighting)
    
    X_target_train = candsets_train[target_name][feature]
    y_target_train = candsets_train[target_name]['label']
    
    X_target_test = candsets_test[target_name][feature]
    y_target_test = candsets_test[target_name]['label']
    
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
        
        train_acc, train_f1, test_acc, test_f1, test_p, test_r, model_, runt, model_pred_prob,\
        model_feature_import, model_depth_tree = run_atl(train_ds,test_ds,lbr,model,qs,quota,n_labeled)
        #train_acc, train_f1, test_acc, test_f1, model_, runt = run_atl(train_ds,test_ds,lbr,model,qs,quota,n_labeled)

        training_accuracy_scores.append(train_acc)
        training_f1_scores.append(train_f1)
        test_accuracy_scores.append(test_acc)
        test_f1_scores.append(test_f1)
        test_precision.append(test_p)
        test_recall.append(test_r)
        model_pred_prob_start.append(model_pred_prob[0])
        model_feature_import_start.append(model_feature_import[0])
        model_pred_prob_end.append(model_pred_prob[1])
        model_feature_import_end.append(model_feature_import[1])
        if(model.name == 'rf' or model.name == 'dt'):
            model_depth_tree_start.append(model_depth_tree[0])
            model_depth_tree_end.append(model_depth_tree[1])
        
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
                                                                  'test_f1_scores':test_f1_scores,
                                                                  'test_precision':test_precision,
                                                                  'test_recall':test_recall,
                                                                  'model_pred_prob_start':model_pred_prob_start,
                                                                  'model_feature_import_start':model_feature_import_start,
                                                                  'model_depth_tree_start':model_depth_tree_start,
                                                                  'model_pred_prob_end':model_pred_prob_end,
                                                                  'model_feature_import_end':model_feature_import_end,
                                                                  'model_depth_tree_end':model_depth_tree_end}}}}}
    else:
        d = {key:{estimator_name:{query_strategy:{weighting:{'quota':quota,'n_runs':n,'n_init_labeled':n_labeled,
                                                             'model_params':model_.get_params(),'avg_runtime':runt,
                                                              'training_accuracy_scores':training_accuracy_scores,
                                                              'training_f1_scores':training_f1_scores,
                                                              'test_accuracy_scores':test_accuracy_scores,
                                                              'test_f1_scores':test_f1_scores,
                                                              'test_precision':test_precision,
                                                              'test_recall':test_recall,
                                                              'model_pred_prob_start':model_pred_prob_start,
                                                              'model_feature_import_start':model_feature_import_start,
                                                              'model_depth_tree_start':model_depth_tree_start,
                                                              'model_pred_prob_end':model_pred_prob_end,
                                                              'model_feature_import_end':model_feature_import_end,
                                                              'model_depth_tree_end':model_depth_tree_end,
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
    E_out_P, E_out_R = [], []
    
    model_pred_prob, model_feature_import, model_depth_tree = [],[],[]
    labels = []
    
    X_test,y_test = test_ds.format_sklearn()
    
    model.train(train_ds)
    
    model_pred_prob.append(model.predict_proba(X_test))
    model_feature_import.append(model.feature_importances_())
    if(model.name == 'dt'):
        model_depth_tree.append(model.get_tree_max_depth())
    if(model.name == 'rf'):
        model_depth_tree.append(model.get_trees_max_depth())
    
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
        
        
        E_out = np.append(E_out, model.score(test_ds))
        prec, recall, f1score, support = precision_recall_fscore_support(y_test, model.predict(X_test), average='binary')
        
#        if(i==0):
#            model_pred_prob.append(model.predict_proba(X_test))
#            model_feature_import.append(model.feature_importances_())
#            if(model.name == 'dt'):
#                model_depth_tree.append(model.get_tree_max_depth())
#            if(model.name == 'rf'):
#                model_depth_tree.append(model.get_trees_max_depth())
        
        if(i==quota-1):
            model_pred_prob.append(model.predict_proba(X_test))
            model_feature_import.append(model.feature_importances_())
            if(model.name == 'dt'):
                model_depth_tree.append(model.get_tree_max_depth())
            if(model.name == 'rf'):
                model_depth_tree.append(model.get_trees_max_depth())
        
        E_out_f1 = np.append(E_out_f1, f1score)
        E_out_P = np.append(E_out_P, prec)
        E_out_R = np.append(E_out_R, recall)
                    
        # Update Progress Bar
        sup.printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
        
    
    runt = time.time() - start_time
    print('Runtime: {:.2f} seconds'.format(runt))
    
    return E_in, E_in_f1, E_out, E_out_f1, E_out_P, E_out_R, model, runt, model_pred_prob, model_feature_import, model_depth_tree


#%%
    
def awtl_single(candsets,candsets_train,candsets_test,source_name,target_name,feature,estimators,query_strategies,
                                           quota,weighting=[None],disagreement='vote',n=5):
    """
    Run Active Transfer Learning with different settings as specified in estimators and query_strategies!
    """
    d = {}
    key = '{}_{}'.format(source_name,target_name)
    for est in estimators:
        print('Start with Estimator: {}'.format(est))
        for qs in query_strategies:
            print('Start with Query Strategy: {}'.format(qs))
            for weight in weighting:
                print('Start with Weighting Strategy: {}'.format(weight))
                temp = active_weighted_transfer_learning(candsets,candsets_train,candsets_test,source_name,target_name,feature,est,qs,
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

#%%
    
def awtl_all_combinations(candsets,candsets_train,candsets_test,estimator_names,query_strategies,
                          quota,all_feature,dense_features_dict=None,weighting=[None],disagreement='vote',n=5,switch_roles=False):
    
    d = {}
    
    combinations = []
    for combo in itertools.combinations(candsets, 2):
        if((combo[0].split('_')[0] in combo[1].split('_')) or (combo[0].split('_')[1] in combo[1].split('_'))):
            combinations.append(combo)
    if(switch_roles):
        for combo in combinations:
            source_name = combo[0]
            target_name = combo[1]
            if(dense_features_dict is None):
                # if dense_features_dict is None then use all features for each combination
                feature = all_feature
            else:
                # else only use the dense features and forget about the all_feature parameter
                dense_feature_key = '_'.join(sorted(set(source_name.split('_')+target_name.split('_'))))
                feature = dense_features_dict[dense_feature_key]
        
            print('Start with ATL using different settings for source {} and target {}'.format(source_name,target_name))
        
            temp = awtl_single(candsets,candsets_train,candsets_test,source_name,target_name,feature,estimator_names,query_strategies,
                               quota,weighting,disagreement,n)

            d.update(temp)
            # switch roles: now the previous target serves as source
            source_name = combo[1]
            target_name = combo[0]
        
            print('Start with ATL using different settings for source {} and target {}'.format(source_name,target_name))
        
            temp = awtl_single(candsets,candsets_train,candsets_test,source_name,target_name,feature,estimator_names,query_strategies,
                               quota,weighting,disagreement,n)

            d.update(temp)
    else:
        for combo in combinations:
            source_name = combo[0]
            target_name = combo[1]
            if(dense_features_dict is None):
                # if dense_features_dict is None then use all features for each combination
                feature = all_feature
            else:
                dense_feature_key = '_'.join(sorted(set(source_name.split('_')+target_name.split('_'))))
                feature = dense_features_dict[dense_feature_key]
            
            print('Start with ATL using different settings for source {} and target {}'.format(source_name,target_name))
            
            temp = awtl_single(candsets,candsets_train,candsets_test,source_name,target_name,feature,estimator_names,query_strategies,
                               quota,weighting,disagreement,n)
            
            d.update(temp)
    
    return d

#%%
    
def awtl_certain_combinations(candsets,candsets_train,candsets_test,combinations,estimator_names,query_strategies,
                                           quota,all_feature,dense_features_dict=None,weighting=[None],
                                           disagreement='vote',n=5,switch_roles=False):
    
    d = {}
    
    if(switch_roles):
        for combo in combinations:
            source_name = combo[0]
            target_name = combo[1]
            if(dense_features_dict is None):
                # if dense_features_dict is None then use all features for each combination
                feature = all_feature
            else:
                # else only use the dense features and forget about the all_feature parameter
                dense_feature_key = '_'.join(sorted(set(source_name.split('_')+target_name.split('_'))))
                feature = dense_features_dict[dense_feature_key]
        
            print('Start with ATL using different settings for source {} and target {}'.format(source_name,target_name))
        
            temp = awtl_single(candsets,candsets_train,candsets_test,source_name,target_name,feature,estimator_names,query_strategies,
                               quota,weighting,disagreement,n)

            d.update(temp)
            # switch roles: now the previous target serves as source
            source_name = combo[1]
            target_name = combo[0]
        
            print('Start with ATL using different settings for source {} and target {}'.format(source_name,target_name))
        
            temp = awtl_single(candsets,candsets_train,candsets_test,source_name,target_name,feature,estimator_names,query_strategies,
                               quota,weighting,disagreement,n)

            d.update(temp)
    else:
        for combo in combinations:
            source_name = combo[0]
            target_name = combo[1]
            if(dense_features_dict is None):
                # if dense_features_dict is None then use all features for each combination
                feature = all_feature
            else:
                # else only use the dense features and forget about the all_feature parameter
                dense_feature_key = '_'.join(sorted(set(source_name.split('_')+target_name.split('_'))))
                feature = dense_features_dict[dense_feature_key]
            
            print('Start with ATL using different settings for source {} and target {}'.format(source_name,target_name))
            
            temp = awtl_single(candsets,candsets_train,candsets_test,source_name,target_name,feature,estimator_names,query_strategies,
                                           quota,weighting,disagreement,n)
            
            d.update(temp)
    
    return d