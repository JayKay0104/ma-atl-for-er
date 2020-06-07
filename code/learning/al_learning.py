# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:49:40 2020

@author: jonas
"""

#import logging
#LOGGER = logging.getLogger(__name__)
import numpy as np
import al_learningalgos as la
import al_committees as com
from libact.base.dataset import Dataset
from libact.labelers import IdealLabeler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.utils import shuffle
import time
import sys
sys.path.append('../')
import support_utils as sup

#%%  

def active_learning(candsets,target_name,feature,estimator_name,query_strategy,n_labeled,quota,disagreement='vote',n=5):
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
    
    X_target = candsets[target_name][feature].to_numpy()
    y_target = candsets[target_name]['label'].to_numpy()
    # stratified train_test_split and random_state fixed (i.e. set to 42) in order to ensure comparable results
    X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(X_target,y_target,test_size=0.33,
                                                                                    random_state=42,stratify=y_target)
    
    print('Train_test_split: random_state = 42, stratified ; LR solver: liblinear')
    # test set 
    test_ds = Dataset(X=X_target_test,y=y_target_test)
    
    print('Starting AL Experiments (no transfer!) for candset {}'.format(target_name))
    for i in range(n):
        print('{}. Run of {}'.format(i+1,n))
        
        train_ds, fully_labeled_trn_ds = initializeALPool(X_target_train, y_target_train, n_labeled)
        
        # if quota -1 it means it is not a fixed amount
        # create the quota which is the amount of all instances 
        # in the training pool minus the amount of already labeled ones
        if(quota == -1): 
            quota = train_ds.len_unlabeled()
            
        # cerate the IdealLabeler with the full labeled training set
        lbr = IdealLabeler(fully_labeled_trn_ds)
        
        model = la.getLearningModel(estimator_name)
        
        qs = com.getQueryStrategy(query_strategy, train_ds, disagreement, estimator_name)
            
        train_acc, train_f1, test_acc, test_f1, model_, runt = run_al(train_ds,test_ds,lbr,model,qs,quota,n_labeled)

        training_accuracy_scores.append(train_acc)
        training_f1_scores.append(train_f1)
        test_accuracy_scores.append(test_acc)
        test_f1_scores.append(test_f1)
        runtimes.append(runt)
    
    runt = np.mean(runtimes)
    d = {target_name:{estimator_name:{query_strategy:{'quota':quota,'n_runs':n,'n_init_labeled':n_labeled,
                                     'model_params':model_.get_params(),'avg_runtime':runt,
                                     'training_accuracy_scores':training_accuracy_scores,
                                     'training_f1_scores':training_f1_scores,
                                     'test_accuracy_scores':test_accuracy_scores,
                                     'test_f1_scores':test_f1_scores}}}}
    return d

#%%
    
def initializeALPool(X_target_train, y_target_train, n_labeled):
    # share of true matches at beginning. Because we have to sample n_labled instances at
    # the beginning to kick start the active learning process we have to ensure that
    # the initial training set does not only contain one class. Hence, we calculate
    # the share of true matches and force it to be between 0.4 and 0.6
    # here we also make a train test split

    share_tm = np.sum(y_target_train[:n_labeled])/n_labeled
    while not (0.4<=share_tm<=0.6):
        # shuffle the target training set so that n_labeled instances are balanced at start
        X_target_train, y_target_train = shuffle(X_target_train, y_target_train)
        share_tm = np.sum(y_target_train[:n_labeled])/n_labeled
    
    # only target instances
    X_train = X_target_train
    y_train = y_target_train
    # train_ds is the whole X_train but only the n_labeled instances are labeled
    train_ds = Dataset(X=X_train,y=np.concatenate([y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))

    # here we have the fully labeled training set 
    fully_labeled_trn_ds = Dataset(X=X_train,y=y_train)
    return train_ds, fully_labeled_trn_ds

#%%
def run_al(train_ds,test_ds,lbr,model,qs,quota,n_init_labeled):
    
    start_time = time.time()
    
    E_in, E_in_f1, E_out, E_out_f1, E_out_P, E_out_R = [], [], [], [], [], []
        
    labels = []
    
    for x in range(n_init_labeled):
        E_out_f1 = np.append(E_out_f1, 0.0)
        E_out_P = np.append(E_out_P, 0.0)
        E_out_R = np.append(E_out_R, 0.0)
        E_in_f1 = np.append(E_in_f1, 0.0)
        E_in = np.append(E_in, 0.0)
    
    l = quota-n_init_labeled
    sup.printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
       
    for i in range(quota-n_init_labeled):
        # QBC
        ask_id = qs.make_query()
        # Labeler for QBC on train_ds and Random on train_ds2
        lb = lbr.label(train_ds.data[ask_id][0])         
        # QBC
        train_ds.update(ask_id, lb)
        labels.append(lb)
                
        model.train(train_ds)
        
        X_train_current,y_train_current = train_ds.format_sklearn()
        E_in = np.append(E_in, model.score(train_ds))
        E_in_f1 = np.append(E_in_f1, f1_score(y_train_current, model.predict(X_train_current), pos_label=1, average='binary', sample_weight=None))
        
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
    
def active_learning_different_settings_single(candsets,target_name,feature,estimators,query_strategies,n_labeled,
                                       quota,disagreement='vote',n=5):
    """
    Run Active Learning with different settings as specified in estimators and query_strategies!
    """
    d = {}
    for est in estimators:
        print('Start with Estimator: {}'.format(est))
        for qs in query_strategies:
            print('Start with Query Strategy: {}'.format(qs))
            temp = active_learning(candsets,target_name,feature,est,qs,n_labeled,quota,
                                disagreement,n)
            if(target_name in d.keys()):
                if(est in d[target_name].keys()):
                    d[target_name][est].update(temp[target_name][est])
                else:
                    d[target_name].update(temp[target_name])
            else:
                d.update(temp)
    
    return d

#%%
    
def active_learning_different_settings_all(candsets,feature,estimator_names,query_strategies,n_labeled,
                                           quota,disagreement='vote',n=5):
    
    al_results = {}
    for candset in candsets:
        print('Start with AL using different settings for {}'.format(candset))
        target_name = candset
        temp = active_learning_different_settings_single(candsets,target_name,feature,estimator_names,query_strategies,
                                                         n_labeled,quota,disagreement,n)
        if(target_name in al_results):
            al_results[target_name].update(temp[target_name])
        else:
            al_results.update(temp)
    
    return al_results