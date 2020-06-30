# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:09:47 2020

@author: jonas
"""

import logging
LOGGER = logging.getLogger(__name__)
import numpy as np
import al_learningalgos as la
import al_committees as com
from libact.base.dataset import Dataset
from libact.labelers import IdealLabeler
from libact.query_strategies import UncertaintySampling
from sklearn.metrics import f1_score, precision_recall_fscore_support
import time
import itertools
import da_weighting as da
import la_ext as le

import sys
sys.path.append('../')
import support_utils as sup
        

#%%  

def atlx(candsets,candsets_train,candsets_test,source_name,target_name,feature,bootstrap_clf,
           query_strategy,quota,warm_start,n_bootstrapped_samples=2,weighting=None,disagreement='vote',n=5):
    """
    query_strategy: 
        Possible strategies are: 
            Baselines: 'uncertainty', 'random'
            Heterogeneous Committees: 'lr_lrcv_rf_dt', 'lr_lsvc_dt_gpc'
            Homogeneous Committees: 'homogeneous_committee' (it will then take the specified committee for the model used)
    """
    
    training_accuracy_scores, training_f1_scores, test_accuracy_scores, test_f1_scores, test_precision, test_recall = [],[],[],[],[],[]
    model_pred_prob_start, model_feature_import_start, model_depth_tree_start = [],[],[]
    model_pred_prob_end, model_feature_import_end, model_depth_tree_end = [],[],[]
    runtimes = []
    share_noise_labeled_set_pos_lst, share_noise_labeled_set_neg_lst = [],[]
    #n_labeled = 0
    
    X_source = candsets[source_name][feature].to_numpy()
    y_source = candsets[source_name]['label'].to_numpy()
    X_target = candsets[target_name][feature].to_numpy()
    X_target_train = candsets_train[target_name][feature].to_numpy()
    y_target_train = candsets_train[target_name]['label'].to_numpy()
    X_target_test = candsets_test[target_name][feature].to_numpy()
    y_target_test = candsets_test[target_name]['label'].to_numpy()

    n_labeled = n_bootstrapped_samples  
    
    # check if domain adaptation is desired
    if(weighting is None):
        print('No Unsupervised Domain Adaptation performed')
        sample_weight = None
    else:
        print('Unsupervised Domain Adaptation: Calculate sample_weight for the source instances using {}'.format(weighting))
        # unsupervised domain adaptation so we use the whole unlabeled source and target data
        sample_weight = da.getSampleWeightsOfDomainAdaptation(X_source, X_target, weighting)
    
    # create libact DataSet Object containting the validation set
    test_ds = Dataset(X=X_target_test,y=y_target_test)
    
    print('Starting ATL Experiments (WITH transfer!) source {} and target {}'.format(source_name,target_name))
    for i in range(n):
        print('{}. Run of {}'.format(i+1,n))
        
        train_ds, fully_labeled_trn_ds, n_labeled_, share_noise_labeled_set_pos, share_noise_labeled_set_neg = initializeATLPool(X_source, y_source, X_target_train, 
                                                                                                                     y_target_train, sample_weight, 
                                                                                                                     bootstrap_clf, n_labeled)
        
        share_noise_labeled_set_pos_lst.append(share_noise_labeled_set_pos)
        share_noise_labeled_set_neg_lst.append(share_noise_labeled_set_neg)
        # if quota -1 it means it is not a fixed amount
        # create the quota which is the amount of all instances 
        # in the training pool minus the amount of already labeled ones
        if(quota == -1): 
            quota = train_ds.len_unlabeled()
        
        # cerate the IdealLabeler with the full labeled training set
        lbr = IdealLabeler(fully_labeled_trn_ds)


        
        model = la.RandomForest_(random_state=42,warm_start=warm_start, n_estimators=10)
        
        qs = getQueryStrategy(query_strategy, train_ds, disagreement, 'rf')

            
        train_acc, train_f1, test_acc, test_f1, test_p, test_r, model_, runt, share_of_corrected_labels, model_pred_prob,\
        model_feature_import, model_depth_tree = run_weighted_atl(train_ds,test_ds,lbr,model,qs,quota)

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
        model_depth_tree_start.append(model_depth_tree[0])
        model_depth_tree_end.append(model_depth_tree[1])
        
        runtimes.append(runt)
    
    runt = np.mean(runtimes)
    
    key = '{}_{}'.format(source_name,target_name)
    if(weighting is None):
        # append weighting strategy to query_strategy name to be able to distinguish 
        d = {key:{'rf':{query_strategy:{'no_weighting':{'quota':quota,'n_runs':n,'n_init_labeled':n_labeled,
                                                        'share_noise_labeled_set_pos':share_noise_labeled_set_pos, 
                                                        'share_noise_labeled_set_neg':share_noise_labeled_set_neg,
                                                                  'share_of_corrected_labels':share_of_corrected_labels,
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
    else:
        d = {key:{'rf':{query_strategy:{weighting:{'quota':quota,'n_runs':n,'n_init_labeled':n_labeled,
                                                   'share_noise_labeled_set_pos':share_noise_labeled_set_pos, 
                                                        'share_noise_labeled_set_neg':share_noise_labeled_set_neg,
                                                             'share_of_corrected_labels':share_of_corrected_labels,
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
    
def initializeATLPool(X_source, y_source, X_target_train, y_target_train, sample_weights=None, bootstrap_clf='rf', n_labeled=2):
    print('Initialize Dataset Object')
    
    # Train a LRCV model on source (either with DA (sample weights) or not) 
    # and run predict_proba on all target instances. Then select one instance
    # where the prdict_proba for the pos. class is the highest and then the same 
    # for the neg. class in order to kick-start the committee.
#    lr = la.LogisticRegressionCV_(random_state=42,solver='liblinear',max_iter=1000)
#    if(sample_weights is None):
#        lr.train(le.AWTLDataset(X_source,y_source))
#    else:
#        lr.train(le.AWTLDataset(X_source,y_source,sample_weights))
#        
#    predict_proba_target = lr.predict_proba(X_target_train)
    
    # Train a model (bootstrap_clf parameter to choose but default rf with default sklearn setting)
    # on source (either with DA (sample weights) or not) 
    # and run predict_proba on all target instances. Then select one instance
    # where the prdict_proba for the pos. class is the highest and then the same 
    # for the neg. class in order to kick-start the committee.
    clf = la.getLearningModel(bootstrap_clf)
    if(sample_weights is None):
        clf.train(le.AWTLDataset(X_source,y_source),no_weights=True)
    else:
        clf.train(le.AWTLDataset(X_source,y_source,sample_weights))
        
    predict_proba_target = clf.predict_proba(X_target_train)
    
    # as the classes shall be balanced everything smaller than 4 will be assigned 1 instance per class, so n_labeled == 2
    k = 0
    if(n_labeled>=4):
        neg_proba = predict_proba_target[:,0]
        pos_proba = predict_proba_target[:,1]
        
        # balance the class
        n_per_class = int(n_labeled/2)
        # check whether 
        if (((pos_proba[np.argpartition(pos_proba, -n_per_class)[-n_per_class:]]>0.9).all()==True)\
           and ((neg_proba[np.argpartition(neg_proba, -n_per_class)[-n_per_class:]]>0.9).all()==True)):
            k = n_per_class
            pos_idx = np.argpartition(pos_proba, -k)[-k:]
            neg_idx = np.argpartition(neg_proba, -k)[-k:]
        else:
            # get the amount of class positive with pred proba > 0.9
            k = np.argwhere(pos_proba>0.9).shape[0]
            # check whether the k largesst class negatives have pred proba > 0.9, if yes take them
            if(k==0):
                pass
            elif ((neg_proba[np.argpartition(neg_proba, -k)[-k:]]>0.9).all()==True):
                pos_idx = np.argpartition(pos_proba, -k)[-k:]
                neg_idx = np.argpartition(neg_proba, -k)[-k:]  
            else:
                # if not reduce to the amount of k to the amount of largest neg with pred proba > 0.9
                k = np.argwhere(neg_proba>0.9).shape[0]
                # take as many (k) pos as neg from pred proba
                pos_idx = np.argpartition(pos_proba, -k)[-k:]
                neg_idx = np.argpartition(neg_proba, -k)[-k:]
        
        if(k!=0):
            print('n_labeled max, as specified upfront is {}'.format(n_labeled))
            n_labeled = k*2 # if n_labeled was not even now it is ensured
            print('The actual amount of instances with a predict proba higher than 90% is {}'.format(n_labeled))
            
            X_init_neg = X_target_train[neg_idx]
            y_init_neg_true = y_target_train[neg_idx]
            X_init_pos = X_target_train[pos_idx]
            y_init_pos_true = y_target_train[pos_idx]
            
            X_init_labeled = np.vstack([X_init_pos,X_init_neg])
            y_init_labeled = np.append(np.ones(k),np.zeros(k))
            
            y_init_labeled_true = np.append(y_init_pos_true,y_init_neg_true)
            # check whether noise in labeled set
            share_noise_labeled_set_pos = (k-(np.sum(y_init_pos_true)))/k
            share_noise_labeled_set_neg = ((np.sum(y_init_neg_true))/k)
        else:
            pass

    
    if(k==0):
        n_labeled = 2 
        # get the target instance with the highest predict_proba for neg. class
        neg_idx = np.argmax(predict_proba_target[:,0])
        X_init_neg = X_target_train[neg_idx]
        y_init_neg_true = y_target_train[neg_idx]
        # get the target instance with the highest predict_proba for pos. class
        pos_idx = np.argmax(predict_proba_target[:,1])
        X_init_pos = X_target_train[pos_idx]
        y_init_pos_true = y_target_train[pos_idx]
        
        X_init_labeled = np.vstack([X_init_pos,X_init_neg])
        y_init_labeled = np.array([1,0])
        
        y_init_labeled_true = np.append(y_init_pos_true,y_init_neg_true)
        
        share_noise_labeled_set_pos = 1 - y_init_pos_true
        share_noise_labeled_set_neg = y_init_neg_true
        

    # delete the two target instances added to the labeled set from the training set.
    X_target_train = np.delete(X_target_train,[pos_idx,neg_idx],axis=0)
    y_target_train = np.delete(y_target_train,[pos_idx,neg_idx])
    
    # add the labels of the initial label set at the beginnning
    X_train = np.vstack([X_init_labeled,X_target_train])
    y_train = np.append(y_init_labeled,y_target_train)
    
    y_train_true = np.append(y_init_labeled_true,y_target_train)

    # TBD: we first need to adapt the ATLDatasetObject
    # but the idea is to predict labels for all target train instances using the model we trained on only
    # source instances and use those labels when querying instances (namely querying those not only
    # with highest diagreement among committee members but also where the majority vote differs from the
    # label predicted by the model only trained on source instances)
    
    y_transfer_labels = clf.predict(X_train) 
    y_transfer_predict_proba = clf.predict_proba(X_train)
    
    # train_ds is the whole X_train but only the n_labeled (all source) instances are labeled
    
    # TBD: Adapt the ATLDataset Object so that we can store the y_transfer_labels in it.
    # We want to adapt the vote strategy (i.e. make_query function) so that we query the example
    # not only with the highest disagreement among them but also where the majority vote 
    # (maybe we need then 5 instead of 4 members) is other than the one initially predicted by the 
    # model only trained on source instances, so that we query those examples where it seems that 
    # they differ from the source.
    train_ds = le.SourceATLDataset(X_train,np.concatenate([y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]),
                                  X_source, y_source, y_transfer_labels, y_transfer_predict_proba, sample_weights)
    
    # here we have the fully labeled training set 
    fully_labeled_trn_ds = Dataset(X=X_train,y=y_train_true)
    return train_ds, fully_labeled_trn_ds, n_labeled, share_noise_labeled_set_pos, share_noise_labeled_set_neg


#%%
    
def getQueryStrategy(query_strategy, train_ds, disagreement, estimator_name=None):
    print('Initialize Query Strategy')
    if query_strategy == 'lr_lsvc_rf_dt' or query_strategy == 'default':
        if disagreement == 'kl_divergence':
            raise ValueError('when using kl_divergence lsvc cannot be in the committee as linearSVC does not provide predict_proba().\
                             Use svc instead or change disagreement to vote!')
        qs = le.SourceQueryByCommittee_(train_ds, models=[la.RandomForest_(),la.DecisionTree_(),
                                                   la.LogisticRegression_(solver='liblinear',max_iter=1000),
                                                   la.LinearSVC_()],disagreement=disagreement)
        # committee of five 
    elif query_strategy == 'lr_svc_rf_dt_xgb':
        qs = le.SourceQueryByCommittee_(train_ds, models=[la.LogisticRegression_(solver='liblinear',max_iter=1000),
                                                la.SVC_(kernel='linear',probability=True),la.RandomForest_(),la.DecisionTree_(),
                                                la.XGBClassifier_(objective="binary:logistic")],disagreement=disagreement)
    elif query_strategy == 'lr_lsvc_rf_dt_xgb':
        if disagreement == 'kl_divergence':
            raise ValueError('when using kl_divergence lsvc cannot be in the committee as linearSVC does not provide predict_proba().\
                             Use svc instead or change disagreement to vote!')
        qs = le.SourceQueryByCommittee_(train_ds, models=[la.RandomForest_(),la.DecisionTree_(),
                                                   la.LogisticRegression_(solver='liblinear',max_iter=1000),
                                                   la.LinearSVC_(),la.XGBClassifier_(objective="binary:logistic")],disagreement=disagreement)
    elif query_strategy == 'lr_svc_dt_xgb':
        qs = le.SourceQueryByCommittee_(train_ds, models=[la.LogisticRegression_(solver='liblinear',max_iter=1000),
                                                la.SVC_(kernel='linear',probability=True),la.DecisionTree_(),
                                                la.XGBClassifier_(objective="binary:logistic")],
            disagreement=disagreement)
    elif query_strategy == 'uncertainty':
        qs = UncertaintySampling(train_ds, method='lc', model=la.LogisticRegression_())
    elif query_strategy == 'random':
        qs = le.RandomSampling_(train_ds)
    # committee with probabilistic models (SVC with prob=True used here instead of LinearSVC)
    elif query_strategy == 'lr_svc_rf_dt':
        qs = le.SourceQueryByCommittee_(train_ds, models=[la.RandomForest_(),la.DecisionTree_(),
                                                la.LogisticRegression_(solver='liblinear',max_iter=1000),
                                                la.SVC_(kernel='linear',probability=True)],
                    disagreement=disagreement)
    elif query_strategy == 'lr_lsvc_dt_gpc':
        if disagreement == 'kl_divergence':
            raise ValueError('when using kl_divergence lsvc cannot be in the committee as linearSVC does not provide predict_proba().\
                             Use svc instead or change disagreement to vote!')
        qs = le.SourceQueryByCommittee_(train_ds, models=[la.LogisticRegression_(solver='liblinear',max_iter=1000),
                                                    la.LinearSVC_(),la.DecisionTree_(),la.GaussianProcess_()],
                    disagreement=disagreement)
    elif query_strategy == 'lr_lsvc_dt_xgb':
        if disagreement == 'kl_divergence':
            raise ValueError('when using kl_divergence lsvc cannot be in the committee as linearSVC does not provide predict_proba().\
                             Use svc instead or change disagreement to vote!')
        qs = le.SourceQueryByCommittee_(train_ds, models=[la.LogisticRegression_(solver='liblinear',max_iter=1000),
                                                    la.LinearSVC_(),la.DecisionTree_(),
                                                    la.XGBClassifier_(objective="binary:logistic")],
            disagreement=disagreement)
    elif query_strategy == 'homogeneous_committee':
        committee = com.CommitteeModels(estimator_name)
        qs = le.SourceQueryByCommittee_(train_ds,models= committee.committee['models'])
    else:
        print("Query strategy not defined!")
    return qs

##%%
#    
#def getLearningModel(estimator_name, warm_start=True):
#    print('Initialize Learning Model')
#    if estimator_name == 'rf': 
#        model = la.RandomForest_(random_state=42,warm_start=warm_start, n_estimators=10)
#    elif estimator_name =='lr': 
#        model = la.LogisticRegression_(random_state=42,solver='liblinear', max_iter=1000)
#    elif estimator_name == 'dt':
#        model = la.DecisionTree_(random_state=42)
#    elif estimator_name == 'lsvc':
#        model = la.LinearSVC_(random_state=42)
#    elif estimator_name == 'svc':
#        model = la.SVC_(random_state=42,kernel='linear',probability=True)
#    elif estimator_name == 'xgb':
#        model = la.XGBClassifier_(random_state=42,objective="binary:logistic")
#    elif estimator_name == 'gpc':
#        model = la.GaussianProcess_(random_state=42)
#    elif estimator_name == 'lrcv':
#        model = la.LogisticRegressionCV_(random_state=42,cv=5, solver='liblinear', max_iter=1000)
#    else:
#        print('Unknown model type!')
#    return model

#%%
def run_weighted_atl(train_ds,test_ds,lbr,model,qs,quota):
    
    start_time = time.time()
    
    E_in, E_in_f1, E_out, E_out_f1 = [], [], [], []
    E_out_P, E_out_R = [], []
    
    model_pred_prob, model_feature_import, model_depth_tree = [],[],[]
    
    labels, corrected_labels = [],[]
    
    X_test,y_test = test_ds.format_sklearn()
    
    l = quota
    sup.printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    for i in range(quota):
        # QBC
        ask_id = qs.make_query()
        # Oracle
        lb = lbr.label(train_ds.data[ask_id][0])         
        # QBC
        train_ds.update(ask_id, lb)
        labels.append(lb)
        
        corrected = 0
        if(train_ds._y_transfer_labels[ask_id] != lb):
            corrected = 1
            train_ds.update_transfer_labels(ask_id, lb)
                 
        corrected_labels.append(corrected)
        
        if(model.model.warm_start and 1 in train_ds._y and 0 in train_ds._y and i>0):
            # for RF if warm_start = True and i > 0 then there will be two trees added
            # to the forest which are trained only on the target instances queried. In the 
            # case i == 1 then only the two bootstrapped target instances (one neg. and one pos. target instances)
            # are in the training set. For the subsequent runs the queried target instances will be in the training set 
            model.model.n_estimators +=2
            model.train(train_ds,no_weights=True)
            
            # calculating the training score
            X_train_current,y_train_current = train_ds.format_sklearn(no_weights=True)
            E_in = np.append(E_in, model.score(Dataset(X=X_train_current,y=y_train_current)))
            E_in_f1 = np.append(E_in_f1, f1_score(y_train_current, model.predict(X_train_current), pos_label=1, average='binary'))
        elif(model.model.warm_start and i==0):
            # for RF with warm_start = True in the first itertation a forest with 10 trees
            # is trained on the source instances. If the source instances are weighted
            # based on importance weighting of domain adaptation then it is trained with them
            # as sample_weights however if no weighting was specified when initializing the
            # Dataset (i.e. SourceATLDataset) object, then it it trained without domain adaptation
            model.train(train_ds.get_source_training_data())
            # calculating the training score
            X_train_current,y_train_current = train_ds.format_sklearn(no_weights=True)
            E_in = np.append(E_in, model.score(Dataset(X=X_train_current,y=y_train_current)))
            E_in_f1 = np.append(E_in_f1, f1_score(y_train_current, model.predict(X_train_current), pos_label=1, average='binary'))
            
            # get info about the RF model like pred prob on target test, feature importance, and depth of trees
            model_pred_prob.append(model.predict_proba(X_test))
            model_feature_import.append(model.feature_importances_())
            model_depth_tree.append(model.get_trees_max_depth())
        else:
            # for the case that we use a model other than RF as active learning model, we cannot use the warm_start
            # approch and hence we always learn a model on the source instance + the current labeled set. This is also
            # the case if we use RF with warm_start = False.
            X_source, y_source, sample_weights = train_ds.get_source_training_data().format_sklearn()
            X_target_current, y_target_current = train_ds.format_sklearn(no_weights=True)
            X_train_current = np.vstack([X_source,X_target_current])
            y_train_current = np.append(y_source, y_target_current)
            # assign a weight of 1 to each target instance
            sample_weights = np.concatenate([sample_weights,[1]*(y_target_current.shape[0])]) 
            model.train(le.AWTLDataset(X_train_current,y_train_current,sample_weights))
            # calculating the training score
            E_in = np.append(E_in, model.score(Dataset(X=X_train_current,y=y_train_current)))
            E_in_f1 = np.append(E_in_f1, f1_score(y_train_current, model.predict(X_train_current), pos_label=1, average='binary'))
        
        # calculating the test score for this iteration. This is actually the interesting part!!!
        E_out = np.append(E_out, model.score(test_ds))
        prec, recall, f1score, support = precision_recall_fscore_support(y_test, model.predict(X_test), average='binary')
        
        if(i==quota-1):
            model_pred_prob.append(model.predict_proba(X_test))
            model_feature_import.append(model.feature_importances_())
            model_depth_tree.append(model.get_trees_max_depth())
            print('Last iteration Performance on Target Test Set: F1 {:.2f}; Prec {:.2f}; Recall {:.2f}'.format(E_out_f1[-1],E_out_P[-1],E_out_R[-1]))
            print('Average depth of trees at start (iteration 0): {} at last iteration {}'.format(np.mean(model_depth_tree[0]),np.mean(model_depth_tree[1])))
    
        
        E_out_f1 = np.append(E_out_f1, f1score)
        E_out_P = np.append(E_out_P, prec)
        E_out_R = np.append(E_out_R, recall)
                    
        # Update Progress Bar
        sup.printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
        
            
        
    share_of_corrected_labeles = sum(corrected_labels)/quota
    runt = time.time() - start_time
    print('Runtime: {:.2f} seconds'.format(runt))
    print('Corrected labels from transfer: {}'.format(sum(corrected_labels)))
    
    return E_in, E_in_f1, E_out, E_out_f1, E_out_P, E_out_R, model, runt, share_of_corrected_labeles, model_pred_prob, model_feature_import, model_depth_tree


#%%
    
def atlx_all(candsets,candsets_train,candsets_test,dense_features_dict,bootstrap_clf,
               query_strategies,quota,warm_start,n_bootstrapped_samples=2,weighting=[None],
               disagreement='vote',n=5,switch_roles=False):
    
    d = {}
    
    combinations = []
    for combo in itertools.combinations(candsets, 2):
        if((combo[0].split('_')[0] in combo[1].split('_')) or (combo[0].split('_')[1] in combo[1].split('_'))):
            combinations.append(combo)
    if(switch_roles):
        for combo in combinations:
            source_name = combo[0]
            target_name = combo[1]
            dense_feature_key = '_'.join(sorted(set(source_name.split('_')+target_name.split('_'))))
            feature = dense_features_dict[dense_feature_key]
        
            print('Start with ATL using different settings for source {} and target {}'.format(source_name,target_name))
        
            temp = atlx_single(candsets,candsets_train,candsets_test,source_name,target_name,
                                                          feature,bootstrap_clf,query_strategies,quota,warm_start,
                                                          n_bootstrapped_samples,weighting,disagreement,n)

            d.update(temp)
            # switch roles: now the previous target serves as source
            source_name = combo[1]
            target_name = combo[0]
        
            print('Start with ATL using different settings for source {} and target {}'.format(source_name,target_name))
        
            temp = atlx_single(candsets,candsets_train,candsets_test,source_name,target_name,
                                                          feature,bootstrap_clf,query_strategies,quota,warm_start,
                                                          n_bootstrapped_samples,weighting,disagreement,n)

            d.update(temp)
    else:
        for combo in combinations:
            source_name = combo[0]
            target_name = combo[1]
            dense_feature_key = '_'.join(sorted(set(source_name.split('_')+target_name.split('_'))))
            feature = dense_features_dict[dense_feature_key]
            
            print('Start with ATL using different settings for source {} and target {}'.format(source_name,target_name))
            
            temp = atlx_single(candsets,candsets_train,candsets_test,source_name,target_name,
                                                          feature,bootstrap_clf,query_strategies,quota,warm_start,
                                                          n_bootstrapped_samples,weighting,disagreement,n)

            d.update(temp)
    
    return d


#%%
    
def atlx_combos(candsets,candsets_train,candsets_test,combinations,dense_features_dict,bootstrap_clf,
                  query_strategies,quota,warm_start,n_bootstrapped_samples=2,weighting=[None],disagreement='vote',n=5,switch_roles=False):
    d = {} 
    if(switch_roles):
        for combo in combinations:
            source_name = combo[0]
            target_name = combo[1]
            dense_feature_key = '_'.join(sorted(set(source_name.split('_')+target_name.split('_'))))
            feature = dense_features_dict[dense_feature_key]
        
            print('Start with ATL using different settings for source {} and target {}'.format(source_name,target_name))
        
            temp = atlx_single(candsets,candsets_train,candsets_test,source_name,target_name,
                                 feature,bootstrap_clf,query_strategies,quota,warm_start,
                                 n_bootstrapped_samples,weighting,disagreement,n)
            d.update(temp)
            # switch roles: now the previous target serves as source
            source_name = combo[1]
            target_name = combo[0]
        
            print('Start with ATL using different settings for source {} and target {}'.format(source_name,target_name))
        
            temp = atlx_single(candsets,candsets_train,candsets_test,source_name,target_name,
                                 feature,bootstrap_clf,query_strategies,quota,warm_start,
                                 n_bootstrapped_samples,weighting,disagreement,n)

            d.update(temp)
    else:
        for combo in combinations:
            source_name = combo[0]
            target_name = combo[1]
            dense_feature_key = '_'.join(sorted(set(source_name.split('_')+target_name.split('_'))))
            feature = dense_features_dict[dense_feature_key]
            
            print('Start with ATL using different settings for source {} and target {}'.format(source_name,target_name))
            
            temp = atlx_single(candsets,candsets_train,candsets_test,source_name,target_name,
                                 feature,bootstrap_clf,query_strategies,quota,warm_start,
                                 n_bootstrapped_samples,weighting,disagreement,n)

            d.update(temp)
    
    return d

#%%
    
def atlx_single(candsets,candsets_train,candsets_test,source_name,target_name,features,bootstrap_clf,
                  query_strategies,quota,warm_start,n_bootstrapped_samples=2,weighting=[None],disagreement='vote',n=5):
    """
    Run The final Active Transfer Learning Method with RF (expanding if warm_start=True) and query_strategies!
    """
    d = {}
    key = '{}_{}'.format(source_name,target_name)
    # for this final method the active learning model is RandomForest, but we want to
    # stick to the same structure for the dictionary with the results
    estimators = ['rf']
    for est in estimators:
        print('Start with Estimator: {}'.format(est))
        for qs in query_strategies:
            print('Start with Query Strategy: {}'.format(qs))
            for weight in weighting:
                print('Start with Weighting Strategy: {}'.format(weight))
                temp = atlx(candsets,candsets_train,candsets_test,source_name,target_name,features,
                              bootstrap_clf,qs,quota,warm_start,n_bootstrapped_samples,weight,disagreement,n)
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

