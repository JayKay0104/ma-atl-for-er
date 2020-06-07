# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:18:27 2020

@author: jonas
"""

import al_learningalgos as la
from libact.query_strategies import RandomSampling, UncertaintySampling, QueryByCommittee

#%%
    
def getQueryStrategy(query_strategy, train_ds, disagreement, estimator_name=None):
    print('Initialize Query Strategy')
    # no committee but baseline query strategy
    if query_strategy == 'uncertainty':
        qs = UncertaintySampling(train_ds, method='lc', model=la.LogisticRegression_())
    # no committee but baseline query strategy
    elif query_strategy == 'random':
        qs = RandomSampling(train_ds)
    elif query_strategy == 'lr_lsvc_rf_dt':
        if disagreement == 'kl_divergence':
            raise ValueError('when using kl_divergence lsvc cannot be in the committee as linearSVC does not provide predict_proba().\
                             Use svc instead or change disagreement to vote!')
        qs = QueryByCommittee(train_ds, models=[la.RandomForest_(),la.DecisionTree_(),
                                                la.LogisticRegression_(solver='liblinear',max_iter=1000),
                                                la.LinearSVC_()],
                    disagreement=disagreement)
    # committee with probabilistic models (SVC with prob=True used here instead of LinearSVC)
    elif query_strategy == 'lr_svc_rf_dt':
        qs = QueryByCommittee(train_ds, models=[la.RandomForest_(),la.DecisionTree_(),
                                                la.LogisticRegression_(solver='liblinear',max_iter=1000),
                                                la.SVC_(kernel='linear',probability=True)],
                    disagreement=disagreement)
    elif query_strategy == 'lr_svc_dt_xgb':
        qs = QueryByCommittee(train_ds, models=[la.LogisticRegression_(solver='liblinear',max_iter=1000),
                                                la.SVC_(kernel='linear',probability=True),la.DecisionTree_(),
                                                la.XGBClassifier_(objective="binary:logistic")],
            disagreement=disagreement)
    # committee of five 
    elif query_strategy == 'lr_svc_dt_xgb_rf':
        qs = QueryByCommittee(train_ds, models=[la.LogisticRegression_(solver='liblinear',max_iter=1000),
                                                la.SVC_(kernel='linear',probability=True),la.DecisionTree_(),
                                                la.XGBClassifier_(objective="binary:logistic"),la.RandomForest_()],
            disagreement=disagreement)
    elif query_strategy == 'lr_lsvc_dt_gpc':
        if disagreement == 'kl_divergence':
            raise ValueError('when using kl_divergence lsvc cannot be in the committee as linearSVC does not provide predict_proba().\
                             Use svc instead or change disagreement to vote!')
        qs = QueryByCommittee(train_ds, models=[la.LogisticRegression_(solver='liblinear',max_iter=1000),
                                                la.LinearSVC_(),la.DecisionTree_(),la.GaussianProcess_()],
                    disagreement=disagreement)
    elif query_strategy == 'lr_lsvc_dt_xgb':
        if disagreement == 'kl_divergence':
            raise ValueError('when using kl_divergence lsvc cannot be in the committee as linearSVC does not provide predict_proba().\
                             Use svc instead or change disagreement to vote!')
        qs = QueryByCommittee(train_ds, models=[la.LogisticRegression_(solver='liblinear',max_iter=1000),
                                                la.LinearSVC_(),la.DecisionTree_(),
                                                la.XGBClassifier_(objective="binary:logistic")],
            disagreement=disagreement)
    elif query_strategy == 'homogeneous_committee':
        committee = CommitteeModels(estimator_name)
        qs = QueryByCommittee(train_ds,models= committee.committee['models'])
    else:
        print("Query strategy not defined!")
        return None
    return qs 

#%%
class CommitteeModels(object):
    
    def __init__(self, learning_algorithm):
        self.committee = dict() # this will be filled with the committee of models ad their cost
        self.learning_algorithm = learning_algorithm  
        if self.learning_algorithm == 'dt' : self.getDecisionTreeCommittee()
        elif self.learning_algorithm == 'rf' : self.getRandomForrestCommittee()
        elif self.learning_algorithm == 'lr' : self.getLogisticRegressionCommittee()
        elif self.learning_algorithm == 'lrcv' : self.getLogisticRegressionCVCommittee()
        elif self.learning_algorithm == 'xgb' : self.getXGBoostCommittee()
        elif self.learning_algorithm == 'lsvc' : self.getLinearSVCCommittee()
        elif self.learning_algorithm == 'gpc' : self.getGaussianProcessCommittee()
                
   
    def getDecisionTreeCommittee(self):
        max_depth_list = [None, 3, 5, 10, 15]
        min_samples_leaf_list = [3, 5, 10]
        criterion_list = ['gini', 'entropy']
        models = []
        
        for max_depth_ in max_depth_list:
                for criterion_ in criterion_list:
                        for min_samples_leaf_ in min_samples_leaf_list:
                            models.append(la.DecisionTree_(random_state=1, max_depth=max_depth_,criterion=criterion_, min_samples_leaf=min_samples_leaf_))
                    
        self.committee['models'] = models
        
    def getRandomForrestCommittee(self):
        n_estimators_list = [10, 50, 100]
        max_depth_list = [None, 3, 5, 10, 15]
        min_samples_leaf_list = [3, 5, 10]
        models = []

        for n_estimators_ in n_estimators_list:
            for max_depth_ in max_depth_list:
                for min_samples_leaf_ in min_samples_leaf_list:
                    models.append(la.RandomForest_(random_state=1, max_depth=max_depth_, n_estimators=n_estimators_, min_samples_leaf=min_samples_leaf_))
                        
        self.committee['models'] = models
    
    def getLogisticRegressionCommittee(self):
        penalty_list = ['l1', 'l2']
        fit_intercept_list = [True, False]
        solver_list = ['liblinear', 'saga']
        #max_iter_list = [50, 100, 150]
        max_iter_list = [500,1000,1500]
        models = []
        
        for penalty_ in penalty_list:
            for fit_intercept_ in fit_intercept_list:
                for solver_ in solver_list:
                    for max_iter_ in max_iter_list:
                        models.append(la.LogisticRegression_(penalty=penalty_, fit_intercept=fit_intercept_, solver=solver_, max_iter=max_iter_))
                        
        self.committee['models'] = models

    def getLogisticRegressionCVCommittee(self):
        penalty_list = ['l1', 'l2']
        fit_intercept_list = [True, False]
        solver_list = ['liblinear', 'saga']
        #max_iter_list = [50, 100, 150]
        max_iter_list = [500,1000,1500]
        models = []
        
        for penalty_ in penalty_list:
            for fit_intercept_ in fit_intercept_list:
                for solver_ in solver_list:
                    for max_iter_ in max_iter_list:
                        models.append(la.LogisticRegressionCV_(penalty=penalty_, fit_intercept=fit_intercept_, solver=solver_, max_iter=max_iter_))
                        
        self.committee['models'] = models

    def getXGBoostCommittee(self):
        n_estimators_list = [100, 150, 200]
        learning_rate_list = [0.05, 0.1, 0.15]
        max_depth_list = [3, 5, 7]
        models = []
        
        for n_estimators_ in n_estimators_list:
            for learning_rate_ in learning_rate_list:
                for max_depth_ in max_depth_list:
                    models.append(la.XGBClassifier_(random_state=1, n_estimators=n_estimators_,learning_rate=learning_rate_,  max_depth=max_depth_))
                    #models.append(XGBClassifier_(random_state=1, n_estimators=n_estimators_,learning_rate=learning_rate_))
                    
        self.committee['models'] = models
        
    def getLinearSVCCommittee(self):
        penalty_list = ['l1', 'l2']
        fit_intercept_list = [True, False]
        C_list = [0.5, 1, 5, 10]
        models = []
        
        for penalty_ in penalty_list:
            for fit_intercept_ in fit_intercept_list:
                for C_ in C_list:
                    models.append(la.LinearSVC_(random_state=1, penalty=penalty_, fit_intercept=fit_intercept_, C=C_))
        
        self.committee['models'] = models
        
    def getGaussianProcessCommittee(self):
        max_iter_predict_list = [50, 100, 150]
        models = []
        
        for max_iter_ in max_iter_predict_list:
            models.append(la.GaussianProcess_(max_iter_predict=max_iter_))
        
        self.committee['models'] = models
