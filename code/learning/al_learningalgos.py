# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:20:23 2020

@author: jonas
"""

import numpy as np
import logging
LOGGER = logging.getLogger(__name__)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
import xgboost as xgb
from libact.base.interfaces import ProbabilisticModel
from libact.base.interfaces import ContinuousModel


#%%
    
def getLearningModel(estimator_name):
    print('Initialize Learning Model')
    if estimator_name == 'rf': 
        model = RandomForest_(random_state=42)
    elif estimator_name =='lr': 
        model = LogisticRegression_(random_state=42,solver='liblinear', max_iter=1000)
    elif estimator_name == 'dt':
        model = DecisionTree_(random_state=42)
    elif estimator_name == 'lsvc':
        model = LinearSVC_(random_state=42)
    elif estimator_name == 'svc':
        model = SVC_(random_state=42,kernel='linear',probability=True)
    elif estimator_name == 'xgb':
        model = XGBClassifier_(random_state=42,objective="binary:logistic")
    elif estimator_name == 'gpc':
        model = GaussianProcess_(random_state=42)
    elif estimator_name == 'lrcv':
        model = LogisticRegressionCV_(random_state=42,cv=5, solver='liblinear', max_iter=1000)
    else:
        print('Unknown model type!')
    return model

class RandomForest_(ProbabilisticModel):

    """Random Forest Classifier
    """

    def __init__(self, *args, **kwargs):
        self.model = RandomForestClassifier(*args, **kwargs)
        self.name = "rf"        

    def train(self, dataset, no_weights=False, *args, **kwargs):
        if(no_weights):
            return self.model.fit(*(dataset.format_sklearn(no_weights=True) + args), **kwargs)
        else:
            return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)
    def predict_proba(self, feature, *args, **kwargs):
        return self.model.predict_proba(feature, *args, **kwargs)
    
    def feature_importances_(self):
        return self.model.feature_importances_
    
    def get_params(self):
        return self.model.get_params()
    
    def get_trees_max_depth(self):
        return [est.get_depth() for est in self.model.estimators_]
    
class LogisticRegression_(ProbabilisticModel):

    """LogisticRegression Classifier
    """

    def __init__(self, *args, **kwargs):
        # add solver liblinear
        self.model = LogisticRegression(*args, **kwargs)
        self.name = "lr"        

    def train(self, dataset, no_weights=False, *args, **kwargs):
        if(no_weights):
            return self.model.fit(*(dataset.format_sklearn(no_weights) + args), **kwargs)
        else:
            return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)
    
    def predict_real(self, feature, *args, **kwargs):
        dvalue = self.model.decision_function(feature, *args, **kwargs)
        if len(np.shape(dvalue)) == 1:  # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T
        else:
            return dvalue
    
    def predict_proba(self, feature, *args, **kwargs):
        return self.model.predict_proba(feature, *args, **kwargs)
    
    def feature_importances_(self):
        return self.model.coef_.ravel()
    
    def get_params(self):
        return self.model.get_params()
    
    
class DecisionTree_(ProbabilisticModel):

    """DecisionTree Classifier
    """

    def __init__(self, *args, **kwargs):
        self.model = DecisionTreeClassifier(*args, **kwargs)
        self.name = "dt"        

    def train(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)
    def predict_proba(self, feature, *args, **kwargs):
        return self.model.predict_proba(feature, *args, **kwargs)
    
    def feature_importances_(self):
        return self.model.feature_importances_
    
    def get_params(self):
        return self.model.get_params()
    
    def get_tree_max_depth(self):
        return self.model.tree_.max_depth
    
    
class LinearSVC_(ContinuousModel):

    """Random Forest Classifier
    """

    def __init__(self, *args, **kwargs):
        self.model = LinearSVC(*args, **kwargs)
        self.name = "lsvc"        

    def train(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)
    
    def predict_real(self, feature, *args, **kwargs):
        dvalue = self.model.decision_function(feature, *args, **kwargs)
        if len(np.shape(dvalue)) == 1:  # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T
        else:
            if self.decision_function_shape != 'ovr':
                LOGGER.warn("SVM model support only 'ovr' for multiclass"
                            "predict_real.")
            return dvalue
    
    def feature_importances_(self):
        return self.model.coef_.ravel()
    
    def get_params(self):
        return self.model.get_params()

    
class SVC_(ProbabilisticModel):

    """SVC
    """

    def __init__(self, *args, **kwargs):
        self.model = SVC(*args, **kwargs)
        self.name = "svm"

    def train(self, dataset, *args, **kwargs):
        #if self.noisy: 
        #   return self.model.fit(*(dataset.unsupervised_format_sklearn() + args), **kwargs)
        #else:
        #   return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)
        return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)
    def predict_proba(self, feature, *args, **kwargs):
        return self.model.predict_proba(feature, *args, **kwargs)
    
    def feature_importances_(self):
        # only works if kernel is linear!
        return self.model.coef_.ravel()
    
    def get_params(self):
        return self.model.get_params
    
    def kernel(self):
        return self.model.kernel
    
    
class GaussianProcess_(ProbabilisticModel):

    """GaussianProcess Classifier
    """

    def __init__(self, *args, **kwargs):
        self.model = GaussianProcessClassifier(*args, **kwargs)
        self.name = "gpc"        

    def train(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)
    def predict_proba(self, feature, *args, **kwargs):
        return self.model.predict_proba(feature, *args, **kwargs)
    
    def feature_importances_(self):
        LOGGER.warn("GPC model does not support feature_importance")
        return None
    
    def get_params(self):
        return self.model.get_params()
    

class XGBClassifier_(ProbabilisticModel):

    """Gradient Boosting Classifier
    """

    def __init__(self, *args, **kwargs):
        self.model = xgb.XGBClassifier(*args, **kwargs)
        self.name = "xgb"

    def train(self, dataset, no_weights=False, *args, **kwargs):
        if(no_weights):
            return self.model.fit(*(dataset.format_sklearn(no_weights) + args), **kwargs)
        else:
            return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        feature_array = np.asarray(feature)
        return self.model.predict(feature_array, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)
    
    def predict_proba(self, feature, *args, **kwargs):
        feature_array = np.asarray(feature)
        return self.model.predict_proba(feature_array, *args, **kwargs)
    
    def feature_importances_(self):
        return self.model.feature_importances_
    
    def get_params(self):
        return self.model.get_params()
    

class LogisticRegressionCV_(ProbabilisticModel):

    """LogisticRegressionCV Classifier
    """

    def __init__(self, *args, **kwargs):
        self.model = LogisticRegressionCV(*args, **kwargs)
        self.name = "lrcv"        

    def train(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)
    
    def predict_real(self, feature, *args, **kwargs):
        dvalue = self.model.decision_function(feature, *args, **kwargs)
        if len(np.shape(dvalue)) == 1:  # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T
        else:
            return dvalue
    
    def predict_proba(self, feature, *args, **kwargs):
        return self.model.predict_proba(feature, *args, **kwargs)
    
    def feature_importances_(self):
        return self.model.coef_.ravel()
    
    def get_params(self):
        return self.model.get_params