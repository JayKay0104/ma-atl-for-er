# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:00:59 2020

@author: jonas
"""
from __future__ import division
#from sklearn.metrics import f1_score, precision_recall_fscore_support, make_scorer
import numpy as np
#import libact 
from libact.utils import inherit_docstring_from, seed_random_state
#import scipy.sparse as sp
#import libact.models 
#from libact.query_strategies import *
from libact.base.dataset import Dataset
from libact.base.interfaces import QueryStrategy
from libact.query_strategies import QueryByCommittee
#from collections import Counter
#import copy
#from itertools import compress
import math
#from copy import deepcopy
#import time
#from al_learningalgos import *
#import random

import logging
LOGGER = logging.getLogger(__name__)

class AWTLDataset(Dataset):
    
    def __init__(self, X=None, y=None, da_weights=None):       
        
        super().__init__(X, y)

        if(da_weights is None):
            self._sample_weights = np.full(self.__len__(), 1)
        else:
            if(self.len_labeled()!=da_weights.shape[0]):
                raise ValueError('Length not correct: da_weights need to contain the importance weights for all source instances')
            #every source data point is weighted based on DA importance weighting. All unlabeled (not yet labled) get weight 1
            else: self._sample_weights = np.concatenate([da_weights,[1]*(self.len_unlabeled())])  
    
    def get_sample_weights(self):
        return self._sample_weights[self.get_labeled_mask()]
    
    def update_weights (self, weight):
        self._sample_weights[self.get_labeled_mask()] = weight
    
    def update_single_weight (self, entry_id, weight):
        self._sample_weights[entry_id] = weight
   
    def format_sklearn(self,no_weights=False):
        if(no_weights):
            X, y = self.get_labeled_entries()
            return X, np.array(y)
        else:
            X, y = self.get_labeled_entries()
            weights = self.get_sample_weights()
            return X, np.array(y), np.array(weights)
    
    def format_sklearn_with_domain_col(self):
        X, y = self.get_labeled_entries_with_domain_col()
        #weights = self.get_sample_weights()
        # no sample weights for TrAdaBoost, as it is calculating them 
        # by its own using also already labeled target instances
        # here the domain column shall stay
        return X, np.array(y)#, np.array(weights)
    
    def labeled_uniform_sample(self, sample_size, replace=True):
        """Returns a Dataset object with labeled data only, which is
        resampled uniformly with given sample size.
        Parameter `replace` decides whether sampling with replacement or not.
        Parameters
        ----------
        sample_size
        """
        idx = np.random.choice(np.where(self.get_labeled_mask())[0],
                               size=sample_size, replace=replace )
        return AWTLDataset(self._X[idx], self._y[idx], da_weights=self._sample_weights[idx])
    
    def get_labeled_entries(self):
        """
        Returns list of labeled feature and their label
        Returns
        -------
        X: numpy array or scipy matrix, shape = ( n_sample labeled, n_features )
        y: list, shape = (n_samples lebaled)
        """
        #X = self._X[self.get_labeled_mask()]
        #return X[:,:-1], self._y[self.get_labeled_mask()].tolist()
        return self._X[self.get_labeled_mask()], self._y[self.get_labeled_mask()].tolist()

    def get_unlabeled_entries_(self):
        """
        Returns list of unlabeled features, along with their entry_ids
        Returns
        -------
        idx: numpy array, shape = (n_samples unlebaled)
        X: numpy array or scipy matrix, shape = ( n_sample unlabeled, n_features )
        """
        #X = self._X[~self.get_labeled_mask()]
        #return np.where(~self.get_labeled_mask())[0], X[:,:-1]
        return np.where(~self.get_labeled_mask())[0], self._X[~self.get_labeled_mask()]
    
    def get_labeled_entries_with_domain_col(self):
        """
        Returns list of labeled feature and their label
        Returns
        -------
        X: numpy array or scipy matrix, shape = ( n_sample labeled, n_features )
        y: list, shape = (n_samples lebaled)
        """
        return self._X[self.get_labeled_mask()], self._y[self.get_labeled_mask()].tolist()

    def get_unlabeled_entries_with_domain_col(self):
        """
        Returns list of unlabeled features, along with their entry_ids
        Returns
        -------
        idx: numpy array, shape = (n_samples unlebaled)
        X: numpy array or scipy matrix, shape = ( n_sample unlabeled, n_features )
        """
        return np.where(~self.get_labeled_mask())[0], self._X[~self.get_labeled_mask()]


    def _vote_disagreement(self, votes, models_count):
        ret = []
        for candidate in votes:
            ret.append(0.0)
            lab_count = {}
            for lab in candidate:
                lab_count[lab] = lab_count.setdefault(lab, 0) + 1

            # Using vote entropy to measure disagreement
            for lab in lab_count.keys():
                ret[-1] -= float(lab_count[lab]) / float(models_count) * \
                    math.log(float(lab_count[lab]) / float(models_count))
        return ret
    
#%%
        
    
class SourceATLDataset(Dataset):
    
    def __init__(self, X, y, X_source, y_source, y_transfer_labels, y_transfer_predict_proba=None, da_weights=None):       
        
        super().__init__(X, y)
        self._y_transfer_labels = np.array(y_transfer_labels)
        if y_transfer_predict_proba is None:
            self._y_transfer_predict_proba = np.array([])
        else:
            self._y_transfer_predict_proba = np.array(y_transfer_predict_proba)
        self._X_source = X_source
        self._y_source = np.array(y_source)
        if(da_weights is None):
            self._sample_weights = None
        else:
            if(self._y_source.shape[0]!=da_weights.shape[0]):
                raise ValueError('Length not correct: da_weights need to contain the importance weights for all source instances')
            # every source data point is weighted based on DA importance weighting. This applys only for
            # source data which is used at the beginning to train the active learning model
            else: self._sample_weights = da_weights
    
    def get_source_training_data(self):
        return AWTLDataset(self._X_source, self._y_source, da_weights=self._sample_weights)
    
    def get_sample_weights(self):
        return self._sample_weights
    
#    def get_transfer_labeled_mask(self):
#        return ~np.fromiter((e is None for e in self._y_transfer_labels), dtype=bool)
        
    def get_transfer_labeled_entries(self):
        return np.where(~self.get_labeled_mask())[0], self._y_transfer_labels[~self.get_labeled_mask()]
        #return self._X[self.get_transfer_labeled_mask()], self._y_transfer_labels[self.get_transfer_labeled_mask()].tolist()
    
    def get_transfer_predict_proba_entries(self):
        return np.where(~self.get_labeled_mask())[0], self._y_transfer_predict_proba[~self.get_labeled_mask()]
        #return self._X[self.get_transfer_labeled_mask()], self._y_transfer_predict_proba[self.get_transfer_labeled_mask()].tolist()
    
    def len_transfer_labeled(self):
        return self.get_transfer_labeled_mask().sum()
    
    def get_transfer_label(self, entry_id):
        return self._y_transfer_labels[entry_id]
        
    def update_transfer_labels(self, entry_id, new_label):
        self._y_transfer_labels[entry_id] = new_label
    
#    def transfer_labels_format_sklearn(self):
#        X, y = self.get_transfer_labeled_entries()
#        return X, np.array(y)
    
#    def transfer_predict_proba_format_sklearn(self):
#        X, y = self.get_transfer_predict_proba_entries()
#        return X, np.array(y)
    
    def format_sklearn(self,no_weights=False):
        if(no_weights):
            X, y = self.get_labeled_entries()
            return X, np.array(y)
        # watch out here actually the weights are only for the source instances
        # and hence it actually does not make sense to leave it as it is.
        # anyway it is considered in each function call from other modules
        # that one has to hand it over with the parameter no_weights=True
        # TBD: Check if it can be removed!
        else:
            X, y = self.get_labeled_entries()
            weights = self.get_sample_weights()
            return X, np.array(y), np.array(weights)

#    def _vote_disagreement(self, votes, models_count):
#        ret = []
#        for candidate in votes:
#            ret.append(0.0)
#            lab_count = {}
#            for lab in candidate:
#                lab_count[lab] = lab_count.setdefault(lab, 0) + 1
#
#            # Using vote entropy to measure disagreement
#            for lab in lab_count.keys():
#                ret[-1] -= float(lab_count[lab]) / float(models_count) * \
#                    math.log(float(lab_count[lab]) / float(models_count))
#        return ret
    
#%%

class RandomSampling_(QueryStrategy):

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        
        random_state = kwargs.pop('random_state', None)
        self.random_state_ = seed_random_state(random_state)

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        unlabeled_entry_ids, _ = dataset.get_unlabeled_entries()
        entry_id = unlabeled_entry_ids[
                self.random_state_.randint(0, len(unlabeled_entry_ids))]
        return entry_id

#%%
        
"""QueryByCommittee of the Final ATL RF method
Extension of libact QueryByCommittee Strategy
"""


class SourceQueryByCommittee_(QueryByCommittee):
    
    def teach_students(self):
        """
        Train each model (student) with the labeled data using bootstrap
        aggregating (bagging).
        """
        dataset = self.dataset
        for student in self.students:
            bag = self._labeled_uniform_sample(int(dataset.len_labeled()))
            while bag.get_num_of_labels() != dataset.get_num_of_labels():
                bag = self._labeled_uniform_sample(int(dataset.len_labeled()))
#                LOGGER.warning('There is student receiving only one label,'
#                               're-sample the bag.')
            student.train(bag)
    
    def _kl_divergence_disagreement(self, proba):
        """
        Calculate the Kullback-Leibler (KL) divergence disaagreement measure.
        Parameters
        ----------
        proba : array-like, shape=(n_samples, n_students, n_class)
        Returns
        -------
        disagreement : list of float, shape=(n_samples)
            The kl_divergence of the given probability.
        """
        n_students = np.shape(proba)[1]
        consensus = np.mean(proba, axis=1) # shape=(n_samples, n_class)
        # average probability of each class across all students
        consensus = np.tile(consensus, (n_students, 1, 1)).transpose(1, 0, 2)
        # add eps to proba and consensus in order to avoid division by zero
        eps = 0.0000001
        proba = proba+eps
        consensus = consensus+eps
        kl = np.sum(proba * np.log(proba / consensus), axis=2)
        return np.mean(kl, axis=1)
    
    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
        
        y_transfer_labeled_ids, y_transfer_labeled = dataset.get_transfer_labeled_entries()
        y_transfer_predict_proba_ids, y_transfer_predict_proba = dataset.get_transfer_predict_proba_entries()
        
        assert (~np.array_equal(unlabeled_entry_ids,y_transfer_labeled_ids)) | (~np.array_equal(unlabeled_entry_ids,y_transfer_predict_proba_ids)), 'mask of transfer labels/predict_proba wrong'
        
        if self.disagreement == 'vote':
            # Let the trained students vote for unlabeled data
            # +1 so we can append the transfer label to it before we hand it over for calculating the vote entropy
            votes = np.zeros((len(X_pool), len(self.students)+1))
            for i, student in enumerate(self.students):
                votes[:, i] = student.predict(X_pool)
            # add the transfer_labels as last column 
            votes[:, -1] = y_transfer_labeled
            vote_entropy = self._vote_disagreement(votes)
            
            ask_idx = self.random_state_.choice(
                        np.where(np.isclose(vote_entropy, np.max(vote_entropy)))[0])
            
        elif self.disagreement == 'kl_divergence':
            proba = []
            for student in self.students:
                pred_proba = student.predict_proba(X_pool)
                #if pred_proba.__contains__(0): raise ValueError('{}'.format(student.name))
                proba.append(pred_proba)
                #proba.append(student.predict_proba(X_pool))
            # add the predict proba from the transfer
            proba.append(y_transfer_predict_proba)
            proba = np.array(proba).transpose(1, 0, 2).astype(float)
            #if proba.__contains__(0): raise ValueError
            avg_kl = self._kl_divergence_disagreement(proba)
            ask_idx = self.random_state_.choice(
                    np.where(np.isclose(avg_kl, np.max(avg_kl)))[0])
        else:
            print('Disagremeent measure not implemented')
            return None
        return unlabeled_entry_ids[ask_idx]
    
#%%
        
"""Query by committee for AWTL (incorporating all source data in the labeled set)
This module contains a class that implements Query by committee active learning
algorithm.
"""


class QueryByCommittee_(QueryByCommittee):
    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _vote_disagreement(self, votes):
        """
        Return the disagreement measurement of the given number of votes.
        It uses the vote vote to measure the disagreement.
        Parameters
        ----------
        votes : list of int, shape==(n_samples, n_students)
            The predictions that each student gives to each sample.
        Returns
        -------
        disagreement : list of float, shape=(n_samples)
            The vote entropy of the given votes.
        """
        ret = []
        for candidate in votes:
            ret.append(0.0)
            lab_count = {}
            for lab in candidate:
                lab_count[lab] = lab_count.setdefault(lab, 0) + 1

            # Using vote entropy to measure disagreement
            for lab in lab_count.keys():
                ret[-1] -= lab_count[lab] / self.n_students * \
                    math.log(float(lab_count[lab]) / self.n_students)

        return ret


    def _labeled_uniform_sample(self, sample_size):
        """sample labeled entries uniformly"""
        X, y = self.dataset.get_labeled_entries()
        sample_weights = self.dataset.get_sample_weights()
        samples_idx = [self.random_state_.randint(0, X.shape[0]) for _ in range(sample_size)]
        return AWTLDataset(X[samples_idx], np.array(y)[samples_idx], sample_weights[samples_idx])

    def teach_students(self):
        """
        Train each model (student) with the labeled data using bootstrap
        aggregating (bagging).
        """
        dataset = self.dataset
        for student in self.students:
            bag = self._labeled_uniform_sample(int(dataset.len_labeled()))
            while bag.get_num_of_labels() != dataset.get_num_of_labels():
                bag = self._labeled_uniform_sample(int(dataset.len_labeled()))
                LOGGER.warning('There is student receiving only one label,'
                               're-sample the bag.')
            student.train(bag) #,sample_weight=np.array(bag.get_sample_weights())

    @inherit_docstring_from(QueryStrategy)
    def update(self, entry_id, label):
        # Train each model with newly updated label.
        self.teach_students()

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()

        if self.disagreement == 'vote':
            # Let the trained students vote for unlabeled data
            votes = np.zeros((len(X_pool), len(self.students)))
            for i, student in enumerate(self.students):
                votes[:, i] = student.predict(X_pool)

            vote_entropy = self._vote_disagreement(votes)
            ask_idx = self.random_state_.choice(
                    np.where(np.isclose(vote_entropy, np.max(vote_entropy)))[0])

        elif self.disagreement == 'kl_divergence':
            proba = []
            for student in self.students:
                proba.append(student.predict_proba(X_pool))
            proba = np.array(proba).transpose(1, 0, 2).astype(float)

            avg_kl = self._kl_divergence_disagreement(proba)
            ask_idx = self.random_state_.choice(
                    np.where(np.isclose(avg_kl, np.max(avg_kl)))[0])

        return unlabeled_entry_ids[ask_idx]
    
#%% following code from Anna https://github.com/aprimpeli/UnsupervisedBootAL/blob/master/code/libact_datasetext.py
        
class UnsupervisedPoolDataset(Dataset):
    
    def __init__(self, X=None, y=None, y_unsupervised = None, pool_unsupervised_weights=[], reweight='default'):       
        
        Dataset.__init__(self, X, y)

        if y_unsupervised is None: y_unsupervised = []
        y_unsupervised = np.array(y_unsupervised)   
        
        self._y_unsupervised = y_unsupervised
        self._sample_weights = np.full(len(self._y_unsupervised), 1)
        
        if reweight=='score_based': 
            self._sample_weights = pool_unsupervised_weights #every unsupervised data point is weighted by its distance to the decision boundary
    
    def get_unsupervised_labeled_mask(self):
        return ~np.fromiter((e is None for e in self._y_unsupervised), dtype=bool)
        
    def get_unsupervised_labeled_entries(self):
        return self._X[self.get_unsupervised_labeled_mask()], self._y_unsupervised[self.get_unsupervised_labeled_mask()].tolist()
    
    def get_sample_weights(self):
        return self._sample_weights[self.get_unsupervised_labeled_mask()]

    def len_unsupervised_labeled(self):
        return self.get_unsupervised_labeled_mask().sum()
    
    def update_unsupervised(self, entry_id, new_label):
        self._y_unsupervised[entry_id] = new_label
        
    def update_weights (self, weight):
        self._sample_weights[self.get_labeled_mask()] = weight
    
    def update_single_weight (self, entry_id, weight):
        self._sample_weights[entry_id] = weight
   
    def unsupervised_format_sklearn(self):
        X, y = self.get_unsupervised_labeled_entries()
        return X, np.array(y)
    


    def _vote_disagreement(self, votes, models_count):
        ret = []
        for candidate in votes:
            ret.append(0.0)
            lab_count = {}
            for lab in candidate:
                lab_count[lab] = lab_count.setdefault(lab, 0) + 1

            # Using vote entropy to measure disagreement
            for lab in lab_count.keys():
                ret[-1] -= float(lab_count[lab]) / float(models_count) * \
                    math.log(float(lab_count[lab]) / float(models_count))
        return ret
    
#%%
        
class QueryByCommitteeUBoot_(QueryByCommittee):

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
        
        X_un, y_un = self.dataset.get_unsupervised_labeled_entries()

        if self.disagreement == 'vote':
            # Let the trained students vote for unlabeled data
            votes = np.zeros((len(X_pool), len(self.students)))
            
            for i, student in enumerate(self.students):
                votes[:, i] = student.predict(X_pool)
               

            vote_entropy = self._vote_disagreement(votes)
                   
          
            committee_pool = list(zip(unlabeled_entry_ids, np.array(y_un)[unlabeled_entry_ids], votes, dataset.get_sample_weights()[unlabeled_entry_ids],vote_entropy))
            max_entropy = np.max(vote_entropy)
            
            
            candidates = filter(lambda x, max_entropy=max_entropy: x[4]==max_entropy and len(np.where(x[2]==x[1])[0])<0.5*len(x[2]), committee_pool)

            candidates_ids = list(map(lambda x:x[0], candidates))
            
            if len(candidates_ids)==0 :
            
                ask_idx = self.random_state_.choice(
                    np.where(np.isclose(vote_entropy, np.max(vote_entropy)))[0])
                return unlabeled_entry_ids[ask_idx]
            
            else:  
            
                ask_idx = self.random_state_.choice(candidates_ids)
                return ask_idx
            
        elif self.disagreement == 'kl_divergence':
            proba = []
            for student in self.students:
                proba.append(student.predict_proba(X_pool))
            proba = np.array(proba).transpose(1, 0, 2).astype(float)

            avg_kl = self._kl_divergence_disagreement(proba)
            ask_idx = np.random.choice(
                    np.where(np.isclose(avg_kl, np.max(avg_kl)))[0])
            return unlabeled_entry_ids[ask_idx]