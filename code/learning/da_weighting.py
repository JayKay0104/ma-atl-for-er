# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:19:21 2020

@author: jonas
"""

import numpy as np
from sklearn.model_selection import cross_val_predict
import scipy.stats as st
from scipy.spatial.distance import cdist
#import sklearn as sk
#from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
#from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, \
#    RidgeClassifier, RidgeClassifierCV

from cvxopt import matrix, solvers


#%%
def getSampleWeightsOfDomainAdaptation(X_source, X_target, weighting):
    if(weighting == 'lr_predict_proba'):
        sample_weight = iwe_BasedOnPredictProba(X_source, X_target,'lr')
    elif(weighting == 'lrcv_predict_proba'):
        sample_weight = iwe_BasedOnPredictProba(X_source, X_target,'lrcv')
    elif(weighting == 'xgb_predict_proba'):
        sample_weight = iwe_BasedOnPredictProba(X_source, X_target,'xgb')
    elif(weighting == 'lr_disc'):
        sample_weight = iwe_logistic_discrimination(X_source, X_target, l2=0.01)
    elif(weighting == 'ratio_gaussians'):
        sample_weight = iwe_ratio_gaussians(X_source, X_target)
    elif(weighting == 'kmm'):
        sample_weight = iwe_kernel_mean_matching(X_source, X_target, kernel_type='rbf', bandwidth=1)
    elif(weighting == 'nn'):
        # fixed parameters smoothing and clip. Need to be changed in the code.
        sample_weight = iwe_nearest_neighbours(X_source, X_target, smoothing=True, clip=1)
    else: 
        raise ValueError
    return sample_weight

#%%
    
def iwe_BasedOnPredictProba(X,Z,classifier='lr'):
    """
    Estimate importance weights based on predict_proba of classifier
    The source (X) and target (Z) get combined and a label 
    (source label 1 and target label 0) indicating from where an instance 
    is coming from is created. Then the classifier is trained to distinguish 
    the instances between coming from source or target. The predict_proba that
    a source sample has for label 0 (i.e. coming from target) is used as weight 
    for that sample.
    A source sample where the predict_proba that it is coming from the target is
    high (close to 1) hence gets a higher weight than a source sample where the 
    classifier is quite sure that it is coming from the source and not the target.
    Such a source sample probably does not seem to represent the distribution or
    characteristics of the target samples (hence the small weight).
    Parameters
    ----------
    X : array
        source data (N samples by D features)
    Z : array
        target data (M samples by D features)
    Classifier : String (either 'lr','lrcv', or 'xgb')
    
    Returns
    -------
    array
        importance weights (N samples by 1)
    """
    # Data shapes
    N, DX = X.shape
    M, DZ = Z.shape

    # Assert equivalent dimensionalities
    if not DX == DZ:
        raise ValueError('Dimensionalities of X and Z should be equal.')

    # Make label for source label=1 and target label=0
    y = np.concatenate([np.ones(N),np.zeros(M)])

    # Concatenate data
    XZ = np.concatenate((X, Z), axis=0)

    if(classifier=='lr'):
        clf = LogisticRegression(random_state=42,solver='liblinear',max_iter=1000,class_weight='balanced')
    elif(classifier=='lrcv'):
        clf = LogisticRegressionCV(random_state=42,cv=5,solver='liblinear',max_iter=1000,class_weight='balanced')
    elif(classifier=='xgb'):
        clf = xgb.XGBClassifier(random_state=42,objective="binary:logistic")
    else:
        raise ValueError('Classifier not defined')
    # reload source dataset

    # calculate predict_proba (that will be used as sample weights)
    proba = cross_val_predict(clf, XZ, y, cv=5, method='predict_proba')
    # posterior of classifier that instance is coming from target (which is class 0)
    # even though it is an acutal source instance
    # the intention behind is, that if the classifier considers an actual source instance
    # as coming from the target with high probability it will be more valuable for
    # the transfer as it is similar to the target instances-
    # only get the predict_probas for label = 0 (target) of the source instances (hence up to N)
    weights_source = proba[:N,0]
    return weights_source


#### FOLLOWING CODE FROM https://github.com/wmkouw/libTLDA/blob/master/libtlda/iw.py #######
#### it belongs to the libtlda domain adaptation package. Slightly modefied to fit the requirements here
#### it was done so that the whole package due to version conflicts of packages does not need to be installed.

#%%
def is_pos_def(X):
    """Check for positive definiteness."""
    return np.all(np.linalg.eigvals(X) > 0)
    
def iwe_ratio_gaussians(X, Z):
    """
    Estimate importance weights based on a ratio of Gaussian distributions.
    Parameters
    ----------
    X : array
        source data (N samples by D features)
    Z : array
        target data (M samples by D features)
    Returns
    -------
    iw : array
        importance weights (N samples by 1)
    """
    # Data shapes
    N, DX = X.shape
    M, DZ = Z.shape

    # Assert equivalent dimensionalities
    if not DX == DZ:
        raise ValueError('Dimensionalities of X and Z should be equal.')

    # Sample means in each domain
    mu_X = np.mean(X, axis=0)
    mu_Z = np.mean(Z, axis=0)

    # Sample covariances
    Si_X = np.cov(X.T)
    Si_Z = np.cov(Z.T)

    # Check for positive-definiteness of covariance matrices
    if not (is_pos_def(Si_X) or is_pos_def(Si_Z)):
        print('Warning: covariate matrices not PSD.')

        regct = -6
        while not (is_pos_def(Si_X) or is_pos_def(Si_Z)):
            print('Adding regularization: ' + str(1**regct))

            # Add regularization
            Si_X += np.eye(DX)*10.**regct
            Si_Z += np.eye(DZ)*10.**regct

            # Increment regularization counter
            regct += 1

    # Compute probability of X under each domain
    pT = st.multivariate_normal.pdf(X, mu_Z, Si_Z)
    pS = st.multivariate_normal.pdf(X, mu_X, Si_X)

    # Check for numerical problems
    if np.any(np.isnan(pT)) or np.any(pT == 0):
        raise ValueError('Source probabilities are NaN or 0.')
    if np.any(np.isnan(pS)) or np.any(pS == 0):
        raise ValueError('Target probabilities are NaN or 0.')

    # Return the ratio of probabilities
    return pT / pS

def iwe_kernel_densities(X, Z):
    """
    Estimate importance weights based on kernel density estimation.
    Parameters
    ----------
    X : array
        source data (N samples by D features)
    Z : array
        target data (M samples by D features)
    Returns
    -------
    array
        importance weights (N samples by 1)
    """
    # Data shapes
    N, DX = X.shape
    M, DZ = Z.shape

    # Assert equivalent dimensionalities
    if not DX == DZ:
        raise ValueError('Dimensionalities of X and Z should be equal.')

    # Compute probabilities based on source kernel densities
    pT = st.gaussian_kde(Z.T).pdf(X.T)
    pS = st.gaussian_kde(X.T).pdf(X.T)

    # Check for numerical problems
    if np.any(np.isnan(pT)) or np.any(pT == 0):
        raise ValueError('Source probabilities are NaN or 0.')
    if np.any(np.isnan(pS)) or np.any(pS == 0):
        raise ValueError('Target probabilities are NaN or 0.')

    # Return the ratio of probabilities
    return pT / pS

def iwe_logistic_discrimination(X, Z, l2):
    """
    Estimate importance weights based on logistic regression.
    Parameters
    ----------
    X : array
        source data (N samples by D features)
    Z : array
        target data (M samples by D features)
    Returns
    -------
    array
        importance weights (N samples by 1)
    """
    # Data shapes
    N, DX = X.shape
    M, DZ = Z.shape

    # Assert equivalent dimensionalities
    if not DX == DZ:
        raise ValueError('Dimensionalities of X and Z should be equal.')

    # Make domain-label variable
    y = np.concatenate((np.zeros((N, 1)),
                        np.ones((M, 1))), axis=0)

    # Concatenate data
    XZ = np.concatenate((X, Z), axis=0)

    # Call a logistic regressor
    if l2:

        lr = LogisticRegression(random_state=42,C=l2, solver='liblinear',max_iter=1000)

    else:
        lr = LogisticRegressionCV(random_state=42,cv=5, solver='liblinear',max_iter=1000)

    # Predict probability of belonging to target using cross-validation
    preds = cross_val_predict(lr, XZ, y[:, 0], cv=5)

    # Return predictions for source samples
    return preds[:N]

def iwe_nearest_neighbours(X, Z,smoothing=True,clip=1):
    """
    Estimate importance weights based on nearest-neighbours.
    Parameters
    ----------
    X : array
        source data (N samples by D features)
    Z : array
        target data (M samples by D features)
    Returns
    -------
    iw : array
        importance weights (N samples by 1)
    """
    # Data shapes
    N, DX = X.shape
    M, DZ = Z.shape

    # Assert equivalent dimensionalities
    if not DX == DZ:
        raise ValueError('Dimensionalities of X and Z should be equal.')

    # Compute Euclidean distance between samples
    d = cdist(X, Z, metric='euclidean')

    # Count target samples within each source Voronoi cell
    ix = np.argmin(d, axis=1)
    iw, _ = np.array(np.histogram(ix, np.arange(N+1)))

    # Laplace smoothing
    if smoothing:
        iw = (iw + 1.) / (N + 1)

    # Weight clipping
    if clip > 0:
        iw = np.minimum(clip, np.maximum(0, iw))

    # Return weights
    return iw

#%%
    
def iwe_kernel_mean_matching(X, Z, kernel_type='rbf', bandwidth=1):
    """
    Estimate importance weights based on kernel mean matching.
    Parameters
    ----------
    X : array
        source data (N samples by D features)
    Z : array
        target data (M samples by D features)
    Returns
    -------
    iw : array
        importance weights (N samples by 1)
    """
    # Data shapes
    N, DX = X.shape
    M, DZ = Z.shape

    # Assert equivalent dimensionalities
    if not DX == DZ:
        raise ValueError('Dimensionalities of X and Z should be equal.')

    # Compute sample pairwise distances
    KXX = cdist(X, X, metric='euclidean')
    KXZ = cdist(X, Z, metric='euclidean')

    # Check non-negative distances
    if not np.all(KXX >= 0):
        raise ValueError('Non-positive distance in source kernel.')
    if not np.all(KXZ >= 0):
        raise ValueError('Non-positive distance in source-target kernel.')

    # Compute kernels
    if kernel_type == 'rbf':
        # Radial basis functions
        KXX = np.exp(-KXX / (2*bandwidth**2))
        KXZ = np.exp(-KXZ / (2*bandwidth**2))

    # Collapse second kernel and normalize
    KXZ = N/M * np.sum(KXZ, axis=1)

    # Prepare for CVXOPT
    Q = matrix(KXX, tc='d')
    p = matrix(KXZ, tc='d')
    G = matrix(np.concatenate((np.ones((1, N)), -1*np.ones((1, N)),
                               -1.*np.eye(N)), axis=0), tc='d')
    h = matrix(np.concatenate((np.array([N/np.sqrt(N) + N], ndmin=2),
                               np.array([N/np.sqrt(N) - N], ndmin=2),
                               np.zeros((N, 1))), axis=0), tc='d')

    # Call quadratic program solver
    sol = solvers.qp(Q, p, G, h)

    # Return optimal coefficients as importance weights
    return np.array(sol['x'])[:, 0]