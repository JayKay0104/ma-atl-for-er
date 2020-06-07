# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:02:49 2020

@author: jonas
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.metrics import make_scorer,f1_score,matthews_corrcoef
#from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_validate, StratifiedShuffleSplit, cross_val_score
import random
import support_utils as sup
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
import statistics
import warnings
import itertools
import numpy.matlib

#%%

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
    clf = LogisticRegressionCV(cv=5, solver='liblinear')
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


#%%
    
def calcCorrWithLabel(source_name,target_name,candsets,dense_features_dict):
    dense_feature_key = '_'.join(sorted(set(source_name.split('_')+target_name.split('_'))))
    feature = dense_features_dict[dense_feature_key]
    fig, axes = plt.subplots(1,2,figsize=(16,14),constrained_layout=True)
    source_corr = pd.DataFrame({'corr_with_label':candsets[source_name][feature].corrwith(candsets[source_name]['label']).sort_values(ascending=False)})
    target_corr = pd.DataFrame({'corr_with_label':candsets[target_name][feature].corrwith(candsets[target_name]['label']).sort_values(ascending=False)})
    g_source = sns.heatmap(source_corr,vmin=0, vmax=1,annot=True, fmt=".2g",ax=axes[0],cmap='coolwarm')
    g_source.set_title(source_name)
    g_target = sns.heatmap(target_corr,vmin=0, vmax=1,annot=True, fmt=".2g",ax=axes[1],cmap='coolwarm')
    g_target.set_title(target_name)
    return None

#%% supervised matching utils
# check whether 


#%%
def returnPassiveLearningResultsHoldoutSet(estimator,candset,features,random_state=42,test_size=0.33):
    """
    Train estimator on candset using the features in rel_columns and perform cross validation
    in order to use all instances for training and testing for the model evaluation. 
    
    Estimator: sklearn estimator that shall be used to train a classifier (Exp: LogisticRegression())
    Candset: feature vector of potential correspondences (Exp: ban_half)
    features: Columns (features) that shall be used for training (Exp: list of all similarity features)
    CV: Amount of folds used for cross validation. Default: 10
    Scoring: Scoring function that shall be used to measure performance. Default: 'f1' (F1-score)
    """
    
    X = candset[features].to_numpy()
    y = candset['label'].to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, random_state=random_state,
                                                        stratify=y)
    estimator.fit(X_train,y_train)
    pred = estimator.predict(X_test)
    return f1_score(y_test, pred, pos_label=1, average='binary')

#%%
    
def returnSuperBMsInDict(candsets, estimators, features, random_state=42, test_size=0.33, progress_bar=True):
    """
    For each candest in candsets dictionary caluclate the cross validated performance of each
    estimator provided in estimators dictionary on the columns specified in rel_columns.
    
    Estimators: dictionary of sklearn estimators that shall be used to train a classifier (Exp: {'logreg':LogisticRegression(),'dectree':DecisionTree()})
    Candsets: dictionary of feature vectors (DataFrames) of potential correspondences (Exp: {'ban_half':ban_half,'ban_bx':ban_bx})
    Rel_Columns: Columns (features) that shall be used for training (Exp: list of all similarity features)
    CV: Amount of folds used for cross validation. Default: 10
    Scoring: Scoring function that shall be used to measure performance. Default: 'f1' (F1-score)
    Progress_bar: Boolean if progress bar shall be printed to track progress. Default: True
    """
    d={}
    
    if(progress_bar):
        l = len(candsets.keys())
        for i, candset in enumerate(candsets.keys()):
            for clf in estimators:
                res = returnPassiveLearningResultsHoldoutSet(estimators[clf],candsets[candset],features,random_state,test_size)
                if(candset not in d):
                    d.update({'{}'.format(candset):{clf:res}})
                else:
                    d[candset].update({clf:res})
            # Update Progress Bar
            sup.printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    else:
        for candset in candsets:
            for clf in estimators:
                res = returnPassiveLearningResultsHoldoutSet(estimators[clf],candsets[candset],features,random_state,test_size)
                if(candset not in d):
                    d.update({'{}'.format(candset):{clf:res}})
                else:
                    d[candset].update({clf:res})
    return d

##%% Unsupervised Matching
#
#def elbow_threshold(similarities, labels):
#    """
#    This code is taken from Anna Primpeli's code at https://github.com/aprimpeli/UnsupervisedBootAL/blob/master/code/similarityutils.py 
#    and she has it from https://dataplatform.cloud.ibm.com/analytics/notebooks/54d79c2a-f155-40ec-93ec-ed05b58afa39/view?access_token=6d8ec910cf2a1b3901c721fcb94638563cd646fe14400fecbb76cea6aaae2fb1 (accessed:12.09.2019)
#    The function returns the elbow_threshold
#    """
#    sim_list = list(similarities)
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
#
#    #print "Knee of the curve is at index =",idxOfBestPoint
#    #print "Knee value =", similarities[idxOfBestPoint]
#       
#    return similarities[idxOfBestPoint],idxOfBestPoint

#%%

# help function for elbow threshold calculation
def returnSortedDataFrame(feature_vector, label='label', agg_sim='sum_weighted_sim', ids=[], final_id='id'):
    """
    Sorts the sum of the (weighted) similarity scores and returns a dataframe containing label, ids, sim, and idx.
    
    Parameters:
    feature_vector: Dataframe where the data is stored
    label: String with the column name where the labels are stored
        Default: 'label'
    ids: When in the dataframe currently no ID for the pot. corr. are stored but rather two single ids for
         each datasource then the two column names can be handed in a list as this argument and the two IDs
         get combined to a single ID (which will be stored in the column with the name of the String from
         final_id argument). If the dataframe already has an ID that identifies pot. corr. an empty list
         can be handed to ids and the final_id argument needs to specify the correct name of the ID
        Default: []
    final_id: Name of the single ID which is the combination of the IDs from the two datasources. If the
              dataframe already has this final_id and does not need to be created based on the two single 
              IDs then the ids argument needs to be set to an empty list and final_id needs to have the 
              name of the column where the ID is stored. Otherwise final_id is just the name of the column
              where the newly created ID will be stored.
    agg_sim: Name of the column where the sum of the (weighted) similarity scores is stored.
    """
    if(len(ids)==2):    # in the dataframe an ID for each source is stored, hence we combine them to one
        feature_vector[final_id] = feature_vector.apply(lambda row: '{}_{}'.format(row[ids[0]],row[ids[1]]),axis=1)
    
    sorted_data = list(zip(feature_vector[label], feature_vector[final_id], feature_vector[agg_sim], np.arange(feature_vector[final_id].size)))
    random.Random(1).shuffle(sorted_data)
    sorted_data.sort(key = lambda t: t[2])
    sorted_label,sorted_ids,sorted_sim,sorted_idx = zip(*sorted_data)
    df_sorted = pd.DataFrame({'label':sorted_label,'ids':sorted_ids,'sim':sorted_sim,'idx':sorted_idx})
    return df_sorted

#%%
    
def returnUnsuperBMsInDict(candsets, label='label', test_size=0.33, random_state=42):
    """
    Calculate unsupervised results using the elbow threshold on the sum of weighted similarities (density).
    Returns dictionary with the results of form {'ban_half':0.7316810344827587,'bx_half': 0.6263565891472868,...}
    
    Candsets: dictionary of feature vectors (DataFrames) of potential correspondences (Exp: {'ban_half':ban_half,'ban_bx':ban_bx})
    Label: Column name where the true labels are stored. Default: 'label'
    test_size: in order to create comparable results this should not be changes unless for each experiment it gets changed in the same way.
        Default: 0.33
    random_state: in order to create comparable results this should not be changes unless for each experiment it gets changed in the same way.
        Default: 42
    """
    
    d={}

    
    agg_sim='sum_weighted_sim'
    
    #l=len(candsets.keys())
    #sup.printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    for i, candset in enumerate(candsets.keys()):
        
#            X = candsets[candset].to_numpy()
#            y = candsets[candset]['label'].to_numpy()
        
        # make the same train/test split
        X_train, X_test, y_train, y_test = train_test_split(candsets[candset],candsets[candset][label],
                                                            test_size=test_size, random_state=random_state,
                                                            stratify=candsets[candset][label])
        X_test = X_test.copy()
        X_test[agg_sim] = calcWeightedSumOfSimScores(X_test)
        X_test[label]=y_test.copy()
        #print(X_test.head())
        X_test.sort_values(by=agg_sim,axis=0,ascending=True,inplace=True)
        #Sprint(X_test.head())
        #sorted_df = returnSortedDataFrame(X_test,label=label,agg_sim=agg_sim,ids=ids,final_id=final_id)
        elbow_th, index = elbow_threshold(np.array(X_test[agg_sim]))
        print('Elbow_Threshold: {} and Index: {}'.format(elbow_th,index))
        
        elb_labels = np.zeros(X_test.iloc[:index].shape[0])
        elb_labels = np.append(elb_labels,np.ones(X_test.iloc[index:].shape[0]))
        
        #elbow_label = X_test[agg_sim].apply(lambda x: 1 if (x>=elbow_th) else 0)
        
        res = f1_score(X_test[label],elb_labels)
        d.update({candset:res})
        print('{} und {}'.format(candset,res))
        # Update Progress Bar
        #sup.printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    return d

#%%

def calcWeightedSumOfSimScores(feature_vector):
    
    columns = [x for x in feature_vector.columns.tolist() if 'sim' in x]
    if('sum_weighted_sim' in columns):
        columns.remove('sum_weighted_sim')
        print('sum_weighted_sim was in columns and hence got removed. Only the non-aggregated similarity scores containing sim as substring\
              are consider')
    #print(columns)
    rel_columns = feature_vector[columns]
    rel_columns = rel_columns.replace(-1,np.nan)
    
    column_weights = []
    for column in rel_columns:
        nan_values = rel_columns[column].isna().sum()   # get amount of missing values in column
        ratio = float(nan_values)/float(len(rel_columns[column]))
        column_weights.append(1.0-ratio)
    
    #print(column_weights)
    #logger.debug(column_weights)
    weighted_columns = rel_columns*column_weights
    #logger.debug(weighted_columns.iloc[0])
    
    rel_columns_sum = weighted_columns.sum(axis=1, skipna=True)
    rel_columns_mean = rel_columns_sum/len(rel_columns.columns)
          
    #rescale
    sum_weighted_similarity = np.interp(rel_columns_mean, (rel_columns_mean.min(), rel_columns_mean.max()), (0, +1))
    
    return sum_weighted_similarity

#%%
    
def elbow_threshold(sorted_similarities):
    """
    This code is taken from Anna Primpeli's code at https://github.com/aprimpeli/UnsupervisedBootAL/blob/master/code/similarityutils.py 
    and she has it from https://dataplatform.cloud.ibm.com/analytics/notebooks/54d79c2a-f155-40ec-93ec-ed05b58afa39/view?access_token=6d8ec910cf2a1b3901c721fcb94638563cd646fe14400fecbb76cea6aaae2fb1 (accessed:12.09.2019)
    The function returns the elbow_threshold
    """
    sim_list = list(sorted_similarities)
    nPoints = len(sim_list)
    allCoord = np.vstack((range(nPoints), sim_list)).T
    
    firstPoint = allCoord[0]
    # get vector between first and last point - this is the line
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    idxOfBestPoint = np.argmax(distToLine)
       
    return sorted_similarities[idxOfBestPoint],idxOfBestPoint


#%%
def plotDistOfFeature(source_name,target_name,features,candsets):
    n_rows = round(len(features)/2+0.5)
    source = candsets[source_name]
    target = candsets[target_name]
    if(len(features)==1):
        fig,ax = plt.subplots(figsize=(16,8))
        feature = features[0]
        sns.distplot(source[feature],ax=ax,label=source_name)
        sns.distplot(target[feature],ax=ax,label=target_name)
        ax.set_title('Distribution of feature {}'.format(feature))
        ax.legend()
    elif(len(features)==2):
        fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(16,10))
        for i,fea in enumerate(features):
            sns.distplot(source[fea],ax=ax[i],label=source_name)
            sns.distplot(target[fea],ax=ax[i],label=target_name)
            ax[i].set_title('Distribution of feature {}'.format(fea))
            ax[i].legend()
    else:
        fig,ax = plt.subplots(nrows=n_rows,ncols=2,figsize=(16,10))
        k = 0
        for i,fea in enumerate(features):
            sns.distplot(source[fea],ax=ax[k,i%2],label=source_name)
            sns.distplot(target[fea],ax=ax[k,i%2],label=target_name)
            ax[k,i%2].set_title('Distribution of feature {}'.format(fea))
            ax[k,i%2].legend()
            if(i%2==1):
                k += 1
    plt.tight_layout()
    return None

#%%
def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys

def plotECDFsForFeatures(source_name,target_name,features,candsets):
    n_rows = round(len(features)/2+0.5)
    source = candsets[source_name]
    target = candsets[target_name]
    if(len(features)==1):
        fig,ax = plt.subplots(figsize=(16,8))
        x_source,y_source = ecdf(source[features[0]])
        x_target,y_target = ecdf(target[features[0]])
        ax.plot(x_source,y_source,label=source_name)
        ax.plot(x_target,y_target,label=target_name)
        ax.set_title('ecdfs of feature {}'.format(features[0]))
        ax.legend()
    elif(len(features)==2):
        fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(16,10))
        for i,fea in enumerate(features):
            x_source,y_source = ecdf(source[fea])
            x_target,y_target = ecdf(target[fea])
            ax[i].plot(x_source,y_source,label=source_name)
            ax[i].plot(x_target,y_target,label=target_name)
            ax[i].set_title('ecdfs of feature {}'.format(fea))
            ax[i].legend()
    else:
        fig,ax = plt.subplots(nrows=n_rows,ncols=2,figsize=(16,10))
        k = 0
        for i,fea in enumerate(features):
            x_source,y_source = ecdf(source[fea])
            x_target,y_target = ecdf(target[fea])
            ax[k,i%2].plot(x_source,y_source,label=source_name)
            ax[k,i%2].plot(x_target,y_target,label=target_name)
            ax[k,i%2].set_title('ecdfs of feature {}'.format(fea))
            ax[k,i%2].legend()
            if(i%2==1):
                k += 1
    plt.tight_layout()
    return None

#%%
# PROFILING DIMENSIONS
#%% corner case identification approach as explained in Primpeli et al 2020
    
# code from Anna Primpeli and taken from her gihub: https://github.com/aprimpeli/EntityMatchingTaskProfiler/blob/master/profiling.py
# slightly modified to fit here
#def getCornerCaseswithOptimalThreshold(candsets, candset_name, attributes):
#    candset = candsets[candset_name]
#    
#    positives = copy.copy(candset[candset['label']==True])
#    negatives = copy.copy(candset[candset['label']==False])
#    
#    positives = positives.replace(-1, 0)
#    negatives = negatives.replace(-1, 0)
#    
#    positive_values = positives[attributes].mean(axis=1).values
#    negative_values = negatives[attributes].mean(axis=1).values
#    
#    thresholds = []
#    fp_fn = []
#    for t in np.arange(0.0, 1.0, 0.01):
#        fn = len(np.where(positive_values<t)[0])
#        fp = len(np.where(negative_values>=t)[0])
#        thresholds.append(t)
#        fp_fn.append(fn+fp)
#    #optimal_threshold = thresholds[fp_fn.index(min(fp_fn))]
#    hard_cases = min(fp_fn)
#    groups_positives = positives[attributes].groupby(attributes).size().reset_index()
#   
#    return hard_cases/len(positive_values),groups_positives.shape[0]/len(positive_values)


def getCornerCaseswithOptimalThreshold(feature_vector, attributes):
    #print(feature_vector.iloc[0])
    positives = feature_vector[feature_vector['label']==1].copy()
    negatives = feature_vector[feature_vector['label']==0].copy()
    
    positives = positives.replace(-1, 0)
    negatives = negatives.replace(-1, 0)
    
    positive_values = positives[attributes].mean(axis=1).values
    negative_values = negatives[attributes].mean(axis=1).values
    
    thresholds = []
    fp_fn = []
    for t in np.arange(0.0, 1.0, 0.01):
        fn = len(np.where(positive_values<t)[0])
        fp = len(np.where(negative_values>=t)[0])
        thresholds.append(t)
        fp_fn.append(fn+fp)
    #optimal_threshold = thresholds[fp_fn.index(min(fp_fn))]
    hard_cases = min(fp_fn)
    groups_positives = positives[attributes].groupby(attributes).size().reset_index()
   
    return hard_cases/len(positive_values),groups_positives.shape[0]/len(positive_values)

#%%
def get_model_importances(model,classifierName=None):
 
    if classifierName == 'logr':
        importances = model.coef_.ravel()
    elif classifierName == 'svm':
        if model.kernel != 'linear':
            print("Cannot print feature importances without a linear kernel")
            return
        else: importances = model.coef_.ravel()
    else:
        importances = model.feature_importances_
    
    return importances

#%%
    
def showFeatureImportances(column_names, model, classifierName):
      
    importances = get_model_importances(model, classifierName)
       
    column_names = [c.replace('<http://schema.org/Product/', '').replace('>','') for c in column_names]
    sorted_zipped = sorted(list(zip(column_names, importances)), key = lambda x: x[1], reverse=True)[:50]
   
    plt.figure(figsize=(18,3))
    plt.title('Feature importances for classifier %s (max. top 50 features)' % classifierName)
    plt.bar(range(len(sorted_zipped)), [val[1] for val in sorted_zipped], align='center', width = 0.8)
    plt.xticks(range(len(sorted_zipped)), [val[0] for val in sorted_zipped])
    plt.xticks(rotation=90)
    plt.show() 
    features_in_order = [val[0] for val in sorted_zipped]
    feature_weights_in_order = [round(val[1],2) for val in sorted_zipped]
    return features_in_order,feature_weights_in_order


#%%
    
def get_cor_attribute(common_attributes, pairwiseatt):
    for c_att in common_attributes:
        if  pairwiseatt.startswith(c_att): 
            return c_att


#%%
    
def getFeatureDensities(feature_vector, common_attributes, show=False):
    feature_vector_with_nulls = copy.copy(feature_vector)
    feature_vector_with_nulls = feature_vector_with_nulls.replace({-1: None})
    #non_null_values = feature_vector_with_nulls.count()
    density = round(feature_vector_with_nulls.count()/len(feature_vector_with_nulls.index),2)
    visited = []
    overall_density = []
    if show:
        print("*****Feature densities*****")
    for feat in feature_vector_with_nulls.columns:
        if (feat not in ['source_id', 'target_id', 'pair_id', 'label']):
            for common_attr in common_attributes:
                if (feat.startswith(common_attr) and common_attr not in visited):
                    visited.append(common_attr)
                    overall_density.append(density[feat])
                    if show: print(common_attr+": "+str(density[feat]))
    return statistics.mean(overall_density)

#%%
    
def getAvgLength(feature_vector, column, mode):
    column_values= copy.copy(feature_vector[column])
    column_values.fillna('nan')
    lengths = []       
    for i in column_values.values:
        if i!='nan': 
            if mode == 'tokens' : lengths.append(len(str(i)) )
            elif mode == 'words' : lengths.append(len(str(i).split()))

    avg = 0 if len(lengths) == 0 else round(float(sum(lengths) / len(lengths)),2)
    return avg

#%%
    
class DataProfiling():
    
    def __init__(self, ds1, ds2, candset, features, common_attributes_names, datatype_dict):
        
        #ds1 and ds2 need to be in alphabetic order. so f.i. ds1 = 'ban' and ds2 = 'half'
        self.ds1 = ds1.copy()
        self.ds2 = ds2.copy()
        self.feature_vector = candset[features+['label']].copy()
        # label is 0 for non-matches and 1 for matches
        # id is either ids or id
        if('id' not in candset.columns):
            ids = 'ids'
        else:
            ids = 'id'
        gs = candset[[ids,'label']].copy()
        # the record ids are incorporated in the id (e.g. 'ban_2132_half_123' where ban_2132 is id of record in ds1 and half_123 od record from ds2)
        gs['ds1'] = gs[ids].apply(lambda s: '_'.join(s.split('_')[:2]))
        gs['ds2'] = gs[ids].apply(lambda s: '_'.join(s.split('_')[2:]))
        gs.drop(ids,axis=1,inplace=True)
        self.gs = gs
        
        self.datatype_dict = datatype_dict
        self.common_attributes_names = common_attributes_names
               
        self.important_attr_names = None
        self.important_features = None
        #self.label_attr = self.metadata_mt.get("primattr")
        
        self.ds1_subset=ds1[ds1.filter(regex='id').isin(gs['ds1'])]
        self.ds2_subset=ds2[ds2.filter(regex='id').isin(gs['ds2'])]
        self.ds1_subset_match=ds1[ds1.filter(regex='id').isin(gs[gs['label']==1]['ds1'])]
        self.ds2_subset_match=ds2[ds2.filter(regex='id').isin(gs[gs['label']==1]['ds2'])]
        self.gs_prediction_scores = None
        
        self.dict_summary =  {}
        self.dict_important_features_profiling = {}
        self.dict_profiling_features = {}
        
    #returns profiling features for matching task - considers only the source and target subset that appears in the correspondences     
    def getSummaryFeatures(self):
        try:
            #general
            self.dict_summary['#ds1'] = self.ds1.shape[0]
            self.dict_summary['#ds2'] = self.ds2.shape[0]
            count_record_pairs = self.gs.shape[0]
            self.dict_summary['#record_pairs'] = count_record_pairs
            self.dict_summary['#attr'] = len(self.common_attributes_names) 
            self.dict_summary['#non-match'] = len(self.gs[self.gs['label']==0])
            self.dict_summary['#match'] = sum(self.gs['label'])

            #self.dict_summary['ratio_pos'] = len(self.gs_feature_vector[self.gs_feature_vector['label']==True])/count_record_pairs
            #self.dict_summary['ratio_neg'] = len(self.gs_feature_vector[self.gs_feature_vector['label']==False])/count_record_pairs
            short_string_attr = []
            #long_string_attr = []
            num_attr = []
            date_attr = []
            for attr in self.common_attributes_names:
                if attr+"_lev_sim" in self.feature_vector.columns : short_string_attr.append(attr)
                #elif attr+"_cosine_tfidf"  in self.feature_vector.columns : long_string_attr.append(attr) 
                elif attr+"_num_sim" in self.feature_vector.columns : num_attr.append(attr)
                elif attr+"_years_sim" in self.feature_vector.columns : date_attr.append(attr)
            
            self.dict_summary['#short_string_attr'] = len(short_string_attr)
            #self.dict_summary['#long_string_attr'] = len(long_string_attr)
            self.dict_summary['#numeric_attr'] = len(num_attr)
            self.dict_summary['#date_attr'] = len(date_attr)

            #density features
            self.dict_summary['avg_density_all'] = getFeatureDensities(self.feature_vector, self.common_attributes_names)
            #self.dict_summary['density_label'] = getFeatureDensities(self.feature_vector, [self.label_attr])
           
        except:  import pdb; pdb.set_trace();
        
        
    def getProfilingFeatures(self):
        #get identifying features
        #we drop the cosine_tfidf because we want to get single attribute related features (and importances)
        X = self.feature_vector.copy()
        y = self.feature_vector['label'].copy()
        X.drop('label',axis=1,inplace=True)
        #y = self.gs['label']
        
        clf = RandomForestClassifier(random_state=42, min_samples_leaf=2)
        model = clf.fit(X,y)     
        features_in_order, feature_weights = showFeatureImportances(X.columns.values,model,'rf') 
        # all features that are relevant for the matching
        self.dict_profiling_features['matching_relevant_features'] = []
        matching_relevant_attributes = []
        for feat, weight in zip(features_in_order, feature_weights):
            if weight>0: 
                self.dict_profiling_features['matching_relevant_features'].append(feat)
                matching_relevant_attributes.append(get_cor_attribute(self.common_attributes_names,feat))
                
        self.dict_profiling_features['matching_relevant_attributes_datatypes'] = self.getAttributeDistinctDatatypes(matching_relevant_attributes)
        self.dict_profiling_features['matching_relevant_attributes'] = set(matching_relevant_attributes)
        self.dict_profiling_features['matching_relevant_attributes_count'] = len(set(matching_relevant_attributes))
        self.dict_profiling_features['matching_relevant_attributes_density'] = round(getFeatureDensities(self.feature_vector, self.dict_profiling_features['matching_relevant_features']),2)
        #max results
        xval_scoring = {'f1_score' : make_scorer(f1_score)}       
        max_result = cross_validate(clf, X, y, cv=StratifiedShuffleSplit(n_splits=4,random_state =1),  scoring=xval_scoring, n_jobs=-1)
        max_f1_score = round(np.mean(max_result['test_f1_score']),2)
        
        #gather features that are relevant for 95% of the max f1 score
        sub_result = 0.0
        for i in range(1,len(features_in_order)+1):
            results_subvector = cross_validate(clf, X[features_in_order[:i]], y, cv=StratifiedShuffleSplit(n_splits=4,random_state =1),  scoring=xval_scoring, n_jobs=-1)
            sub_result = round(np.mean(results_subvector ['test_f1_score']),2)
            if (sub_result>0.95*max_f1_score): break;
        
        important_features = features_in_order[:i]
    
        self.dict_profiling_features['top_matching_relevant_features_count'] = len(important_features)
        self.dict_profiling_features['F1_xval_max'] = max_f1_score
        self.dict_profiling_features['F1_xval_top_matching_relevant_features'] = sub_result
        self.dict_profiling_features['top_matching_relevant_features'] = important_features
        mapped_ident_features = []
        for attr in important_features:  mapped_ident_features.append(get_cor_attribute(self.common_attributes_names,attr))
        self.dict_profiling_features['top_relevant_attributes'] =  set(mapped_ident_features)
        self.dict_profiling_features['top_relevant_attributes_datatypes'] =  self.getAttributeDistinctDatatypes(set(mapped_ident_features))
        self.dict_profiling_features['top_relevant_attributes_count'] = len(self.dict_profiling_features['top_relevant_attributes'])
        self.dict_profiling_features['top_relevant_attributes_density']=round(getFeatureDensities(self.feature_vector, important_features),2)
        
        avg_length_tokens_ident_feature = []
        avg_length_words_ident_feature = []
        for attr in self.dict_profiling_features['top_relevant_attributes']: 
            #check if it is string
            if (self.datatype_dict[attr]=='str'):
                avg_length_tokens_ident_feature.append(np.mean([getAvgLength(self.ds1_subset, attr, 'tokens'), getAvgLength(self.ds2_subset, attr, 'tokens')]))
                avg_length_words_ident_feature.append(np.mean([getAvgLength(self.ds1_subset, attr, 'words'), getAvgLength(self.ds2_subset, attr, 'words')]))
        
        self.dict_profiling_features['avg_length_tokens_top_relevant_attributes'] = round(sum(avg_length_tokens_ident_feature),2)
        self.dict_profiling_features['avg_length_words_top_relevant_attributes'] = round(sum(avg_length_words_ident_feature),2)

        
        #corner cases
        interstingness,uniqueness = getCornerCaseswithOptimalThreshold(self.feature_vector,important_features)
        self.dict_profiling_features['corner_cases_top_matching_relevant_features'] = round(interstingness,2)
        self.dict_profiling_features['avg_uniqueness_top_matching_relevant_features'] = round(uniqueness,2)
        
        return None
    
    def getAttributeDistinctDatatypes(self, attr_list):
        dt=[]
        for att in attr_list:
            dt.append(self.datatype_dict[att])
        return set(dt)