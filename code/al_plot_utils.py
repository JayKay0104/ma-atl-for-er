# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:08:48 2020

@author: jonas
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_color_codes()
#import glob
#import re
#import json
import itertools
from IPython.display import display
#import copy

# HELP FUNCTIONS for other functions

#%%
#generator function to create tuples of every n element
def group(lst, n):
    for i in range(0, len(lst), n):
        val = lst[i:i+n]
        if len(val) == n:
            yield tuple(val)

#%%
    
def returnDFWithSuper(candsets_super_results,number_of_estimators,filename=None):
    innerkeys =  [innerkey for k,innerdict in candsets_super_results.items() for innerkey, values in innerdict.items()]
    values = [round(value,3) for k,innerdict in candsets_super_results.items() for innerkey, value in innerdict.items()]
    df_super = pd.DataFrame(list(group(values,number_of_estimators)),index=candsets_super_results.keys(),columns=innerkeys[:number_of_estimators])
    ###########################################################################
    # style functions as nested functions
    # function for pandas styler. Highlight all max values
    def highlight_max(s):
        '''
        highlight the maximum in a Series yellow.
        '''
        is_max = s == s.max()
        return ['background-color: #FBFF75' if v else '' for v in is_max]
    ###########################################################################
    # specify the styles for the html output
    styles=[
        {'selector': 'th','props': [
            ('border-style', 'solid'),
            ('border-color', '#D3D3D3'),
            ('vertical-align','top'),
            ('text-align','center')]}]
    html = (df_super.style.\
            apply(highlight_max,axis=1)).set_table_styles(styles).set_precision(3)
    display(html)
    if filename is not None:
        with open('{}.html'.format(filename), 'w') as f:
            f.write(html.render())
    return df_super

#%%
    
def returnDFWithUnsuper(candsets_unsuper_results,filename=None):
    unsuper_ser = pd.Series(candsets_unsuper_results)
    df_unsuper = pd.DataFrame(unsuper_ser,columns=['Elbow Results'])
    ###########################################################################
    # style functions as nested functions
    # function for pandas styler. Highlight all max values
    def highlight_max(s):
        '''
        highlight the maximum in a Series yellow.
        '''
        is_max = s == s.max()
        return ['background-color: #FBFF75' if v else '' for v in is_max]
    ###########################################################################
    # specify the styles for the html output
    styles=[
        {'selector': 'th','props': [
            ('border-style', 'solid'),
            ('border-color', '#D3D3D3'),
            ('vertical-align','top'),
            ('text-align','center')]}]
    html = (df_unsuper.style.\
            apply(highlight_max,axis=0)).set_table_styles(styles).set_precision(3)
    display(html)
    if filename is not None:
        with open('{}.html'.format(filename), 'w') as f:
            f.write(html.render())
    return df_unsuper

#%%
    
def plotATLResults(atl_results,source,target,quota,candsets,candsets_super_results,n,selected_estimator=None,al_results=None,
                   selected_qs=None,errorbars=False,ylim=None,saveFig=True,path_for_output='./graphics/custom_plots/'):
    """
    Plots a customized plot of the results. Here it can be selected which source-target combination, which feature and especially which estimators are requested to be plotted.
    If many estimators were used for the experiments it can be too confusing if all the estimator results are plotted in the graphic (as it is the case with returnTLExpResultPlotsInDict()).
    Hence, here one can select a subset of estimators were the results shall show up in the grapic.
    When specifying the name of source and target, it is important that it needs to be a valid combination. So only when source and target share on original dataset with each other they were
    considered for the Experiments. So source = 'ban_half' and target = 'bx_wor' is not valid but source = 'ban_half' and target = 'wor_half' would be valid because both share 'half' with each other.
    
    @parameters
    atl_results: Dictionary with the results of the ATL Experiments. Either result of returnF1TLResultsFromDictWithPlot() function or importJSONFileInDict() when the results were imported from hard disk.
    source: Specify the name of the source. Exp: 'bx_half'
    target: Specify the name of the source. Exp: 'bx_wor'
    candsets: Dictionary containing all candidate sets (pot. correspondences)
    candsets_unsuper_results: Dictionary containing all the Results of Unsupervised Matching
    candsets_super_results: Dictionary containing all the Results of Supervised Matching for each Estimator
    save_fig: If True the plot gets saved on hard disk
    path_for_output: If save_fig == True then the path to the directory needs to be specified. Default: './graphics/TL/custom_plots/'
    """
    x = np.arange(0,quota+1)
    d = atl_results
    d2 = al_results
    
    keys = list(d.keys())
    if(not isinstance(keys[0],tuple)):
        source_target = '{}_{}'.format(source,target)
    else:
        source_target = (source,target)
    
    qss,estimators = [],[]
    fig,ax = plt.subplots(figsize=(16,8))
    plt.subplots_adjust(right=0.7)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # unsupervised bm plot
    #y_target_unsup_bm = list(itertools.repeat(candsets_unsuper_results[target],x.shape[0]))
    #ax.plot(x,y_target_unsup_bm,linestyle='dashdot',color='g',label='target unsupervised (elbow) benchmark')
    ax.set_xlabel('x target instances used for training',fontsize=14)
    ax.set_ylabel('Avg. F1-Score',fontsize=14)

    if(selected_estimator is None):
        for clf in d[source_target]:
            estimators.append(clf)
            if(clf == 'lr'):
                clf_old = 'logreg'
            elif(clf == 'rf'):
                clf_old = 'randforest'
            elif(clf == 'lrcv'):
                clf_old = 'logregcv'
            elif(clf == 'dt'):
                clf_old = 'dectree'
            else:
                clf_old = clf
            if(selected_qs is None):
                for qs in d[source_target][clf]:
                    qss.append(qs)
                    plt.close()
                    # insert first result of transfer one more time at beginning to also plot point when x = 0.
                    #x_atl_results = d[source_target][clf][qs]['x'].copy()
                    max_quota = d[source_target][clf][qs]['quota']
                    x_atl_results = np.arange(1, max_quota + 1)
                    atl_test_f1_scores = np.array(d[source_target][clf][qs]['test_f1_scores'])
                    y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                    std_atl_results = np.std(atl_test_f1_scores, axis=0)
                    if(errorbars):
                        ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                    label='{}: ATL {} source inst. & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                    else:
                        ax.plot(x_atl_results,y_atl_results,linewidth=2,
                            label='{}: ATL {} source inst. & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                    #ax.plot(x,y_target_results,linewidth=2,linestyle='dashed',
                    #        label='target results ({}) when trained on x target instances and tested on the rest'.format(clf))
                    #if(int(d[source_target][qs]['x_target_exceed'])!=0):
                    #    #idx = x.index(d[source_target][feature][clf]['x_target_exceed'])
                    #    idx = int(d[source_target][qs]['x_target_exceed'])
                    #    if(idx!=0):
                    #        ax.plot(x_atl_results[idx], y_atl_results[idx], 'ro')
                if(al_results is not None):
                    d2 = al_results
                    for qs in d2[target][clf]:
                        plt.close()
                        al_max_quota = d2[target][clf][qs]['quota']
                        x_al_results = np.arange(1, al_max_quota + 1)
                        al_test_f1_scores = np.array(d2[target][clf][qs]['test_f1_scores'])
                        y_al_results = np.mean(al_test_f1_scores,axis=0)
                        n_init_labeled = d2[target][clf][qs]['n_init_labeled']
                        std_al_results = np.std(al_test_f1_scores, axis=0)
                        if(errorbars):
                            ax.errorbar(x_al_results, y_al_results, yerr=std_al_results,
                                        label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
                        else:
                            ax.plot(x_al_results,y_al_results,linewidth=2,linestyle='dashed',
                                    label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
            else:
                for qs in selected_qs:
                    qss.append(qs)
                    plt.close()
                    # insert first result of transfer one more time at beginning to also plot point when x = 0.
                    #x_atl_results = d[source_target][clf][qs]['x'].copy()
                    max_quota = d[source_target][clf][qs]['quota']
                    x_atl_results = np.arange(1, max_quota + 1)
                    atl_test_f1_scores = np.array(d[source_target][clf][qs]['test_f1_scores'])
                    y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                    std_atl_results = np.std(atl_test_f1_scores, axis=0)
                    if(errorbars):
                        ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                    label='{}: ATL {} source inst. & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                    else:
                        ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                label='{}: ATL {} source inst. & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                    #ax.plot(x,y_target_results,linewidth=2,linestyle='dashed',
                    #        label='target results ({}) when trained on x target instances and tested on the rest'.format(clf))
                    #if(int(d[source_target][qs]['x_target_exceed'])!=0):
                    #    #idx = x.index(d[source_target][feature][clf]['x_target_exceed'])
                    #    idx = int(d[source_target][qs]['x_target_exceed'])
                    #    if(idx!=0):
                    #        ax.plot(x_atl_results[idx], y_atl_results[idx], 'ro')
                
                    if(al_results is not None):
                        d2 = al_results
                        al_max_quota = d2[target][clf][qs]['quota']
                        x_al_results = np.arange(1, al_max_quota + 1)
                        al_test_f1_scores = np.array(d2[target][clf][qs]['test_f1_scores'])
                        y_al_results = np.mean(al_test_f1_scores,axis=0)
                        n_init_labeled = d2[target][clf][qs]['n_init_labeled']
                        std_al_results = np.std(al_test_f1_scores, axis=0)
                        if(errorbars):
                            ax.errorbar(x_al_results, y_al_results, yerr=std_al_results,
                                        label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
                            ax.plot(x_al_results,y_al_results,linewidth=2,linestyle='dashed',
                                    label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
            # benchmark plots
            # supervised
            f1_target_bm = candsets_super_results[target][clf_old]
            y_target_sup_bm = list(itertools.repeat(f1_target_bm,len(x)))
            ax.plot(x,y_target_sup_bm,linewidth=3,linestyle='dotted',label='target supervised ({}) benchmark {} instances F1: {:.2f}'.format(clf,candsets[target].shape[0],f1_target_bm))
    else:
        for clf in selected_estimator:
            estimators.append(clf)
            if(clf == 'lr'):
                clf_old = 'logreg'
            elif(clf == 'rf'):
                clf_old = 'randforest'
            elif(clf == 'lrcv'):
                clf_old = 'logregcv'
            elif(clf == 'dt'):
                clf_old = 'dectree'
            else:
                clf_old = clf
            if(selected_qs is None):
                for qs in d[source_target][clf]:
                    qss.append(qs)
                    plt.close()
                    # insert first result of transfer one more time at beginning to also plot point when x = 0.
                    #x_atl_results = d[source_target][clf][qs]['x'].copy()
                    max_quota = d[source_target][clf][qs]['quota']
                    x_atl_results = np.arange(1, max_quota + 1)
                    atl_test_f1_scores = np.array(d[source_target][clf][qs]['test_f1_scores'])
                    y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                    std_atl_results = np.std(atl_test_f1_scores, axis=0)
                    if(errorbars):
                        ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                    label='{}: ATL {} source inst. & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                    else:
                        ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                label='{}: ATL {} source inst. & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                    #ax.plot(x,y_target_results,linewidth=2,linestyle='dashed',
                    #        label='target results ({}) when trained on x target instances and tested on the rest'.format(clf))
                    #if(int(d[source_target][qs]['x_target_exceed'])!=0):
                    #    #idx = x.index(d[source_target][feature][clf]['x_target_exceed'])
                    #    idx = int(d[source_target][qs]['x_target_exceed'])
                    #    if(idx!=0):
                    #        ax.plot(x_atl_results[idx], y_atl_results[idx], 'ro')
                if(al_results is not None):
                    d2 = al_results
                    for qs in d2[target][clf]:
                        plt.close()
                        al_max_quota = d2[target][clf][qs]['quota']
                        x_al_results = np.arange(1, al_max_quota + 1)
                        al_test_f1_scores = np.array(d2[target][clf][qs]['test_f1_scores'])
                        y_al_results = np.mean(al_test_f1_scores,axis=0)
                        n_init_labeled = d2[target][clf][qs]['n_init_labeled']
                        std_al_results = np.std(al_test_f1_scores, axis=0)
                        if(errorbars):
                            ax.errorbar(x_al_results, y_al_results, yerr=std_al_results,
                                        label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
                        else:
                            ax.plot(x_al_results,y_al_results,linewidth=2,linestyle='dashed',
                                    label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
            else:
                for qs in selected_qs:
                    qss.append(qs)
                    plt.close()
                    # insert first result of transfer one more time at beginning to also plot point when x = 0.
                    #x_atl_results = d[source_target][clf][qs]['x'].copy()
                    max_quota = d[source_target][clf][qs]['quota']
                    x_atl_results = np.arange(1, max_quota + 1)
                    atl_test_f1_scores = np.array(d[source_target][clf][qs]['test_f1_scores'])
                    y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                    std_atl_results = np.std(atl_test_f1_scores, axis=0)
                    if(errorbars):
                        ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                    label='{}: ATL {} source inst. & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                    else:
                        ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                label='{}: ATL {} source inst. & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                    #ax.plot(x,y_target_results,linewidth=2,linestyle='dashed',
                    #        label='target results ({}) when trained on x target instances and tested on the rest'.format(clf))
                    #if(int(d[source_target][qs]['x_target_exceed'])!=0):
                    #    #idx = x.index(d[source_target][feature][clf]['x_target_exceed'])
                    #    idx = int(d[source_target][qs]['x_target_exceed'])
                    #    if(idx!=0):
                    #        ax.plot(x_atl_results[idx], y_atl_results[idx], 'ro')
                
                    if(al_results is not None):
                        d2 = al_results
                        al_max_quota = d2[target][clf][qs]['quota']
                        x_al_results = np.arange(1, al_max_quota + 1)
                        al_test_f1_scores = np.array(d2[target][clf][qs]['test_f1_scores'])
                        y_al_results = np.mean(al_test_f1_scores,axis=0)
                        n_init_labeled = d2[target][clf][qs]['n_init_labeled']
                        std_al_results = np.std(al_test_f1_scores, axis=0)
                        if(errorbars):
                            ax.errorbar(x_al_results, y_al_results, yerr=std_al_results,
                                        label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
                        else:
                            ax.plot(x_al_results,y_al_results,linewidth=2,linestyle='dashed',
                                    label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
            # benchmark plots
            # supervised
            f1_target_bm = candsets_super_results[target][clf_old]
            y_target_sup_bm = list(itertools.repeat(f1_target_bm,len(x)))
            ax.plot(x,y_target_sup_bm,linewidth=3,linestyle='dotted',label='target supervised ({}) benchmark {} instances F1: {:.2f}'.format(clf,candsets[target].shape[0],f1_target_bm))
        
            
    # add legend to plot
    ax.legend(fontsize=10)
    # add a text box with info
    info_text = 'ATL was tested on a\nhold-out test set with\n33% of target instances\nand avg. over n={} runs.'.format(n)
    info_text_2 = 'Results of benchmark\nare calculated on\nthe same test set.'
    #info_text_3 = 'The ATL results were\navg. over n={} runs\nusing different random\ntest sets.'.format(n)
    textstr = 'INFO:\n{}\n{}'.format(info_text,info_text_2)
    #textstr = '#Source instances used for training: {}\n{}\nINFO: {}'.format(source.shape[0],info_text_2,info_text)
    ax.text(0.71, 0.85, textstr, transform=fig.transFigure, fontsize=14,verticalalignment='top', bbox=props)
    features_used = 'Only dense features across source and target were used. Lowers risk of negative transfer (finding from TL Experiment)'
    ax.set_title('Active-Transfer Learning (ATL) Results with Naive Transfer (no DA):\nInitially trained on only source {} then iteratively queried target {} instances and added to training set\n{}'.format(source,target,features_used))
    
    if(saveFig):
        fig.savefig('{}{}_{}_{}_{}_n{}.png'.format(path_for_output,source,target,'_'.join(estimators),'_'.join(qss),n),bbox_inches='tight')
    
    if(ylim is not None):
        ax.set_ylim(ylim)
    return fig

#%%
    
def plotAWTLResults(awtl_results,source,target,quota,candsets,candsets_super_results,n,selected_estimator=None,al_results=None,
                   selected_qs=None,selected_weights=None,errorbars=False,ylim=None,saveFig=True,path_for_output='./graphics/custom_plots/'):
    """
    Plots a customized plot of the results. Here it can be selected which source-target combination, which feature and especially which estimators are requested to be plotted.
    If many estimators were used for the experiments it can be too confusing if all the estimator results are plotted in the graphic (as it is the case with returnTLExpResultPlotsInDict()).
    Hence, here one can select a subset of estimators were the results shall show up in the grapic.
    When specifying the name of source and target, it is important that it needs to be a valid combination. So only when source and target share on original dataset with each other they were
    considered for the Experiments. So source = 'ban_half' and target = 'bx_wor' is not valid but source = 'ban_half' and target = 'wor_half' would be valid because both share 'half' with each other.
    
    @parameters
    atl_results: Dictionary with the results of the ATL Experiments. Either result of returnF1TLResultsFromDictWithPlot() function or importJSONFileInDict() when the results were imported from hard disk.
    source: Specify the name of the source. Exp: 'bx_half'
    target: Specify the name of the source. Exp: 'bx_wor'
    candsets: Dictionary containing all candidate sets (pot. correspondences)
    candsets_unsuper_results: Dictionary containing all the Results of Unsupervised Matching
    candsets_super_results: Dictionary containing all the Results of Supervised Matching for each Estimator
    save_fig: If True the plot gets saved on hard disk
    path_for_output: If save_fig == True then the path to the directory needs to be specified. Default: './graphics/TL/custom_plots/'
    """
    x = np.arange(0,quota+1)
    d = awtl_results
    d2 = al_results
    
    keys = list(d.keys())
    if(not isinstance(keys[0],tuple)):
        source_target = '{}_{}'.format(source,target)
    else:
        source_target = (source,target)
    
    qss,estimators = [],[]
    fig,ax = plt.subplots(figsize=(16,8))
    plt.subplots_adjust(right=0.7)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # unsupervised bm plot
    #y_target_unsup_bm = list(itertools.repeat(candsets_unsuper_results[target],x.shape[0]))
    #ax.plot(x,y_target_unsup_bm,linestyle='dashdot',color='g',label='target unsupervised (elbow) benchmark')
    ax.set_xlabel('x target instances used for training',fontsize=14)
    ax.set_ylabel('Avg. F1-Score',fontsize=14)

    if(selected_estimator is None):
        for clf in d[source_target]:
            estimators.append(clf)
            if(clf == 'lr'):
                clf_old = 'logreg'
            elif(clf == 'rf'):
                clf_old = 'randforest'
            elif(clf == 'lrcv'):
                clf_old = 'logregcv'
            elif(clf == 'dt'):
                clf_old = 'dectree'
            else:
                clf_old = clf
            if(selected_qs is None):
                for qs in d[source_target][clf]:
                    qss.append(qs)
                    if selected_weights is None:
                        for weight in d[source_target][clf][qs]:
                            plt.close()
                            # insert first result of transfer one more time at beginning to also plot point when x = 0.
                            #x_atl_results = d[source_target][clf][qs]['x'].copy()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                            label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                    label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            #ax.plot(x,y_target_results,linewidth=2,linestyle='dashed',
                            #        label='target results ({}) when trained on x target instances and tested on the rest'.format(clf))
                            #if(int(d[source_target][qs]['x_target_exceed'])!=0):
                            #    #idx = x.index(d[source_target][feature][clf]['x_target_exceed'])
                            #    idx = int(d[source_target][qs]['x_target_exceed'])
                            #    if(idx!=0):
                            #        ax.plot(x_atl_results[idx], y_atl_results[idx], 'ro')
                    else:
                        for weight in selected_weights:
                            plt.close()
                            # insert first result of transfer one more time at beginning to also plot point when x = 0.
                            #x_atl_results = d[source_target][clf][qs]['x'].copy()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                            label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                        label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                    
                    if(al_results is not None):
                        d2 = al_results
                        plt.close()
                        al_max_quota = d2[target][clf][qs]['quota']
                        x_al_results = np.arange(1, al_max_quota + 1)
                        al_test_f1_scores = np.array(d2[target][clf][qs]['test_f1_scores'])
                        y_al_results = np.mean(al_test_f1_scores,axis=0)
                        n_init_labeled = d2[target][clf][qs]['n_init_labeled']
                        std_al_results = np.std(al_test_f1_scores, axis=0)
                        if(errorbars):
                            ax.errorbar(x_al_results, y_al_results, yerr=std_al_results,
                                        label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
                        else:
                            ax.plot(x_al_results,y_al_results,linewidth=2,linestyle='dashed',
                                    label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
            else:
                for qs in selected_qs:
                    qss.append(qs)
                    if selected_weights is None:
                        for weight in d[source_target][clf][qs]:
                            plt.close()
                            # insert first result of transfer one more time at beginning to also plot point when x = 0.
                            #x_atl_results = d[source_target][clf][qs]['x'].copy()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                            label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                        label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            #ax.plot(x,y_target_results,linewidth=2,linestyle='dashed',
                            #        label='target results ({}) when trained on x target instances and tested on the rest'.format(clf))
                            #if(int(d[source_target][qs]['x_target_exceed'])!=0):
                            #    #idx = x.index(d[source_target][feature][clf]['x_target_exceed'])
                            #    idx = int(d[source_target][qs]['x_target_exceed'])
                            #    if(idx!=0):
                            #        ax.plot(x_atl_results[idx], y_atl_results[idx], 'ro')
                    else:
                        for weight in selected_weights:
                            plt.close()
                            # insert first result of transfer one more time at beginning to also plot point when x = 0.
                            #x_atl_results = d[source_target][clf][qs]['x'].copy()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                            label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                        label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                
                    if(al_results is not None):
                        d2 = al_results
                        al_max_quota = d2[target][clf][qs]['quota']
                        x_al_results = np.arange(1, al_max_quota + 1)
                        al_test_f1_scores = np.array(d2[target][clf][qs]['test_f1_scores'])
                        y_al_results = np.mean(al_test_f1_scores,axis=0)
                        n_init_labeled = d2[target][clf][qs]['n_init_labeled']
                        std_al_results = np.std(al_test_f1_scores, axis=0)
                        if(errorbars):
                            ax.errorbar(x_al_results, y_al_results, yerr=std_al_results,
                                        label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
                            ax.plot(x_al_results,y_al_results,linewidth=2,linestyle='dashed',
                                    label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
            # benchmark plots
            # supervised
            f1_target_bm = candsets_super_results[target][clf_old]
            y_target_sup_bm = list(itertools.repeat(f1_target_bm,len(x)))
            ax.plot(x,y_target_sup_bm,linewidth=3,linestyle='dotted',label='target supervised ({}) benchmark {} instances F1: {:.2f}'.format(clf,candsets[target].shape[0],f1_target_bm))
    else:
        for clf in selected_estimator:
            estimators.append(clf)
            if(clf == 'lr'):
                clf_old = 'logreg'
            elif(clf == 'rf'):
                clf_old = 'randforest'
            elif(clf == 'lrcv'):
                clf_old = 'logregcv'
            elif(clf == 'dt'):
                clf_old = 'dectree'
            else:
                clf_old = clf
            if(selected_qs is None):
                for qs in d[source_target][clf]:
                    qss.append(qs)
                    if selected_weights is None:
                        for weight in d[source_target][clf][qs]:
                            plt.close()
                            # insert first result of transfer one more time at beginning to also plot point when x = 0.
                            #x_atl_results = d[source_target][clf][qs]['x'].copy()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                            label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                        label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            #ax.plot(x,y_target_results,linewidth=2,linestyle='dashed',
                            #        label='target results ({}) when trained on x target instances and tested on the rest'.format(clf))
                            #if(int(d[source_target][qs]['x_target_exceed'])!=0):
                            #    #idx = x.index(d[source_target][feature][clf]['x_target_exceed'])
                            #    idx = int(d[source_target][qs]['x_target_exceed'])
                            #    if(idx!=0):
                            #        ax.plot(x_atl_results[idx], y_atl_results[idx], 'ro')
                    else:
                        for weight in selected_weights:
                            plt.close()
                            # insert first result of transfer one more time at beginning to also plot point when x = 0.
                            #x_atl_results = d[source_target][clf][qs]['x'].copy()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                            label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                        label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                                
                if(al_results is not None):
                    d2 = al_results
                    plt.close()
                    al_max_quota = d2[target][clf][qs]['quota']
                    x_al_results = np.arange(1, al_max_quota + 1)
                    al_test_f1_scores = np.array(d2[target][clf][qs]['test_f1_scores'])
                    y_al_results = np.mean(al_test_f1_scores,axis=0)
                    n_init_labeled = d2[target][clf][qs]['n_init_labeled']
                    std_al_results = np.std(al_test_f1_scores, axis=0)
                    if(errorbars):
                        ax.errorbar(x_al_results, y_al_results, yerr=std_al_results,
                                    label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
                    else:
                        ax.plot(x_al_results,y_al_results,linewidth=2,linestyle='dashed',
                                label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
            else:
                for qs in selected_qs:
                    qss.append(qs)
                    if selected_weights is None:
                        for weight in d[source_target][clf][qs]:
                            plt.close()
                            # insert first result of transfer one more time at beginning to also plot point when x = 0.
                            #x_atl_results = d[source_target][clf][qs]['x'].copy()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                            label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                        label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            #ax.plot(x,y_target_results,linewidth=2,linestyle='dashed',
                            #        label='target results ({}) when trained on x target instances and tested on the rest'.format(clf))
                            #if(int(d[source_target][qs]['x_target_exceed'])!=0):
                            #    #idx = x.index(d[source_target][feature][clf]['x_target_exceed'])
                            #    idx = int(d[source_target][qs]['x_target_exceed'])
                            #    if(idx!=0):
                            #        ax.plot(x_atl_results[idx], y_atl_results[idx], 'ro')
                    else:
                        for weight in selected_weights:
                            plt.close()
                            # insert first result of transfer one more time at beginning to also plot point when x = 0.
                            #x_atl_results = d[source_target][clf][qs]['x'].copy()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                            label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                        label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                
                    if(al_results is not None):
                        d2 = al_results
                        al_max_quota = d2[target][clf][qs]['quota']
                        x_al_results = np.arange(1, al_max_quota + 1)
                        al_test_f1_scores = np.array(d2[target][clf][qs]['test_f1_scores'])
                        y_al_results = np.mean(al_test_f1_scores,axis=0)
                        n_init_labeled = d2[target][clf][qs]['n_init_labeled']
                        std_al_results = np.std(al_test_f1_scores, axis=0)
                        if(errorbars):
                            ax.errorbar(x_al_results, y_al_results, yerr=std_al_results,
                                        label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
                        else:
                            ax.plot(x_al_results,y_al_results,linewidth=2,linestyle='dashed',
                                    label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
            # benchmark plots
            # supervised
            f1_target_bm = candsets_super_results[target][clf_old]
            y_target_sup_bm = list(itertools.repeat(f1_target_bm,len(x)))
            ax.plot(x,y_target_sup_bm,linewidth=3,linestyle='dotted',label='target supervised ({}) benchmark {} instances F1: {:.2f}'.format(clf,candsets[target].shape[0],f1_target_bm))
        
            
    # add legend to plot
    ax.legend(fontsize=10)
    # add a text box with info
    info_text = 'ATL was tested on a\nhold-out test set with\n33% of target instances\nand avg. over n={} runs.'.format(n)
    info_text_2 = 'Results of benchmark\nare calculated on\nthe same test set.'
    #info_text_3 = 'The ATL results were\navg. over n={} runs\nusing different random\ntest sets.'.format(n)
    textstr = 'INFO:\n{}\n{}'.format(info_text,info_text_2)
    #textstr = '#Source instances used for training: {}\n{}\nINFO: {}'.format(source.shape[0],info_text_2,info_text)
    ax.text(0.71, 0.85, textstr, transform=fig.transFigure, fontsize=14,verticalalignment='top', bbox=props)
    features_used = 'Only dense features across source and target were used. Lowers risk of negative transfer (finding from TL Experiment)'
    ax.set_title('Active-Transfer Learning (ATL) Results with DA :\nInitially trained on only source {} then iteratively queried target {} instances and added to training set\n{}'.format(source,target,features_used))
    
    if(saveFig):
        fig.savefig('{}{}_{}_{}_{}_n{}.png'.format(path_for_output,source,target,'_'.join(estimators),'_'.join(qss),n),bbox_inches='tight')
    
    if(ylim is not None):
        ax.set_ylim(ylim)
    return fig

#%%
    
#%%
    
def plotATLRFALUnsupResults(atl_rf_results,source,target,quota,candsets,candsets_super_results,n,warm_start,selected_estimator=None,al_results=None,
                   selected_qs=None,selected_weights=None,errorbars=False,ylim=None,saveFig=True,path_for_output='./graphics/custom_plots/'):
    """
    Plots a customized plot of the results. Here it can be selected which source-target combination, which feature and especially which estimators are requested to be plotted.
    If many estimators were used for the experiments it can be too confusing if all the estimator results are plotted in the graphic (as it is the case with returnTLExpResultPlotsInDict()).
    Hence, here one can select a subset of estimators were the results shall show up in the grapic.
    When specifying the name of source and target, it is important that it needs to be a valid combination. So only when source and target share on original dataset with each other they were
    considered for the Experiments. So source = 'ban_half' and target = 'bx_wor' is not valid but source = 'ban_half' and target = 'wor_half' would be valid because both share 'half' with each other.
    
    @parameters
    atl_results: Dictionary with the results of the ATL Experiments. Either result of returnF1TLResultsFromDictWithPlot() function or importJSONFileInDict() when the results were imported from hard disk.
    source: Specify the name of the source. Exp: 'bx_half'
    target: Specify the name of the source. Exp: 'bx_wor'
    candsets: Dictionary containing all candidate sets (pot. correspondences)
    candsets_unsuper_results: Dictionary containing all the Results of Unsupervised Matching
    candsets_super_results: Dictionary containing all the Results of Supervised Matching for each Estimator
    save_fig: If True the plot gets saved on hard disk
    path_for_output: If save_fig == True then the path to the directory needs to be specified. Default: './graphics/TL/custom_plots/'
    """
    x = np.arange(0,quota+1)
    d = atl_rf_results
    d2 = al_results
    
    if(warm_start):
        warm_start = 'ws'
    else:
        warm_start = 'no ws'
    keys = list(d.keys())
    if(not isinstance(keys[0],tuple)):
        source_target = '{}_{}'.format(source,target)
    else:
        source_target = (source,target)
    
    qss,estimators = [],[]
    fig,ax = plt.subplots(figsize=(20,10))
    plt.subplots_adjust(right=0.7)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # unsupervised bm plot
    #y_target_unsup_bm = list(itertools.repeat(candsets_unsuper_results[target],x.shape[0]))
    #ax.plot(x,y_target_unsup_bm,linestyle='dashdot',color='g',label='target unsupervised (elbow) benchmark')
    ax.set_xlabel('x target instances used for training',fontsize=14)
    ax.set_ylabel('Avg. F1-Score',fontsize=14)
    if(selected_estimator is None):
        for clf in d[source_target]:
            estimators.append(clf)
            if(clf == 'lr'):
                clf_old = 'logreg'
            elif(clf == 'rf'):
                clf_old = 'randforest'
            elif(clf == 'lrcv'):
                clf_old = 'logregcv'
            elif(clf == 'dt'):
                clf_old = 'dectree'
            else:
                clf_old = clf
            if(selected_qs is None):
                for qs in d[source_target][clf]:
                    qss.append(qs)
                    if selected_weights is None:
                        for weight in d[source_target][clf][qs]:
                            plt.close()
                            # insert first result of transfer one more time at beginning to also plot point when x = 0.
                            #x_atl_results = d[source_target][clf][qs]['x'].copy()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                    label='{}: ATL_RF {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                    label='{}: ATL_RF {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            #ax.plot(x,y_target_results,linewidth=2,linestyle='dashed',
                            #        label='target results ({}) when trained on x target instances and tested on the rest'.format(clf))
                            #if(int(d[source_target][qs]['x_target_exceed'])!=0):
                            #    #idx = x.index(d[source_target][feature][clf]['x_target_exceed'])
                            #    idx = int(d[source_target][qs]['x_target_exceed'])
                            #    if(idx!=0):
                            #        ax.plot(x_atl_results[idx], y_atl_results[idx], 'ro')
                    else:
                        for weight in selected_weights:
                            plt.close()
                            # insert first result of transfer one more time at beginning to also plot point when x = 0.
                            #x_atl_results = d[source_target][clf][qs]['x'].copy()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                            label='{}: ATL_RF {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                        label='{}: ATL_RF {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                    
                    if(al_results is not None):
                        d2 = al_results
                        plt.close()
                        al_max_quota = d2[target][clf][qs]['quota']
                        x_al_results = np.arange(1, al_max_quota + 1)
                        al_test_f1_scores = np.array(d2[target][clf][qs]['test_f1_scores'])
                        y_al_results = np.mean(al_test_f1_scores,axis=0)
                        n_init_labeled = d2[target][clf][qs]['n_init_labeled']
                        std_al_results = np.std(al_test_f1_scores, axis=0)
                        if(errorbars):
                            ax.errorbar(x_al_results, y_al_results, yerr=std_al_results,
                                        label='{}: AL_RF {}: init. {} labeled (UnsupBoot) & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
                        else:
                            ax.plot(x_al_results,y_al_results,linewidth=2,linestyle='dashed',
                                    label='{}: AL_RF {}: init. {} labeled (UnsupBoot) & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
            else:
                for qs in selected_qs:
                    qss.append(qs)
                    if selected_weights is None:
                        for weight in d[source_target][clf][qs]:
                            plt.close()
                            # insert first result of transfer one more time at beginning to also plot point when x = 0.
                            #x_atl_results = d[source_target][clf][qs]['x'].copy()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                            label='{}: ATL_RF {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                        label='{}: ATL_RF {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            #ax.plot(x,y_target_results,linewidth=2,linestyle='dashed',
                            #        label='target results ({}) when trained on x target instances and tested on the rest'.format(clf))
                            #if(int(d[source_target][qs]['x_target_exceed'])!=0):
                            #    #idx = x.index(d[source_target][feature][clf]['x_target_exceed'])
                            #    idx = int(d[source_target][qs]['x_target_exceed'])
                            #    if(idx!=0):
                            #        ax.plot(x_atl_results[idx], y_atl_results[idx], 'ro')
                    else:
                        for weight in selected_weights:
                            plt.close()
                            # insert first result of transfer one more time at beginning to also plot point when x = 0.
                            #x_atl_results = d[source_target][clf][qs]['x'].copy()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                            label='{}: ATL_RF {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                        label='{}: ATL_RF {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                
                    if(al_results is not None):
                        d2 = al_results
                        al_max_quota = d2[target][clf][qs]['quota']
                        x_al_results = np.arange(1, al_max_quota + 1)
                        al_test_f1_scores = np.array(d2[target][clf][qs]['test_f1_scores'])
                        y_al_results = np.mean(al_test_f1_scores,axis=0)
                        n_init_labeled = d2[target][clf][qs]['n_init_labeled']
                        std_al_results = np.std(al_test_f1_scores, axis=0)
                        if(errorbars):
                            ax.errorbar(x_al_results, y_al_results, yerr=std_al_results,
                                        label='{}: AL_RF {}: init. {} labeled (UnsupBoot) & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
                            ax.plot(x_al_results,y_al_results,linewidth=2,linestyle='dashed',
                                    label='{}: AL_RF {}: init. {} labeled (UnsupBoot) & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
            # benchmark plots
            # supervised
            f1_target_bm = candsets_super_results[target][clf_old]
            y_target_sup_bm = list(itertools.repeat(f1_target_bm,len(x)))
            ax.plot(x,y_target_sup_bm,linewidth=3,linestyle='dotted',label='target supervised ({}) benchmark {} instances F1: {:.2f}'.format(clf,candsets[target].shape[0],f1_target_bm))
    else:
        for clf in selected_estimator:
            estimators.append(clf)
            if(clf == 'lr'):
                clf_old = 'logreg'
            elif(clf == 'rf'):
                clf_old = 'randforest'
            elif(clf == 'lrcv'):
                clf_old = 'logregcv'
            elif(clf == 'dt'):
                clf_old = 'dectree'
            else:
                clf_old = clf
            if(selected_qs is None):
                for qs in d[source_target][clf]:
                    qss.append(qs)
                    if selected_weights is None:
                        for weight in d[source_target][clf][qs]:
                            plt.close()
                            # insert first result of transfer one more time at beginning to also plot point when x = 0.
                            #x_atl_results = d[source_target][clf][qs]['x'].copy()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                            label='{}: ATL_RF {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                        label='{}: ATL_RF {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            #ax.plot(x,y_target_results,linewidth=2,linestyle='dashed',
                            #        label='target results ({}) when trained on x target instances and tested on the rest'.format(clf))
                            #if(int(d[source_target][qs]['x_target_exceed'])!=0):
                            #    #idx = x.index(d[source_target][feature][clf]['x_target_exceed'])
                            #    idx = int(d[source_target][qs]['x_target_exceed'])
                            #    if(idx!=0):
                            #        ax.plot(x_atl_results[idx], y_atl_results[idx], 'ro')
                    else:
                        for weight in selected_weights:
                            plt.close()
                            # insert first result of transfer one more time at beginning to also plot point when x = 0.
                            #x_atl_results = d[source_target][clf][qs]['x'].copy()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                            label='{}: ATL_RF {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                        label='{}: ATL_RF {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                                
                if(al_results is not None):
                    d2 = al_results
                    plt.close()
                    al_max_quota = d2[target][clf][qs]['quota']
                    x_al_results = np.arange(1, al_max_quota + 1)
                    al_test_f1_scores = np.array(d2[target][clf][qs]['test_f1_scores'])
                    y_al_results = np.mean(al_test_f1_scores,axis=0)
                    n_init_labeled = d2[target][clf][qs]['n_init_labeled']
                    std_al_results = np.std(al_test_f1_scores, axis=0)
                    if(errorbars):
                        ax.errorbar(x_al_results, y_al_results, yerr=std_al_results,
                                    label='{}: AL_RF {}: init. {} labeled (UnsupBoot) & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
                    else:
                        ax.plot(x_al_results,y_al_results,linewidth=2,linestyle='dashed',
                                label='{}: AL_RF {}: init. {} labeled (UnsupBoot) & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
            else:
                for qs in selected_qs:
                    qss.append(qs)
                    if selected_weights is None:
                        for weight in d[source_target][clf][qs]:
                            plt.close()
                            # insert first result of transfer one more time at beginning to also plot point when x = 0.
                            #x_atl_results = d[source_target][clf][qs]['x'].copy()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                            label='{}: ATL_RF {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                        label='{}: ATL_RF {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            #ax.plot(x,y_target_results,linewidth=2,linestyle='dashed',
                            #        label='target results ({}) when trained on x target instances and tested on the rest'.format(clf))
                            #if(int(d[source_target][qs]['x_target_exceed'])!=0):
                            #    #idx = x.index(d[source_target][feature][clf]['x_target_exceed'])
                            #    idx = int(d[source_target][qs]['x_target_exceed'])
                            #    if(idx!=0):
                            #        ax.plot(x_atl_results[idx], y_atl_results[idx], 'ro')
                    else:
                        for weight in selected_weights:
                            plt.close()
                            # insert first result of transfer one more time at beginning to also plot point when x = 0.
                            #x_atl_results = d[source_target][clf][qs]['x'].copy()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                            label='{}: ATL_RF {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                        label='{}: ATL_RF {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                
                    if(al_results is not None):
                        d2 = al_results
                        al_max_quota = d2[target][clf][qs]['quota']
                        x_al_results = np.arange(1, al_max_quota + 1)
                        al_test_f1_scores = np.array(d2[target][clf][qs]['test_f1_scores'])
                        y_al_results = np.mean(al_test_f1_scores,axis=0)
                        n_init_labeled = d2[target][clf][qs]['n_init_labeled']
                        std_al_results = np.std(al_test_f1_scores, axis=0)
                        if(errorbars):
                            ax.errorbar(x_al_results, y_al_results, yerr=std_al_results,
                                        label='{}: AL_RF {}: init. {} labeled (UnsupBoot) & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
                        else:
                            ax.plot(x_al_results,y_al_results,linewidth=2,linestyle='dashed',
                                    label='{}: AL_RF {}: init. {} labeled (UnsupBoot) & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
            # benchmark plots
            # supervised
            f1_target_bm = candsets_super_results[target][clf_old]
            y_target_sup_bm = list(itertools.repeat(f1_target_bm,len(x)))
            ax.plot(x,y_target_sup_bm,linewidth=3,linestyle='dotted',label='target supervised ({}) benchmark {} instances F1: {:.2f}'.format(clf,candsets[target].shape[0],f1_target_bm))
        
            
    # add legend to plot
    ax.legend(fontsize=13)
    # add a text box with info
    features_used = 'Only dense features\nacross source and\ntarget were used.\nLowers risk of negative\ntransfer (finding from\nTL Experiment)'
    info_text = 'ATL was tested on a\nhold-out test set with\n33% of target instances\nand avg. over n={} runs.'.format(n)
    info_text_2 = 'Results of benchmark\nare calculated on\nthe same test set.'
    #info_text_3 = 'The ATL results were\navg. over n={} runs\nusing different random\ntest sets.'.format(n)
    textstr = 'INFO:\n{}\n{}\n{}'.format(features_used,info_text,info_text_2)
    #textstr = '#Source instances used for training: {}\n{}\nINFO: {}'.format(source.shape[0],info_text_2,info_text)
    ax.text(0.71, 0.85, textstr, transform=fig.transFigure, fontsize=14,verticalalignment='top', bbox=props)
    
    ax.set_title('Active-Transfer Learning (ATL_RF {}):\nIncorporating labeled source data then iteratively queried target instances and added to training set\nSource: {} and Target {}'.format(warm_start,source,target),fontsize=14)
    
    if(saveFig):
        fig.savefig('{}{}_{}_{}_{}_n{}.png'.format(path_for_output,source,target,'_'.join(estimators),'_'.join(qss),n),bbox_inches='tight')
    
    if(ylim is not None):
        ax.set_ylim(ylim)
    return fig

#%%
    
def createDFwithAlandATLResults(al_results,atl_results,candsets_super_results,filename='al_and_atl_results'):
    """
    This function creates a DataFrame with the results of each experiment. Per estimator the TL_avg (Transfer Learning Avg Result),
    Tar_max (Target max result when trained on 500 target instances), Tar_exc (Amount of Target Instances needed to exceed TL results),
    when trained on all features and dense features, are reported. Hence, amount of estimators*2*3 columns + 1 for unsupervised results
    of target are in the resulting DataFrame. It also stores a html file where important information is highlighted in the DataFrame.
    
    tl_results: Dictionary containting the results of the Transfer Learning experiements (as output by returnF1TLResultsFromDictWithPlot())
    candsets_unsuper_results: Dictionary containing the unsupervised results which act as benchmark (in the form of {'ban_half':0.732,'bx_half': 0.626,...})
    number_of_estimators: Integer indicating the amount of different estimators that were used for the TL experiments
    filename: Name of the html file that gets saved and contains the DataFrame with important information highlighted.
        Default: 'tl_results'
    """
    d = atl_results
    reform = {(outerKey, innerKey): values for outerKey, innerDict in d.items() for innerKey, values in innerDict.items()}
    reform_2 = {(outerKey, innerKey): values for outerKey, innerDict in reform.items() for innerKey, values in innerDict.items()}
    mean_2iter = [round(np.mean(v['test_f1_scores'],axis=0)[1],3) for k,v in reform_2.items()]
    mean_10iter = [round(np.mean(v['test_f1_scores'],axis=0)[9],3) for k,v in reform_2.items()]
    mean_100iter = [round(np.mean(v['test_f1_scores'],axis=0)[99],3) for k,v in reform_2.items()]
    al_mean_2iter,al_mean_10iter,al_mean_100iter = [],[],[]
    clf_org,clfs,qss,column_list = [],[],[],[]
    
    for key in d:
        target = '_'.join(key.split('_')[2:])
        for clf in d[key]:
            clf_org.append(clf)
            if(clf=='lr'):
                clfs.append('logreg')
            elif(clf=='rf'):
                clfs.append('randforest')
            elif(clf=='dt'):
                clfs.append('dectree')
            else:
                clfs.append(clf)
            for qs in d[key][clf]:
                qss.append(qs)
                al_test_f1_scores = al_results[target][clf][qs]['test_f1_scores']
                n_init_labeled = al_results[target][clf][qs]['n_init_labeled']
                al_mean_test_f1_scores = np.mean(al_test_f1_scores,axis=0)
                al_mean_2iter.append(round(al_mean_test_f1_scores[1+n_init_labeled],3))
                column_list.append((clf,qs,'2nd','ATL'))
                column_list.append((clf,qs,'2nd','AL'))
                al_mean_10iter.append(round(al_mean_test_f1_scores[9+n_init_labeled],3))
                column_list.append((clf,qs,'10th','ATL'))
                column_list.append((clf,qs,'10th','AL'))
                al_mean_100iter.append(round(al_mean_test_f1_scores[99+n_init_labeled],3))
                column_list.append((clf,qs,'100th','ATL'))
                column_list.append((clf,qs,'100th','AL'))
    al_atl_mean_2iter = list(zip(mean_2iter,al_mean_2iter))
    al_atl_mean_10iter = list(zip(mean_10iter,al_mean_10iter))
    al_atl_mean_100iter = list(zip(mean_100iter,al_mean_100iter))
    
    clf_org = list(set(clf_org))
    number_of_estimators = len(clf_org)
    
    clfs = list(set(clfs))
    # get the amount of different query strategies
    number_of_qs = len(list(set(qss)))
    # ensure that the order is maintained 
    qss = qss[:number_of_qs]
    n = number_of_estimators*number_of_qs*3*2 #number_of_estimators*number_of_qs*Iterations (3) *AL Results and ATL Results (2)
    data = list(group(list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(list(zip(al_atl_mean_2iter,al_atl_mean_10iter,al_atl_mean_100iter)))))),n))
    col_tuple_list = list(group(column_list,n))
    single_col_tuple = list(set(col_tuple_list))
    columns_list = [item for t in single_col_tuple for item in t]
    keys = [('_'.join(k.split('_')[:2]),'_'.join(k.split('_')[2:])) for k in d]
    df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(keys,names=['Source','Target']), columns=pd.MultiIndex.from_tuples(columns_list,names=['Estimators','QS','Iteration','Results']))
    
    values = [round(candsets_super_results[key][clf],3) for key in candsets_super_results.keys() for clf in clfs]
    df_super = pd.DataFrame(list(group(values,len(clfs))),index=candsets_super_results.keys(),columns=clfs)
    
    if('logreg' in df_super.columns):
        df_super.rename(columns={'logreg':'lr'},inplace=True)
    if('randforest' in df_super.columns):
        df_super.rename(columns={'randforest':'rf'},inplace=True)
    if('logregcv' in df_super.columns):
        df_super.rename(columns={'logregcv':'lrcv'},inplace=True)
    if('dectree' in df_super.columns):
        df_super.rename(columns={'dectree':'dt'},inplace=True)
    
    df_super = df_super[clf_org] 
    #columns_before = list(df_super.columns)
    #print(columns_before)
    df_super = pd.concat([df_super,df_super,df_super],axis=1)
    df_super = df_super[clf_org]
    #print(qss)
    df_super.columns = pd.MultiIndex.from_product([clf_org,qss,['all'],['Tar_sup']],names=['Estimators','QS','Iterations','Results'])
    df = pd.merge(df,df_super,how='left',left_on='Target',right_index=True)
    df = df.reindex(columns=clf_org,level=0)
    df = df.reindex(columns=qss,level=1)
    df = df.reindex(columns=['2nd','10th','100th','all'],level=2)
    col_al_atl = [col for col in df.columns if col[3]=='ATL' or col[3]=='AL']
    
    ###########################################################################################
    # style functions as nested functions
    # function for pandas styler. Highlight all max values of TL_avg
    def highlight_max(data):
        attr_max = 'background-color: #FBFF75'
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = data == data.max()
            return [attr_max if v else '' for v in is_max]
        else: 
            is_max = data.groupby(['AL','ATL']).transform('max') == data
            return pd.DataFrame(np.where(is_max, attr_max, ''),
                                    index=data.index, columns=data.columns)
    # another function for pandas style      
    def highlight_al_worse_than_atl(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL']>row[:,:,:,'AL'])
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i!=0 and i%3==0):
                k += 1
            if(b):
                lst[k] = 'color:#000000;background-color:rgba(51, 204, 51, .1);'
            k += 2
        return lst
    # another function for pandas style      
    def highlight_atl_worse_than_al(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL']<row[:,:,:,'AL'])
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 1
        for i, b in enumerate(ser):
            if(i!=0 and i%3==0):
                k += 1
            if(b):
                lst[i+k] = 'background-color: #FF7070'
            k += 1
        return lst
    ############################################################################################
    # specify the styles for the html output
    styles=[{'selector': 'th','props': [
            ('border-style', 'solid'),
            ('border-color', '#D3D3D3'),
            ('vertical-align','top'),
            ('text-align','center')]}]
   
    html = (df.style.\
            apply(highlight_al_worse_than_atl,axis=1).\
            apply(highlight_atl_worse_than_al,axis=1).\
            apply(highlight_max,axis=1,subset=pd.IndexSlice[:,col_al_atl])).set_table_styles(styles).set_precision(3)
    display(html)
    with open('{}.html'.format(filename), 'w') as f:
        f.write(html.render())
    return df

#%%
    
def createDFwithAlandATLResultsUpToMaxQuota(al_results,atl_results,candsets_super_results,max_quota=200,filename='al_and_atl_results'):
    """
    This function creates a DataFrame with the results of each experiment. Per estimator the TL_avg (Transfer Learning Avg Result),
    Tar_max (Target max result when trained on 500 target instances), Tar_exc (Amount of Target Instances needed to exceed TL results),
    when trained on all features and dense features, are reported. Hence, amount of estimators*2*3 columns + 1 for unsupervised results
    of target are in the resulting DataFrame. It also stores a html file where important information is highlighted in the DataFrame.
    
    tl_results: Dictionary containting the results of the Transfer Learning experiements (as output by returnF1TLResultsFromDictWithPlot())
    candsets_unsuper_results: Dictionary containing the unsupervised results which act as benchmark (in the form of {'ban_half':0.732,'bx_half': 0.626,...})
    number_of_estimators: Integer indicating the amount of different estimators that were used for the TL experiments
    filename: Name of the html file that gets saved and contains the DataFrame with important information highlighted.
        Default: 'tl_results'
    """
    d = atl_results
    reform = {(outerKey, innerKey): values for outerKey, innerDict in d.items() for innerKey, values in innerDict.items()}
    reform_2 = {(outerKey, innerKey): values for outerKey, innerDict in reform.items() for innerKey, values in innerDict.items()}
    mean_2iter = [round(np.mean(v['test_f1_scores'],axis=0)[1],3) for k,v in reform_2.items()]
    mean_10iter = [round(np.mean(v['test_f1_scores'],axis=0)[9],3) for k,v in reform_2.items()]
    mean_50iter = [round(np.mean(v['test_f1_scores'],axis=0)[49],3) for k,v in reform_2.items()]
    mean_100iter = [round(np.mean(v['test_f1_scores'],axis=0)[99],3) for k,v in reform_2.items()]
    mean_maxiter = [round(np.mean(v['test_f1_scores'],axis=0)[max_quota-1],3) for k,v in reform_2.items()]
    al_mean_2iter,al_mean_10iter,al_mean_50iter,al_mean_100iter,al_mean_maxiter = [],[],[],[],[]
    clf_org,clfs,qss,column_list = [],[],[],[]
    max_str = '{}th'.format(max_quota)
    
    for key in d:
        target = '_'.join(key.split('_')[2:])
        for clf in d[key]:
            clf_org.append(clf)
            if(clf=='lr'):
                clfs.append('logreg')
            elif(clf=='rf'):
                clfs.append('randforest')
            elif(clf=='dt'):
                clfs.append('dectree')
            else:
                clfs.append(clf)
            for qs in d[key][clf]:
                qss.append(qs)
                al_test_f1_scores = al_results[target][clf][qs]['test_f1_scores']
                n_init_labeled = al_results[target][clf][qs]['n_init_labeled']
                al_mean_test_f1_scores = np.mean(al_test_f1_scores,axis=0)
                al_mean_2iter.append(round(al_mean_test_f1_scores[1+n_init_labeled],3))
                column_list.append((clf,qs,'2nd','ATL'))
                column_list.append((clf,qs,'2nd','AL'))
                al_mean_10iter.append(round(al_mean_test_f1_scores[9+n_init_labeled],3))
                column_list.append((clf,qs,'10th','ATL'))
                column_list.append((clf,qs,'10th','AL'))
                al_mean_50iter.append(round(al_mean_test_f1_scores[49+n_init_labeled],3))
                column_list.append((clf,qs,'50th','ATL'))
                column_list.append((clf,qs,'50th','AL'))
                al_mean_100iter.append(round(al_mean_test_f1_scores[99+n_init_labeled],3))
                column_list.append((clf,qs,'100th','ATL'))
                column_list.append((clf,qs,'100th','AL'))
                al_mean_maxiter.append(round(al_mean_test_f1_scores[max_quota-1],3))
                column_list.append((clf,qs,max_str,'ATL'))
                column_list.append((clf,qs,max_str,'AL'))
    al_atl_mean_2iter = list(zip(mean_2iter,al_mean_2iter))
    al_atl_mean_10iter = list(zip(mean_10iter,al_mean_10iter))
    al_atl_mean_50iter = list(zip(mean_50iter,al_mean_50iter))
    al_atl_mean_100iter = list(zip(mean_100iter,al_mean_100iter))
    al_atl_mean_maxiter = list(zip(mean_maxiter,al_mean_maxiter))
    
    clf_org = list(set(clf_org))
    #print(clf_org)
    number_of_estimators = len(clf_org)
    #print('QSS before: {}'.format(qss))
    clfs = list(set(clfs))
    # get the amount of different query strategies
    number_of_qs = len(list(set(qss)))
    # ensure that the order is maintained 
    qss = qss[:number_of_qs]
    n = number_of_estimators*number_of_qs*5*2 #number_of_estimators*number_of_qs*Iterations (5) *AL Results and ATL Results (2)
    data = list(group(list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(list(zip(al_atl_mean_2iter,al_atl_mean_10iter,al_atl_mean_50iter,al_atl_mean_100iter,al_atl_mean_maxiter)))))),n))
    col_tuple_list = list(group(column_list,n))
    single_col_tuple = list(set(col_tuple_list))
    columns_list = [item for t in single_col_tuple for item in t]
    keys = [('_'.join(k.split('_')[:2]),'_'.join(k.split('_')[2:])) for k in d]
    df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(keys,names=['Source','Target']), columns=pd.MultiIndex.from_tuples(columns_list,names=['Estimators','QS','Iteration','Results']))
    
    values = [round(candsets_super_results[key][clf],3) for key in candsets_super_results.keys() for clf in clfs]
    df_super = pd.DataFrame(list(group(values,len(clfs))),index=candsets_super_results.keys(),columns=clfs)
    
    if('logreg' in df_super.columns):
        df_super.rename(columns={'logreg':'lr'},inplace=True)
    if('randforest' in df_super.columns):
        df_super.rename(columns={'randforest':'rf'},inplace=True)
    if('logregcv' in df_super.columns):
        df_super.rename(columns={'logregcv':'lrcv'},inplace=True)
    if('dectree' in df_super.columns):
        df_super.rename(columns={'dectree':'dt'},inplace=True)
    df_super = df_super[clf_org] 
    #columns_before = list(df_super.columns)
    #print(columns_before)
    df_super = pd.concat([df_super,df_super,df_super],axis=1)
    df_super = df_super[clf_org]
    #print(qss)
    df_super.columns = pd.MultiIndex.from_product([clf_org,qss,['all'],['Tar_sup']],names=['Estimators','QS','Iterations','Results'])
    df = pd.merge(df,df_super,how='left',left_on='Target',right_index=True)
    df = df.reindex(columns=clf_org,level=0)
    df = df.reindex(columns=qss,level=1)
    df = df.reindex(columns=['2nd','10th','50th','100th',max_str,'all'],level=2)
    #col_al_atl = [col for col in df.columns if col[3]=='ATL' or col[3]=='AL']
    col_tar_sup = [col for col in df.columns if col[3]=='Tar_sup']
    col_tar_sup_format = {col:lambda x: '<font color=\'#000000\'><b>{}</b></font>'.format(x) for col in col_tar_sup}
    ###########################################################################################
    # style functions as nested functions
    # function for pandas styler. Highlight all max values of TL_avg
    def highlight_max(data):
        attr_max = 'background-color: #FBFF75'
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = data == data.max()
            return [attr_max if v else '' for v in is_max]
        else: 
            is_max = data.groupby(['AL','ATL']).transform('max') == data
            return pd.DataFrame(np.where(is_max, attr_max, ''),
                                    index=data.index, columns=data.columns)
    # another function for pandas style      
    def highlight_atl_exceed_bm(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL'].reset_index(drop=True)>=row[:,:,'all','Tar_sup'].repeat(5).reset_index(drop=True))
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i!=0 and i%5==0):
                k += 1
            if(b):
                lst[k] = 'background-color: #FBFF75'
            k += 2
        return lst
    
    # another function for pandas style      
    def highlight_al_worse_than_atl(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL']>row[:,:,:,'AL'])
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i!=0 and i%5==0):
                k += 1
            if(b):
                lst[k] = 'color:#000000;background-color:rgba(51, 204, 51, .1);'
            k += 2
        return lst
    # another function for pandas style      
    def highlight_atl_worse_than_al(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL']<row[:,:,:,'AL'])
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 1
        for i, b in enumerate(ser):
            if(i!=0 and i%5==0):
                k += 1
            if(b):
                lst[k] = 'background-color: #FF7070'
            k += 2
        return lst
    ############################################################################################
    # specify the styles for the html output
    styles=[{'selector': 'th','props': [
            ('border-style', 'solid'),
            ('border-color', '#D3D3D3'),
            ('vertical-align','top'),
            ('text-align','center')]}]
   
    html = (df.style.\
            apply(highlight_al_worse_than_atl,axis=1).\
            apply(highlight_atl_worse_than_al,axis=1).\
            #apply(highlight_max,axis=1,subset=pd.IndexSlice[:,col_al_atl]))
            apply(highlight_atl_exceed_bm,axis=1)).set_table_styles(styles).set_precision(3).format(col_tar_sup_format)
    display(html)
    with open('{}.html'.format(filename), 'w') as f:
        f.write(html.render())
    return df

#%%
    
def createDFwithAlandATLResultsMoreItersUpToMaxQuota(al_results,atl_results,candsets_super_results,max_quota=200,filename='al_and_atl_results'):
    """
    This function creates a DataFrame with the results of each experiment. Per estimator the TL_avg (Transfer Learning Avg Result),
    Tar_max (Target max result when trained on 500 target instances), Tar_exc (Amount of Target Instances needed to exceed TL results),
    when trained on all features and dense features, are reported. Hence, amount of estimators*2*3 columns + 1 for unsupervised results
    of target are in the resulting DataFrame. It also stores a html file where important information is highlighted in the DataFrame.
    
    tl_results: Dictionary containting the results of the Transfer Learning experiements (as output by returnF1TLResultsFromDictWithPlot())
    candsets_unsuper_results: Dictionary containing the unsupervised results which act as benchmark (in the form of {'ban_half':0.732,'bx_half': 0.626,...})
    number_of_estimators: Integer indicating the amount of different estimators that were used for the TL experiments
    filename: Name of the html file that gets saved and contains the DataFrame with important information highlighted.
        Default: 'tl_results'
    """
    d = atl_results
    reform = {(outerKey, innerKey): values for outerKey, innerDict in d.items() for innerKey, values in innerDict.items()}
    reform_2 = {(outerKey, innerKey): values for outerKey, innerDict in reform.items() for innerKey, values in innerDict.items()}
    mean_2iter = [round(np.mean(v['test_f1_scores'],axis=0)[1],3) for k,v in reform_2.items()]
    mean_10iter = [round(np.mean(v['test_f1_scores'],axis=0)[9],3) for k,v in reform_2.items()]
    mean_20iter = [round(np.mean(v['test_f1_scores'],axis=0)[19],3) for k,v in reform_2.items()]
    mean_30iter = [round(np.mean(v['test_f1_scores'],axis=0)[29],3) for k,v in reform_2.items()]
    mean_50iter = [round(np.mean(v['test_f1_scores'],axis=0)[49],3) for k,v in reform_2.items()]
    mean_100iter = [round(np.mean(v['test_f1_scores'],axis=0)[99],3) for k,v in reform_2.items()]
    mean_maxiter = [round(np.mean(v['test_f1_scores'],axis=0)[max_quota-1],3) for k,v in reform_2.items()]
    al_mean_2iter,al_mean_10iter,al_mean_20iter,al_mean_30iter,al_mean_50iter,al_mean_100iter,al_mean_maxiter = [],[],[],[],[],[],[]
    clf_org,clfs,qss,column_list = [],[],[],[]
    max_str = '{}th'.format(max_quota)
    
    for key in d:
        target = '_'.join(key.split('_')[2:])
        for clf in d[key]:
            clf_org.append(clf)
            if(clf=='lr'):
                clfs.append('logreg')
            elif(clf=='rf'):
                clfs.append('randforest')
            elif(clf=='dt'):
                clfs.append('dectree')
            else:
                clfs.append(clf)
            for qs in d[key][clf]:
                qss.append(qs)
                al_test_f1_scores = al_results[target][clf][qs]['test_f1_scores']
                n_init_labeled = al_results[target][clf][qs]['n_init_labeled']
                al_mean_test_f1_scores = np.mean(al_test_f1_scores,axis=0)
                al_mean_2iter.append(round(al_mean_test_f1_scores[1+n_init_labeled],3))
                column_list.append((clf,qs,'2nd','ATL'))
                column_list.append((clf,qs,'2nd','AL'))
                al_mean_10iter.append(round(al_mean_test_f1_scores[9+n_init_labeled],3))
                column_list.append((clf,qs,'10th','ATL'))
                column_list.append((clf,qs,'10th','AL'))
                al_mean_20iter.append(round(al_mean_test_f1_scores[19+n_init_labeled],3))
                column_list.append((clf,qs,'20th','ATL'))
                column_list.append((clf,qs,'20th','AL'))
                al_mean_30iter.append(round(al_mean_test_f1_scores[29+n_init_labeled],3))
                column_list.append((clf,qs,'30th','ATL'))
                column_list.append((clf,qs,'30th','AL'))
                al_mean_50iter.append(round(al_mean_test_f1_scores[49+n_init_labeled],3))
                column_list.append((clf,qs,'50th','ATL'))
                column_list.append((clf,qs,'50th','AL'))
                al_mean_100iter.append(round(al_mean_test_f1_scores[99+n_init_labeled],3))
                column_list.append((clf,qs,'100th','ATL'))
                column_list.append((clf,qs,'100th','AL'))
                al_mean_maxiter.append(round(al_mean_test_f1_scores[max_quota-1],3))
                column_list.append((clf,qs,max_str,'ATL'))
                column_list.append((clf,qs,max_str,'AL'))
    al_atl_mean_2iter = list(zip(mean_2iter,al_mean_2iter))
    al_atl_mean_10iter = list(zip(mean_10iter,al_mean_10iter))
    al_atl_mean_20iter = list(zip(mean_20iter,al_mean_20iter))
    al_atl_mean_30iter = list(zip(mean_30iter,al_mean_30iter))
    al_atl_mean_50iter = list(zip(mean_50iter,al_mean_50iter))
    al_atl_mean_100iter = list(zip(mean_100iter,al_mean_100iter))
    al_atl_mean_maxiter = list(zip(mean_maxiter,al_mean_maxiter))
    
    clf_org = list(set(clf_org))
    #print(clf_org)
    number_of_estimators = len(clf_org)
    #print('QSS before: {}'.format(qss))
    clfs = list(set(clfs))
    # get the amount of different query strategies
    number_of_qs = len(list(set(qss)))
    # ensure that the order is maintained 
    qss = qss[:number_of_qs]
    n = number_of_estimators*number_of_qs*7*2 #number_of_estimators*number_of_qs*Iterations (5) *AL Results and ATL Results (2)
    data = list(group(list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(list(zip(al_atl_mean_2iter,al_atl_mean_10iter,al_atl_mean_20iter,al_atl_mean_30iter,al_atl_mean_50iter,al_atl_mean_100iter,al_atl_mean_maxiter)))))),n))
    col_tuple_list = list(group(column_list,n))
    single_col_tuple = list(set(col_tuple_list))
    columns_list = [item for t in single_col_tuple for item in t]
    keys = [('_'.join(k.split('_')[:2]),'_'.join(k.split('_')[2:])) for k in d]
    df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(keys,names=['Source','Target']), columns=pd.MultiIndex.from_tuples(columns_list,names=['Estimators','QS','Iteration','Results']))
    
    values = [round(candsets_super_results[key][clf],3) for key in candsets_super_results.keys() for clf in clfs]
    df_super = pd.DataFrame(list(group(values,len(clfs))),index=candsets_super_results.keys(),columns=clfs)
    
    if('logreg' in df_super.columns):
        df_super.rename(columns={'logreg':'lr'},inplace=True)
    if('randforest' in df_super.columns):
        df_super.rename(columns={'randforest':'rf'},inplace=True)
    if('logregcv' in df_super.columns):
        df_super.rename(columns={'logregcv':'lrcv'},inplace=True)
    if('dectree' in df_super.columns):
        df_super.rename(columns={'dectree':'dt'},inplace=True)
    
    df_super = df_super[clf_org] 
    #columns_before = list(df_super.columns)
    #print(columns_before)
    df_super = pd.concat([df_super,df_super,df_super],axis=1)
    df_super = df_super[clf_org]
    #print(qss)
    df_super.columns = pd.MultiIndex.from_product([clf_org,qss,['all'],['Tar_sup']],names=['Estimators','QS','Iterations','Results'])
    df = pd.merge(df,df_super,how='left',left_on='Target',right_index=True)
    df = df.reindex(columns=clf_org,level=0)
    df = df.reindex(columns=qss,level=1)
    df = df.reindex(columns=['2nd','10th','20th','30th','50th','100th',max_str,'all'],level=2)
    #col_al_atl = [col for col in df.columns if col[3]=='ATL' or col[3]=='AL']
    col_tar_sup = [col for col in df.columns if col[3]=='Tar_sup']
    col_tar_sup_format = {col:lambda x: '<font color=\'#000000\'><b>{}</b></font>'.format(x) for col in col_tar_sup}
    ###########################################################################################
    # style functions as nested functions
    # function for pandas styler. Highlight all max values of TL_avg
    def highlight_max(data):
        attr_max = 'background-color: #FBFF75'
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = data == data.max()
            return [attr_max if v else '' for v in is_max]
        else: 
            is_max = data.groupby(['AL','ATL']).transform('max') == data
            return pd.DataFrame(np.where(is_max, attr_max, ''),
                                    index=data.index, columns=data.columns)
    # another function for pandas style      
    def highlight_atl_exceed_bm(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL'].reset_index(drop=True)>=row[:,:,'all','Tar_sup'].repeat(7).reset_index(drop=True))
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i!=0 and i%7==0):
                k += 1
            if(b):
                lst[k] = 'background-color: #FBFF75'
            k += 2
        return lst
    
    # another function for pandas style      
    def highlight_al_worse_than_atl(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL']>row[:,:,:,'AL'])
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i!=0 and i%7==0):
                k += 1
            if(b):
                lst[k] = 'color:#000000;background-color:rgba(51, 204, 51, .1);'
            k += 2
        return lst
    # another function for pandas style      
    def highlight_atl_worse_than_al(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL']<row[:,:,:,'AL'])
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 1
        for i, b in enumerate(ser):
            if(i!=0 and i%7==0):
                k += 1
            if(b):
                lst[k] = 'background-color: #FF7070'
            k += 2
        return lst
    ############################################################################################
    # specify the styles for the html output
    styles=[{'selector': 'th','props': [
            ('border-style', 'solid'),
            ('border-color', '#D3D3D3'),
            ('vertical-align','top'),
            ('text-align','center')]}]
   
    html = (df.style.\
            apply(highlight_al_worse_than_atl,axis=1).\
            apply(highlight_atl_worse_than_al,axis=1).\
            #apply(highlight_max,axis=1,subset=pd.IndexSlice[:,col_al_atl]))
            apply(highlight_atl_exceed_bm,axis=1)).set_table_styles(styles).set_precision(3).format(col_tar_sup_format)
    display(html)
    with open('{}.html'.format(filename), 'w') as f:
        f.write(html.render())
    return df

#%%
    
def highlightDFWithATLandALResults(df,number_iters,filename=None):
    col_tar_sup = [col for col in df.columns if col[3]=='Tar_sup']
    col_tar_sup_format = {col:lambda x: '<font color=\'#000000\'><b>{}</b></font>'.format(x) for col in col_tar_sup}
    ###########################################################################################
    # style functions as nested functions
    # function for pandas styler. Highlight all max values of TL_avg
    def highlight_max(data):
        attr_max = 'background-color: #FBFF75'
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = data == data.max()
            return [attr_max if v else '' for v in is_max]
        else: 
            is_max = data.groupby(['AL','ATL']).transform('max') == data
            return pd.DataFrame(np.where(is_max, attr_max, ''),
                                    index=data.index, columns=data.columns)
    # another function for pandas style      
    def highlight_atl_exceed_bm(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL'].reset_index(drop=True)>=row[:,:,'all','Tar_sup'].repeat(number_iters).reset_index(drop=True))
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i!=0 and i%number_iters==0):
                k += 1
            if(b):
                lst[k] = 'background-color: #FBFF75'
            k += 2
        return lst
    
    # another function for pandas style      
    def highlight_al_worse_than_atl(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL']>row[:,:,:,'AL'])
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i!=0 and i%number_iters==0):
                k += 1
            if(b):
                lst[k] = 'color:#000000;background-color:rgba(51, 204, 51, .1);'
            k += 2
        return lst
    # another function for pandas style      
    def highlight_atl_worse_than_al(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL']<row[:,:,:,'AL'])
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 1
        for i, b in enumerate(ser):
            if(i!=0 and i%number_iters==0):
                k += 1
            if(b):
                lst[k] = 'background-color: #FF7070'
            k += 2
        return lst
    ############################################################################################
    # specify the styles for the html output
    styles=[{'selector': 'th','props': [
            ('border-style', 'solid'),
            ('border-color', '#D3D3D3'),
            ('vertical-align','top'),
            ('text-align','center')]}]
   
    html = (df.style.\
            apply(highlight_al_worse_than_atl,axis=1).\
            apply(highlight_atl_worse_than_al,axis=1).\
            #apply(highlight_max,axis=1,subset=pd.IndexSlice[:,col_al_atl]))
            apply(highlight_atl_exceed_bm,axis=1)).set_table_styles(styles).set_precision(3).format(col_tar_sup_format)
    display(html)
    if(isinstance(filename,str)):
        with open('{}.html'.format(filename), 'w') as f:
            f.write(html.render())
    
    return None

#%%
    
def createDFwithAlandATLResultsOnlyOneEstimator(al_results,atl_results,estimator_name,candsets_super_results,filename='al_and_atl_results'):
    """
    This function creates a DataFrame with the results of each experiment. Per estimator the TL_avg (Transfer Learning Avg Result),
    Tar_max (Target max result when trained on 500 target instances), Tar_exc (Amount of Target Instances needed to exceed TL results),
    when trained on all features and dense features, are reported. Hence, amount of estimators*2*3 columns + 1 for unsupervised results
    of target are in the resulting DataFrame. It also stores a html file where important information is highlighted in the DataFrame.
    
    tl_results: Dictionary containting the results of the Transfer Learning experiements (as output by returnF1TLResultsFromDictWithPlot())
    candsets_unsuper_results: Dictionary containing the unsupervised results which act as benchmark (in the form of {'ban_half':0.732,'bx_half': 0.626,...})
    number_of_estimators: Integer indicating the amount of different estimators that were used for the TL experiments
    filename: Name of the html file that gets saved and contains the DataFrame with important information highlighted.
        Default: 'tl_results'
    """
    d = atl_results
    reform = {(outerKey, innerKey): values for outerKey, innerDict in d.items() for innerKey, values in innerDict.items()}
    reform_2 = {(outerKey, innerKey): values for outerKey, innerDict in reform.items() for innerKey, values in innerDict.items()}
    mean_2iter = [round(np.mean(v['test_f1_scores'],axis=0)[1],3) for k,v in reform_2.items()]
    mean_10iter = [round(np.mean(v['test_f1_scores'],axis=0)[9],3) for k,v in reform_2.items()]
    mean_100iter = [round(np.mean(v['test_f1_scores'],axis=0)[99],3) for k,v in reform_2.items()]
    al_mean_2iter,al_mean_10iter,al_mean_100iter = [],[],[]
    qss,column_list = [],[]
    clf = estimator_name
    for key in d:
        target = '_'.join(key.split('_')[2:])
        for qs in d[key][clf]:
            qss.append(qs)
            al_test_f1_scores = al_results[target][clf][qs]['test_f1_scores']
            n_init_labeled = al_results[target][clf][qs]['n_init_labeled']
            al_mean_test_f1_scores = np.mean(al_test_f1_scores,axis=0)
            al_mean_2iter.append(round(al_mean_test_f1_scores[1+n_init_labeled],3))
            column_list.append((clf,qs,'2nd','ATL'))
            column_list.append((clf,qs,'2nd','AL'))
            al_mean_10iter.append(round(al_mean_test_f1_scores[9+n_init_labeled],3))
            column_list.append((clf,qs,'10th','ATL'))
            column_list.append((clf,qs,'10th','AL'))
            al_mean_100iter.append(round(al_mean_test_f1_scores[99+n_init_labeled],3))
            column_list.append((clf,qs,'100th','ATL'))
            column_list.append((clf,qs,'100th','AL'))
    al_atl_mean_2iter = list(zip(mean_2iter,al_mean_2iter))
    al_atl_mean_10iter = list(zip(mean_10iter,al_mean_10iter))
    al_atl_mean_100iter = list(zip(mean_100iter,al_mean_100iter))
    
    number_of_estimators = 1 #only one estimator selected in this function
    
    # get the amount of different query strategies
    number_of_qs = len(list(set(qss)))
    # ensure that the order is maintained 
    qss = qss[:number_of_qs]
    n = number_of_estimators*number_of_qs*3*2 #number_of_estimators*number_of_qs*Iterations (3) *AL Results and ATL Results (2)
    data = list(group(list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(list(zip(al_atl_mean_2iter,al_atl_mean_10iter,al_atl_mean_100iter)))))),n))
    col_tuple_list = list(group(column_list,n))
    single_col_tuple = list(set(col_tuple_list))
    columns_list = [item for t in single_col_tuple for item in t]
    keys = [('_'.join(k.split('_')[:2]),'_'.join(k.split('_')[2:])) for k in d]
    df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(keys,names=['Source','Target']), columns=pd.MultiIndex.from_tuples(columns_list,names=['Estimators','QS','Iteration','Results']))
    
    if(clf == 'lr'):
        clf_old = 'logreg'
    elif(clf == 'rf'):
        clf_old = 'randforest'
    elif(clf == 'lrcv'):
        clf_old = 'logregcv'
    elif(clf == 'dt'):
        clf_old = 'dectree'
    else:
        clf_old = clf
        
    values = [round(candsets_super_results[key][clf_old],3) for key in candsets_super_results.keys()]
    df_super = pd.DataFrame(values,index=candsets_super_results.keys(),columns=clf)
    
    if('logreg' in df_super.columns):
        df_super.rename(columns={'logreg':'lr'},inplace=True)
    if('randforest' in df_super.columns):
        df_super.rename(columns={'randforest':'rf'},inplace=True)
    if('logregcv' in df_super.columns):
        df_super.rename(columns={'logregcv':'lrcv'},inplace=True)
    if('dectree' in df_super.columns):
        df_super.rename(columns={'dectree':'dt'},inplace=True)
    
    #df_super = df_super[clf_org] 
    #columns_before = list(df_super.columns)
    #print(columns_before)
    df_super = pd.concat([df_super,df_super,df_super],axis=1)
    #df_super = df_super[clf_org]
    #print(qss)
    df_super.columns = pd.MultiIndex.from_product([clf,qss,['all'],['Tar_sup']],names=['Estimators','QS','Iterations','Results'])
    df = pd.merge(df,df_super,how='left',left_on='Target',right_index=True)
    df = df.reindex(columns=clf,level=0)
    df = df.reindex(columns=qss,level=1)
    df = df.reindex(columns=['2nd','10th','100th','all'],level=2)
    col_al_atl = [col for col in df.columns if col[3]=='ATL' or col[3]=='AL']
    
    ###########################################################################################
    # style functions as nested functions
    # function for pandas styler. Highlight all max values of TL_avg
    def highlight_max(data):
        attr_max = 'background-color: #FBFF75'
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = data == data.max()
            return [attr_max if v else '' for v in is_max]
        else: 
            is_max = data.groupby(['AL','ATL']).transform('max') == data
            return pd.DataFrame(np.where(is_max, attr_max, ''),
                                    index=data.index, columns=data.columns)
    # another function for pandas style      
    def highlight_al_worse_than_atl(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL']>row[:,:,:,'AL'])
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i!=0 and i%3==0):
                k += 1
            if(b):
                lst[k] = 'color:#000000;background-color:rgba(51, 204, 51, .1);'
            k += 2
        return lst
    # another function for pandas style      
    def highlight_atl_worse_than_al(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL']<row[:,:,:,'AL'])
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 1
        for i, b in enumerate(ser):
            if(i!=0 and i%3==0):
                k += 1
            if(b):
                lst[i+k] = 'background-color: #FF7070'
            k += 1
        return lst
    ############################################################################################
    # specify the styles for the html output
    styles=[{'selector': 'th','props': [
            ('border-style', 'solid'),
            ('border-color', '#D3D3D3'),
            ('vertical-align','top'),
            ('text-align','center')]}]
   
    html = (df.style.\
            apply(highlight_al_worse_than_atl,axis=1).\
            apply(highlight_atl_worse_than_al,axis=1).\
            apply(highlight_max,axis=1,subset=pd.IndexSlice[:,col_al_atl])).set_table_styles(styles).set_precision(3)
    display(html)
    with open('{}.html'.format(filename), 'w') as f:
        f.write(html.render())
    return df

#%%
    
def createDFwithAlandATLResultsUpToMaxQuotaOnlyOneEstimator(al_results,atl_results,estimator_name,candsets_super_results,max_quota=200,filename='al_and_atl_results'):
    """
    This function creates a DataFrame with the results of each experiment. Per estimator the TL_avg (Transfer Learning Avg Result),
    Tar_max (Target max result when trained on 500 target instances), Tar_exc (Amount of Target Instances needed to exceed TL results),
    when trained on all features and dense features, are reported. Hence, amount of estimators*2*3 columns + 1 for unsupervised results
    of target are in the resulting DataFrame. It also stores a html file where important information is highlighted in the DataFrame.
    
    tl_results: Dictionary containting the results of the Transfer Learning experiements (as output by returnF1TLResultsFromDictWithPlot())
    candsets_unsuper_results: Dictionary containing the unsupervised results which act as benchmark (in the form of {'ban_half':0.732,'bx_half': 0.626,...})
    number_of_estimators: Integer indicating the amount of different estimators that were used for the TL experiments
    filename: Name of the html file that gets saved and contains the DataFrame with important information highlighted.
        Default: 'tl_results'
    """
    d = atl_results
    reform = {(outerKey, innerKey): values for outerKey, innerDict in d.items() for innerKey, values in innerDict.items()}
    reform_2 = {(outerKey, innerKey): values for outerKey, innerDict in reform.items() for innerKey, values in innerDict.items()}
    mean_2iter = [round(np.mean(v['test_f1_scores'],axis=0)[1],3) for k,v in reform_2.items()]
    mean_10iter = [round(np.mean(v['test_f1_scores'],axis=0)[9],3) for k,v in reform_2.items()]
    mean_50iter = [round(np.mean(v['test_f1_scores'],axis=0)[49],3) for k,v in reform_2.items()]
    mean_100iter = [round(np.mean(v['test_f1_scores'],axis=0)[99],3) for k,v in reform_2.items()]
    mean_maxiter = [round(np.mean(v['test_f1_scores'],axis=0)[max_quota-1],3) for k,v in reform_2.items()]
    al_mean_2iter,al_mean_10iter,al_mean_50iter,al_mean_100iter,al_mean_maxiter = [],[],[],[],[]
    qss,column_list = [],[]
    max_str = '{}th'.format(max_quota)
    clf = estimator_name
    for key in d:
        target = '_'.join(key.split('_')[2:])
        for qs in d[key][clf]:
            qss.append(qs)
            al_test_f1_scores = al_results[target][clf][qs]['test_f1_scores']
            n_init_labeled = al_results[target][clf][qs]['n_init_labeled']
            al_mean_test_f1_scores = np.mean(al_test_f1_scores,axis=0)
            al_mean_2iter.append(round(al_mean_test_f1_scores[1+n_init_labeled],3))
            column_list.append((clf,qs,'2nd','ATL'))
            column_list.append((clf,qs,'2nd','AL'))
            al_mean_10iter.append(round(al_mean_test_f1_scores[9+n_init_labeled],3))
            column_list.append((clf,qs,'10th','ATL'))
            column_list.append((clf,qs,'10th','AL'))
            al_mean_50iter.append(round(al_mean_test_f1_scores[49+n_init_labeled],3))
            column_list.append((clf,qs,'50th','ATL'))
            column_list.append((clf,qs,'50th','AL'))
            al_mean_100iter.append(round(al_mean_test_f1_scores[99+n_init_labeled],3))
            column_list.append((clf,qs,'100th','ATL'))
            column_list.append((clf,qs,'100th','AL'))
            al_mean_maxiter.append(round(al_mean_test_f1_scores[max_quota-1],3))
            column_list.append((clf,qs,max_str,'ATL'))
            column_list.append((clf,qs,max_str,'AL'))
    al_atl_mean_2iter = list(zip(mean_2iter,al_mean_2iter))
    al_atl_mean_10iter = list(zip(mean_10iter,al_mean_10iter))
    al_atl_mean_50iter = list(zip(mean_50iter,al_mean_50iter))
    al_atl_mean_100iter = list(zip(mean_100iter,al_mean_100iter))
    al_atl_mean_maxiter = list(zip(mean_maxiter,al_mean_maxiter))
    
    number_of_estimators = 1

    # get the amount of different query strategies
    number_of_qs = len(list(set(qss)))
    # ensure that the order is maintained 
    qss = qss[:number_of_qs]
    n = number_of_estimators*number_of_qs*5*2 #number_of_estimators*number_of_qs*Iterations (5) *AL Results and ATL Results (2)
    data = list(group(list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(list(zip(al_atl_mean_2iter,al_atl_mean_10iter,al_atl_mean_50iter,al_atl_mean_100iter,al_atl_mean_maxiter)))))),n))
    col_tuple_list = list(group(column_list,n))
    single_col_tuple = list(set(col_tuple_list))
    columns_list = [item for t in single_col_tuple for item in t]
    keys = [('_'.join(k.split('_')[:2]),'_'.join(k.split('_')[2:])) for k in d]
    df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(keys,names=['Source','Target']), columns=pd.MultiIndex.from_tuples(columns_list,names=['Estimators','QS','Iteration','Results']))
    
    if(clf == 'lr'):
        clf_old = 'logreg'
    elif(clf == 'rf'):
        clf_old = 'randforest'
    elif(clf == 'lrcv'):
        clf_old = 'logregcv'
    elif(clf == 'dt'):
        clf_old = 'dectree'
    else:
        clf_old = clf
        
    values = [round(candsets_super_results[key][clf_old],3) for key in candsets_super_results.keys()]
    df_super = pd.DataFrame(values,index=candsets_super_results.keys(),columns=clf)
    
    if('logreg' in df_super.columns):
        df_super.rename(columns={'logreg':'lr'},inplace=True)
    if('randforest' in df_super.columns):
        df_super.rename(columns={'randforest':'rf'},inplace=True)
    if('logregcv' in df_super.columns):
        df_super.rename(columns={'logregcv':'lrcv'},inplace=True)
    if('dectree' in df_super.columns):
        df_super.rename(columns={'dectree':'dt'},inplace=True)
    
    #df_super = df_super[clf_org] 
    #columns_before = list(df_super.columns)
    #print(columns_before)
    df_super = pd.concat([df_super,df_super,df_super],axis=1)
    #df_super = df_super[clf_org]
    #print(qss)
    df_super.columns = pd.MultiIndex.from_product([clf,qss,['all'],['Tar_sup']],names=['Estimators','QS','Iterations','Results'])
    df = pd.merge(df,df_super,how='left',left_on='Target',right_index=True)
    df = df.reindex(columns=clf,level=0)
    df = df.reindex(columns=qss,level=1)
    df = df.reindex(columns=['2nd','10th','50th','100th',max_str,'all'],level=2)
    #col_al_atl = [col for col in df.columns if col[3]=='ATL' or col[3]=='AL']
    col_tar_sup = [col for col in df.columns if col[3]=='Tar_sup']
    col_tar_sup_format = {col:lambda x: '<font color=\'#000000\'><b>{}</b></font>'.format(x) for col in col_tar_sup}
    ###########################################################################################
    # style functions as nested functions
    # function for pandas styler. Highlight all max values of TL_avg
    def highlight_max(data):
        attr_max = 'background-color: #FBFF75'
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = data == data.max()
            return [attr_max if v else '' for v in is_max]
        else: 
            is_max = data.groupby(['AL','ATL']).transform('max') == data
            return pd.DataFrame(np.where(is_max, attr_max, ''),
                                    index=data.index, columns=data.columns)
    # another function for pandas style      
    def highlight_atl_exceed_bm(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL'].reset_index(drop=True)>=row[:,:,'all','Tar_sup'].repeat(5).reset_index(drop=True))
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i!=0 and i%5==0):
                k += 1
            if(b):
                lst[k] = 'background-color: #FBFF75'
            k += 2
        return lst
    
    # another function for pandas style      
    def highlight_al_worse_than_atl(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL']>row[:,:,:,'AL'])
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i!=0 and i%5==0):
                k += 1
            if(b):
                lst[k] = 'color:#000000;background-color:rgba(51, 204, 51, .1);'
            k += 2
        return lst
    # another function for pandas style      
    def highlight_atl_worse_than_al(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL']<row[:,:,:,'AL'])
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 1
        for i, b in enumerate(ser):
            if(i!=0 and i%5==0):
                k += 1
            if(b):
                lst[k] = 'background-color: #FF7070'
            k += 2
        return lst
    ############################################################################################
    # specify the styles for the html output
    styles=[{'selector': 'th','props': [
            ('border-style', 'solid'),
            ('border-color', '#D3D3D3'),
            ('vertical-align','top'),
            ('text-align','center')]}]
   
    html = (df.style.\
            apply(highlight_al_worse_than_atl,axis=1).\
            apply(highlight_atl_worse_than_al,axis=1).\
            #apply(highlight_max,axis=1,subset=pd.IndexSlice[:,col_al_atl]))
            apply(highlight_atl_exceed_bm,axis=1)).set_table_styles(styles).set_precision(3).format(col_tar_sup_format)
    display(html)
    with open('{}.html'.format(filename), 'w') as f:
        f.write(html.render())
    return df

#%%
    
def createResultsDFOneEstimator(al_results,atl_results,estimator_name,candsets_super_results,max_quota=200,filename='al_and_atl_results'):
    """
    This function creates a DataFrame with the results of each experiment. Per estimator the TL_avg (Transfer Learning Avg Result),
    Tar_max (Target max result when trained on 500 target instances), Tar_exc (Amount of Target Instances needed to exceed TL results),
    when trained on all features and dense features, are reported. Hence, amount of estimators*2*3 columns + 1 for unsupervised results
    of target are in the resulting DataFrame. It also stores a html file where important information is highlighted in the DataFrame.
    
    tl_results: Dictionary containting the results of the Transfer Learning experiements (as output by returnF1TLResultsFromDictWithPlot())
    candsets_unsuper_results: Dictionary containing the unsupervised results which act as benchmark (in the form of {'ban_half':0.732,'bx_half': 0.626,...})
    number_of_estimators: Integer indicating the amount of different estimators that were used for the TL experiments
    filename: Name of the html file that gets saved and contains the DataFrame with important information highlighted.
        Default: 'tl_results'
    """
    d = atl_results
    mean_2iter,mean_10iter,mean_20iter,mean_30iter,mean_50iter,mean_100iter,mean_maxiter = [],[],[],[],[],[],[]
    al_mean_2iter,al_mean_10iter,al_mean_20iter,al_mean_30iter,al_mean_50iter,al_mean_100iter,al_mean_maxiter = [],[],[],[],[],[],[]
    qss,column_list = [],[]
    max_str = '{}th'.format(max_quota)
    clf = estimator_name
    
    for key in d:
        target = '_'.join(key.split('_')[2:])
        for qs in d[key][clf]:
            qss.append(qs)
            mean_test_f1_score = np.mean(d[key][clf][qs]['test_f1_scores'],axis=0)
            mean_2iter.append(round(mean_test_f1_score[1],3))
            mean_10iter.append(round(mean_test_f1_score[9],3))
            mean_20iter.append(round(mean_test_f1_score[19],3))
            mean_30iter.append(round(mean_test_f1_score[29],3))
            mean_50iter.append(round(mean_test_f1_score[49],3))
            mean_100iter.append(round(mean_test_f1_score[99],3))
            mean_maxiter.append(round(mean_test_f1_score[max_quota-1],3))
            al_test_f1_scores = al_results[target][clf][qs]['test_f1_scores']
            n_init_labeled = al_results[target][clf][qs]['n_init_labeled']
            al_mean_test_f1_scores = np.mean(al_test_f1_scores,axis=0)
            al_mean_2iter.append(round(al_mean_test_f1_scores[1+n_init_labeled],3))
            column_list.append((clf,qs,'2nd','ATL'))
            column_list.append((clf,qs,'2nd','AL'))
            al_mean_10iter.append(round(al_mean_test_f1_scores[9+n_init_labeled],3))
            column_list.append((clf,qs,'10th','ATL'))
            column_list.append((clf,qs,'10th','AL'))
            al_mean_20iter.append(round(al_mean_test_f1_scores[19+n_init_labeled],3))
            column_list.append((clf,qs,'20th','ATL'))
            column_list.append((clf,qs,'20th','AL'))
            al_mean_30iter.append(round(al_mean_test_f1_scores[29+n_init_labeled],3))
            column_list.append((clf,qs,'30th','ATL'))
            column_list.append((clf,qs,'30th','AL'))
            al_mean_50iter.append(round(al_mean_test_f1_scores[49+n_init_labeled],3))
            column_list.append((clf,qs,'50th','ATL'))
            column_list.append((clf,qs,'50th','AL'))
            al_mean_100iter.append(round(al_mean_test_f1_scores[99+n_init_labeled],3))
            column_list.append((clf,qs,'100th','ATL'))
            column_list.append((clf,qs,'100th','AL'))
            al_mean_maxiter.append(round(al_mean_test_f1_scores[max_quota-1],3))
            column_list.append((clf,qs,max_str,'ATL'))
            column_list.append((clf,qs,max_str,'AL'))
    al_atl_mean_2iter = list(zip(mean_2iter,al_mean_2iter))
    al_atl_mean_10iter = list(zip(mean_10iter,al_mean_10iter))
    al_atl_mean_20iter = list(zip(mean_20iter,al_mean_20iter))
    al_atl_mean_30iter = list(zip(mean_30iter,al_mean_30iter))
    al_atl_mean_50iter = list(zip(mean_50iter,al_mean_50iter))
    al_atl_mean_100iter = list(zip(mean_100iter,al_mean_100iter))
    al_atl_mean_maxiter = list(zip(mean_maxiter,al_mean_maxiter))
    
    number_of_estimators = 1

    # get the amount of different query strategies
    number_of_qs = len(list(set(qss)))
    # ensure that the order is maintained 
    qss = qss[:number_of_qs]
    n = number_of_estimators*number_of_qs*7*2 #number_of_estimators*number_of_qs*Iterations (5) *AL Results and ATL Results (2)
    data = list(group(list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(list(zip(al_atl_mean_2iter,al_atl_mean_10iter,al_atl_mean_20iter,al_atl_mean_30iter,al_atl_mean_50iter,al_atl_mean_100iter,al_atl_mean_maxiter)))))),n))
    col_tuple_list = list(group(column_list,n))
    single_col_tuple = list(set(col_tuple_list))
    columns_list = [item for t in single_col_tuple for item in t]
    keys = [('_'.join(k.split('_')[:2]),'_'.join(k.split('_')[2:])) for k in d]
    df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(keys,names=['Source','Target']), columns=pd.MultiIndex.from_tuples(columns_list,names=['Estimators','QS','Iteration','Results']))
    
    if(clf == 'lr'):
        clf_old = 'logreg'
    elif(clf == 'rf'):
        clf_old = 'randforest'
    elif(clf == 'lrcv'):
        clf_old = 'logregcv'
    elif(clf == 'dt'):
        clf_old = 'dectree'
    else:
        clf_old = clf
        
    values = [round(candsets_super_results[key][clf_old],3) for key in candsets_super_results.keys()]
    df_super = pd.DataFrame(values,index=candsets_super_results.keys(),columns=[clf])
    
    if('logreg' in df_super.columns):
        df_super.rename(columns={'logreg':'lr'},inplace=True)
    if('randforest' in df_super.columns):
        df_super.rename(columns={'randforest':'rf'},inplace=True)
    if('logregcv' in df_super.columns):
        df_super.rename(columns={'logregcv':'lrcv'},inplace=True)
    if('dectree' in df_super.columns):
        df_super.rename(columns={'dectree':'dt'},inplace=True)
    
    #df_super = df_super[clf_org] 
    #columns_before = list(df_super.columns)
    #print(columns_before)
    df_super = pd.concat([df_super,df_super,df_super],axis=1)
    #df_super = df_super[clf_org]
    #print(qss)
    df_super.columns = pd.MultiIndex.from_product([[clf],qss,['all'],['Tar_sup']],names=['Estimators','QS','Iterations','Results'])
    df = pd.merge(df,df_super,how='left',left_on='Target',right_index=True)
    df = df.reindex(columns=[clf],level=0)
    df = df.reindex(columns=qss,level=1)
    df = df.reindex(columns=['2nd','10th','20th','30th','50th','100th',max_str,'all'],level=2)
    #col_al_atl = [col for col in df.columns if col[3]=='ATL' or col[3]=='AL']
    col_tar_sup = [col for col in df.columns if col[3]=='Tar_sup']
    col_tar_sup_format = {col:lambda x: '<font color=\'#000000\'><b>{}</b></font>'.format(x) for col in col_tar_sup}
    ###########################################################################################
    # style functions as nested functions
    # function for pandas styler. Highlight all max values of TL_avg
    def highlight_max(data):
        attr_max = 'background-color: #FBFF75'
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = data == data.max()
            return [attr_max if v else '' for v in is_max]
        else: 
            is_max = data.groupby(['AL','ATL']).transform('max') == data
            return pd.DataFrame(np.where(is_max, attr_max, ''),
                                    index=data.index, columns=data.columns)
    # another function for pandas style      
    def highlight_atl_exceed_bm(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL'].reset_index(drop=True)>=row[:,:,'all','Tar_sup'].repeat(7).reset_index(drop=True))
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i!=0 and i%7==0):
                k += 1
            if(b):
                lst[k] = 'background-color: #FBFF75'
            k += 2
        return lst
    
    # another function for pandas style      
    def highlight_al_worse_than_atl(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL']>row[:,:,:,'AL'])
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i!=0 and i%7==0):
                k += 1
            if(b):
                lst[k] = 'color:#000000;background-color:rgba(51, 204, 51, .1);'
            k += 2
        return lst
    # another function for pandas style      
    def highlight_atl_worse_than_al(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL']<row[:,:,:,'AL'])
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 1
        for i, b in enumerate(ser):
            if(i!=0 and i%7==0):
                k += 1
            if(b):
                lst[k] = 'background-color: #FF7070'
            k += 2
        return lst
    ############################################################################################
    # specify the styles for the html output
    styles=[{'selector': 'th','props': [
            ('border-style', 'solid'),
            ('border-color', '#D3D3D3'),
            ('vertical-align','top'),
            ('text-align','center')]}]
   
    html = (df.style.\
            apply(highlight_al_worse_than_atl,axis=1).\
            apply(highlight_atl_worse_than_al,axis=1).\
            #apply(highlight_max,axis=1,subset=pd.IndexSlice[:,col_al_atl]))
            apply(highlight_atl_exceed_bm,axis=1)).set_table_styles(styles).set_precision(3).format(col_tar_sup_format)
    display(html)
    with open('{}.html'.format(filename), 'w') as f:
        f.write(html.render())
    return df


#%%
    
def createWeightedResultsDFOneEstimator(al_results,atl_results,weighting,estimator_name,candsets_super_results,max_quota=200,filename='al_and_atl_results'):
    """
    This function creates a DataFrame with the results of each experiment. Per estimator the TL_avg (Transfer Learning Avg Result),
    Tar_max (Target max result when trained on 500 target instances), Tar_exc (Amount of Target Instances needed to exceed TL results),
    when trained on all features and dense features, are reported. Hence, amount of estimators*2*3 columns + 1 for unsupervised results
    of target are in the resulting DataFrame. It also stores a html file where important information is highlighted in the DataFrame.
    
    tl_results: Dictionary containting the results of the Transfer Learning experiements (as output by returnF1TLResultsFromDictWithPlot())
    candsets_unsuper_results: Dictionary containing the unsupervised results which act as benchmark (in the form of {'ban_half':0.732,'bx_half': 0.626,...})
    number_of_estimators: Integer indicating the amount of different estimators that were used for the TL experiments
    filename: Name of the html file that gets saved and contains the DataFrame with important information highlighted.
        Default: 'tl_results'
    """
    d = atl_results
    mean_2iter,mean_10iter,mean_20iter,mean_30iter,mean_50iter,mean_100iter,mean_maxiter = [],[],[],[],[],[],[]
    al_mean_2iter,al_mean_10iter,al_mean_20iter,al_mean_30iter,al_mean_50iter,al_mean_100iter,al_mean_maxiter = [],[],[],[],[],[],[]
    qss,column_list = [],[]
    max_str = '{}th'.format(max_quota)
    clf = estimator_name
    
    for key in d:
        target = '_'.join(key.split('_')[2:])
        for qs in d[key][clf]:
            qss.append(qs)
            mean_test_f1_score = np.mean(d[key][clf][qs][weighting]['test_f1_scores'],axis=0)
            mean_2iter.append(round(mean_test_f1_score[1],3))
            mean_10iter.append(round(mean_test_f1_score[9],3))
            mean_20iter.append(round(mean_test_f1_score[19],3))
            mean_30iter.append(round(mean_test_f1_score[29],3))
            mean_50iter.append(round(mean_test_f1_score[49],3))
            mean_100iter.append(round(mean_test_f1_score[99],3))
            mean_maxiter.append(round(mean_test_f1_score[max_quota-1],3))
            al_test_f1_scores = al_results[target][clf][qs]['test_f1_scores']
            n_init_labeled = al_results[target][clf][qs]['n_init_labeled']
            al_mean_test_f1_scores = np.mean(al_test_f1_scores,axis=0)
            al_mean_2iter.append(round(al_mean_test_f1_scores[1+n_init_labeled],3))
            column_list.append((clf,qs,'2nd','ATL'))
            column_list.append((clf,qs,'2nd','AL'))
            al_mean_10iter.append(round(al_mean_test_f1_scores[9+n_init_labeled],3))
            column_list.append((clf,qs,'10th','ATL'))
            column_list.append((clf,qs,'10th','AL'))
            al_mean_20iter.append(round(al_mean_test_f1_scores[19+n_init_labeled],3))
            column_list.append((clf,qs,'20th','ATL'))
            column_list.append((clf,qs,'20th','AL'))
            al_mean_30iter.append(round(al_mean_test_f1_scores[29+n_init_labeled],3))
            column_list.append((clf,qs,'30th','ATL'))
            column_list.append((clf,qs,'30th','AL'))
            al_mean_50iter.append(round(al_mean_test_f1_scores[49+n_init_labeled],3))
            column_list.append((clf,qs,'50th','ATL'))
            column_list.append((clf,qs,'50th','AL'))
            al_mean_100iter.append(round(al_mean_test_f1_scores[99+n_init_labeled],3))
            column_list.append((clf,qs,'100th','ATL'))
            column_list.append((clf,qs,'100th','AL'))
            al_mean_maxiter.append(round(al_mean_test_f1_scores[max_quota-1],3))
            column_list.append((clf,qs,max_str,'ATL'))
            column_list.append((clf,qs,max_str,'AL'))
    al_atl_mean_2iter = list(zip(mean_2iter,al_mean_2iter))
    al_atl_mean_10iter = list(zip(mean_10iter,al_mean_10iter))
    al_atl_mean_20iter = list(zip(mean_20iter,al_mean_20iter))
    al_atl_mean_30iter = list(zip(mean_30iter,al_mean_30iter))
    al_atl_mean_50iter = list(zip(mean_50iter,al_mean_50iter))
    al_atl_mean_100iter = list(zip(mean_100iter,al_mean_100iter))
    al_atl_mean_maxiter = list(zip(mean_maxiter,al_mean_maxiter))
    
    number_of_estimators = 1

    # get the amount of different query strategies
    number_of_qs = len(list(set(qss)))
    # ensure that the order is maintained 
    qss = qss[:number_of_qs]
    n = number_of_estimators*number_of_qs*7*2 #number_of_estimators*number_of_qs*Iterations (5) *AL Results and ATL Results (2)
    data = list(group(list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(list(zip(al_atl_mean_2iter,al_atl_mean_10iter,al_atl_mean_20iter,al_atl_mean_30iter,al_atl_mean_50iter,al_atl_mean_100iter,al_atl_mean_maxiter)))))),n))
    col_tuple_list = list(group(column_list,n))
    single_col_tuple = list(set(col_tuple_list))
    columns_list = [item for t in single_col_tuple for item in t]
    keys = [('_'.join(k.split('_')[:2]),'_'.join(k.split('_')[2:])) for k in d]
    df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(keys,names=['Source','Target']), columns=pd.MultiIndex.from_tuples(columns_list,names=['Estimators','QS','Iteration','Results']))
    
    if(clf == 'lr'):
        clf_old = 'logreg'
    elif(clf == 'rf'):
        clf_old = 'randforest'
    elif(clf == 'lrcv'):
        clf_old = 'logregcv'
    elif(clf == 'dt'):
        clf_old = 'dectree'
    else:
        clf_old = clf
        
    values = [round(candsets_super_results[key][clf_old],3) for key in candsets_super_results.keys()]
    df_super = pd.DataFrame(values,index=candsets_super_results.keys(),columns=[clf])
    
    if('logreg' in df_super.columns):
        df_super.rename(columns={'logreg':'lr'},inplace=True)
    if('randforest' in df_super.columns):
        df_super.rename(columns={'randforest':'rf'},inplace=True)
    if('logregcv' in df_super.columns):
        df_super.rename(columns={'logregcv':'lrcv'},inplace=True)
    if('dectree' in df_super.columns):
        df_super.rename(columns={'dectree':'dt'},inplace=True)
    
    #df_super = df_super[clf_org] 
    #columns_before = list(df_super.columns)
    #print(columns_before)
    df_super = pd.concat([df_super,df_super,df_super],axis=1)
    #df_super = df_super[clf_org]
    #print(qss)
    df_super.columns = pd.MultiIndex.from_product([[clf],qss,['all'],['Tar_sup']],names=['Estimators','QS','Iterations','Results'])
    df = pd.merge(df,df_super,how='left',left_on='Target',right_index=True)
    df = df.reindex(columns=[clf],level=0)
    df = df.reindex(columns=qss,level=1)
    df = df.reindex(columns=['2nd','10th','20th','30th','50th','100th',max_str,'all'],level=2)
    #col_al_atl = [col for col in df.columns if col[3]=='ATL' or col[3]=='AL']
    col_tar_sup = [col for col in df.columns if col[3]=='Tar_sup']
    col_tar_sup_format = {col:lambda x: '<font color=\'#000000\'><b>{}</b></font>'.format(x) for col in col_tar_sup}
    ###########################################################################################
    # style functions as nested functions
    # function for pandas styler. Highlight all max values of TL_avg
    def highlight_max(data):
        attr_max = 'background-color: #FBFF75'
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = data == data.max()
            return [attr_max if v else '' for v in is_max]
        else: 
            is_max = data.groupby(['AL','ATL']).transform('max') == data
            return pd.DataFrame(np.where(is_max, attr_max, ''),
                                    index=data.index, columns=data.columns)
    # another function for pandas style      
    def highlight_atl_exceed_bm(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL'].reset_index(drop=True)>=row[:,:,'all','Tar_sup'].repeat(7).reset_index(drop=True))
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i!=0 and i%7==0):
                k += 1
            if(b):
                lst[k] = 'background-color: #FBFF75'
            k += 2
        return lst
    
    # another function for pandas style      
    def highlight_al_worse_than_atl(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL']>row[:,:,:,'AL'])
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i!=0 and i%7==0):
                k += 1
            if(b):
                lst[k] = 'color:#000000;background-color:rgba(51, 204, 51, .1);'
            k += 2
        return lst
    # another function for pandas style      
    def highlight_atl_worse_than_al(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL']<row[:,:,:,'AL'])
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 1
        for i, b in enumerate(ser):
            if(i!=0 and i%7==0):
                k += 1
            if(b):
                lst[k] = 'background-color: #FF7070'
            k += 2
        return lst
    ############################################################################################
    # specify the styles for the html output
    styles=[{'selector': 'th','props': [
            ('border-style', 'solid'),
            ('border-color', '#D3D3D3'),
            ('vertical-align','top'),
            ('text-align','center')]}]
   
    html = (df.style.\
            apply(highlight_al_worse_than_atl,axis=1).\
            apply(highlight_atl_worse_than_al,axis=1).\
            #apply(highlight_max,axis=1,subset=pd.IndexSlice[:,col_al_atl]))
            apply(highlight_atl_exceed_bm,axis=1)).set_table_styles(styles).set_precision(3).format(col_tar_sup_format)
    display(html)
    with open('{}.html'.format(filename), 'w') as f:
        f.write(html.render())
    return df
#%%
    
def highlightDFWithATLandALResultsWithOnlyOneEstimator(df,number_iters,filename=None):
    col_tar_sup = [col for col in df.columns if col[3]=='Tar_sup']
    col_tar_sup_format = {col:lambda x: '<font color=\'#000000\'><b>{}</b></font>'.format(x) for col in col_tar_sup}
    ###########################################################################################
    # style functions as nested functions
    # function for pandas styler. Highlight all max values of TL_avg
    def highlight_max(data):
        attr_max = 'background-color: #FBFF75'
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = data == data.max()
            return [attr_max if v else '' for v in is_max]
        else: 
            is_max = data.groupby(['AL','ATL']).transform('max') == data
            return pd.DataFrame(np.where(is_max, attr_max, ''),
                                    index=data.index, columns=data.columns)
    # another function for pandas style      
    def highlight_atl_exceed_bm(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]

        ser = (row[:,:,:,'ATL'].reset_index(drop=True)>=row[:,:,'all','Tar_sup'].repeat(number_iters).reset_index(drop=True))
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i!=0 and i%number_iters==0):
                k += 1
            if(b):
                lst[k] = 'background-color: #FBFF75'
            k += 2
        return lst
    
    # another function for pandas style      
    def highlight_al_worse_than_atl(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL']>row[:,:,:,'AL'])
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i!=0 and i%number_iters==0):
                k += 1
            if(b):
                lst[k] = 'color:#000000;background-color:rgba(51, 204, 51, .1);'
            k += 2
        return lst
    # another function for pandas style      
    def highlight_atl_worse_than_al(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL']<row[:,:,:,'AL'])
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 1
        for i, b in enumerate(ser):
            if(i!=0 and i%number_iters==0):
                k += 1
            if(b):
                lst[k] = 'background-color: #FF7070'
            k += 2
        return lst
    ############################################################################################
    # specify the styles for the html output
    styles=[{'selector': 'th','props': [
            ('border-style', 'solid'),
            ('border-color', '#D3D3D3'),
            ('vertical-align','top'),
            ('text-align','center')]}]
   
    html = (df.style.\
            apply(highlight_al_worse_than_atl,axis=1).\
            apply(highlight_atl_worse_than_al,axis=1).\
            #apply(highlight_max,axis=1,subset=pd.IndexSlice[:,col_al_atl]))
            apply(highlight_atl_exceed_bm,axis=1)).set_table_styles(styles).set_precision(3).format(col_tar_sup_format)
    display(html)
    if(isinstance(filename,str)):
        with open('{}.html'.format(filename), 'w') as f:
            f.write(html.render())
    
    return None