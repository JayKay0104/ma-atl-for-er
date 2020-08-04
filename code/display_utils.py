# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:05:26 2020

@author: jonas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
#sns.set_context("paper")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 3,"font.size":19, "axes.labelsize":19,
    "axes.titlesize":19})
#, "xtick.labelsize","ytick.labelsize", "legend.fontsize"
sns.set_color_codes()
import itertools
import math
import copy
from IPython.display import display




#%%
# generator function to create tuples of every n element
# from https://stackoverflow.com/questions/15480483/split-string-into-list-of-tuples
def group(lst, n):
    for i in range(0, len(lst), n):
        val = lst[i:i+n]
        if len(val) == n:
            yield tuple(val)

#%%            
### Help Functions to display DataFrames with the Baseline Results (Passive Learning and Unsupervised Matching) ###
         
def returnDFWithSuper(candsets_super_results,filename=None):
    innerkeys =  [innerkey for k,innerdict in candsets_super_results.items() for innerkey, values in innerdict.items()]
    values = [round(value['f1'],3) for k,innerdict in candsets_super_results.items() for innerkey, value in innerdict.items()]
    number_of_estimators = len(set(innerkeys))
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
    
def returnDFWithUnsuper(candsets_unsuper_results,filename=None):
    f1 = [round(value['f1'],3) for k,value in candsets_unsuper_results.items()]
    elb = [round(value['elbow_threshold'],3) for k,value in candsets_unsuper_results.items()]
    data = {'F1 score':f1,'Elbow Threshold':elb}
    df_unsuper = pd.DataFrame(data,index=candsets_unsuper_results.keys())
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
            apply(highlight_max,axis=0,subset=pd.IndexSlice['F1 score'])).set_table_styles(styles).set_precision(3)
    display(html)
    if filename is not None:
        with open('{}.html'.format(filename), 'w') as f:
            f.write(html.render())
    return df_unsuper

#%%
    
def plotATLXALUnsupResultsAll(atlx_results,quota,candsets,candsets_super_results,n,warm_start,selected_estimator=None,al_results=None,
                   selected_qs=None,selected_weights=None,errorbars=False,saveFig=True,path_for_output='./graphics/custom_plots/',nrows=False,ncols=2,figsize=(16,20)):
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
    d = atlx_results
    
    number_combos = len(d.keys())
        
    qss,estimators = [],[]
    if nrows==False:
        fig,ax = plt.subplots(nrows=int(number_combos/ncols), ncols=ncols, figsize=figsize,constrained_layout=True)
    else:
        fig,ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,constrained_layout=True)
        
    keys = list(d.keys())
    k = 0
    for i,combo in enumerate(d):
        if(not isinstance(keys[0],tuple)):
            source = '_'.join(combo.split('_')[:2])
            target = '_'.join(combo.split('_')[2:])
        else:
            source = combo[0]
            target = combo[1]
        
        ax[k,i%ncols].set_xlabel('x target instances queried',fontsize=17)
        ax[k,i%ncols].set_ylabel('Avg. F1-Score',fontsize=17)


        if(selected_estimator is None):
            for clf in d[combo]:
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
                    for qs in d[combo][clf]:
                        qss.append(qs)
                        if selected_weights is None:
                            for weight in d[combo][clf][qs]:
                                plt.close()
                                max_quota = d[combo][clf][qs][weight]['quota']
                                x_atl_results = np.arange(1, max_quota + 1)
                                atl_test_f1_scores = np.array(d[combo][clf][qs][weight]['test_f1_scores'])
                                y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                                std_atl_results = np.std(atl_test_f1_scores, axis=0)
                                #max_quota = d[combo][clf][qs][weight]['quota']
                                if(errorbars):
                                    ax[k,i%ncols].errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                        label='ATLX: {}; {}'.format(weight,qs))
                                else:
                                    ax[k,i%ncols].plot(x_atl_results,y_atl_results,linewidth=2,
                                        label='ATLX: {}; {}'.format(weight,qs))
                        else:
                            for weight in selected_weights:
                                plt.close()
                                max_quota = d[combo][clf][qs][weight]['quota']
                                x_atl_results = np.arange(1, max_quota + 1)
                                atl_test_f1_scores = np.array(d[combo][clf][qs][weight]['test_f1_scores'])
                                y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                                std_atl_results = np.std(atl_test_f1_scores, axis=0)
                                if(errorbars):
                                    ax[k,i%ncols].errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                                label='ATLX: {}; {}'.format(weight,qs))
                                else:
                                    ax[k,i%ncols].plot(x_atl_results,y_atl_results,linewidth=2,
                                            label='ATLX: {}; {}'.format(weight,qs))

                        if(al_results is not None):
                            d2 = al_results
                            plt.close()
                            al_max_quota = d2[target][clf][qs]['quota']
                            x_al_results = np.arange(1, al_max_quota + 1)
                            al_test_f1_scores = np.array(d2[target][clf][qs]['test_f1_scores'])
                            y_al_results = np.mean(al_test_f1_scores,axis=0)
                            #n_init_labeled = d2[target][clf][qs]['n_init_labeled']
                            std_al_results = np.std(al_test_f1_scores, axis=0)
                            if(errorbars):
                                ax[k,i%ncols].errorbar(x_al_results, y_al_results, yerr=std_al_results,
                                            label='AL Unsup Boot')
                            else:
                                ax[k,i%ncols].plot(x_al_results,y_al_results,linewidth=2,linestyle='dashed',
                                        label='AL Unsup Boot')
                else:
                    for qs in selected_qs:
                        qss.append(qs)
                        if selected_weights is None:
                            for weight in d[combo][clf][qs]:
                                plt.close()
                                max_quota = d[combo][clf][qs][weight]['quota']
                                x_atl_results = np.arange(1, max_quota + 1)
                                atl_test_f1_scores = np.array(d[combo][clf][qs][weight]['test_f1_scores'])
                                y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                                std_atl_results = np.std(atl_test_f1_scores, axis=0)
                                if(errorbars):
                                    ax[k,i%ncols].errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                                label='ATLX: {}; {}'.format(weight,qs))
                                else:
                                    ax[k,i%ncols].plot(x_atl_results,y_atl_results,linewidth=2,
                                            label='ATLX: {}; {}'.format(weight,qs))
                        else:
                            for weight in selected_weights:
                                plt.close()
                                max_quota = d[combo][clf][qs][weight]['quota']
                                x_atl_results = np.arange(1, max_quota + 1)
                                atl_test_f1_scores = np.array(d[combo][clf][qs][weight]['test_f1_scores'])
                                y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                                std_atl_results = np.std(atl_test_f1_scores, axis=0)
                                if(errorbars):
                                    ax[k,i%ncols].errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                                label='ATLX: {}; {}'.format(weight,qs))
                                else:
                                    ax[k,i%ncols].plot(x_atl_results,y_atl_results,linewidth=2,
                                            label='ATLX: {}; {}'.format(weight,qs))

                        if(al_results is not None):
                            d2 = al_results
                            al_max_quota = d2[target][clf][qs]['quota']
                            x_al_results = np.arange(1, al_max_quota + 1)
                            al_test_f1_scores = np.array(d2[target][clf][qs]['test_f1_scores'])
                            y_al_results = np.mean(al_test_f1_scores,axis=0)
                            #n_init_labeled = d2[target][clf][qs]['n_init_labeled']
                            std_al_results = np.std(al_test_f1_scores, axis=0)
                            if(errorbars):
                                ax[k,i%ncols].errorbar(x_al_results, y_al_results, yerr=std_al_results,
                                            label='AL Unsup Boot')
                                ax[k,i%ncols].plot(x_al_results,y_al_results,linewidth=2,linestyle='dashed',
                                        label='AL Unsup Boot')
                # benchmark plots
                # supervised
                f1_target_bm = candsets_super_results[target][clf_old]['f1']
                y_target_sup_bm = list(itertools.repeat(f1_target_bm,len(x)))
                ax[k,i%ncols].plot(x,y_target_sup_bm,linewidth=3,linestyle='dotted',label='target supervised ({}) benchmark {} instances F1: {:.2f}'.format(clf,candsets[target].shape[0],f1_target_bm))
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
                    for qs in d[combo][clf]:
                        qss.append(qs)
                        if selected_weights is None:
                            for weight in d[combo][clf][qs]:
                                plt.close()
                                max_quota = d[combo][clf][qs][weight]['quota']
                                x_atl_results = np.arange(1, max_quota + 1)
                                atl_test_f1_scores = np.array(d[combo][clf][qs][weight]['test_f1_scores'])
                                y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                                std_atl_results = np.std(atl_test_f1_scores, axis=0)
                                if(errorbars):
                                    ax[k,i%ncols].errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                                label='ATLX: {}; {}'.format(weight,qs))
                                else:
                                    ax[k,i%ncols].plot(x_atl_results,y_atl_results,linewidth=2,
                                            label='ATLX: {}; {}'.format(weight,qs))
                        else:
                            for weight in selected_weights:
                                plt.close()
                                max_quota = d[combo][clf][qs][weight]['quota']
                                x_atl_results = np.arange(1, max_quota + 1)
                                atl_test_f1_scores = np.array(d[combo][clf][qs][weight]['test_f1_scores'])
                                y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                                std_atl_results = np.std(atl_test_f1_scores, axis=0)
                                if(errorbars):
                                    ax[k,i%ncols].errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                                label='ATLX: {}; {}'.format(weight,qs))
                                else:
                                    ax[k,i%ncols].plot(x_atl_results,y_atl_results,linewidth=2,
                                            label='ATLX: {}; {}'.format(weight,qs))

                    if(al_results is not None):
                        d2 = al_results
                        plt.close()
                        al_max_quota = d2[target][clf][qs]['quota']
                        x_al_results = np.arange(1, al_max_quota + 1)
                        al_test_f1_scores = np.array(d2[target][clf][qs]['test_f1_scores'])
                        y_al_results = np.mean(al_test_f1_scores,axis=0)
                        #n_init_labeled = d2[target][clf][qs]['n_init_labeled']
                        std_al_results = np.std(al_test_f1_scores, axis=0)
                        if(errorbars):
                            ax[k,i%ncols].errorbar(x_al_results, y_al_results, yerr=std_al_results,
                                        label='AL Unsup Boot')
                        else:
                            ax[k,i%ncols].plot(x_al_results,y_al_results,linewidth=2,linestyle='dashed',
                                    label='AL Unsup Boot')
                else:
                    for qs in selected_qs:
                        qss.append(qs)
                        if selected_weights is None:
                            for weight in d[combo][clf][qs]:
                                plt.close()
                                max_quota = d[combo][clf][qs][weight]['quota']
                                x_atl_results = np.arange(1, max_quota + 1)
                                atl_test_f1_scores = np.array(d[combo][clf][qs][weight]['test_f1_scores'])
                                y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                                std_atl_results = np.std(atl_test_f1_scores, axis=0)
                                if(errorbars):
                                    ax[k,i%ncols].errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                                label='ATLX: {}; {}'.format(weight,qs))
                                else:
                                    ax[k,i%ncols].plot(x_atl_results,y_atl_results,linewidth=2,
                                            label='ATLX: {}; {}'.format(weight,qs))
                        else:
                            for weight in selected_weights:
                                plt.close()
                                max_quota = d[combo][clf][qs][weight]['quota']
                                x_atl_results = np.arange(1, max_quota + 1)
                                atl_test_f1_scores = np.array(d[combo][clf][qs][weight]['test_f1_scores'])
                                y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                                std_atl_results = np.std(atl_test_f1_scores, axis=0)
                                if(errorbars):
                                    ax[k,i%ncols].errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                                label='ATLX: {}; {}'.format(weight,qs))
                                else:
                                    ax[k,i%ncols].plot(x_atl_results,y_atl_results,linewidth=2,
                                            label='ATLX: {}; {}'.format(weight,qs))

                        if(al_results is not None):
                            d2 = al_results
                            al_max_quota = d2[target][clf][qs]['quota']
                            x_al_results = np.arange(1, al_max_quota + 1)
                            al_test_f1_scores = np.array(d2[target][clf][qs]['test_f1_scores'])
                            y_al_results = np.mean(al_test_f1_scores,axis=0)
                            #n_init_labeled = d2[target][clf][qs]['n_init_labeled']
                            std_al_results = np.std(al_test_f1_scores, axis=0)
                            if(errorbars):
                                ax[k,i%ncols].errorbar(x_al_results, y_al_results, yerr=std_al_results,
                                            label='AL Unsup Boot')
                            else:
                                ax[k,i%ncols].plot(x_al_results,y_al_results,linewidth=2,linestyle='dashed',
                                        label='AL Unsup Boot')
                # benchmark plots
                # supervised
                f1_target_bm = candsets_super_results[target][clf_old]['f1']
                y_target_sup_bm = list(itertools.repeat(f1_target_bm,len(x)))
                ax[k,i%ncols].plot(x,y_target_sup_bm,linewidth=3,linestyle='dotted',label='passive learning target {} inst.'.format(candsets[target].shape[0]))
            
            # add legend to plot
            #ax[k,i%ncols].legend(fontsize=17)
            #ax[k,i%ncols].set_title('source: {}, target = {}'.format(source,target),fontsize=21)
            if(i%ncols==2):
                k += 1
    
    #lines, labels = fig.axes[-1].get_legend_handles_labels()
    
    #fig.legend(lines, labels, loc='lower left', bbox_to_anchor= (0.2, 1.01), ncol=2,
    #           borderaxespad=0, frameon=False,fontsize=21)
    #fig.suptitle('ATLX Evaluation',fontsize=21)
    #fig.tight_layout()
    
    if(saveFig):
        fig.savefig(path_for_output,dpi=600,bbox_inches='tight')
    
    return fig


#%%
### Help functions to plot and display the ATL RF compared to AL Unsupervised Bootstrapping Method ###
    
def plotATLXALUnsupResults(atlx_results,source,target,quota,candsets,candsets_super_results,n,warm_start,selected_estimator=None,al_results=None,
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
    d = atlx_results
    
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
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            #max_quota = d[source_target][clf][qs][weight]['quota']
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                    label='{}: ATLX {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                    label='{}: ATLX {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                    else:
                        for weight in selected_weights:
                            plt.close()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                            label='{}: ATLX {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                        label='{}: ATLX {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                    
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
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                            label='{}: ATLX {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                        label='{}: ATLX {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                    else:
                        for weight in selected_weights:
                            plt.close()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                            label='{}: ATLX {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                        label='{}: ATLX {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                
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
            f1_target_bm = candsets_super_results[target][clf_old]['f1']
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
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                            label='{}: ATLX {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                        label='{}: ATLX {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                    else:
                        for weight in selected_weights:
                            plt.close()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                            label='{}: ATLX {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                        label='{}: ATLX {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                                
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
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                            label='{}: ATLX {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                        label='{}: ATLX {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                    else:
                        for weight in selected_weights:
                            plt.close()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,
                                            label='{}: ATLX {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,linewidth=2,
                                        label='{}: ATLX {}: {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,warm_start,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                
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
            f1_target_bm = candsets_super_results[target][clf_old]['f1']
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
    
    ax.set_title('Active-Transfer Learning (ATLX {}):\nIncorporating labeled source data then iteratively query target instances and add to training set\nSource: {} and Target {}'.format(warm_start,source,target),fontsize=14)
    
    if(saveFig):
        fig.savefig('{}{}_{}_{}_{}_n{}.png'.format(path_for_output,source,target,'_'.join(estimators),'_'.join(qss),n),bbox_inches='tight')
    
    if(ylim is not None):
        ax.set_ylim(ylim)
    return fig

    
def createDFwithAlUnsupandATLXResultsMoreItersUpToMaxQuota(atlx_results,al_unsup_results,selected_weighting,candsets_super_results,selected_qs=None,max_quota=100,filename=None):
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
    d = atlx_results
    
    mean_2iter,mean_10iter,mean_20iter,mean_30iter,mean_50iter,mean_100iter = [],[],[],[],[],[]
    
    al_mean_2iter,al_mean_10iter,al_mean_20iter,al_mean_30iter,al_mean_50iter,al_mean_100iter = [],[],[],[],[],[]
    qss,column_list = [],[]
    
    keys = []
    for key in d:       
        if(not isinstance(key,tuple)):
            target = '_'.join(key.split('_')[2:])
            keys.append(('_'.join(key.split('_')[:2]),'_'.join(key.split('_')[2:])))
        else:
            target = key[1]
            keys.append(key)
        clf = 'rf'
        if(selected_qs is None):
            for qs in d[key][clf]:
                qss.append(qs)
                al_test_f1_scores = al_unsup_results[target][clf][qs]['test_f1_scores']
                al_mean_test_f1_scores = np.mean(al_test_f1_scores,axis=0)
                al_mean_2iter.append(round(al_mean_test_f1_scores[1],3))
                column_list.append((clf,qs,'2nd','ATLX'))
                column_list.append((clf,qs,'2nd','AL'))
                al_mean_10iter.append(round(al_mean_test_f1_scores[9],3))
                column_list.append((clf,qs,'10th','ATLX'))
                column_list.append((clf,qs,'10th','AL'))
                al_mean_20iter.append(round(al_mean_test_f1_scores[19],3))
                column_list.append((clf,qs,'20th','ATLX'))
                column_list.append((clf,qs,'20th','AL'))
                al_mean_30iter.append(round(al_mean_test_f1_scores[29],3))
                column_list.append((clf,qs,'30th','ATLX'))
                column_list.append((clf,qs,'30th','AL'))
                al_mean_50iter.append(round(al_mean_test_f1_scores[49],3))
                column_list.append((clf,qs,'50th','ATLX'))
                column_list.append((clf,qs,'50th','AL'))
                al_mean_100iter.append(round(al_mean_test_f1_scores[99],3))
                column_list.append((clf,qs,'100th','ATLX'))
                column_list.append((clf,qs,'100th','AL'))
                mean_2iter.append(round(np.mean(d[key][clf][qs][selected_weighting]['test_f1_scores'],axis=0)[1],3))
                mean_10iter.append(round(np.mean(d[key][clf][qs][selected_weighting]['test_f1_scores'],axis=0)[9],3))
                mean_20iter.append(round(np.mean(d[key][clf][qs][selected_weighting]['test_f1_scores'],axis=0)[19],3))
                mean_30iter.append(round(np.mean(d[key][clf][qs][selected_weighting]['test_f1_scores'],axis=0)[29],3))
                mean_50iter.append(round(np.mean(d[key][clf][qs][selected_weighting]['test_f1_scores'],axis=0)[49],3))
                mean_100iter.append(round(np.mean(d[key][clf][qs][selected_weighting]['test_f1_scores'],axis=0)[99],3))
        
        elif(isinstance(selected_qs,list)):
            for qs in selected_qs:
                qss.append(qs)
                al_test_f1_scores = al_unsup_results[target][clf][qs]['test_f1_scores']
                al_mean_test_f1_scores = np.mean(al_test_f1_scores,axis=0)
                al_mean_2iter.append(round(al_mean_test_f1_scores[1],3))
                column_list.append((clf,qs,'2nd','ATLX'))
                column_list.append((clf,qs,'2nd','AL'))
                al_mean_10iter.append(round(al_mean_test_f1_scores[9],3))
                column_list.append((clf,qs,'10th','ATLX'))
                column_list.append((clf,qs,'10th','AL'))
                al_mean_20iter.append(round(al_mean_test_f1_scores[19],3))
                column_list.append((clf,qs,'20th','ATLX'))
                column_list.append((clf,qs,'20th','AL'))
                al_mean_30iter.append(round(al_mean_test_f1_scores[29],3))
                column_list.append((clf,qs,'30th','ATLX'))
                column_list.append((clf,qs,'30th','AL'))
                al_mean_50iter.append(round(al_mean_test_f1_scores[49],3))
                column_list.append((clf,qs,'50th','ATLX'))
                column_list.append((clf,qs,'50th','AL'))
                al_mean_100iter.append(round(al_mean_test_f1_scores[99],3))
                column_list.append((clf,qs,'100th','ATLX'))
                column_list.append((clf,qs,'100th','AL'))
                mean_2iter.append(round(np.mean(d[key][clf][qs][selected_weighting]['test_f1_scores'],axis=0)[1],3))
                mean_10iter.append(round(np.mean(d[key][clf][qs][selected_weighting]['test_f1_scores'],axis=0)[9],3))
                mean_20iter.append(round(np.mean(d[key][clf][qs][selected_weighting]['test_f1_scores'],axis=0)[19],3))
                mean_30iter.append(round(np.mean(d[key][clf][qs][selected_weighting]['test_f1_scores'],axis=0)[29],3))
                mean_50iter.append(round(np.mean(d[key][clf][qs][selected_weighting]['test_f1_scores'],axis=0)[49],3))
                mean_100iter.append(round(np.mean(d[key][clf][qs][selected_weighting]['test_f1_scores'],axis=0)[99],3))
        else:
            qss.append(selected_qs)
            al_test_f1_scores = al_unsup_results[target][clf][selected_qs]['test_f1_scores']
            al_mean_test_f1_scores = np.mean(al_test_f1_scores,axis=0)
            al_mean_2iter.append(round(al_mean_test_f1_scores[1],3))
            column_list.append((clf,selected_qs,'2nd','ATLX'))
            column_list.append((clf,selected_qs,'2nd','AL'))
            al_mean_10iter.append(round(al_mean_test_f1_scores[9],3))
            column_list.append((clf,selected_qs,'10th','ATLX'))
            column_list.append((clf,selected_qs,'10th','AL'))
            al_mean_20iter.append(round(al_mean_test_f1_scores[19],3))
            column_list.append((clf,selected_qs,'20th','ATLX'))
            column_list.append((clf,selected_qs,'20th','AL'))
            al_mean_30iter.append(round(al_mean_test_f1_scores[29],3))
            column_list.append((clf,selected_qs,'30th','ATLX'))
            column_list.append((clf,selected_qs,'30th','AL'))
            al_mean_50iter.append(round(al_mean_test_f1_scores[49],3))
            column_list.append((clf,selected_qs,'50th','ATLX'))
            column_list.append((clf,selected_qs,'50th','AL'))
            al_mean_100iter.append(round(al_mean_test_f1_scores[99],3))
            column_list.append((clf,selected_qs,'100th','ATLX'))
            column_list.append((clf,selected_qs,'100th','AL'))
            mean_2iter.append(round(np.mean(d[key][clf][selected_qs][selected_weighting]['test_f1_scores'],axis=0)[1],3))
            mean_10iter.append(round(np.mean(d[key][clf][selected_qs][selected_weighting]['test_f1_scores'],axis=0)[9],3))
            mean_20iter.append(round(np.mean(d[key][clf][selected_qs][selected_weighting]['test_f1_scores'],axis=0)[19],3))
            mean_30iter.append(round(np.mean(d[key][clf][selected_qs][selected_weighting]['test_f1_scores'],axis=0)[29],3))
            mean_50iter.append(round(np.mean(d[key][clf][selected_qs][selected_weighting]['test_f1_scores'],axis=0)[49],3))
            mean_100iter.append(round(np.mean(d[key][clf][selected_qs][selected_weighting]['test_f1_scores'],axis=0)[99],3))
                

    al_atl_mean_2iter = list(zip(mean_2iter,al_mean_2iter))
    al_atl_mean_10iter = list(zip(mean_10iter,al_mean_10iter))
    al_atl_mean_20iter = list(zip(mean_20iter,al_mean_20iter))
    al_atl_mean_30iter = list(zip(mean_30iter,al_mean_30iter))
    al_atl_mean_50iter = list(zip(mean_50iter,al_mean_50iter))
    al_atl_mean_100iter = list(zip(mean_100iter,al_mean_100iter))

    
    # get the amount of different query strategies
    number_of_qs = len(list(set(qss)))
    # ensure that the order is maintained 
    qss = qss[:number_of_qs]
    n = number_of_qs*6*2 #number_of_qs*Iterations (6) *AL Unsup Results and ATLX RF Results (2)
    data = list(group(list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(list(zip(al_atl_mean_2iter,al_atl_mean_10iter,al_atl_mean_20iter,al_atl_mean_30iter,al_atl_mean_50iter,al_atl_mean_100iter)))))),n))
    col_tuple_list = list(group(column_list,n))
    single_col_tuple = list(set(col_tuple_list))
    columns_list = [item for t in single_col_tuple for item in t]

    df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(keys,names=['Source','Target']), columns=pd.MultiIndex.from_tuples(columns_list,names=['Estimator','QS','Iteration','Results']))
    
    values = [round(candsets_super_results[key]['randforest']['f1'],3) for key in candsets_super_results.keys()]
    df_super = pd.DataFrame(values,index=candsets_super_results.keys(),columns=[clf])
    

    if(number_of_qs==2):
        df_super = pd.concat([df_super,df_super],axis=1)
        df_super = df_super[[clf]]
        
    if(number_of_qs==3):
        df_super = pd.concat([df_super,df_super,df_super],axis=1)
        df_super = df_super[[clf]]

    df_super.columns = pd.MultiIndex.from_product([[clf],qss,['all'],['Tar_sup']],names=['Estimator','QS','Iterations','Results'])
    df = pd.merge(df,df_super,how='left',left_on='Target',right_index=True)
    df = df.reindex(columns=[clf],level=0)
    df = df.reindex(columns=qss,level=1)
    df = df.reindex(columns=['2nd','10th','20th','30th','50th','100th','all'],level=2)
    #col_al_atl = [col for col in df.columns if col[3]=='ATL' or col[3]=='AL']
    print(f'Weighting: {selected_weighting} and Query Strategy: {selected_qs}')
    
    styleDFwithAlUnsupandATLXResultsMoreItersUpToMaxQuota(df,qss,number_of_qs,filename)
            
    return df

def styleDFwithAlUnsupandATLXResultsMoreItersUpToMaxQuota(df,qss,number_of_qs,filename):
    col_tar_sup = [col for col in df.columns if col[3]=='Tar_sup']
    col_tar_sup_format = {col:lambda x: '<font color=\'#000000\'><b>{}</b></font>'.format(x) for col in col_tar_sup}
                                               
    ###########################################################################################
    # style functions as nested functions
   
    def highlight_atl_exceed_bm(row):
        # initiate list with the size of the row with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. 
        ser = (row[:,:,:,'ATLX'].reset_index(drop=True)>=row[:,:,'all','Tar_sup'].repeat(6).reset_index(drop=True))
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 6 (i+k). K has to be incremented by 2 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i!=0 and i%6==0):
                k += 1
            if(b):
                lst[k] = 'background-color: #FBFF75'
            k += 2
        return lst
    
    # another function for pandas style      
    def highlight_al_worse_than_atl(row):
        # initiate list with the size of the row with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. 
        ser = (row[:,:,:,'ATLX']-row[:,:,:,'AL'])<0.01
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 6 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i!=0 and i%6==0):
                k += 1
            if(b):
                lst[k] = 'color:#000000;background-color:rgba(51, 204, 51, .1);'
            k += 2
        return lst
    
    def highlight_qs_worse_than_rs(row):
        # initiate list with the size of the row with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. 
        ser = (row[:,qss[0],:,'ATLX']-row[:,'random',:,'ATLX'])>0.01
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 6 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i<6 and not b):
                # yellow: #FFFF00
                lst[k] = 'background-color:#FFA500'
            #elif(i<6 and b):
            #    lst[k] = 'color:#000000;background-color:rgba(51, 204, 51, .1);'
            else:
                lst[k] = ''
            k += 2
        return lst
    
    def highlight_qs2_worse_than_rs(row):
        # initiate list with the size of the row with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. 
        ser1 = (row[:,qss[0],:,'ATLX']-row[:,'random',:,'ATLX'])>0.01
        ser2 = (row[:,qss[1],:,'ATLX']-row[:,'random',:,'ATLX'])>0.01
        ser = ser1.append(ser2)
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 6 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i<6 and not b):
                # yellow: #FFFF00
                lst[k] = 'background-color:#FFA500'
            #elif(i<6 and b):
            #    lst[k] = 'color:#000000;background-color:rgba(51, 204, 51, .1);'
            elif(i>=6 and i<12 and not b):
                # yellow: #FFFF00
                if(i==6):
                    k += 1
                lst[k] = 'background-color:#FFA500'
            elif(i==6):
                k += 1
            else:
                lst[k] = ''
            k += 2
        return lst
    
#    # another function for pandas style      
    def highlight_atl_worse_than_al(row):
        # initiate list with the size of the row with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATLX']<row[:,:,:,'AL'])
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 6 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 1
        for i, b in enumerate(ser):
            if(i!=0 and i%6==0):
                k += 1
            if(b):
                lst[k] = 'background-color: #FF7070'
            k += 2
        return lst
#    ############################################################################################
    # specify the styles for the html output
    styles=[{'selector': 'th','props': [
            ('border-style', 'solid'),
            ('border-color', '#D3D3D3'),
            ('vertical-align','top'),
            ('text-align','center')]}]
   
    if(number_of_qs==3):
        html = (df.style.\
            apply(highlight_qs2_worse_than_rs,axis=1).\
            apply(highlight_atl_worse_than_al,axis=1).\
            apply(highlight_atl_exceed_bm,axis=1)).set_table_styles(styles).set_precision(3).format(col_tar_sup_format)
        
    else:
        html = (df.style.\
                apply(highlight_qs_worse_than_rs,axis=1).\
                #apply(highlight_al_worse_than_atl,axis=1).\
                apply(highlight_atl_worse_than_al,axis=1).\
                apply(highlight_atl_exceed_bm,axis=1)).set_table_styles(styles).set_precision(3).format(col_tar_sup_format)
    
    display(html)
    if(filename is not None):
        with open('{}.html'.format(filename), 'w') as f:
            f.write(html.render())


def createDFwithAlUnsupandATLXResultsToCompareDA(atlx_results,al_unsup_results,candsets_super_results,selected_qs='lr_lsvc_rf_dt',max_quota=100,filename=None):
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
    d = atlx_results
    #selected_qs = 'lr_lsvc_rf_dt'
    mean_2iter,mean_20iter,mean_50iter,mean_100iter = [],[],[],[]
    
    column_list = []
    
    keys = []
    for key in d:       
        if(not isinstance(key,tuple)):
            target = '_'.join(key.split('_')[2:])
            keys.append(('_'.join(key.split('_')[:2]),'_'.join(key.split('_')[2:])))
        else:
            target = key[1]
            keys.append(key)
    
        clf = 'rf'
        al_mean_test_f1_scores = np.mean(al_unsup_results[target][clf][selected_qs]['test_f1_scores'],axis=0)
        mean_test_f1_scores = np.mean(d[key][clf][selected_qs]['no_weighting']['test_f1_scores'],axis=0)
        mean_test_f1_scores_nn = np.mean(d[key][clf][selected_qs]['nn']['test_f1_scores'],axis=0)
        mean_test_f1_scores_lp = np.mean(d[key][clf][selected_qs]['lrcv_predict_proba']['test_f1_scores'],axis=0)
    
        mean_2iter.append(round(mean_test_f1_scores[1],3))
        column_list.append((clf,selected_qs,'2nd','ATLX'))
        mean_2iter.append(round(mean_test_f1_scores_nn[1],3))
        column_list.append((clf,selected_qs,'2nd','_NN'))
        mean_2iter.append(round(mean_test_f1_scores_lp[1],3))
        column_list.append((clf,selected_qs,'2nd','_LP'))
        mean_2iter.append(round(al_mean_test_f1_scores[1],3))
        column_list.append((clf,selected_qs,'2nd','AL'))
        mean_20iter.append(round(mean_test_f1_scores[19],3))
        column_list.append((clf,selected_qs,'20th','ATLX'))
        mean_20iter.append(round(mean_test_f1_scores_nn[19],3))
        column_list.append((clf,selected_qs,'20th','_NN'))
        mean_20iter.append(round(mean_test_f1_scores_lp[19],3))
        column_list.append((clf,selected_qs,'20th','_LP'))
        mean_20iter.append(round(al_mean_test_f1_scores[19],3))
        column_list.append((clf,selected_qs,'20th','AL'))
        mean_50iter.append(round(mean_test_f1_scores[49],3))
        column_list.append((clf,selected_qs,'50th','ATLX'))
        mean_50iter.append(round(mean_test_f1_scores_nn[49],3))
        column_list.append((clf,selected_qs,'50th','_NN'))
        mean_50iter.append(round(mean_test_f1_scores_lp[49],3))
        column_list.append((clf,selected_qs,'50th','_LP'))
        mean_50iter.append(round(al_mean_test_f1_scores[49],3))
        column_list.append((clf,selected_qs,'50th','AL'))
        mean_100iter.append(round(mean_test_f1_scores[99],3))
        column_list.append((clf,selected_qs,'100th','ATLX'))
        mean_100iter.append(round(mean_test_f1_scores_nn[99],3))
        column_list.append((clf,selected_qs,'100th','_NN'))
        mean_100iter.append(round(mean_test_f1_scores_lp[99],3))
        column_list.append((clf,selected_qs,'100th','_LP'))
        mean_100iter.append(round(al_mean_test_f1_scores[99],3))
        column_list.append((clf,selected_qs,'100th','AL'))
    
    
    
    data = list(zip(list(group(mean_2iter,4)),list(group(mean_20iter,4)),list(group(mean_50iter,4)),list(group(mean_100iter,4))))
    data = [element for tupl in data for element in tupl]
    data = [element for tupl in data for element in tupl]
    data = list(group(data,16))
    
    col_tuple_list = list(group(column_list,4))
    single_col_tuple = col_tuple_list[:4]
    columns_list = [item for t in single_col_tuple for item in t]
    
    df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(keys,names=['Source','Target']), columns=pd.MultiIndex.from_tuples(columns_list,names=['Estimator','QS','Iteration','Results']))
    
    values = [round(candsets_super_results[key]['randforest']['f1'],3) for key in candsets_super_results.keys()]
    df_super = pd.DataFrame(values,index=candsets_super_results.keys(),columns=[clf])
    
    df_super.columns = pd.MultiIndex.from_product([[clf],[selected_qs],['all'],['Tar_sup']],names=['Estimator','QS','Iterations','Results'])
    df = pd.merge(df,df_super,how='left',left_on='Target',right_index=True)
    df = df.reindex(columns=[clf],level=0)
    df = df.reindex(columns=[selected_qs],level=1)
    df = df.reindex(columns=['2nd','20th','50th','100th','all'],level=2)
    
    print(f'All ATLX results with no_weighting and nn as well as lr_predict_proba (lp) and Query Strategy: {selected_qs}')
    styleDFwithAlUnsupandATLXResultsToCompareDA(df,filename)
    
    return df

def styleDFwithAlUnsupandATLXResultsToCompareDA(df,filename=None):
    col_atl = [col for col in df.columns if col[3]=='ATLX' or col[3]=='_NN' or col[3]=='_LP']
    col_2nd_atl = [col for col in df.columns if (col[2]=='2nd' and (col[3]=='ATLX' or col[3]=='_NN' or col[3]=='_LP'))]
    col_20th_atl = [col for col in df.columns if (col[2]=='20th' and (col[3]=='ATLX' or col[3]=='_NN' or col[3]=='_LP'))]
    col_50th_atl = [col for col in df.columns if (col[2]=='50th' and (col[3]=='ATLX' or col[3]=='_NN' or col[3]=='_LP'))]
    col_100th_atl = [col for col in df.columns if (col[2]=='100th' and (col[3]=='ATLX' or col[3]=='_NN' or col[3]=='_LP'))]
    col_tar_sup = [col for col in df.columns if col[3]=='Tar_sup']
    col_tar_sup_format = {col:lambda x: '<font color=\'#000000\'><b>{}</b></font>'.format(x) for col in col_tar_sup}
                                               
    ###########################################################################################
    # style functions as nested functions
    
    def highlight_max(data, color='yellow'):
        attr = 'background-color: {}'.format(color)
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = data == data.max()
            return [attr if v else '' for v in is_max]
        else:  # from .apply(axis=None)
            is_max = data == data.max().max()
            return pd.DataFrame(np.where(is_max, attr, ''),
                                index=data.index, columns=data.columns)
    
    def highlight_max_per_iter(data):
        attr = 'color:#000000;background-color:rgba(51, 204, 51, .3);'
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = data == data.max()
            return [attr if v else '' for v in is_max]
        else:  # from .apply(axis=None)
            is_max = data == data.max().max()
            return pd.DataFrame(np.where(is_max, attr, ''),
                                index=data.index, columns=data.columns)
        
    def highlight_atl_exceed_bm(row):
        # initiate list with the size of the row with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. 
        ser = (row[:,:,:,'ATLX'].reset_index(drop=True)>=row[:,:,'all','Tar_sup'].repeat(4).reset_index(drop=True))
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 6 (i+k). K has to be incremented by 2 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(b):
                lst[k] = 'background-color: #ff00d9' #magenta
            k += 4
        return lst
    
    def highlight_nn_exceed_bm(row):
        # initiate list with the size of the row with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. 
        ser = (row[:,:,:,'_NN'].reset_index(drop=True)>=row[:,:,'all','Tar_sup'].repeat(4).reset_index(drop=True))
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 6 (i+k). K has to be incremented by 2 for each execution of the loop
        k = 1
        for i, b in enumerate(ser):
            if(b):
                lst[k] = 'background-color: #ff00d9' #magenta
            k += 4
        return lst
    
    def highlight_lp_exceed_bm(row):
        # initiate list with the size of the row with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. 
        ser = (row[:,:,:,'_LP'].reset_index(drop=True)>=row[:,:,'all','Tar_sup'].repeat(4).reset_index(drop=True))
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 6 (i+k). K has to be incremented by 2 for each execution of the loop
        k = 2
        for i, b in enumerate(ser):
            if(b):
                lst[k] = 'background-color: #ff00d9' #magenta
            k += 4
        return lst
    
    def highlight_al_exceed_atl(row):
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. 
        ser = ((row[:,:,:,'ATLX'].reset_index(drop=True)<row[:,:,:,'AL'].reset_index(drop=True))\
              & (row[:,:,:,'_NN'].reset_index(drop=True)<row[:,:,:,'AL'].reset_index(drop=True))\
              & (row[:,:,:,'_LP'].reset_index(drop=True)<row[:,:,:,'AL'].reset_index(drop=True)))
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 6 (i+k). K has to be incremented by 2 for each execution of the loop
        k = 3
        for i, b in enumerate(ser):
            if(b):
                lst[k] = 'background-color: #FF7070' #red
            k += 4
        return lst
    
#    ############################################################################################
    # specify the styles for the html output
    styles=[{'selector': 'th','props': [
            ('border-style', 'solid'),
            ('border-color', '#D3D3D3'),
            ('vertical-align','top'),
            ('text-align','center')]}]
   
    #highlight_max_per_iter
    html = (df.style.\
            apply(highlight_max_per_iter,axis=1,subset=pd.IndexSlice[:,col_2nd_atl]).\
            apply(highlight_max_per_iter,axis=1,subset=pd.IndexSlice[:,col_20th_atl]).\
            apply(highlight_max_per_iter,axis=1,subset=pd.IndexSlice[:,col_50th_atl]).\
            apply(highlight_max_per_iter,axis=1,subset=pd.IndexSlice[:,col_100th_atl]).\
            apply(highlight_max,axis=1,subset=pd.IndexSlice[:,col_atl]).\
            apply(highlight_atl_exceed_bm,axis=1).\
            apply(highlight_nn_exceed_bm,axis=1).\
            apply(highlight_lp_exceed_bm,axis=1).\
            apply(highlight_al_exceed_atl,axis=1)).set_table_styles(styles).set_precision(3).format(col_tar_sup_format)
    
    display(html)
    if(filename is not None):
        with open('{}.html'.format(filename), 'w') as f:
            f.write(html.render())
    

#%%
### Plot and Display AWTL Experiment results ###
    
def plotAWTLResults(awtl_results,source,target,quota,candsets,candsets_super_results,n,selected_estimator=None,al_results=None,
                   selected_qs=None,selected_weights=None,errorbars=False,ylim=None,info_text=True,path_for_output='./graphics/custom_plots/'):
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
    
    keys = list(d.keys())
    if(not isinstance(keys[0],tuple)):
        source_target = '{}_{}'.format(source,target)
    else:
        source_target = (source,target)
    
    qss,estimators = [],[]
    if(info_text):
        fig,ax = plt.subplots(figsize=(20,10))
        plt.subplots_adjust(right=0.7)
    else:
        fig,ax = plt.subplots(figsize=(16,10))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.set_xlabel('x target instances used for training',fontsize=15)
    ax.set_ylabel('Avg. F1-Score',fontsize=15)

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
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,lw=3,
                                            label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,lw=3,
                                    label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                    else:
                        for weight in selected_weights:
                            plt.close()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,lw=3,
                                            label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,lw=3,
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
                            ax.errorbar(x_al_results, y_al_results, yerr=std_al_results,lw=3,
                                        label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
                        else:
                            ax.plot(x_al_results,y_al_results,lw=3,linestyle='dashed',
                                    label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
            else:
                for qs in selected_qs:
                    qss.append(qs)
                    if selected_weights is None:
                        for weight in d[source_target][clf][qs]:
                            plt.close()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,lw=3,
                                            label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,lw=3,
                                        label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                    else:
                        for weight in selected_weights:
                            plt.close()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,lw=3,
                                            label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,lw=3,
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
                            ax.errorbar(x_al_results, y_al_results, yerr=std_al_results,lw=3,
                                        label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
                            ax.plot(x_al_results,y_al_results,lw=3,linestyle='dashed',
                                    label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
            # benchmark plots
            # supervised
            f1_target_bm = candsets_super_results[target][clf_old]['f1']
            y_target_sup_bm = list(itertools.repeat(f1_target_bm,len(x)))
            ax.plot(x,y_target_sup_bm,lw=3,linestyle='dotted',label='target supervised ({}) benchmark {} instances F1: {:.2f}'.format(clf,candsets[target].shape[0],f1_target_bm))
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
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,lw=3,
                                            label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,lw=3,
                                        label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                    else:
                        for weight in selected_weights:
                            plt.close()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,lw=3,
                                            label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,lw=3,
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
                        ax.errorbar(x_al_results, y_al_results, yerr=std_al_results,lw=3,
                                    label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
                    else:
                        ax.plot(x_al_results,y_al_results,lw=3,linestyle='dashed',
                                label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
            else:
                for qs in selected_qs:
                    qss.append(qs)
                    if selected_weights is None:
                        for weight in d[source_target][clf][qs]:
                            plt.close()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,lw=3,
                                            label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,lw=3,
                                        label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                    else:
                        for weight in selected_weights:
                            plt.close()
                            max_quota = d[source_target][clf][qs][weight]['quota']
                            x_atl_results = np.arange(1, max_quota + 1)
                            atl_test_f1_scores = np.array(d[source_target][clf][qs][weight]['test_f1_scores'])
                            y_atl_results = np.mean(atl_test_f1_scores,axis=0)
                            std_atl_results = np.std(atl_test_f1_scores, axis=0)
                            if(errorbars):
                                ax.errorbar(x_atl_results, y_atl_results, yerr=std_atl_results,lw=3,
                                            label='{}: ATL {} source inst. & QS {} Weight. {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,candsets[source].shape[0],qs,weight,y_atl_results[max_quota-1],std_atl_results[max_quota-1]))
                            else:
                                ax.plot(x_atl_results,y_atl_results,lw=3,
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
                            ax.errorbar(x_al_results, y_al_results, yerr=std_al_results,lw=3,
                                        label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
                        else:
                            ax.plot(x_al_results,y_al_results,lw=3,linestyle='dashed',
                                    label='{}: AL init. {} labeled & QS {}. Final Iter. F1: {:.2f} σ {:.2f}'.format(clf,n_init_labeled,qs,y_al_results[max_quota-1],std_al_results[max_quota-1]))
            # benchmark plots
            # supervised
            f1_target_bm = candsets_super_results[target][clf_old]['f1']
            y_target_sup_bm = list(itertools.repeat(f1_target_bm,len(x)))
            ax.plot(x,y_target_sup_bm,lw=3,linestyle='dotted',label='target supervised ({}) benchmark {} instances F1: {:.2f}'.format(clf,candsets[target].shape[0],f1_target_bm))
        
            
    # add legend to plot
    ax.legend(fontsize=15)
    if(info_text):
        # add a text box with info
        info_text = 'ATL was tested on a\nhold-out test set with\n33% of target instances\nand avg. over n={} runs.'.format(n)
        info_text_2 = 'Results of benchmark\nare calculated on\nthe same test set.'
        textstr = 'INFO:\n{}\n{}'.format(info_text,info_text_2)
        ax.text(0.71, 0.85, textstr, transform=fig.transFigure, fontsize=15,verticalalignment='top', bbox=props)
    
    #features_used = 'Only dense features across source and target were used. Lowers risk of negative transfer (finding from TL Experiment)'
    features_used = 'dense features'
    ax.set_title('Active-Transfer Learning (ATL) Results with DA:\nsource {} and target {}\n{}'.format(source,target,features_used), fontsize=15)
    
    if(path_for_output is not None):
        if(errorbars):
            fig.savefig('{}{}_{}_{}_{}_n{}_errorbars.png'.format(path_for_output,source,target,'_'.join(estimators),'_'.join(qss),n),bbox_inches='tight',dpi=600)
        else:
            fig.savefig('{}{}_{}_{}_{}_n{}.png'.format(path_for_output,source,target,'_'.join(estimators),'_'.join(qss),n),bbox_inches='tight',dpi=600)
    
    if(ylim is not None):
        ax.set_ylim(ylim)
    return fig
    
def createDFwithAlandAWTLResultsMoreItersUpToMaxQuota(awtl_results,al_results,selected_weighting,candsets_super_results,selected_qs=None,max_quota=100,filename=None):
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
    d = awtl_results
    
    mean_2iter,mean_10iter,mean_20iter,mean_30iter,mean_50iter,mean_100iter = [],[],[],[],[],[]
    
    al_mean_2iter,al_mean_10iter,al_mean_20iter,al_mean_30iter,al_mean_50iter,al_mean_100iter = [],[],[],[],[],[]
    qss,column_list = [],[]
    
    keys = []
    for key in d:       
        if(not isinstance(key,tuple)):
            target = '_'.join(key.split('_')[2:])
            keys.append(('_'.join(key.split('_')[:2]),'_'.join(key.split('_')[2:])))
        else:
            target = key[1]
            keys.append(key)
        clf = 'rf'
        if(selected_qs is None):
            for qs in d[key][clf]:
                qss.append(qs)
                al_test_f1_scores = al_results[target][clf][qs]['test_f1_scores']
                al_mean_test_f1_scores = np.mean(al_test_f1_scores,axis=0)
                al_mean_2iter.append(round(al_mean_test_f1_scores[1],3))
                column_list.append((clf,qs,'2nd','ATL'))
                column_list.append((clf,qs,'2nd','AL'))
                al_mean_10iter.append(round(al_mean_test_f1_scores[9],3))
                column_list.append((clf,qs,'10th','ATL'))
                column_list.append((clf,qs,'10th','AL'))
                al_mean_20iter.append(round(al_mean_test_f1_scores[19],3))
                column_list.append((clf,qs,'20th','ATL'))
                column_list.append((clf,qs,'20th','AL'))
                al_mean_30iter.append(round(al_mean_test_f1_scores[29],3))
                column_list.append((clf,qs,'30th','ATL'))
                column_list.append((clf,qs,'30th','AL'))
                al_mean_50iter.append(round(al_mean_test_f1_scores[49],3))
                column_list.append((clf,qs,'50th','ATL'))
                column_list.append((clf,qs,'50th','AL'))
                al_mean_100iter.append(round(al_mean_test_f1_scores[99],3))
                column_list.append((clf,qs,'100th','ATL'))
                column_list.append((clf,qs,'100th','AL'))
                mean_2iter.append(round(np.mean(d[key][clf][qs][selected_weighting]['test_f1_scores'],axis=0)[1],3))
                mean_10iter.append(round(np.mean(d[key][clf][qs][selected_weighting]['test_f1_scores'],axis=0)[9],3))
                mean_20iter.append(round(np.mean(d[key][clf][qs][selected_weighting]['test_f1_scores'],axis=0)[19],3))
                mean_30iter.append(round(np.mean(d[key][clf][qs][selected_weighting]['test_f1_scores'],axis=0)[29],3))
                mean_50iter.append(round(np.mean(d[key][clf][qs][selected_weighting]['test_f1_scores'],axis=0)[49],3))
                mean_100iter.append(round(np.mean(d[key][clf][qs][selected_weighting]['test_f1_scores'],axis=0)[99],3))
        else:
            qss.append(selected_qs)
            al_test_f1_scores = al_results[target][clf][selected_qs]['test_f1_scores']
            al_mean_test_f1_scores = np.mean(al_test_f1_scores,axis=0)
            al_mean_2iter.append(round(al_mean_test_f1_scores[1],3))
            column_list.append((clf,selected_qs,'2nd','ATL'))
            column_list.append((clf,selected_qs,'2nd','AL'))
            al_mean_10iter.append(round(al_mean_test_f1_scores[9],3))
            column_list.append((clf,selected_qs,'10th','ATL'))
            column_list.append((clf,selected_qs,'10th','AL'))
            al_mean_20iter.append(round(al_mean_test_f1_scores[19],3))
            column_list.append((clf,selected_qs,'20th','ATL'))
            column_list.append((clf,selected_qs,'20th','AL'))
            al_mean_30iter.append(round(al_mean_test_f1_scores[29],3))
            column_list.append((clf,selected_qs,'30th','ATL'))
            column_list.append((clf,selected_qs,'30th','AL'))
            al_mean_50iter.append(round(al_mean_test_f1_scores[49],3))
            column_list.append((clf,selected_qs,'50th','ATL'))
            column_list.append((clf,selected_qs,'50th','AL'))
            al_mean_100iter.append(round(al_mean_test_f1_scores[99],3))
            column_list.append((clf,selected_qs,'100th','ATL'))
            column_list.append((clf,selected_qs,'100th','AL'))
            mean_2iter.append(round(np.mean(d[key][clf][selected_qs][selected_weighting]['test_f1_scores'],axis=0)[1],3))
            mean_10iter.append(round(np.mean(d[key][clf][selected_qs][selected_weighting]['test_f1_scores'],axis=0)[9],3))
            mean_20iter.append(round(np.mean(d[key][clf][selected_qs][selected_weighting]['test_f1_scores'],axis=0)[19],3))
            mean_30iter.append(round(np.mean(d[key][clf][selected_qs][selected_weighting]['test_f1_scores'],axis=0)[29],3))
            mean_50iter.append(round(np.mean(d[key][clf][selected_qs][selected_weighting]['test_f1_scores'],axis=0)[49],3))
            mean_100iter.append(round(np.mean(d[key][clf][selected_qs][selected_weighting]['test_f1_scores'],axis=0)[99],3))
                

    al_atl_mean_2iter = list(zip(mean_2iter,al_mean_2iter))
    al_atl_mean_10iter = list(zip(mean_10iter,al_mean_10iter))
    al_atl_mean_20iter = list(zip(mean_20iter,al_mean_20iter))
    al_atl_mean_30iter = list(zip(mean_30iter,al_mean_30iter))
    al_atl_mean_50iter = list(zip(mean_50iter,al_mean_50iter))
    al_atl_mean_100iter = list(zip(mean_100iter,al_mean_100iter))

    
    # get the amount of different query strategies
    number_of_qs = len(list(set(qss)))
    # ensure that the order is maintained 
    qss = qss[:number_of_qs]
    n = number_of_qs*6*2 #number_of_qs*Iterations (6) *AL Unsup Results and ATL RF Results (2)
    data = list(group(list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(list(zip(al_atl_mean_2iter,al_atl_mean_10iter,al_atl_mean_20iter,al_atl_mean_30iter,al_atl_mean_50iter,al_atl_mean_100iter)))))),n))
    col_tuple_list = list(group(column_list,n))
    single_col_tuple = list(set(col_tuple_list))
    columns_list = [item for t in single_col_tuple for item in t]

    df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(keys,names=['Source','Target']), columns=pd.MultiIndex.from_tuples(columns_list,names=['Estimator','QS','Iteration','Results']))
    
    values = [round(candsets_super_results[key]['randforest']['f1'],3) for key in candsets_super_results.keys()]
    df_super = pd.DataFrame(values,index=candsets_super_results.keys(),columns=[clf])
    

    if(number_of_qs==2):
        df_super = pd.concat([df_super,df_super],axis=1)
        df_super = df_super[[clf]]

    df_super.columns = pd.MultiIndex.from_product([[clf],qss,['all'],['Tar_sup']],names=['Estimator','QS','Iterations','Results'])
    df = pd.merge(df,df_super,how='left',left_on='Target',right_index=True)
    df = df.reindex(columns=[clf],level=0)
    df = df.reindex(columns=qss,level=1)
    df = df.reindex(columns=['2nd','10th','20th','30th','50th','100th','all'],level=2)
    #col_al_atl = [col for col in df.columns if col[3]=='ATL' or col[3]=='AL']
    col_tar_sup = [col for col in df.columns if col[3]=='Tar_sup']
    col_tar_sup_format = {col:lambda x: '<font color=\'#000000\'><b>{}</b></font>'.format(x) for col in col_tar_sup}
                                               
    ###########################################################################################
    # style functions as nested functions
   
    def highlight_atl_exceed_bm(row):
        # initiate list with the size of the row with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. 
        ser = (row[:,:,:,'ATL'].reset_index(drop=True)>=row[:,:,'all','Tar_sup'].repeat(6).reset_index(drop=True))
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 6 (i+k). K has to be incremented by 2 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i!=0 and i%6==0):
                k += 1
            if(b):
                lst[k] = 'background-color: #FBFF75'
            k += 2
        return lst
    
    # another function for pandas style      
    def highlight_al_worse_than_atl(row):
        # initiate list with the size of the row with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. 
        ser = (row[:,:,:,'ATL']>row[:,:,:,'AL'])
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 6 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i!=0 and i%6==0):
                k += 1
            if(b):
                lst[k] = 'color:#000000;background-color:rgba(51, 204, 51, .1);'
            k += 2
        return lst
    
        # another function for pandas style      
    def highlight_qs_worse_than_rs(row):
        # initiate list with the size of the row with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. 
        ser = (row[:,'lr_lsvc_rf_dt',:,'ATL']-row[:,'random',:,'ATL'])>0.01
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 6 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(i<6 and not b):
                # yellow: #FFFF00
                lst[k] = 'background-color:#FFA500'
            #elif(i<6 and b):
            #    lst[k] = 'color:#000000;background-color:rgba(51, 204, 51, .1);'
            else:
                lst[k] = ''
            k += 2
        return lst
#    # another function for pandas style      
    def highlight_atl_worse_than_al(row):
        # initiate list with the size of the row with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = (row[:,:,:,'ATL']<row[:,:,:,'AL'])
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 6 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 1
        for i, b in enumerate(ser):
            if(i!=0 and i%6==0):
                k += 1
            if(b):
                lst[k] = 'background-color: #FF7070'
            k += 2
        return lst
#    ############################################################################################
    # specify the styles for the html output
    styles=[{'selector': 'th','props': [
            ('border-style', 'solid'),
            ('border-color', '#D3D3D3'),
            ('vertical-align','top'),
            ('text-align','center')]}]
   
    html = (df.style.\
            apply(highlight_qs_worse_than_rs,axis=1).\
            apply(highlight_atl_worse_than_al,axis=1).\
            apply(highlight_atl_exceed_bm,axis=1)).set_table_styles(styles).set_precision(3).format(col_tar_sup_format)
    print(f'Weighting: {selected_weighting} and Query Strategy: {selected_qs}')
    display(html)
    if(filename is not None):
        with open('{}.html'.format(filename), 'w') as f:
            f.write(html.render())
            
    return df


def createDFwithAlandAWTLResultsToCompareDA(awtl_results,al_results,candsets_super_results,selected_qs='lr_lsvc_rf_dt',max_quota=100,filename=None):
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
    d = awtl_results
    #selected_qs = 'lr_lsvc_rf_dt'
    mean_2iter,mean_20iter,mean_50iter,mean_100iter = [],[],[],[]
    
    column_list = []
    
    keys = []
    for key in d:       
        if(not isinstance(key,tuple)):
            target = '_'.join(key.split('_')[2:])
            keys.append(('_'.join(key.split('_')[:2]),'_'.join(key.split('_')[2:])))
        else:
            target = key[1]
            keys.append(key)
    
        clf = 'rf'
        al_mean_test_f1_scores = np.mean(al_results[target][clf][selected_qs]['test_f1_scores'],axis=0)
        mean_test_f1_scores = np.mean(d[key][clf][selected_qs]['no_weighting']['test_f1_scores'],axis=0)
        mean_test_f1_scores_nn = np.mean(d[key][clf][selected_qs]['nn']['test_f1_scores'],axis=0)
        mean_test_f1_scores_lp = np.mean(d[key][clf][selected_qs]['lr_predict_proba']['test_f1_scores'],axis=0)
    
        mean_2iter.append(round(mean_test_f1_scores[1],3))
        column_list.append((clf,selected_qs,'2nd','ATL'))
        mean_2iter.append(round(mean_test_f1_scores_nn[1],3))
        column_list.append((clf,selected_qs,'2nd','_NN'))
        mean_2iter.append(round(mean_test_f1_scores_lp[1],3))
        column_list.append((clf,selected_qs,'2nd','_LP'))
        mean_2iter.append(round(al_mean_test_f1_scores[1],3))
        column_list.append((clf,selected_qs,'2nd','AL'))
        mean_20iter.append(round(mean_test_f1_scores[19],3))
        column_list.append((clf,selected_qs,'20th','ATL'))
        mean_20iter.append(round(mean_test_f1_scores_nn[19],3))
        column_list.append((clf,selected_qs,'20th','_NN'))
        mean_20iter.append(round(mean_test_f1_scores_lp[19],3))
        column_list.append((clf,selected_qs,'20th','_LP'))
        mean_20iter.append(round(al_mean_test_f1_scores[19],3))
        column_list.append((clf,selected_qs,'20th','AL'))
        mean_50iter.append(round(mean_test_f1_scores[49],3))
        column_list.append((clf,selected_qs,'50th','ATL'))
        mean_50iter.append(round(mean_test_f1_scores_nn[49],3))
        column_list.append((clf,selected_qs,'50th','_NN'))
        mean_50iter.append(round(mean_test_f1_scores_lp[49],3))
        column_list.append((clf,selected_qs,'50th','_LP'))
        mean_50iter.append(round(al_mean_test_f1_scores[49],3))
        column_list.append((clf,selected_qs,'50th','AL'))
        mean_100iter.append(round(mean_test_f1_scores[99],3))
        column_list.append((clf,selected_qs,'100th','ATL'))
        mean_100iter.append(round(mean_test_f1_scores_nn[99],3))
        column_list.append((clf,selected_qs,'100th','_NN'))
        mean_100iter.append(round(mean_test_f1_scores_lp[99],3))
        column_list.append((clf,selected_qs,'100th','_LP'))
        mean_100iter.append(round(al_mean_test_f1_scores[99],3))
        column_list.append((clf,selected_qs,'100th','AL'))
    
    
    
    data = list(zip(list(group(mean_2iter,4)),list(group(mean_20iter,4)),list(group(mean_50iter,4)),list(group(mean_100iter,4))))
    data = [element for tupl in data for element in tupl]
    data = [element for tupl in data for element in tupl]
    data = list(group(data,16))
    
    col_tuple_list = list(group(column_list,4))
    single_col_tuple = col_tuple_list[:4]
    columns_list = [item for t in single_col_tuple for item in t]
    
    df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(keys,names=['Source','Target']), columns=pd.MultiIndex.from_tuples(columns_list,names=['Estimator','QS','Iteration','Results']))
    
    values = [round(candsets_super_results[key]['randforest']['f1'],3) for key in candsets_super_results.keys()]
    df_super = pd.DataFrame(values,index=candsets_super_results.keys(),columns=[clf])
    
    df_super.columns = pd.MultiIndex.from_product([[clf],[selected_qs],['all'],['Tar_sup']],names=['Estimator','QS','Iterations','Results'])
    df = pd.merge(df,df_super,how='left',left_on='Target',right_index=True)
    df = df.reindex(columns=[clf],level=0)
    df = df.reindex(columns=[selected_qs],level=1)
    df = df.reindex(columns=['2nd','20th','50th','100th','all'],level=2)
    
    styleDFwithAlandAWTLResultsToCompareDA(df,filename,selected_qs)
            
    return df


def styleDFwithAlandAWTLResultsToCompareDA(df,filename=None,selected_qs='lr_lsvc_rf_dt'):
    col_atl = [col for col in df.columns if col[3]=='ATL' or col[3]=='_NN' or col[3]=='_LP']
    col_2nd_atl = [col for col in df.columns if (col[2]=='2nd' and (col[3]=='ATL' or col[3]=='_NN' or col[3]=='_LP'))]
    col_20th_atl = [col for col in df.columns if (col[2]=='20th' and (col[3]=='ATL' or col[3]=='_NN' or col[3]=='_LP'))]
    col_50th_atl = [col for col in df.columns if (col[2]=='50th' and (col[3]=='ATL' or col[3]=='_NN' or col[3]=='_LP'))]
    col_100th_atl = [col for col in df.columns if (col[2]=='100th' and (col[3]=='ATL' or col[3]=='_NN' or col[3]=='_LP'))]
    col_tar_sup = [col for col in df.columns if col[3]=='Tar_sup']
    col_tar_sup_format = {col:lambda x: '<font color=\'#000000\'><b>{}</b></font>'.format(x) for col in col_tar_sup}
                          
    ###########################################################################################
    # style functions as nested functions
    
    def highlight_max(data, color='yellow'):
        attr = 'background-color: {}'.format(color)
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = data == data.max()
            return [attr if v else '' for v in is_max]
        else:  # from .apply(axis=None)
            is_max = data == data.max().max()
            return pd.DataFrame(np.where(is_max, attr, ''),
                                index=data.index, columns=data.columns)
    
    def highlight_max_per_iter(data):
        attr = 'color:#000000;background-color:rgba(51, 204, 51, .3);'
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = data == data.max()
            return [attr if v else '' for v in is_max]
        else:  # from .apply(axis=None)
            is_max = data == data.max().max()
            return pd.DataFrame(np.where(is_max, attr, ''),
                                index=data.index, columns=data.columns)
        
    def highlight_atl_exceed_bm(row):
        # initiate list with the size of the row with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. 
        ser = (row[:,:,:,'ATL'].reset_index(drop=True)>=row[:,:,'all','Tar_sup'].repeat(4).reset_index(drop=True))
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 6 (i+k). K has to be incremented by 2 for each execution of the loop
        k = 0
        for i, b in enumerate(ser):
            if(b):
                lst[k] = 'background-color: #ff00d9' #magenta
            k += 4
        return lst
    
    def highlight_nn_exceed_bm(row):
        # initiate list with the size of the row with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. 
        ser = (row[:,:,:,'_NN'].reset_index(drop=True)>=row[:,:,'all','Tar_sup'].repeat(4).reset_index(drop=True))
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 6 (i+k). K has to be incremented by 2 for each execution of the loop
        k = 1
        for i, b in enumerate(ser):
            if(b):
                lst[k] = 'background-color: #ff00d9' #magenta
            k += 4
        return lst
    
    def highlight_lp_exceed_bm(row):
        # initiate list with the size of the row with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. 
        ser = (row[:,:,:,'_LP'].reset_index(drop=True)>=row[:,:,'all','Tar_sup'].repeat(4).reset_index(drop=True))
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 6 (i+k). K has to be incremented by 2 for each execution of the loop
        k = 2
        for i, b in enumerate(ser):
            if(b):
                lst[k] = 'background-color: #ff00d9' #magenta
            k += 4
        return lst
    
    def highlight_al_exceed_atl(row):
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. 
        ser = ((row[:,:,:,'ATL'].reset_index(drop=True)<row[:,:,:,'AL'].reset_index(drop=True))\
              & (row[:,:,:,'_NN'].reset_index(drop=True)<row[:,:,:,'AL'].reset_index(drop=True))\
              & (row[:,:,:,'_LP'].reset_index(drop=True)<row[:,:,:,'AL'].reset_index(drop=True)))
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 6 (i+k). K has to be incremented by 2 for each execution of the loop
        k = 3
        for i, b in enumerate(ser):
            if(b):
                lst[k] = 'background-color: #FF7070' #red
            k += 4
        return lst
    
#    ############################################################################################
    # specify the styles for the html output
    styles=[{'selector': 'th','props': [
            ('border-style', 'solid'),
            ('border-color', '#D3D3D3'),
            ('vertical-align','top'),
            ('text-align','center')]}]
   
    #highlight_max_per_iter
    html = (df.style.\
            apply(highlight_max_per_iter,axis=1,subset=pd.IndexSlice[:,col_2nd_atl]).\
            apply(highlight_max_per_iter,axis=1,subset=pd.IndexSlice[:,col_20th_atl]).\
            apply(highlight_max_per_iter,axis=1,subset=pd.IndexSlice[:,col_50th_atl]).\
            apply(highlight_max_per_iter,axis=1,subset=pd.IndexSlice[:,col_100th_atl]).\
            apply(highlight_max,axis=1,subset=pd.IndexSlice[:,col_atl]).\
            apply(highlight_atl_exceed_bm,axis=1).\
            apply(highlight_nn_exceed_bm,axis=1).\
            apply(highlight_lp_exceed_bm,axis=1).\
            apply(highlight_al_exceed_atl,axis=1)).set_table_styles(styles).set_precision(3).format(col_tar_sup_format)
    print(f'All ATL results with no_weighting and nn as well as lr_predict_proba (lp) and Query Strategy: {selected_qs}')
    display(html)
    
    if(filename is not None):
        with open('{}.html'.format(filename), 'w') as f:
            f.write(html.render())
    

#%%
### Plot and display function for TL Experiments ###

def plotTLResults(tl_results,source_name,target_name,feature,selected_estimators,selected_da_weighting,candsets,
                  candsets_super_results,candsets_unsuper_results,plot_target,info_text=True,path_for_output=None):
    
    x=[0,10,14,20,24,28,32,38,44,50,60,70,80,90,100,120,140,160,180,200,300,500]
    d = tl_results
    
    keys = list(d.keys())
    if(not isinstance(keys[0],tuple)):
        combo= '{}_{}'.format(source_name,target_name)
    else:
        combo = (source_name,target_name)
                         
    #combo = '{}_{}'.format(source_name,target_name)
    
    if(info_text):
        fig,ax = plt.subplots(figsize=(20,10))
        plt.subplots_adjust(right=0.7)
    else:
        fig,ax = plt.subplots(figsize=(16,10))
    plt.close(fig)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # unsupervised bm plot
    y_target_unsup_bm = list(itertools.repeat(candsets_unsuper_results[target_name]['f1'],len(x)))
    ax.plot(x,y_target_unsup_bm,linestyle='dashdot',color='g',lw=3,label='target unsupervised (elbow) benchmark. F1: {}'.format(round(candsets_unsuper_results[target_name]['f1'],2)))
    ax.set_xlabel('x target instances used for training',fontsize=15)
    ax.set_ylabel('Avg. F1-Score',fontsize=15)
    
    colors = ['b','r','y','c','m','k']
    for da in selected_da_weighting: 
        if selected_estimators is None:
            for i,clf in enumerate(d[combo][feature]):
                n = d[combo][feature][da][clf]['n_runs']
                # insert first result of transfer one more time at beginning to also plot point when x = 0.
                y_transfer_results = d[combo][feature][da][clf]['y_transfer_results'].copy()
                y_transfer_results.insert(0,y_transfer_results[0])
                transfer_result_avg = round(np.mean(y_transfer_results),2)
                # insert 0 at beginning to the plot beginns at origin
                y_target_results = d[combo][feature][da][clf]['y_target_results'].copy()
                y_target_results.insert(0,0)
                ax.plot(x,y_transfer_results,lw=3,
                        label='transfer learning results ({}) when trained on all {} source instances. F1: {}'.format(clf,candsets[source_name].shape[0],transfer_result_avg),
                        color=colors[i%len(colors)])
                if(plot_target):
                    ax.plot(x,y_target_results,linestyle='dashed',lw=3,
                            label='target results ({}): trained on x target instances'.format(clf),
                            color=colors[i%len(colors)])
                    if(not math.isnan(float(d[combo][feature][da][clf]['x_target_exceed']))):
                        idx = x.index(d[combo][feature][da][clf]['x_target_exceed'])
                        ax.plot(x[idx], y_target_results[idx], 'ro') 
                # benchmark plots
                # supervised
                y_target_sup_bm = list(itertools.repeat(candsets_super_results[target_name][clf]['f1'],len(x)))
                ax.plot(x,y_target_sup_bm,linestyle='dotted',lw=3,label='target supervised ({}) benchmark. F1: {}'.format(clf,round(candsets_super_results[target_name][clf]['f1'],2)),
                        color=colors[i%len(colors)])
        else:
            for i,clf in enumerate(selected_estimators):
                n = d[combo][feature][da][clf]['n_runs']
                # insert first result of transfer one more time at beginning to also plot point when x = 0.
                y_transfer_results = d[combo][feature][da][clf]['y_transfer_results'].copy()
                y_transfer_results.insert(0,y_transfer_results[0])
                transfer_result_avg = round(np.mean(y_transfer_results),2)
                # insert 0 at beginning to the plot beginns at origin
                y_target_results = d[combo][feature][da][clf]['y_target_results'].copy()
                y_target_results.insert(0,0)
                ax.plot(x,y_transfer_results,lw=3,
                        label='transfer learning results ({}) when trained on all {} source instances. F1: {}'.format(clf,candsets[source_name].shape[0],transfer_result_avg),
                        color=colors[i%len(colors)])
                if(plot_target):
                    ax.plot(x,y_target_results,lw=3,linestyle='dashed',
                            label='target results ({}): trained on x target instances'.format(clf),
                            color=colors[i%len(colors)])
                    if(not math.isnan(float(d[combo][feature][da][clf]['x_target_exceed']))):
                        idx = x.index(d[combo][feature][da][clf]['x_target_exceed'])
                        ax.plot(x[idx], y_target_results[idx], 'ro') 
                # benchmark plots
                # supervised
                y_target_sup_bm = list(itertools.repeat(candsets_super_results[target_name][clf]['f1'],len(x)))
                ax.plot(x,y_target_sup_bm,linestyle='dotted',lw=3,label='target supervised ({}) benchmark. F1: {}'.format(clf,round(candsets_super_results[target_name][clf]['f1'],2)),
                        color=colors[i%len(colors)])
        
        
    # add legend to plot
    ax.legend(fontsize=15)
    # add a text box with info
    if(info_text):
        info_text = 'Transfer Learning (TL)\nand the target results\nwere tested on the same\ntest set from the target.'
        if(plot_target):
            info_text_2 = 'The target results were\ntrained on random\nsamples of size x\nfrom the target\ntraining set and avg.\nover n={} random\nsamples of x.'.format(n)
            textstr = 'INFO:\n{}\n{}'.format(info_text,info_text_2)
        else:
            textstr = 'INFO:\n{}'.format(info_text)
        ax.text(0.71, 0.85, textstr, transform=fig.transFigure, fontsize=15,verticalalignment='top', bbox=props)
    features_used = '{} features were used'.format(feature)
    ax.set_title('Results of Naive Transfer: Trained on source {} and tested on target {}\n{}'.format(source_name,
                 target_name,features_used), fontsize=15)
    if(path_for_output is not None):
        fig.savefig('{}{}_{}_{}_n{}.png'.format(path_for_output,source_name,target_name,feature,n),bbox_inches='tight',dpi=600)
    
    return fig 

def createDFwithTLResults(tl_results,candsets_super_results,candsets_unsuper_results,
                               estimators,feature_sets=['all','dense'],da_weighting='no_weighting',filename='tl_results'):
    """
    This function creates a DataFrame with the results of each experiment. Per estimator the TL_avg (Transfer Learning Avg Result),
    Tar_max (Target max result when trained on 500 target instances), Tar_exc (Amount of Target Instances needed to exceed TL results),
    when trained on all features and dense features, are reported. Hence, amount of estimators*2*3 columns + 1 for unsupervised results
    of target are in the resulting DataFrame. It also stores a html file where important information is highlighted in the DataFrame.
    
    tl_results: Dictionary containting the results of the Transfer Learning experiements (as output by returnF1TLResultsFromDictWithPlot())
    candsets_unsuper_results: Dictionary containing the unsupervised results which act as benchmark (in the form of {'ban_half':0.732,'bx_half': 0.626,...})
    estimators: List of the estimators that were used for the transfer learning experiments
    filename: Name of the html file that gets saved and contains the DataFrame with important information highlighted.
        Default: 'tl_results'
    """
    d = tl_results
    
    ###########################################################################################
    # style functions as nested functions
    # function for pandas styler. Highlight all max values of TL_avg
    def highlight_max(data):
        attr_max = 'background-color: #FBFF75'
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = data == data.max()
            return [attr_max if v else '' for v in is_max]
        else: 
            is_max = data.groupby('TL_avg').transform('max') == data
            return pd.DataFrame(np.where(is_max, attr_max, ''),
                                index=data.index, columns=data.columns)
    # another function for pandas style. Highlight Tar_exc with green            
    def highlight_tar_exc(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = row[:,:,'TL_avg']>row[:,:,'Tar_max']
        # counter if True at Pos 0 (i==0) in ser then lst at pos 2 (i+k) has to be changed
        # if i==1 then pos 6 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 2
        for i, b in enumerate(ser):
            if(b):
                lst[i+k] = 'background-color: #A4FB95'
            k = k + 3
        return lst
    # another function for pandas style. Highlight Tar_exc with green            
    def highlight_tl_super_same(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = ((row[:,:,'TL_avg']>row[:,:,'Tar_sup']) | ((row[:,:,'Tar_sup']-row[:,:,'TL_avg'])<=0.01))
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 3
        for i, b in enumerate(ser):
            if(b):
                lst[i+k] = 'background-color: #A4FB95'
            k = k + 3
        return lst
    ############################################################################################
    # specify the styles for the html output
    styles=[
        {'selector': 'th','props': [
            ('border-style', 'solid'),
            ('border-color', '#D3D3D3'),
            ('vertical-align','top'),
            ('text-align','center')]}]
    # each estimator has 6 results because of being trained on all features and only dense ones and results are TL_avg, Tar_max and Tar_exc
    # number of feature sets
    number_of_feature_sets = len(feature_sets)
    number_of_estimators = len(estimators)
    n = number_of_estimators*number_of_feature_sets*3 
    # getting the source and target keys (first keys in dictionary)
    keys = list(d.keys())
    # when the transfer learning results are saved in a dictionary from json file (dictionary from the experiments was stored in json)
    # then the keys have to be changed back to tuples (for Multiindex)
    if(not isinstance(keys[0],tuple)):
        keys = [('_'.join(k.split('_')[:2]),'_'.join(k.split('_')[2:])) for k in keys]

    # bringing the transfer_results, feature and estimator in tuple format
    test_list = []
    clfs = []
    list_transfer_results,list_x_target_exceed,list_x_target_max_500 = [],[],[]
    for key in d:
        for fea in d[key]:
            for clf in d[key][fea][da_weighting]:
                clfs.append(clf)
                v = d[key][fea][da_weighting][clf]
                list_transfer_results.append(round(v['transfer_avg_result'],3))
                list_x_target_exceed.append(v['x_target_exceed'])
                list_x_target_max_500.append(v['target_max_result'])
                for res in d[key][fea][da_weighting][clf]:
                    if(res in ['transfer_avg_result', 'target_max_result', 'x_target_exceed']):
                        test_list.append((fea,clf,res))
    # bringing the transfer_results in tuple format
    res_tuple_list = list(group(list(itertools.chain.from_iterable(list(zip(list_transfer_results,list_x_target_max_500,list_x_target_exceed)))),n))
    clfs = clfs[:number_of_estimators]
    col_tuple_list = list(group(test_list,n))
    single_col_tuple = list(set(col_tuple_list))
    columns_list = [item for t in single_col_tuple for item in t]
    df = pd.DataFrame(res_tuple_list, index=pd.MultiIndex.from_tuples(keys,names=['Source','Target']), columns=pd.MultiIndex.from_tuples(columns_list,names=['Features','Estimators','Results']))
    df.rename({'transfer_avg_result':'TL_avg','target_max_result':'Tar_max','x_target_exceed':'Tar_exc'},axis=1,inplace=True)
    
    # add supervised benchmarks to dataframe
    tl_candsets_super_results = {}
    for combo in candsets_super_results:
        for clf in candsets_super_results[combo]:
            if(clf in estimators):
                if(combo in tl_candsets_super_results):
                    tl_candsets_super_results[combo].update({clf:copy.deepcopy(candsets_super_results[combo][clf])})
                else:
                    tl_candsets_super_results.update({combo:{clf:copy.deepcopy(candsets_super_results[combo][clf])}})
                
    innerkeys =  [innerkey for k,innerdict in tl_candsets_super_results.items() for innerkey, values in innerdict.items()]
    values = [round(values['f1'],3) for k,innerdict in tl_candsets_super_results.items() for innerkey, values in innerdict.items()]
    df_super = pd.DataFrame(list(group(values,number_of_estimators)),index=tl_candsets_super_results.keys(),columns=innerkeys[:number_of_estimators])
    #list_of_estimators=['logreg','logregcv','dectree','randforest']
    columns_before = list(df_super.columns)
    if(number_of_feature_sets==2):
        df_super = pd.concat([df_super,df_super],axis=1)
    if(number_of_feature_sets>2):
        raise ValueError('Only one feature set supported (either all or dense) or both, but not more!')
    
    df_super.columns = pd.MultiIndex.from_product([feature_sets, columns_before, ['Tar_sup']],names=['Features','Estimators','Results'])
    df = pd.merge(df,df_super,how='left',left_on='Target',right_index=True)
    df = df.reindex(columns=feature_sets,level=0)
    df = df.reindex(columns=clfs,level=1)
    df = df.reindex(columns=['TL_avg','Tar_max','Tar_exc','Tar_sup'], level=2)

    #df['target'] = pd.Series(df.reset_index().apply(lambda x:x[1], axis=1).values, index=df.index)
    f1 = [round(value['f1'],3) for k,value in candsets_unsuper_results.items()]
    unsuper_res_ser = pd.Series(f1,index=candsets_unsuper_results.keys(),name=('','','unsuper_res'))
    df = pd.merge(df,unsuper_res_ser,how='left',left_on='Target',right_index=True)
    #df.drop(columns=[('target', '', '')],inplace=True)
    # round unsupervised results to three decimals
    df[('','','unsuper_res')] = df[('','','unsuper_res')].apply(lambda x:round(x,3))
    #create dictionary to highlight the x_target_exceed values and change NaN to '-'
    col_tar_exc = [col for col in df.columns if col[2]=='Tar_exc']
    col_tar_exc_format = {}
    for col in col_tar_exc:
        col_tar_exc_format.update({col:lambda x: '<b>{0:g}</b>'.format(x) if (not math.isnan(float(x))) else '-'})

    col_tar_exc_format.update({('','','unsuper_res'):lambda x: '<font color=\'#00938B\'><b>{}</b></font>'.format(round(x,3))})
    col_tl_avg = [col for col in df.columns if col[2]=='TL_avg']
    html = (df.style.\
            apply(highlight_max,axis=1,subset=pd.IndexSlice[:,col_tl_avg]).\
            apply(lambda x: ['background: #FF7070' if float(v) < x.iloc[-1] else '' for v in x], axis=1).\
            apply(highlight_tar_exc,axis=1).\
            apply(highlight_tl_super_same,axis=1)).set_table_styles(styles).set_precision(3).format(col_tar_exc_format)
    if(da_weighting == 'no_weighting'):
        print('TL Results when doing naive Transfer (no_weighting) on all feature and only dense features')
    else:
        print('TL Results when doing Transfer with domain adapted ({}) source instances on all feature and only dense features'.format(da_weighting))
    display(html)
    if(filename is not None):
        with open('{}.html'.format(filename), 'w') as f:
            f.write(html.render())
    return df

    
def returnDFwithTLResultsOfOneFeatureSet(df_tl_results,feature,candsets_unsuper_results,filename=None):
    ####################################################
    # function for pandas styler. Highlight all max values of TL_avg
    def highlight_max(data):
        attr_max = 'background-color: #FBFF75'
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = data == data.max()
            return [attr_max if v else '' for v in is_max]
        else: 
            is_max = data.groupby('TL_avg').transform('max') == data
            return pd.DataFrame(np.where(is_max, attr_max, ''),
                                index=data.index, columns=data.columns)
    # another function for pandas style. Highlight Tar_exc with green            
    def highlight_tar_exc_featureset(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = row[:,'TL_avg']>row[:,'Tar_max']
        # counter if True at Pos 0 (i==0) in ser then lst at pos 2 (i+k) has to be changed
        # if i==1 then pos 6 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 2
        for i, b in enumerate(ser):
            if(b):
                lst[i+k] = 'background-color: #A4FB95'
            k = k + 3
        return lst

    # another function for pandas style. Highlight Tar_exc with green            
    def highlight_tl_super_same_featureset(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = ((row[:,'TL_avg']>row[:,'Tar_sup']) | ((row[:,'Tar_sup']-row[:,'TL_avg'])<=0.01))
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 3
        for i, b in enumerate(ser):
            if(b):
                lst[i+k] = 'background-color: #A4FB95'
            k = k + 3
        return lst
    #####################################################
    # specify the styles for the html output
    styles=[
        {'selector': 'th','props': [
            ('border-style', 'solid'),
            ('border-color', '#D3D3D3'),
            ('vertical-align','top'),
            ('text-align','center')]}]
    
    df = df_tl_results.xs(feature,axis=1,level=0).copy()
    f1 = [round(value['f1'],3) for k,value in candsets_unsuper_results.items()]
    unsuper_res_ser = pd.Series(f1,index=candsets_unsuper_results.keys(),name=('','unsuper_res'))
    df = pd.merge(df,unsuper_res_ser,how='left',left_on='Target',right_index=True)
    #import pdb; pdb.set_trace()
    col_tar_exc = [col for col in df.columns if col[1]=='Tar_exc']
    col_tar_exc_format = {}
    for col in col_tar_exc:
        col_tar_exc_format.update({col:lambda x: '<b>{0:g}</b>'.format(x) if (not math.isnan(float(x))) else '-'})

    col_tar_exc_format.update({('','unsuper_res'):lambda x: '<font color=\'#00938B\'><b>{}</b></font>'.format(round(x,3))})
    col_tl_avg = [col for col in df.columns if col[1]=='TL_avg']
    html = (df.style.\
            apply(highlight_max,axis=1,subset=pd.IndexSlice[:,col_tl_avg]).\
            apply(lambda x: ['background: #FF7070' if float(v) < x.iloc[-1] else '' for v in x], axis=1).\
            apply(highlight_tar_exc_featureset,axis=1).\
            apply(highlight_tl_super_same_featureset,axis=1)).set_table_styles(styles).set_precision(3).format(col_tar_exc_format)
    display(html)
    if(filename is not None):
        with open('{}.html'.format(filename), 'w') as f:
            f.write(html.render())
    return df

    
#def returnDFwithTLResultsSubset(df_tl_results,candsets_unsuper_results,feature,selected_estimator,filename=None):
#
#    # specify the styles for the html output
#    styles=[
#        {'selector': 'th','props': [
#            ('border-style', 'solid'),
#            ('border-color', '#D3D3D3'),
#            ('vertical-align','top'),
#            ('text-align','center')]}]
#    
#    df = df_tl_results.xs(feature,axis=1,level=0).copy()
#    if(selected_estimator is not None):
#        print(f'Only displaying the results of {selected_estimator} using {feature} feature')
#        ####################################################
#        # another function for pandas style. Highlight Tar_exc with green            
#        def highlight_tar_exc_featureset(row):
#            # initiate list with the size of the row (length 32) with empty styling
#            lst = ['' for x in row.index]
#            # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
#            ser = row['TL_avg']>row['Tar_max']
#            # counter if True at Pos 0 (i==0) in ser then lst at pos 2 (i+k) has to be changed
#            # if i==1 then pos 6 (i+k). K has to be incremented by 3 for each execution of the loop
#            if(ser):
#                lst[2] = 'background-color: #A4FB95'
#            return lst
#    
#        # another function for pandas style. Highlight Tar_exc with green            
#        def highlight_tl_super_same_featureset(row):
#            # initiate list with the size of the row (length 32) with empty styling
#            lst = ['' for x in row.index]
#            # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
#            ser = ((row['TL_avg']>row['Tar_sup']) | ((row['Tar_sup']-row['TL_avg'])<=0.01))
#            # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
#            # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
#            if(ser):
#                lst[3] = 'background-color: #A4FB95'
#            return lst
#        #####################################################
#    
#        df = df.xs(selected_estimator,axis=1,level=0).copy()
#        f1 = [round(value['f1'],3) for k,value in candsets_unsuper_results.items()]
#        unsuper_res_ser = pd.Series(f1,index=candsets_unsuper_results.keys(),name=('unsuper_res'))
#        df = pd.merge(df,unsuper_res_ser,how='left',left_on='Target',right_index=True)
#        #import pdb; pdb.set_trace()
#        col_tar_exc = [col for col in df.columns if col=='Tar_exc']
#        col_tar_exc_format = {}
#        for col in col_tar_exc:
#            col_tar_exc_format.update({col:lambda x: '<b>{0:g}</b>'.format(x) if (not math.isnan(float(x))) else '-'})
#    
#        col_tar_exc_format.update({'unsuper_res':lambda x: '<font color=\'#00938B\'><b>{}</b></font>'.format(round(x,3))})
#        #col_tl_avg = [col for col in df.columns if col=='TL_avg']
#        html = (df.style.\
#                apply(lambda x: ['background: #FF7070' if float(v) < x.iloc[-1] else '' for v in x], axis=1).\
#                apply(highlight_tar_exc_featureset,axis=1).\
#                apply(highlight_tl_super_same_featureset,axis=1)).set_table_styles(styles).set_precision(3).format(col_tar_exc_format)
#        display(html)
#    else:
#        ####################################################
#        # function for pandas styler. Highlight all max values of TL_avg
#        def highlight_max(data):
#            attr_max = 'background-color: #FBFF75'
#            if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
#                is_max = data == data.max()
#                return [attr_max if v else '' for v in is_max]
#            else: 
#                is_max = data.groupby('TL_avg').transform('max') == data
#                return pd.DataFrame(np.where(is_max, attr_max, ''),
#                                    index=data.index, columns=data.columns)
#        # another function for pandas style. Highlight Tar_exc with green            
#        def highlight_tar_exc_featureset(row):
#            # initiate list with the size of the row (length 32) with empty styling
#            lst = ['' for x in row.index]
#            # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
#            ser = row[:,'TL_avg']>row[:,'Tar_max']
#            # counter if True at Pos 0 (i==0) in ser then lst at pos 2 (i+k) has to be changed
#            # if i==1 then pos 6 (i+k). K has to be incremented by 3 for each execution of the loop
#            k = 2
#            for i, b in enumerate(ser):
#                if(b):
#                    lst[i+k] = 'background-color: #A4FB95'
#                k = k + 3
#            return lst
#    
#        # another function for pandas style. Highlight Tar_exc with green            
#        def highlight_tl_super_same_featureset(row):
#            # initiate list with the size of the row (length 32) with empty styling
#            lst = ['' for x in row.index]
#            # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
#            ser = ((row[:,'TL_avg']>row[:,'Tar_sup']) | ((row[:,'Tar_sup']-row[:,'TL_avg'])<=0.01))
#            # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
#            # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
#            k = 3
#            for i, b in enumerate(ser):
#                if(b):
#                    lst[i+k] = 'background-color: #A4FB95'
#                k = k + 3
#            return lst
#        #####################################################
#        f1 = [round(value['f1'],3) for k,value in candsets_unsuper_results.items()]
#        unsuper_res_ser = pd.Series(f1,index=candsets_unsuper_results.keys(),name=('','unsuper_res'))
#        df = pd.merge(df,unsuper_res_ser,how='left',left_on='Target',right_index=True)
#        #import pdb; pdb.set_trace()
#        col_tar_exc = [col for col in df.columns if col[1]=='Tar_exc']
#        col_tar_exc_format = {}
#        for col in col_tar_exc:
#            col_tar_exc_format.update({col:lambda x: '<b>{0:g}</b>'.format(x) if (not math.isnan(float(x))) else '-'})
#    
#        col_tar_exc_format.update({('','unsuper_res'):lambda x: '<font color=\'#00938B\'><b>{}</b></font>'.format(round(x,3))})
#        col_tl_avg = [col for col in df.columns if col[1]=='TL_avg']
#        html = (df.style.\
#                apply(highlight_max,axis=1,subset=pd.IndexSlice[:,col_tl_avg]).\
#                apply(lambda x: ['background: #FF7070' if float(v) < x.iloc[-1] else '' for v in x], axis=1).\
#                apply(highlight_tar_exc_featureset,axis=1).\
#                apply(highlight_tl_super_same_featureset,axis=1)).set_table_styles(styles).set_precision(3).format(col_tar_exc_format)
#        display(html)
#    
#    if(filename is not None):
#        with open('{}.html'.format(filename), 'w') as f:
#            f.write(html.render())
#    return df 


def returnDFwithTLResultsSelection(df_tl_results,feature,selected_estimator,selected_rows=None,filename=None):
    df = df_tl_results.copy()
    ###########################################################################################
    # style functions as nested functions
    # function for pandas styler. Highlight all max values of TL_avg
    def highlight_max(data):
        attr_max = 'background-color: #FBFF75'
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = data == data.max()
            return [attr_max if v else '' for v in is_max]
        else: 
            is_max = data.groupby('TL_avg').transform('max') == data
            return pd.DataFrame(np.where(is_max, attr_max, ''),
                                index=data.index, columns=data.columns)
    # another function for pandas style. Highlight Tar_exc with green            
    def highlight_tar_exc(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = row[:,:,'TL_avg']>row[:,:,'Tar_max']
        # counter if True at Pos 0 (i==0) in ser then lst at pos 2 (i+k) has to be changed
        # if i==1 then pos 6 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 2
        for i, b in enumerate(ser):
            if(b):
                lst[i+k] = 'background-color: #A4FB95'
            k = k + 3
        return lst
    # another function for pandas style. Highlight Tar_exc with green            
    def highlight_tl_super_same(row):
        # initiate list with the size of the row (length 32) with empty styling
        lst = ['' for x in row.index]
        # get the positions where TL_avg is bigger than Tar_max. pd.Series with length 8
        ser = ((row[:,:,'TL_avg']>row[:,:,'Tar_sup']) | ((row[:,:,'Tar_sup']-row[:,:,'TL_avg'])<=0.01))
        # counter if True at Pos 0 (i==0) in ser then lst at pos 3 (i+k) has to be changed
        # if i==1 then pos 7 (i+k). K has to be incremented by 3 for each execution of the loop
        k = 3
        for i, b in enumerate(ser):
            if(b):
                lst[i+k] = 'background-color: #A4FB95'
            k = k + 3
        return lst
    ############################################################################################

    styles=[
        {'selector': 'th','props': [
            ('border-style', 'solid'),
            ('border-color', '#D3D3D3'),
            ('vertical-align','top'),
            ('text-align','center')]}]

    if(selected_rows is None):
        if(selected_estimator is not None):
            df = df.loc[:,(feature+[''], selected_estimator+[''],slice(None))]
        else:
            df = df.loc[:,(feature+[''], slice(None),slice(None))]
    else:
        if(selected_estimator is not None):
            df = df.loc[selected_rows,(feature+[''], selected_estimator+[''],slice(None))]
        else:
            df = df.loc[selected_rows,(feature+[''], slice(None),slice(None))]    
    col_tar_exc = [col for col in df.columns if col[2]=='Tar_exc']
    col_tar_exc_format = {}
    for col in col_tar_exc:
        col_tar_exc_format.update({col:lambda x: '<b>{0:g}</b>'.format(x) if (not math.isnan(float(x))) else '-'})

    col_tar_exc_format.update({('','','unsuper_res'):lambda x: '<font color=\'#00938B\'><b>{}</b></font>'.format(round(x,3))})
    col_tl_avg = [col for col in df.columns if col[2]=='TL_avg']
    html = (df.style.\
            apply(highlight_max,axis=1,subset=pd.IndexSlice[:,col_tl_avg]).\
            apply(lambda x: ['background: #FF7070' if float(v) < x.iloc[-1] else '' for v in x], axis=1).\
            apply(highlight_tar_exc,axis=1).\
            apply(highlight_tl_super_same,axis=1)).set_table_styles(styles).set_precision(3).format(col_tar_exc_format)

    display(html)
    if(filename is not None):
        with open('{}.html'.format(filename), 'w') as f:
            f.write(html.render())
    return df
#%%
#### Quantifying model change ####

# help function    
def getDenseFeatureForCombo(combo,dense_features_dict):
    if(isinstance(combo,tuple)):
        source_name = combo[0]
        target_name = combo[1]
    else:
        source_name = '_'.join(combo.split('_')[:2])
        target_name = '_'.join(combo.split('_')[2:])
    dense_feature_key = '_'.join(sorted(set(source_name.split('_')+target_name.split('_'))))
    return dense_features_dict[dense_feature_key]

# get df with model parameters comparison at start and end
    
def createDFModelChange(atlx_results,al_unsup_results,candsets_super_results,dense_features_dict,selected_qs=None,clf='rf',display_feature_importance=True,filename=None):
    data = {}
    d = atlx_results
    passive_res = candsets_super_results
    if(selected_qs is None):
        for combo in d:
            source = '_'.join(combo.split('_')[:2])
            target = '_'.join(combo.split('_')[2:])
            feature = getDenseFeatureForCombo(combo,dense_features_dict)
            for qs in d[combo][clf]:
                for ws in d[combo][clf][qs]:
                    n_ls_start = d[combo][clf][qs][ws]['n_init_labeled']
                    share_noise_pos_start = d[combo][clf][qs][ws]['share_noise_labeled_set_pos']
                    share_noise_neg_start = d[combo][clf][qs][ws]['share_noise_labeled_set_neg']
                    share_corrected_labels = d[combo][clf][qs][ws]['share_of_corrected_labels']
                    f1_train_start = round(np.mean(d[combo][clf][qs][ws]['training_f1_scores'],axis=0)[0],3)
                    f1_train_end = round(np.mean(d[combo][clf][qs][ws]['training_f1_scores'],axis=0)[-1],3)
                    f1_test_start = round(np.mean(d[combo][clf][qs][ws]['test_f1_scores'],axis=0)[0],3)
                    f1_test_end = round(np.mean(d[combo][clf][qs][ws]['test_f1_scores'],axis=0)[-1],3)
                    delta_f1_test = f1_test_end - f1_test_start
                    f1_test_end_random = round(np.mean(d[combo][clf]['random'][ws]['test_f1_scores'],axis=0)[-1],3)
                    f1_std_end_random = round(np.std(d[combo][clf]['random'][ws]['test_f1_scores'],axis=0)[-1],3)
                    delta_f1_to_random = f1_test_end - f1_test_end_random
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
                    delta_passive_end = f1_test_end - round(passive_res[target][clf_old]['f1'],3)
                    delta_AL_end = f1_test_end - round(np.mean(al_unsup_results[target][clf][qs]['test_f1_scores'],axis=0)[-1],3)
                    f1_std_AL_end = round(np.std(al_unsup_results[target][clf][qs]['test_f1_scores'],axis=0)[-1],3)
                    #f1_std_test_start = round(np.std(d[combo][clf][qs][ws]['test_f1_scores'],axis=0)[0],3)
                    f1_std_test_end = round(np.std(d[combo][clf][qs][ws]['test_f1_scores'],axis=0)[-1],3)
                    feature_import_start = np.around(np.mean(d[combo][clf][qs][ws]['model_feature_import_start'],axis=0),3)
                    feature_import_end = np.around(np.mean(d[combo][clf][qs][ws]['model_feature_import_end'],axis=0),3)
                    sorted_feature_start = sorted(list(zip(feature, feature_import_start)), key = lambda x: x[1], reverse=True)
                    sorted_feature_end = sorted(list(zip(feature, feature_import_end)), key = lambda x: x[1], reverse=True)
                    top_5_feature_start = sorted_feature_start[:5]
                    top_5_feature_end = sorted_feature_end[:5]
                    ordered_attribute_import_start = list(dict.fromkeys([feature[0].split('_')[0] for feature in sorted_feature_start]))
                    ordered_attribute_import_end = list(dict.fromkeys([feature[0].split('_')[0] for feature in sorted_feature_end]))
                    #neg_pred_start = np.mean(d[combo][clf][qs][ws]['model_pred_prob_start'],axis=0)[:,0]
                    pos_pred_start = np.mean(d[combo][clf][qs][ws]['model_pred_prob_start'],axis=0)[:,1]
                    #neg_pred_end = np.mean(d[combo][clf][qs][ws]['model_pred_prob_end'],axis=0)[:,0]
                    pos_pred_end = np.mean(d[combo][clf][qs][ws]['model_pred_prob_end'],axis=0)[:,1]
                    # custom measure about confidence of the classifier using predict_proba from the test set
                    # checking how far the predict probas are away from threshold 0.5, take the mean of the abs values
                    # a higher value indicates that the model is rather confident in its predictions and a value close to 0
                    # indicates it is rather unconfident. if the predicitons are true is not considered here
                    conf_start = round(np.mean(np.abs(pos_pred_start-0.5)),3)
                    conf_end = round(np.mean(np.abs(pos_pred_end-0.5)),3)
                    avg_depth_tree_start = round(np.mean(np.mean(d[combo][clf][qs][ws]['model_depth_tree_start'],axis=0)),3)
                    avg_depth_tree_end = round(np.mean(np.mean(d[combo][clf][qs][ws]['model_depth_tree_end'],axis=0)),3)
                    tuple_results = (n_ls_start,share_noise_pos_start,share_noise_neg_start,share_corrected_labels,
                                     f1_train_start,f1_train_end,f1_test_start,f1_test_end,delta_f1_test,delta_AL_end,delta_f1_to_random,delta_passive_end,
                                     f1_std_test_end,f1_std_AL_end,f1_std_end_random,conf_start,conf_end,avg_depth_tree_start,
                                     avg_depth_tree_end,top_5_feature_start,top_5_feature_end,
                                     ordered_attribute_import_start,ordered_attribute_import_end)
                    if(ws=='no_weighting'): 
                        data.update({(source,target,qs,'-'):tuple_results})
                    elif(ws=='nn'): 
                        data.update({(source,target,qs,'nn'):tuple_results})
                    else:
                        data.update({(source,target,qs,'lp'):tuple_results})

        df = pd.DataFrame.from_dict(data,orient='index',columns=['ls','n_+','n_-','cor','f1_in_0','f1_in_-1',
                                                             'f1_out_0','f1_out_-1','Δ_out_f1','Δ_AL_f1','Δ_R_f1','Δ_pas_f1',
                                                             'sig_out','sig_AL','sig_R','c_0','c_-1','adt_0','adt_-1',
                                                             'top_5_feature_0','top_5_feature_-1',
                                                             'ordered_attr_import_0','ordered_attr_import_-1'])
        df.index = pd.MultiIndex.from_tuples(df.index,names=['Source','Target','QS','DA'])
    else:
        for combo in d:
            source = '_'.join(combo.split('_')[:2])
            target = '_'.join(combo.split('_')[2:])
            feature = getDenseFeatureForCombo(combo,dense_features_dict)
            for qs in selected_qs:
                for ws in d[combo][clf][qs]:
                    n_ls_start = d[combo][clf][qs][ws]['n_init_labeled']
                    share_noise_pos_start = d[combo][clf][qs][ws]['share_noise_labeled_set_pos']
                    share_noise_neg_start = d[combo][clf][qs][ws]['share_noise_labeled_set_neg']
                    share_corrected_labels = d[combo][clf][qs][ws]['share_of_corrected_labels']
                    f1_train_start = round(np.mean(d[combo][clf][qs][ws]['training_f1_scores'],axis=0)[0],3)
                    f1_train_end = round(np.mean(d[combo][clf][qs][ws]['training_f1_scores'],axis=0)[-1],3)
                    f1_test_start = round(np.mean(d[combo][clf][qs][ws]['test_f1_scores'],axis=0)[0],3)
                    f1_test_end = round(np.mean(d[combo][clf][qs][ws]['test_f1_scores'],axis=0)[-1],3)
                    delta_f1_test = f1_test_end - f1_test_start
                    f1_test_end_random = round(np.mean(d[combo][clf]['random'][ws]['test_f1_scores'],axis=0)[-1],3)
                    delta_f1_to_random = f1_test_end - f1_test_end_random
                    f1_std_end_random = round(np.std(d[combo][clf]['random'][ws]['test_f1_scores'],axis=0)[-1],3)
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
                    delta_passive_end = f1_test_end - round(passive_res[target][clf_old]['f1'],3)
                    delta_AL_end = f1_test_end - round(np.mean(al_unsup_results[target][clf][qs]['test_f1_scores'],axis=0)[-1],3)
                    f1_std_AL_end = round(np.std(al_unsup_results[target][clf][qs]['test_f1_scores'],axis=0)[-1],3)
                    #f1_std_test_start = round(np.std(d[combo][clf][qs][ws]['test_f1_scores'],axis=0)[0],3)
                    f1_std_test_end = round(np.std(d[combo][clf][qs][ws]['test_f1_scores'],axis=0)[-1],3)
                    feature_import_start = np.around(np.mean(d[combo][clf][qs][ws]['model_feature_import_start'],axis=0),3)
                    feature_import_end = np.around(np.mean(d[combo][clf][qs][ws]['model_feature_import_end'],axis=0),3)
                    sorted_feature_start = sorted(list(zip(feature, feature_import_start)), key = lambda x: x[1], reverse=True)
                    sorted_feature_end = sorted(list(zip(feature, feature_import_end)), key = lambda x: x[1], reverse=True)
                    top_5_feature_start = sorted_feature_start[:5]
                    top_5_feature_end = sorted_feature_end[:5]
                    ordered_attribute_import_start = list(dict.fromkeys([feature[0].split('_')[0] for feature in sorted_feature_start]))
                    ordered_attribute_import_end = list(dict.fromkeys([feature[0].split('_')[0] for feature in sorted_feature_end]))
                    #neg_pred_start = np.mean(d[combo][clf][qs][ws]['model_pred_prob_start'],axis=0)[:,0]
                    pos_pred_start = np.mean(d[combo][clf][qs][ws]['model_pred_prob_start'],axis=0)[:,1]
                    #neg_pred_end = np.mean(d[combo][clf][qs][ws]['model_pred_prob_end'],axis=0)[:,0]
                    pos_pred_end = np.mean(d[combo][clf][qs][ws]['model_pred_prob_end'],axis=0)[:,1]
                    # custom measure about confidence of the classifier using predict_proba from the test set
                    # checking how far the predict probas are away from threshold 0.5, take the mean of the abs values
                    # a higher value indicates that the model is rather confident in its predictions and a value close to 0
                    # indicates it is rather unconfident. if the predicitons are true is not considered here
                    conf_start = round(np.mean(np.abs(pos_pred_start-0.5)),3)
                    conf_end = round(np.mean(np.abs(pos_pred_end-0.5)),3)
                    avg_depth_tree_start = round(np.mean(np.mean(d[combo][clf][qs][ws]['model_depth_tree_start'],axis=0)),3)
                    avg_depth_tree_end = round(np.mean(np.mean(d[combo][clf][qs][ws]['model_depth_tree_end'],axis=0)),3)
                    tuple_results = (n_ls_start,share_noise_pos_start,share_noise_neg_start,share_corrected_labels,
                                     f1_train_start,f1_train_end,f1_test_start,f1_test_end,delta_f1_test,delta_AL_end,delta_f1_to_random,delta_passive_end,
                                     f1_std_test_end,f1_std_AL_end,f1_std_end_random,conf_start,conf_end,avg_depth_tree_start,
                                     avg_depth_tree_end,top_5_feature_start,top_5_feature_end,
                                     ordered_attribute_import_start,ordered_attribute_import_end)
                    if(ws=='no_weighting'): 
                        data.update({(source,target,qs,'-'):tuple_results})
                    elif(ws=='nn'): 
                        data.update({(source,target,qs,'nn'):tuple_results})
                    else:
                        data.update({(source,target,qs,'lp'):tuple_results})

        df = pd.DataFrame.from_dict(data,orient='index',columns=['ls','n_+','n_-','cor','f1_in_0','f1_in_-1',
                                                             'f1_out_0','f1_out_-1','Δ_out_f1','Δ_AL_f1','Δ_R_f1','Δ_pas_f1',
                                                             'sig_out','sig_AL','sig_R','c_0','c_-1','adt_0','adt_-1',
                                                             'top_5_feature_0','top_5_feature_-1',
                                                             'ordered_attr_import_0','ordered_attr_import_-1'])
        df.index = pd.MultiIndex.from_tuples(df.index,names=['Source','Target','QS','DA'])
    
    if(display_feature_importance):
        styleDFModelChange(df,False,filename)
    else:
        df_subset = df.drop(columns=['top_5_feature_0', 'top_5_feature_-1',
                                     'ordered_attr_import_0', 'ordered_attr_import_-1'])
        styleDFModelChange(df_subset,False,filename)

    return df

def styleDFModelChange(df,ix_subset=False,filename=None):
    styles=[{'selector': 'th','props': [
        ('border-style', 'solid'),
        ('border-color', '#D3D3D3'),
        ('vertical-align','top'),
        ('text-align','center')]}]
    def highlight_max(data,color= 'yellow'):
        attr = 'background-color: {}'.format(color)

        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = data == data.max()
            return [attr if v else '' for v in is_max]
        else: 
            is_max = data.groupby(level=[0,1]).transform('max') == data
            return pd.DataFrame(np.where(is_max, attr, ''),
                                index=data.index, columns=data.columns)

    def highlight_exceed_x(data,thr,color= 'yellow'):
        attr = 'background-color: {}'.format(color)

        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_greater = data > thr
            return [attr if v else '' for v in is_greater]
        else: 
            is_greater = data > thr
            return pd.DataFrame(np.where(is_greater, attr, ''),
                                index=data.index, columns=data.columns)
    
    def highlight_below_x(data,thr,color= 'yellow'):
        attr = 'background-color: {}'.format(color)

        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_greater = data < thr
            return [attr if v else '' for v in is_greater]
        else: 
            is_greater = data < thr
            return pd.DataFrame(np.where(is_greater, attr, ''),
                                index=data.index, columns=data.columns)
            
    def highlight_min(data,color= 'yellow'):
        attr = 'background-color: {}'.format(color)

        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_min = data == data.min()
            return [attr if v else '' for v in is_min]
        else: 
            is_min = data.groupby(level=[0,1]).transform('min') == data
            return pd.DataFrame(np.where(is_min, attr, ''),
                                index=data.index, columns=data.columns)
            
    if(ix_subset):
        html = (df.style.apply(lambda x: highlight_exceed_x(x,thr=0,color='#ff00d9'),axis=None,subset=pd.IndexSlice[:, ['Δ_pas_f1']]).\
                                                            apply(highlight_min,axis=1,subset=pd.IndexSlice[:, ['sig_out','sig_AL','sig_R']]).\
                                                            apply(lambda x: highlight_exceed_x(x,thr=0.01,color='#66FF99'),axis=None,subset=pd.IndexSlice[:, ['Δ_AL_f1']]).\
                                                            apply(lambda x: highlight_exceed_x(x,thr=0.01,color='#66FF99'),axis=None,subset=pd.IndexSlice[:, ['Δ_R_f1']]).\
                                                            apply(lambda x: highlight_below_x(x,thr=0.01,color='red'),axis=None,subset=pd.IndexSlice[:, ['Δ_out_f1']]).\
                                                            apply(lambda x: highlight_below_x(x,thr=0,color='red'),axis=None,subset=pd.IndexSlice[:, ['Δ_R_f1']]).\
                                                            apply(lambda x: highlight_below_x(x,thr=0,color='red'),axis=None,subset=pd.IndexSlice[:, ['Δ_AL_f1']]).\
                                                            apply(lambda x: highlight_exceed_x(x,thr=0,color='red'),axis=None,subset=pd.IndexSlice[:, ['n_+','n_-']])).set_table_styles(styles).set_precision(3)
    else:
        html = (df.style.apply(highlight_min,axis=1,subset=pd.IndexSlice[:, ['sig_out','sig_AL','sig_R']]).\
                apply(highlight_max,axis=None,subset=pd.IndexSlice[:, ['cor','f1_out_-1']]).\
                apply(lambda x: highlight_exceed_x(x,thr=0.01,color='#66FF99'),axis=None,subset=pd.IndexSlice[:, ['Δ_AL_f1']]).\
                apply(lambda x: highlight_exceed_x(x,thr=0.01,color='#66FF99'),axis=None,subset=pd.IndexSlice[:, ['Δ_R_f1']]).\
                apply(lambda x: highlight_exceed_x(x,thr=0,color='#ff00d9'),axis=None,subset=pd.IndexSlice[:, ['Δ_pas_f1']]).\
                apply(lambda x: highlight_below_x(x,thr=0.01,color='red'),axis=None,subset=pd.IndexSlice[:, ['Δ_out_f1']]).\
                apply(lambda x: highlight_below_x(x,thr=0,color='red'),axis=None,subset=pd.IndexSlice[:, ['Δ_R_f1']]).\
                apply(lambda x: highlight_below_x(x,thr=0,color='red'),axis=None,subset=pd.IndexSlice[:, ['Δ_AL_f1']]).\
                apply(lambda x: highlight_exceed_x(x,thr=0,color='red'),axis=None,subset=pd.IndexSlice[:, ['n_+','n_-']])).set_table_styles(styles).set_precision(3)
                #apply(highlight_max,axis=None,subset=pd.IndexSlice[:, ['cor','f1_out_-1','c_0','c_-1']]).\
                
    display(html)
    
    if(filename is not None):
        with open('{}.html'.format(filename), 'w') as f:
            f.write(html.render())
            
def styleSimple(df,filename):
    styles=[{'selector': 'th','props': [
        ('border-style', 'solid'),
        ('border-color', '#D3D3D3'),
        ('vertical-align','top'),
        ('text-align','center')]}]
    html = df.style.set_table_styles(styles).set_precision(3)
    display(html)
    
    if(filename is not None):
        with open('{}.html'.format(filename), 'w') as f:
            f.write(html.render())