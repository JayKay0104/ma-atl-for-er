# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 08:51:30 2020

@author: jonas
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_color_codes()
import numpy.matlib
import itertools
import math
import copy
from IPython.display import display

#%%
            
def plotTLResults(tl_results,source_name,target_name,feature,selected_estimators,selected_da_weighting,candsets,
                  candsets_super_results,candsets_unsuper_results,plot_target,path_for_output=None):
    
    x=[0,10,14,20,24,28,32,38,44,50,60,70,80,90,100,120,140,160,180,200,300,500]
    d = tl_results
    
    combo = '{}_{}'.format(source_name,target_name)
    fig,ax = plt.subplots(figsize=(16,8))
    plt.subplots_adjust(right=0.7)
    plt.close(fig)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # unsupervised bm plot
    y_target_unsup_bm = list(itertools.repeat(candsets_unsuper_results[target_name],len(x)))
    ax.plot(x,y_target_unsup_bm,linestyle='dashdot',color='g',lw=2,label='target unsupervised (elbow) benchmark. F1: {}'.format(round(candsets_unsuper_results[target_name],2)))
    ax.set_xlabel('x target instances used for training',fontsize=12)
    ax.set_ylabel('Avg. F1-Score',fontsize=12)
    
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
                ax.plot(x,y_transfer_results,linewidth=2,
                        label='transfer learning results ({}) when trained on all {} source instances. F1: {}'.format(clf,candsets[source_name].shape[0],transfer_result_avg),
                        color=colors[i%len(colors)])
                if(plot_target):
                    ax.plot(x,y_target_results,linewidth=2,linestyle='dashed',
                            label='target results ({}): trained on x target instances'.format(clf),
                            color=colors[i%len(colors)])
                    if(not math.isnan(float(d[combo][feature][da][clf]['x_target_exceed']))):
                        idx = x.index(d[combo][feature][da][clf]['x_target_exceed'])
                        ax.plot(x[idx], y_target_results[idx], 'ro') 
                # benchmark plots
                # supervised
                y_target_sup_bm = list(itertools.repeat(candsets_super_results[target_name][clf],len(x)))
                ax.plot(x,y_target_sup_bm,linestyle='dotted',label='target supervised ({}) benchmark. F1: {}'.format(clf,round(candsets_super_results[target_name][clf],2)),
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
                ax.plot(x,y_transfer_results,linewidth=2,
                        label='transfer learning results ({}) when trained on all {} source instances. F1: {}'.format(clf,candsets[source_name].shape[0],transfer_result_avg),
                        color=colors[i%len(colors)])
                if(plot_target):
                    ax.plot(x,y_target_results,linewidth=2,linestyle='dashed',
                            label='target results ({}): trained on x target instances'.format(clf),
                            color=colors[i%len(colors)])
                    if(not math.isnan(float(d[combo][feature][da][clf]['x_target_exceed']))):
                        idx = x.index(d[combo][feature][da][clf]['x_target_exceed'])
                        ax.plot(x[idx], y_target_results[idx], 'ro') 
                # benchmark plots
                # supervised
                y_target_sup_bm = list(itertools.repeat(candsets_super_results[target_name][clf],len(x)))
                ax.plot(x,y_target_sup_bm,linestyle='dotted',label='target supervised ({}) benchmark. F1: {}'.format(clf,round(candsets_super_results[target_name][clf],2)),
                        color=colors[i%len(colors)])
        
        
    # add legend to plot
    ax.legend(fontsize=12)
    # add a text box with info
    info_text = 'Transfer Learning (TL)\nand the target results\nwere tested on the same\ntest set from the target.'
    if(plot_target):
        info_text_2 = 'The target results were\ntrained on random\nsamples of size x\nfrom the target\ntraining set and avg.\nover n={} random\nsamples of x.'.format(n)
        textstr = 'INFO:\n{}\n{}'.format(info_text,info_text_2)
    else:
        textstr = 'INFO:\n{}'.format(info_text)
    ax.text(0.71, 0.85, textstr, transform=fig.transFigure, fontsize=12,verticalalignment='top', bbox=props)
    features_used = '{} features were used'.format(feature)
    ax.set_title('Transfer Learning Results of Naive Transfer: Trained on source {} and tested on target {}\n{}'.format(source_name,
                 target_name,features_used))
    if(path_for_output is not None):
        fig.savefig('{}{}_{}_{}_n{}.png'.format(path_for_output,source_name,target_name,feature,n),bbox_inches='tight')
    
    return fig    

#%%
#generator function to create tuples of every n element
def group(lst, n):
    for i in range(0, len(lst), n):
        val = lst[i:i+n]
        if len(val) == n:
            yield tuple(val)
            

def createDFwithTLResults(tl_results,candsets_super_results,candsets_unsuper_results,
                               estimators,da_weighting='no_weighting',filename='tl_results'):
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
    number_of_estimators = len(estimators)
    n = number_of_estimators*2*3 
    # getting the source and target keys (first keys in dictionary)
    keys = list(d.keys())
    # when the transfer learning results are saved in a dictionary from json file (dictionary from the experiments was stored in json)
    # then the keys have to be changed back to tuples (for Multiindex)
    if(not isinstance(keys[0],tuple)):
        keys = [('_'.join(k.split('_')[:2]),'_'.join(k.split('_')[2:])) for k in keys]
#    # getting the transfer results
#    reform = {(outerKey, innerKey): values for outerKey, innerDict in d.items() for innerKey, values in innerDict.items()}
#    reform_2 = {(outerKey, innerKey): values for outerKey, innerDict in reform.items() for innerKey, values in innerDict.items()}
#    list_transfer_results = [round(v['transfer_avg_result'],3) for k,v in reform_2.items()]    
#    list_x_target_exceed = [v['x_target_exceed'] for k,v in reform_2.items()]
#    list_x_target_max_500 = [v['target_max_result'] for k,v in reform_2.items()]
#    # bringing the transfer_results in tuple format
#    res_tuple_list = list(group(list(itertools.chain.from_iterable(list(zip(list_transfer_results,list_x_target_max_500,list_x_target_exceed)))),n))

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
    values = [round(values,3) for k,innerdict in tl_candsets_super_results.items() for innerkey, values in innerdict.items()]
    df_super = pd.DataFrame(list(group(values,number_of_estimators)),index=tl_candsets_super_results.keys(),columns=innerkeys[:number_of_estimators])
    #list_of_estimators=['logreg','logregcv','dectree','randforest']
    columns_before = list(df_super.columns)
    df_super = pd.concat([df_super,df_super],axis=1)
    df_super.columns = pd.MultiIndex.from_product([['all','dense'], columns_before, ['Tar_sup']],names=['Features','Estimators','Results'])
    df = pd.merge(df,df_super,how='left',left_on='Target',right_index=True)
    df = df.reindex(columns=['all','dense'],level=0)
    df = df.reindex(columns=clfs,level=1)
    df = df.reindex(columns=['TL_avg','Tar_max','Tar_exc','Tar_sup'], level=2)

    #df['target'] = pd.Series(df.reset_index().apply(lambda x:x[1], axis=1).values, index=df.index)
    unsuper_res_ser = pd.Series(candsets_unsuper_results,index=candsets_unsuper_results.keys(),name=('','','unsuper_res'))
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

#%%
    
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
    unsuper_res_ser = pd.Series(candsets_unsuper_results,index=candsets_unsuper_results.keys(),name=('','unsuper_res'))
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


#%%
    
def createDFwithSuperResults(candsets_super_results,number_of_estimators):
    innerkeys =  [innerkey for k,innerdict in candsets_super_results.items() for innerkey, values in innerdict.items()]
    values = [round(values,3) for k,innerdict in candsets_super_results.items() for innerkey, values in innerdict.items()]
    df_super = pd.DataFrame(list(group(values,4)),index=candsets_super_results.keys(),columns=innerkeys[:4])
    return df_super

    
def attachSuperToTLDF(df_tl_results, candsets_super_results, list_of_estimators=['logreg','logregcv','dectree','randforest']):
    df_super = createDFwithSuperResults(candsets_super_results,len(list_of_estimators))
    df_super = df_super.reindex(df_tl_results.index.get_level_values('Target'))
    columns_before = list(df_super.columns)
    df_super = pd.concat([df_super,df_super],axis=1)
    df_super.columns = pd.MultiIndex.from_product([['all','dense'], columns_before, ['Tar_sup']],names=['Features','Estimators','Results'])
    df = pd.merge(df_tl_results,df_super,how='left',left_on='Target',right_index=True)
    df = df.reindex(columns=['all','dense'],level=0)
    df = df.reindex(columns=list_of_estimators,level=1)
    df = df.reindex(columns=['TL_avg','Tar_max','Tar_exc','Tar_sup'], level=2)
    return df