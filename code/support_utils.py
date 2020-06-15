# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:30:53 2020

@author: jonas
"""

#%%
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_color_codes()
import glob
import re
import json


#%% progress bar
# Print iterations progress
# from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

#%%

def readDataInDictionary(path_to_directory = '../datasets/', pattern_of_filename = '(.*)', sep=';'):
    """
    Function to read in the datasets. The datasets need to be stored as csv files!
    
    Parameters:
    
    path_to_directory: Specify the path from the current directory to the directory where the datasets are stored
    Default: \'../datasets/\'
    
    pattern_of_filename: Specify the pattern of the filenames of the datasets. Only the datasets where the filenames
    matches the pattern get read in. The regex need to be enclosed in parentheses. 
    Exp: Files that only have letters in there filename match this pattern \'([a-zA-Z]*)\'
    Default: \'(.*)\' (reads in all csv files contained in the directory)
    
    sep: Specify the delimiter of the csv files. All csv files need to have the same delimiter
    Default: ';'
    """
    pattern_name = '{}{}.csv'.format(path_to_directory,pattern_of_filename)
    #logger.debug('pattern_name: {}'.format(pattern_name))
    file_list = glob.glob('{}*.csv'.format(path_to_directory))
    #logger.debug('file_list: {}'.format(file_list))
    res = [re.findall(pattern_name,x)[0] for x in file_list if bool(re.match(pattern_name,x))]
    file_list = [x for x in file_list if bool(re.match(pattern_name,x))]
    #logger.debug('file_list: {}'.format(file_list))
    
    dfs = {}
    for i in range(len(file_list)):
        dfs.update({res[i]:pd.read_csv(file_list[i],sep=sep,low_memory=False)})
        #logger.info('{} is read in and is stored in the dictionary with they key [\'{}\']'.format(file_list[i],res[i]))
    return dfs
    
    
#%%
    
def getCandsetsWithOrgAttribute(candset_dict,dataset_dict,id_col_name='ids',label_col_name='label'):
    """
    Retrieve the original attributes from the candsets feature set
    """
    candsets_attr = {}
    for cand in candset_dict:
        temp = candset_dict[cand][[id_col_name,label_col_name]].copy()
        temp['l_id'] = temp[id_col_name].apply(lambda s: '_'.join(s.split('_')[:2]))
        temp['r_id'] = temp[id_col_name].apply(lambda s: '_'.join(s.split('_')[2:]))
        l_key = cand.split('_')[0]
        r_key = cand.split('_')[1]
        l_temp_attr = pd.merge(temp,dataset_dict[l_key],left_on='l_id',right_on=l_key+'_id')
        r_l_temp_attr = pd.merge(l_temp_attr,dataset_dict[r_key],left_on='r_id',right_on=r_key+'_id')
        cols = list(r_l_temp_attr.columns.drop([id_col_name,label_col_name,'l_id','r_id']))
        cols.sort(key=(lambda x: '_'.join(x.split('_')[1:])))
        final_cols = [id_col_name,label_col_name]+cols
        candsets_attr.update({cand:r_l_temp_attr[final_cols]})
    return candsets_attr


#%%
# from: https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj) 

def changekeyFromTupleToString(dictionary):
    return {'{}_{}'.format(key[0],key[1]): value for key, value in dictionary.items()}
    
def saveResultsToJSON(results,filename):
    """
    Saves the Dictionary containing the ATL Experiment Results to JSON file on hard disk.
    
    results: Dictionary containing the results of Experiments
    filename: String with the desired filename. Important only filename .json not required. Hence 'atl_results' would be fine.
    """
    keys = list(results.keys())
    # when the transfer learning results are saved in a dictionary from json file (dictionary from the experiments was stored in json)
    # then the keys have to be changed back to tuples (for Multiindex)
    if(isinstance(keys[0],tuple)):
        d = changekeyFromTupleToString(results)
    else:
        d = results
    
    json_results = json.dumps(d, indent=2, cls=NumpyEncoder)
    with open('{}.json'.format(filename),'w') as f:
        f.write(json_results)
    print('Saved in {}.json'.format(filename))
    return None

#%%

def importJSONFileInDict(filename):
    """
    Imports the TL Exp. Results when previously saved as JSON  as dictionary.
    filename: Filename or Path
    """
    if('.json' not in filename):
        with open('{}.json'.format(filename), 'r') as f:
            d = json.load(f,parse_float=float)
    else:
        with open('{}'.format(filename), 'r') as f:
            d = json.load(f,parse_float=float)
    return d

#%%

# remove text within parentheses from string
def remove_paren(string):
    # ensure input is string by converting it to string and lower case it
    string = str(string)
    # first exclude text within parentheses and ensure no whitespace at the beginning or end
    string = re.sub(r'\(.*\)','',string).strip()
    # before returning the string without parentheses also ensure no double whitespaces are in the string
    return re.sub(r'  ',' ',string).strip()