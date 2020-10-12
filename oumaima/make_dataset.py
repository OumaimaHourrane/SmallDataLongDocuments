#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:06:19 2020

@author: jonas

@tile: make_dataset

@description: script to transform taxonomy from excel sheet to machine readable format in python.
"""
#%%
'''import packages'''
import os
import sys

import pandas as pd 
import pickle
#%%

def import_raw_data():
    
    '''
    import and minimal processing of taxonomy from excel    
    '''
    
    taxonomy = pd.read_excel (os.path.abspath(os.path.join('..', 'data/raw'))+'/tagging_table.xlsx')
    # get column names:
    columnNames = taxonomy.iloc[0] 
    taxonomy = taxonomy[1:] 
    taxonomy.columns = columnNames
    print('raw data shape:', taxonomy.shape)
    # delete entries without PIMS ID:
    taxonomy = taxonomy[taxonomy['PIMS #'].notna()]      
    print('only entries with PIMS ID:', taxonomy.shape)                
    # delete columns without names:
    taxonomy = taxonomy.loc[:, taxonomy.columns.notnull()]   
    print('only columns with entries:', taxonomy.shape)                
    # remove white spaces in column names and lowercase names:
    taxonomy.columns = taxonomy.columns.str.replace(' ', '_').str.lower()
    # rename pims id column:
    taxonomy = taxonomy.rename(columns={"pims_#": "PIMS_ID"})
    
    return taxonomy

def import_api_data():
    
    '''
    function that imports data from PIMS+ API. 
    '''
        
def create_training_texts(dataframe, compare_with_API = False):
    
    """
    1. Takes in whole taxonomy and outputs different training data text fields and replaces "nan" with empty spaces. 
    """
    # objectives
    dataframe['objectives'] = dataframe['project_objective'].fillna('').astype(str) + dataframe['project_objective_2'].fillna('').astype(str)
   
    # rename description
    dataframe['description'] = dataframe['project_description'].fillna('').astype(str)

    
    # outcomes
    dataframe['outcomes'] = dataframe['outcome_1'].fillna('').astype(str)
    
    # outputs
    dataframe['outputs'] = dataframe[['output_1.1', 'output_1.2', 'output_1.3',
                            'output_1.4', 'output_1.5', 'outcome_2', 'output_2.1', 'output_2.2',
                            'output_2.3', 'output_2.4', 'output_2.5', 'outcome_3', 'output_3.1',
                            'output_3.2', 'output_3.3', 'output_3.4', 'output_3.5', 'outcome_4',
                            'output_4.1', 'output_4.2', 'output_4.3', 'output_4.4', 'output_4.5', 
                            'outcome_5', 'output_5.1', 'output_5.2', 'output_5.3',
                            'output_5.4_(no_entry)', 'output_5.5_(no_entry)',
                            'outcome_6_(no_entry)', 'output_6.1', 'output_6.2', 'output_6.3',
                            'output_6.4_(no_entry)', 'output_6.5_(no_entry)',
                            'outcome_7_(no_entry)', 'output_7.1', 'output_7.2_(no_entry)',
                            'output_7.3_(no_entry)', 'output_7.4_(no_entry)','output_7.5_(no_entry)']].fillna('').astype(str).agg(' '.join, axis=1)

    
    dataframe['logframe'] = dataframe[['objectives', 'outcomes', 'outputs']].agg(' '.join, axis=1)
    
    dataframe['all_text'] = dataframe['description'] + dataframe['logframe']

    print('extracting and merging done!')
    
    """
    2. Create dataframe with only raw text fields and PimsIDs
    """
    raw_text = taxonomy[['PIMS_ID', 'all_text', 'logframe', 'description', 'objectives', 'outcomes', 'outputs']]
    
    
    """    
    3. if bool is set as True: Compare with downloaded logframes, descriptions and objectives from PIMS+ to see if they 
    match in length and compleetness.
    - Replace empty fields with non-empty fiels if applicable.
    
    """
    
    if compare_with_API == True:
        '''compare with PIMS+ projects/logframes etc and only keep most relevant'''
        print('Pims_plus API is considered to complete training data')
    
    else:
        print('only taxonomy data is considered')
    
    
    '''pickle data'''
    with open(os.path.abspath(os.path.join('..', 'data/interim'))+'/raw_text.pkl', 'wb') as handle:
        pickle.dump(raw_text, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        #pickle data also for personal QA project:
    # with open(os.path.join('/Users/jonas/Google Drive/github_repos/ClosedDomainQA/data')+'/raw_text.pkl', 'wb') as handle:
    #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return raw_text 
    

def create_subset(dataframe, column_title):
    
    '''
    Takes datafram as input and column name and outputs a dataframe with two columns: project_id and column without empty fields.
        - may be appended with more meta_data than only project_id for downstream tasks.
    
    '''
    
    print('deleting all empty fields and creating subset for:', column_title)   
    #keep only projects with column_title that contain alphabetic letter:
    data =  dataframe[dataframe[column_title].str.contains('[A-Za-z]')]



    
    data = data[['PIMS_ID', column_title]]
    print('remaining projects with non empty field', column_title, data.shape)
    
    #reset index
    data = data.reset_index(drop=True)
    
    #rename text column to text
    data = data.rename(columns={column_title: "text"})


    '''pickle data'''
    with open(os.path.abspath(os.path.join('..', 'data/interim'))+'/'+column_title+'.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        #pickle data also for personal QA project:
    # with open(os.path.join('/Users/jonas/Google Drive/github_repos/ClosedDomainQA/data')+'/'+column_title+'.pkl', 'wb') as handle:
    #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data


def labeling_encoding(dataframe, categories, label_powerset = bool):
    '''
    Function that encodes the label and gives option for different classification architectures (for example label_powerset)
    '''
    # generate binary values using get_dummies
    # categories is a list of categories from the main dataframe to be converted into on-hot encoding labels
    df = pd.DataFrame(dataframe, columns=categories)
    dum_df = pd.get_dummies(df, columns=categories)
    # merge with main df on key values
    df = df.join(dum_df)
    return df
    

#%%

taxonomy = import_raw_data()

raw_text = create_training_texts(taxonomy)

data = create_subset(taxonomy, "logframe")

print(data)
#%%

