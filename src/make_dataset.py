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
    
    return taxonomy

def import_api_data():
    
    '''
    function that imports data from PIMS+ API. 
    '''

def create_subset(dataframe, column_title):
    
    '''
    Takes datafram as input and column name and outputs a dataframe with two columns: project_id and column without empty fields.
        - may be appended with more meta_data than only project_id for downstream tasks.
    
    '''
    
    print('deleting all empty fields and creating subset for:', column_title)   
    #keep only projects with column_title not empty:
    data = dataframe[dataframe[column_title].notna()]     
    data = data[['pims_#', column_title]]
    print('remaining projects with non empty field', column_title, data.shape)
    
    #pickle data:
    with open(os.path.abspath(os.path.join('..', 'data/interim'))+'/'+column_title+'.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    #pickle data also for QA project:
    with open(os.path.join('/Users/jonas/Google Drive/jupyter/ClosedDomainQA/data')+'/'+column_title+'.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return data
        

def create_training_texts(dataframem, compare_with_API = bool):
    
    '''
    1. Takes in whole taxonomy and outputs different training data text fields:
        i) descriptions
        ii) objecitves
        iii) outputs
        iv) outcomes
        iii)logframe (objectives + outputs + outcomes)
        iv) all (descriptoins + logframe)
    
    '''
    
    # your code

    '''    
    2. if bool is set as True: Compare with downloaded logframes, descriptions and objectives from PIMS+ to see if they 
    match in length and compleetness.
    - Replace empty fields with non-empty fiels if applicable.
    
    '''
    
    if compare_with_API == True:
        '''compare with PIMS+ projects/logframes etc and only keep most relevant'''
        print('Pims_plus API is considered to complete training data')
    
    if compare_with_API == False:
        print('only taxonomy data is considered')
    

def labeling_encoding(dataframe, label_powerset = bool):
    
    '''
    Function that encodes the label and gives option for different classification architectures (for example label_powerset)
    '''
    
#%%

taxonomy = import_raw_data()

descriptions = create_subset(taxonomy, 'project_description')

