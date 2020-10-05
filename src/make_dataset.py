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

'''import and process taxonomy from excel'''
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
#%%
'''extract relevant data fields''' 
#remove all projects without titles:
taxonomy = taxonomy[taxonomy['title'].notna()]     
print('with title:', taxonomy.shape)

#keep only projects with description:
description = taxonomy[taxonomy['project_description'].notna()]     
description = description[['pims_#', 'project_description']]
print('with description:', description.shape)

#pickle data:
with open(os.path.abspath(os.path.join('..', 'data/interim'))+'/description.pkl', 'wb') as handle:
    pickle.dump(description, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#pickle data also for QA project:
with open(os.path.join('/Users/jonas/Google Drive/jupyter/ClosedDomainQA/data')+'/description.pkl', 'wb') as handle:
    pickle.dump(description, handle, protocol=pickle.HIGHEST_PROTOCOL)