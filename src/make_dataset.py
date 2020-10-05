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
#sys.path.insert(0, os.path.abspath(os.path.join('..', 'src')))

import pandas as pd 

#%%

'''import and process taxonomy from excel'''
taxonomy = pd.read_excel (os.path.abspath(os.path.join('../..', 'Data/raw'))+'/tagging_table.xlsx')
# get column names:
columnNames = taxonomy.iloc[0] 
taxonomy = taxonomy[1:] 
taxonomy.columns = columnNames
# delete entries without PIMS ID:
taxonomy = taxonomy[taxonomy['PIMS #'].notna()]                      
# delete columns without names:
taxonomy = taxonomy.loc[:, taxonomy.columns.notnull()]   
# remove white spaces in column names and lowercase names:
taxonomy.columns = taxonomy.columns.str.replace(' ', '_').str.lower()
#%%
'''extract relevant fields for training data:'''



