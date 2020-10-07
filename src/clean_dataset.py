#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:56:48 2020

@author: jonas

@title: clean_dataset

@descriptions: set of functions that enable different level of data cleaning.
"""
#%%

import pandas as pd
#%%

def clean(dataframe, column_name, lowercase = True, punctuation = True, special_char = True,
          stopwords = False, cotum_stopwords = False, stemming = False, 
          lemmatizing = False, tokenizing = False):
    """
    Allows to input both pre-set english stopwords and costum stopword_list
    
    Probably employs spacy and nltk, has to allow for different levels.
    
    Outputs original data_frame with new_cleaned_coslum.
    """
    
def segment(dataframe, column_name):
    
    '''   
    optional: functions for segmenting:
        
        - costum sentence segmenting (spacy?) and re-merging with costum length (for BERT etc)
        - paragraph splitting (at \n etc)
    
    '''