#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:06:05 2020

@author: jonas

@title: visualisations
"""
import holoviews as hv
from holoviews import opts, dim
import holoviews.plotting.bokeh
import numpy as np
import panel as pn
import pandas as pd


from wordcloud import WordCloud, STOPWORDS
from wordcloud import WordCloud
import wordcloud


#%%

def chord_chard(data):
    
    """
    
    Takes in processed dataframe for multilabel classification problem and computes label co-occurences.
    
    Draws chord chard using bokeh and local server.
    
    """
    hv.extension('bokeh')
    
    hv.output(size=200)
    
    labels_only =  data.drop(labels = ['PIMS_ID', 'language', 'description', 'all_logs', 'text'], axis=1)
    
    
    cooccurrence_matrix = np.dot(labels_only.transpose(),labels_only)
    
    cooccurrence_matrix_diagonal = np.diagonal(cooccurrence_matrix)
    with np.errstate(divide='ignore', invalid='ignore'):
        cooccurrence_matrix_percentage = np.nan_to_num(np.true_divide(cooccurrence_matrix, cooccurrence_matrix_diagonal[:, None]))
    
    
    coocc = labels_only.T.dot(labels_only)
    diagonal = np.diagonal(coocc)
    co_per = np.nan_to_num(np.true_divide(coocc, diagonal[:, None]))
    df_co_per = pd.DataFrame(co_per)
    df_co_per = pd.DataFrame(data=co_per, columns=coocc.columns, index=coocc.index)
    
    #replace diagonal with 0:
    coocc.values[[np.arange(coocc.shape[0])]*2] = 0
    coocc = coocc.mask(np.triu(np.ones(coocc.shape, dtype=np.bool_)))
    coocc = coocc.fillna(0)

    data = hv.Dataset((list(coocc.columns), list(coocc.index), coocc),
                      ['source', 'target'], 'value').dframe()
    data['value'] = data['value'].astype(int)
    chord = hv.Chord(data)

    plot = chord.opts(
        node_color='index', edge_color='source', label_index='index', 
        cmap='Category20', edge_cmap='Category20', width=400, height=400)
    

    bokeh_server = pn.Row(plot).show(port=1234)

def draw_cloud(dataframe, column):

    
    # Join the different processed titles together.
    long_string = ','.join(list(dataframe[column]))
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=6, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    return wordcloud.to_image()