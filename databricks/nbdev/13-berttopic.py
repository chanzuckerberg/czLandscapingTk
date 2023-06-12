# Databricks notebook source
# MAGIC %md # BERT Topic Utilities
# MAGIC
# MAGIC > A simple API to the BertTopic library (https://maartengr.github.io/BERTopic/index.html). 

# COMMAND ----------

#| default_exp berttopic

# COMMAND ----------

#| export
import pandas as pd
import re
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix, f1_score, accuracy_score
from sklearn.decomposition import PCA

from tqdm import tqdm, trange
from enum import Enum
import re
import os
import pytz
from datetime import datetime
import pickle
import hdbscan
import umap
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
import json
import math
import random 
import copy

# COMMAND ----------

#| export

# From: https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
# NOTE - THIS DOES NOT CURRENTLY WORK 
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)
def load_to_cpu_from_file(fpath):
  obj = CPU_Unpickler(fpath).load()
  return obj

def save_to_file(obj, fpath):
  if os.path(fpath).exists():
    os.unlink(fpath)
  with open(fpath, 'wb') as f:
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
def load_from_file(fpath):
  with open(fpath, 'rb') as f:
    obj = pickle.load(f, fix_imports=False, encoding="UTF-8")
  return obj

class DiscourseType(Enum):
    BACKGROUND = 0
    OBJECTIVE = 1
    METHODS = 2
    RESULTS = 2
    CONCLUSIONS = 2
    
class SentenceClusterAnalysis: 
  """
  Analysis functions for a corpus made up of sentences, each with high-dimensional embeddings 
      (expressed in JSON) and assigned discourse types for each sentence.  
  """
  sent_df = None # the pandas dataframe of sentences to be analyzed
  embeddings = []# the high-dimensional embeddings associated with the sentences
  red_vec = {}
  bertopic_model = None
  id_to_order = {}
  order_to_id = {}
    
  #def __init__(self, *args, **kwargs): 

  def load_sent_df(self, sent_df:pd.DataFrame, 
                   id_paper_col='ID_PAPER', 
                   sentence_id_col='SENTENCE_ID', 
                   plain_text_col='text', 
                   json_embeddings_col='json_embeddings',
                   id_col='id'):
    self.sent_df = sent_df.rename(columns={id_paper_col: 'ID_PAPER', 
                                           sentence_id_col: 'SENTENCE_ID', 
                                           plain_text_col: 'text', 
                                           json_embeddings_col: 'json_embeddings',
                                           id_col: 'id'}).sort_values('id')
    self.embeddings = np.array([json.loads(row.json_embeddings) for row in self.sent_df.itertuples()])
    self.red_vec = {}    
    
  def generate_bertopic_model(self, n_dim=5, embedding_model="allenai-specter", 
                              top_n_words=30, min_cluster_size=15, cluster_metric='euclidean',cluster_selection_method='eom'):

    umap_model = umap.UMAP(n_components=n_dim)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=cluster_metric, cluster_selection_method=cluster_selection_method)
    self.bertopic_model = BERTopic(embedding_model=embedding_model, top_n_words=top_n_words, umap_model=umap_model, hdbscan_model=hdbscan_model)
    
    docs = self.sent_df.text.to_list()
    years = self.sent_df.YEAR.to_list()
    topics, probs = self.bertopic_model.fit_transform(docs, self.embeddings)
    
    self.id_to_order = {}
    self.order_to_id = {}
    for i in range(len(topics)):
      self.id_to_order[self.bertopic_model.hdbscan_model.labels_[i]] = topics[i]
      self.order_to_id[topics[i]] = self.bertopic_model.hdbscan_model.labels_[i] 
      
    self.sent_df['cluster_assignments'] = self.bertopic_model.hdbscan_model.labels_
    self.sent_df['cluster_probabilities'] = self.bertopic_model.hdbscan_model.probabilities_    
    
    xy_embed = umap.UMAP(n_components=2).fit_transform(self.embeddings)
    self.sent_df['x'] = [xy[0] for xy in xy_embed] 
    self.sent_df['y'] = [xy[1] for xy in xy_embed] 
        
  def plot_xy(self, red_vec):
    x = [row.x for row in self.sent_df.intertuples()]
    y = [row.y for row in self.sent_df.intertuples()]
    fig, ax = plt.subplots()
    fig.set_figheight(20)
    fig.set_figwidth(20)
    ax.plot(x, y, 'ro',  markersize=0.2)
    display(fig)
    
  def visualize_clusters(self, figsize=(20,20), pointsize=0.01):
    fig, ax = plt.subplots(figsize=figsize)
    outliers = self.sent_df.loc[self.sent_df.cluster_assignments == -1, :]
    clustered = self.sent_df.loc[self.sent_df.cluster_assignments != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=pointsize)
    plt.scatter(clustered.x, clustered.y, c=clustered.cluster_assignments, s=pointsize, cmap='hsv_r')
    cent_x = self.sent_df.groupby(['cluster_assignments']).x.mean().reset_index()
    cent_y = self.sent_df.groupby(['cluster_assignments']).y.mean().reset_index()
    centroids = cent_x.merge(cent_y)
    for c in centroids.itertuples():
      plt.annotate(c.cluster_assignments if c.cluster_assignments!=-1 else '', # this is the text
                 (c.x,c.y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-3), # distance from text to points (x,y)
                 ha='center') # horizonal alignment
    plt.colorbar()
    
  def generate_berttopic_labels(self):
    
    def invert_hex(hex_number):
      inverse = hex(abs(int(hex_number, 16) - 255))[2:]
      # If the number is a single digit add a preceding zero
      if len(inverse) == 1:
          inverse = '0' + inverse
      return inverse

    def float_to_greyscale(f):
      val = '%x' % int(f * 255)
      val = invert_hex(val)
      return '#%s%s%s' % (val, val, val)

    bertopic_model = self.bertopic_model
    ts = bertopic_model.get_topics()
    self.html_labels = {}
    self.labels = {}
    self.label_data = {}
    for t in ts:
      if t==-1:
        
        continue
      txt = '%d: '%(self.order_to_id[t])
      max_weight = ts[t][0][1]
      label_data = {'id':int(self.order_to_id[t]), 'max':float(max_weight)}
      for i, tup in enumerate(ts[t][:5]):
        word = tup[0]
        weight = float(tup[1])/max_weight
        txt += '<span style="color:%s">%s</span> ' % (float_to_greyscale(weight), word.replace(' ', '&nbsp;'))
        label_data[i] = {'word':str(word), 'weight':float(weight)} 
      self.html_labels[self.order_to_id[t]] = txt
      self.labels[self.order_to_id[t]] = '%d: '%(self.order_to_id[t])+'_'.join([tup[0] for tup in ts[t][:5]])
      self.label_data[self.order_to_id[t]] = label_data
      
  def get_cluster_time_series_data(self):
    spy_df = pd.pivot_table( self.sent_df.loc[self.sent_df.cluster_assignments>-1], values='id', 
                          index=['cluster_assignments'], columns=['YEAR'], margins=True, 
                          aggfunc='nunique', fill_value=0).sort_values('All',ascending=False)
    l = []
    l2 = []
    for r in spy_df.index.values:
      #if r=="All":
      #  continue 
      m2 = {c:(spy_df.at[r,c]/spy_df.at['All',c]) for c in spy_df.columns if c!='All'}
      m = {c:(spy_df.at[r,c]) for c in spy_df.columns if c!='All'}
      m['id'] = r
      m2['id'] = r
      l.append(m)
      l2.append(m2)
    topics_time_df = pd.DataFrame(l).set_index('id')
    topics_time_df2 = pd.DataFrame(l2).set_index('id')
    return topics_time_df, topics_time_df2
  
  def plot_cluster_time_series(self, n=-1, width=5):
    if n==-1:
      n=len(sent_claims_df)-1
    (topics_time_df, topics_time_df2) = self.get_cluster_time_series_data()
    topics_time_df = topics_time_df[:n].transpose()
    topics_time_df2 = topics_time_df2[:n].transpose()
    hgt = math.ceil(n*(2.0/width))
    nrows = math.ceil(n/width)
    ax = topics_time_df2.plot(subplots=True, layout=(nrows,width), figsize=(5*width,hgt), title=[self.labels[i][:42] for i in topics_time_df.columns[:n]] ) 
    i = 0
    for arow in ax.tolist():
      for a in arow:
        a2 = a.twinx()
        col_name = topics_time_df.columns[i]
        a2.set_ylim([0, max(topics_time_df[col_name])*1.1])
        a2.plot(topics_time_df[col_name], linestyle=':')
        i += 1
    return topics_time_df, topics_time_df2

  def strip_models(self):
    new_self = copy.copy(self)
    new_self.bertopic_model = None
    new_self.hdbscan_model = None
    return new_self
