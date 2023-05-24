# Databricks notebook source
#|default_exp centaurLabsUtils
from nbdev import *

# COMMAND ----------

# MAGIC %md # CentaurLab Utility Tools 
# MAGIC
# MAGIC > Tools to upload, download, and analyze data from CentaurLabs using their curation interface. 

# COMMAND ----------

#|export

import json
import requests
import pandas as pd
from tqdm import tqdm
import os
import urllib.request 
import spacy
from unidecode import unidecode

# COMMAND ----------

#|export

class CentaurLabsUploadUtils: 
  '''
  Tools to provide capabilities to interface dashboard databases with CentaurLabs' curation 
  systems 
  
  Documentation: https://docs.centaurlabs.com/
  
  Note - this requires that an appropriate scispacy language model be loaded. 
  
  %pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_lg-0.4.0.tar.gz 
  
  Attributes:
    * df: The dataframe being processed - expected columns are (A) ID, (B) title, (C) abstract, (D) year, (E) disease, (F) label. 
    These can be specified with other class attributes
    
  '''
  def __init__(self, df, id_col='ID_PAPER', title_col='Title', abs_col='Abstract', 
               year_col='YEAR', dis_col='DISEASE', label_col='LABEL'):
    self.df = df
    self.id_col = id_col
    self.title_col = title_col
    self.abs_col = abs_col
    self.year_col = year_col
    self.dis_col = dis_col
    self.label_col = label_col
    self.nlp = spacy.load("en_core_sci_lg")
    
  def fix(self, text):
    text = unidecode(str(text))
    text = re.sub('\<', '&lt;', text)
    text = re.sub('\>', '&gt;', text)
    return text
  
  def generate_html_column(self, maxpos, html_col='HTML'):
    final_list = []
    for i, row in tqdm(self.df.iterrows()):
      disease = str(row[self.dis_col])
      id = str(row[self.id_col])
      title = self.fix(row[self.title_col])
      abstract = self.fix(row[self.abs_col])
      text = title + '. ' + abstract 
      doc = self.nlp(text)
      if len(doc)>maxpos:
        title_doc = self.nlp(title)
        title_len = len(title_doc)
        pos = doc[len(doc)+len(title_doc)-maxpos].idx
        abstract = ' ... ' + text[pos:]
      html = '<html>'+\
          '<div style="font-size: 20px; color: white; font-family: sans-serif; font-weight: lighter; line-height: 130%;">'+\
          '<p><b>DISEASE: </b>'+disease+'</p>'+\
          '<p><b>ID: </b>'+id+'</p>'+\
          '<p><b>TITLE: </b>'+title+'</p>'+\
          '<p><b>ABSTRACT: </b>'+abstract+'</p>'+\
          '</div></html>'
      final_list.append(html.replace('\n',' '))
    self.df[html_col] = final_list
    
class CentaurLabsDownmUtils: 
  '''
  Tools to process and evaluate data downloaded from CentaurLabs' curation systems
  
  Attributes:
    * centaur_df: The dataframe downloaded from Centaurlabs. These have a standard format. 
    * text_df: The dataframe uploaded to Centaurlabs (formatted to hmtl) 
    
  '''
  def __init__(self, centaur_df, html_df):
    centaur_df = centaur_df.drop(columns=[c for c in df.columns if c not in ['Case ID', 'Origin', 'URL', 'Labeling State', 'Qualified Reads', 
                                                            'Correct Label', 'Agreement', 'Title', 'Abstract',
                                                            'First Choice Answer', 'First Choice Weight',
                                                            'Second Choice Answer', 'Second Choice Weight',
                                                            'Third Choice Answer', 'Third Choice Weight',
                                                            'Fourth Choice Answer', 'Fourth Choice Weight']])
    centaur_df = centaur_df.rename(columns = {c:re.sub(' ','_',c) for c in centaur_df.columns})
    self.centaur_df = centaur_df[(centaur_df['Agreement'].isnull()==False)]
    self.html_df = html_df
    
  def map_html_to_df(self, df):     
    p1 = '<b>TITLE: </b>(.*?)</p>'
    p2 = '<b>ABSTRACT: </b>(.*?)</p>'
    p3 = '<b>ID: </b>(.*?)</p>'
    titles = [re.search(p1, row.html).group(1) for row in df.itertuples()]
    abstracts = [re.search(p2, row.html).group(1) for row in df.itertuples()]
    ids = [int(re.search(p3, row.html).group(1)) for row in df.itertuples()]
    df2 = pd.DataFrame([(i,j,k) for i,j,k in zip(ids, titles, abstracts)], columns=['ID','Title','Abstract'])
    return df2
  
  def compute_thresholded_f1_scores(self):
    gold_standard = self.centaur_df[(self.centaur_df['Labeling_State']=='Gold Standard') & (self.centaur_df['Agreement']>=0)].sort_values('Agreement')
    categories_to_include = sorted(self.centaur_df['Correct_Label'].unique())
    centaur_categories = ['-1', '0', '1', '2']
    true_cats = [re.search("(\d) - .*", x).group(1) for x in gold_standard['Correct_Label'].to_list()]
    pred_cats = []
    vector_list = []
    agreement = gold_standard['Agreement'].to_list()
    cols = gold_standard.columns.to_list()
    vector_list = []
    for row in gold_standard.itertuples():
      m = {}
      m[row.First_Choice_Answer] = round(row.First_Choice_Weight, 2)
      m[row.Second_Choice_Answer] = round(row.Second_Choice_Weight, 2)
      m[row.Third_Choice_Answer] = round(row.Third_Choice_Weight, 2)
      m[row.Fourth_Choice_Answer] = round(row.Fourth_Choice_Weight, 2)
      vector = [m[c] for c in categories_to_include]
      vector_list.append(vector)
      pred_cats.append(centaur_categories[vector.index(max(vector))])
    gold_standard['vector'] = vector_list 
    gold_standard['predicted'] = pred_cats 
    gold_standard['Correct_Label'] = true_cats
    gold_standard = gold_standard.drop(columns=['First_Choice_Answer', 'First_Choice_Weight', 
                                                'Second_Choice_Answer', 'Second_Choice_Weight', 
                                                'Third_Choice_Answer', 'Third_Choice_Weight', 
                                                'Fourth_Choice_Answer', 'Fourth_Choice_Weight']).fillna('')
    f1 = []
    for thresh in np.linspace(0,1,100):
      i = 0
      for j, a in enumerate(agreement):
        if a>thresh:
          i = j
          break
      t = true_cats[i:]
      p = pred_cats[i:]
      f1.append((thresh, len(t)/len(gold_standard), f1_score(t, p, average='micro'), f1_score(t, p, average='macro')))
    f1.pop()
    f1_df = pd.DataFrame(f1, columns=['agreement', 'Fraction of Data', 'F1_micro', 'F1_macro'])
    f1_df.drop(columns=['Fraction of Data']).plot(x='agreement')
    f1_df.drop(columns=['F1_micro','F1_macro']).plot(x='agreement')
    return f1_df
