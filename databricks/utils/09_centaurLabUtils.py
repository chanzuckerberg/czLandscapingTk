# Databricks notebook source
# default_exp centaurLabsUtils
from nbdev import *

# COMMAND ----------

# MAGIC %md # CentaurLab Utility Tools 
# MAGIC 
# MAGIC > Tools to upload, download, and analyze data from CentaurLabs using their curation interface. 

# COMMAND ----------

#export

import json
import requests
import pandas as pd
from tqdm import tqdm
import os
import urllib.request 
import spacy

class CentaurLabsUtils: 
  '''
  Tools to provide capabilities to interface dashboard databases with CentaurLabs' curation 
  systems Documentation: https://docs.centaurlabs.com/
  
  Note - this requires that an appropriate scispacy language model be loaded. 
  
  %pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_lg-0.4.0.tar.gz 
  
  Attributes:
    * df: The dataframe being processed - expected columns are (A) ID, (B) title, (C) abstract, (D) year, (E) disease, (F) label. These can be specified with other class attributes
    
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
    
  def generate_html_column(self, maxpos, html_col='HTML'):
    final_list = []
    for i, row in tqdm(self.df.iterrows()):
      disease = str(row[self.dis_col])
      id = str(row[self.id_col])
      title = str(row[self.title_col])
      abstract = str(row[self.abs_col])
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
