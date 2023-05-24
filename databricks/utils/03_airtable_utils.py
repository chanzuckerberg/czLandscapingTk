# Databricks notebook source
#|default_exp airtableUtils
from nbdev import *

# COMMAND ----------

# MAGIC %md # Airtable Utilities
# MAGIC
# MAGIC > Simple library to provide lightweight input/output functions for Airtable. Airtable is an excellent vehicle for interacting with users.
# MAGIC
# MAGIC Note - this approach requires manual construction of Airtable notebooks to match the existing format of notebooks so some overhead is needed to check formatting. 

# COMMAND ----------

#|export
import pandas as pd
import json
from urllib.parse import quote
from tqdm import tqdm
import requests
import nltk
from nltk.metrics import agreement
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import masi_distance, binary_distance

# COMMAND ----------

#|export

class AirtableUtils:
  """This class permits simple input / output from airtable
  
  Attributes:
    * api_key: an API key obtained from Airtable to provide authentication
  """
  
  def __init__(self, api_key):
    """ Initialize the interface with an API key. 
    """
    self.api_key = api_key
    self.task = None
    
  def _get_airtable_url(self, file, table):
    return 'https://api.airtable.com/v0/%s/%s?api_key=%s'%(file, table, self.api_key) 

  def read_airtable(self, file, table):
    """ Read an airtable into a Pandas Dataframe. 
    """
    url = self._get_airtable_url(file, table)
    x = requests.get(url)
    js = json.loads(x.text)
    if( js.get('records') is None ):
      raise Exception("Airtable "+url+" not found." )

    df = pd.DataFrame([r.get('fields') for r in js.get('records')])
    df = df.fillna('')

    if 'ID' not in df.columns:
      df.reset_index(inplace=True)
      df = df.rename(columns = {'index':'ID'})

    df = df.sort_values(by=['ID'])
    
    return df

  def read_airtable(self, file, table):
    data_rows = []
    headers = {'Authorization': 'Bearer '+self.api_key, 'Content-Type': 'application/json'}
  #  base_url = start_url + '&maxRecords=100&fields%5B%5D=ID&fields%5B%5D=Title&fields%5B%5D=Abstract' + \
  #    '&fields%5B%5D=Comments&fields%5B%5D=Disease%20Research%20Categories&fields%5B%5D=Irrelevant?' + \
  #    '&fields%5B%5D=TimeofLastCurationAction'
    base_url = self._get_airtable_url(file, table)
    offset = 'GO'
    while offset!='STOP':
      print('.', end = '')
      #print(offset)
      if offset != 'GO':
        url = base_url + '&offset='+offset
      else:
        url = base_url
      #print(url)
      r = requests.get(url, headers=headers)      
      rdata = json.loads(r.text)
      #print(len(rdata.get('records',[])))
      #print(rdata)
      for r in rdata.get('records',[]):
        data_rows.append(r['fields'])
      if( rdata.get('offset') is not None ):
        offset = rdata['offset']
      else: 
        offset = 'STOP'
    df = pd.DataFrame(data_rows)
    df = df.replace('"', '')
    #df = df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=[" "," "], regex=True, inplace=True)
    print('|')
    return df
    
  
  def build_curated_dataframe(self, files, tables):
    curated_df = pd.DataFrame()
    for f in files:
      for t in tables:
        try:
          df = self.read_airtable(f, t)
          df['at_f'] = f
          df['at_t'] = t
          print('%d rows added'%(len(df)))
        except Exception as e:
          print(e)
        curated_df = curated_df.append(df)
    curated_df = curated_df.reset_index(drop=True)
    return curated_df
  
  def send_df_to_airtable(self, file, table, df):
    """ Send a dataframe to an airtable table.
    
    _Note: the dataframe's columns must match the structure of the table exactly_ 
    """
    # note need to check size of payload - 10 JSON records only with 'fields' hash entry
    headers = {'Authorization': 'Bearer '+self.api_key, 'Content-Type': 'application/json'}
    records = []
    for i, row in df.iterrows():
      if i % 10 == 0 and i > 0:
        records = []
        r = requests.post(url, headers=headers, data=payload)      
      records.append(fields_json = json.dumps(mnrow.to_dict()))
    airtable_data = r.content.decode('utf-8')

  def send_records_to_airtable(self, file, table, records):
    url = self._get_airtable_url(file, table)

    # note need to check size of payload - 20 JSON records only with 'fields' hash entry
    headers = {'Authorization': 'Bearer '+self.api_key, 'Content-Type': 'application/json'}
    rec_set = []
    for i, row in tqdm(enumerate(records)):
      if i % 10 == 0 and i > 0:
        payload = json.dumps({'records':rec_set}) 
        r = requests.post(url, headers=headers, data=payload)  
        print(r.text)
        rec_set = []
      rec_set.append({'fields':row})   
    if len(records)>0:
      payload = json.dumps({'records':rec_set}) 
      r = requests.post(url, headers=headers, data=payload)   
      print(r.text)
      

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
  def build_nltk_annotation_task_from_curated_df(self, df, 
                                                 doc_id_column, 
                                                 category_column, 
                                                 curator_column,
                                                 distance_function=masi_distance):
    document_task_data = []
    curators = {}
    docs = {}
    categories = {}

    for i, row in df.iterrows():
      category_array = str(row[category_column]).split(',')
      item = str(row[doc_id_column])
      td_row = (row[curator_column], item, frozenset(category_array))
      document_task_data.append(td_row)
      if docs.get(item) is None:
        docs[item] = 1
      else:
        docs[item] = docs[item] + 1

      if curators.get(row[curator_column]) is None:
        curators[row[curator_column]] = 1
      else:
        curators[row[curator_column]] = curators[row[curator_column]] + 1
      for c in str(row[category_column]).split(','):
        if categories.get(c) is None:
          categories[c] = 1
        else:
          categories[c] = categories[c] + 1

    doc_task = AnnotationTask(distance = distance_function)
    doc_task.load_array(document_task_data)
    self.docs = docs
    self.curators
    return docs, curators, categories, doc_task
  
  def _get_avg_doc_agr(item, curators, task):
    temp_list = [] 
    sum = 0.0
    cnt = 0.0
    for i in range(len(curators)):
      for j in range(i):
        try:
          sum += task.agr(curators[i], curators[j], str(item))
          cnt += 1.0
        except StopIteration:
          # No need to do anything - we get this error if attempting to compute agreement 
          # between curators where one of them never entered a score. 
          print('', end = '')
    if cnt > 0.0:
      avg = sum/cnt
    else: 
      avg = 0.0;
    return avg

  def _get_consensus(self, item, curators, task):
    """ 
    """
    result = []
    best = 0.0 
    for i in range(len(curators)):
      for j in range(i):
        try:
          agr = task.agr(curators[i], curators[j], str(item))
          if agr == 1.0:
            l = [x for x in doc_task.data if x['coder']==curators[i] and x['item']==item]
            return list(l[0]['labels'])[0]
        except StopIteration:
          # No need to do anything - we get this error if attempting to compute agreement 
          # between curators where one of them never entered a score. 
          print('', end = '')
    return None

  def get_consensus_per_doc(df, task):

    cat_list = sorted(list({c:0 for cc in df.CATEGORIES for c in cc.split(',')}.keys()))
    curators = df.CURATOR.unique()

    item_curator_dict = {cc: {c: '' for c in curators} for cc in df.ID_PAPER}
    for row in df.itertuples():
      item_curator_dict[row.ID_PAPER][row.CURATOR] = row.CATEGORIES

    sdf = df.drop(['DISEASE_NAME', 'URI', 'TIMESTAMP','CATEGORIES','IRRELEVANT', \
                   'COMMENTS','CURATOR', 'WORKSHEET', \
                   'TITLE','ABSTRACT'], axis=1).drop_duplicates()
    sdf = sdf.reset_index(drop=True)

    #cat_count_dict = {cc: {c: 0 for c in cat_list} for cc in df.ID_PAPER}
    #for row in df.itertuples():
    #  for t in row.CATEGORIES.split(','):
    #    cat_count_dict[row.ID_PAPER][t] = cat_count_dict.get(row.ID_PAPER).get(t) + 1
    #cat_counts = [[cat_count_dict[row.ID_PAPER][c] for c in cat_list ] for row in sdf.itertuples()]
    #sdf['CATEGORY_COUNTS'] = cat_counts

    sdf['AVG_AGREEMENT'] = [get_avg_doc_agr(str(row.ID_PAPER), curators, task) for row in sdf.itertuples()]
    sdf['CONSENSUS'] = [get_consensus(str(row.ID_PAPER), curators, task) for row in sdf.itertuples()]

    return sdf

