# Databricks notebook source
# default_exp airtableUtils
from nbdev import *

# COMMAND ----------

# MAGIC %md # Airtable Utilities
# MAGIC 
# MAGIC > Simple library to provide lightweight input/output functions for Airtable. Airtable is an excellent vehicle for interacting with users.
# MAGIC 
# MAGIC Note - this approach requires manual construction of Airtable notebooks to match the existing format of notebooks so some overhead is needed to check formatting. 

# COMMAND ----------

#export
import pandas as pd
import json
from urllib.parse import quote
from tqdm import tqdm
import requests

class AirtableUtils:
  """This class permits simple input / output from airtable
  
  Attributes:
    * api_key: an API key obtained from Airtable to provide authentication
  """
  
  def __init__(self, api_key):
    """ Initialize the interface with an API key. 
    """
    self.api_key = api_key
    
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
    headers = {'Authorization': 'Bearer '+self.api.key, 'Content-Type': 'application/json'}
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
      

# COMMAND ----------

show_doc(AirtableUtils.__init__)

# COMMAND ----------

show_doc(AirtableUtils.read_airtable)

# COMMAND ----------

show_doc(AirtableUtils.send_df_to_airtable)

# COMMAND ----------

show_doc(AirtableUtils.send_records_to_airtable)
