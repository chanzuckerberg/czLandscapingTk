# Databricks notebook source
# default_exp generalUtils
from nbdev import *

# COMMAND ----------

# MAGIC %md # Databricks Utilities 
# MAGIC > Tools for use within databricks notebooks. 

# COMMAND ----------

#export

import pytz
from datetime import datetime
import re

def dump_data_to_disk(df, file_stem, databricks_url='https://ie-meta-prod-databricks-workspace.cloud.databricks.com'):
  ''' 
  Save a Pandas's Dataframe to disk and display a HTTP link to download it
  
  `file_stem`: the location to save the file in the databricks file system. String must start with '/dbfs/FileStore/'
  `databricks_url`: is the URL for the Databricks environment being used. Defaults to the CZI workspace. 
  '''
  tz = pytz.timezone('America/Los_Angeles')
  now = datetime.now(tz=tz)
  datestamp = now.strftime("%Y_%m_%d")
  data_path = file_stem+datestamp+'.tsv'
  df = df.replace(r'\\\\n', ' ', regex=True)
  df.to_csv(data_path, sep='\t', index=False)
  databricks_stem = databricks_url + '/files/'
  url = re.sub('/dbfs/FileStore/', databricks_stem, data_path)
  name = stem
  name_match = re.search('/([a-zA-Z0-9_]+)_*$', stem)
  if name_match:
    name = name_match.group(1)
  print(name + ': '+url)
  displayHTML('<a href="'+url+'" >LINK</a>' )

# COMMAND ----------

show_doc(dump_data_to_disk)
