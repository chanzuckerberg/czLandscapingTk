# Databricks notebook source
# default_exp generalUtils
from nbdev import *

# COMMAND ----------

import pytz
from datetime import datetime
import re

def dump_data_to_disk(df, stem):
  tz = pytz.timezone('America/Los_Angeles')
  now = datetime.now(tz=tz)
  datestamp = now.strftime("%Y_%m_%d")
  data_path = stem+datestamp+'.tsv'
  df = df.replace(r'\\\\n', ' ', regex=True)
  df.to_csv(data_path, sep='\t', index=False)
  url = re.sub('/dbfs/FileStore/','https://ie-meta-prod-databricks-workspace.cloud.databricks.com/files/',data_path)
  
  name = stem
  name_match = re.search('/([a-zA-Z0-9_]+)_*$', stem)
  if name_match:
    name = name_match.group(1)
  print(name + ': '+url)
  displayHTML('<a href="'+url+'" >LINK</a>' )
