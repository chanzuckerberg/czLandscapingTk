# Databricks notebook source
# MAGIC %conda install -c conda-forge graph-tool

# COMMAND ----------

# MAGIC %pip install kgtk

# COMMAND ----------

# MAGIC %sh kgtk -h

# COMMAND ----------

import os
import os.path

from kgtk.configure_kgtk_notebooks import ConfigureKGTK

# COMMAND ----------

# Parameters

# Folders on local machine where to create the output and temporary files:
input_path = '/dbfs/FileStore/user/gully/mondo'
output_path = "/dbfs/FileStore/user/gully/kgtk"
project_name = "mondo-kypher"

# COMMAND ----------

# MAGIC %sh mv /dbfs/FileStore/user/gully/mondo /dbfs/FileStore/user/gully/mondo_input

# COMMAND ----------

# MAGIC %sh mkdir /dbfs/FileStore/user/gully/mondo /dbfs/FileStore/user/gully/mondo_output

# COMMAND ----------

big_files = [
    "all",
    "label",
    "pagerank_undirected",
]

ck = ConfigureKGTK(big_files, kgtk_path='/dbfs/FileStore/user/gully/kgtk')
ck.configure_kgtk(input_graph_path='/dbfs/FileStore/user/gully/mondo_input', 
                  output_path='/dbfs/FileStore/user/gully/mondo_output', 
                  project_name="mondo-kgtk")

# COMMAND ----------

ck.print_env_variables()

# COMMAND ----------

# MAGIC %sh ls /dbfs/FileStore/user/gully/mondo_input

# COMMAND ----------

# MAGIC %sh kgtk import-ntriples -i /dbfs/FileStore/user/gully/mondo_input/mondo.nt -o /dbfs/FileStore/user/gully/mondo_output/mondo_kg.tsv 

# COMMAND ----------

# MAGIC %sh head /dbfs/FileStore/user/gully/mondo_output/mondo_kg.tsv 
