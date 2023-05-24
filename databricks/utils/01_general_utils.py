# Databricks notebook source
# default_exp generalUtils
from nbdev import *

# COMMAND ----------

# MAGIC %md # General Utilities  
# MAGIC
# MAGIC > Utility classes for the Landscaping Platform

# COMMAND ----------

#export

from enum import Enum  

# COMMAND ----------

#export

class ID_Type(Enum):
    """
    Types of universal IDs / Accession numbers used in this library
    """
    pmid = 0
    doi = 1
    s2ag = 2
