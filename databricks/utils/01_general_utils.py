# Databricks notebook source
# MAGIC %md # General Utilities  
# MAGIC
# MAGIC > Utility classes for the Landscaping Platform

# COMMAND ----------

#| default_exp generalUtils

# COMMAND ----------

#| hide
from nbdev import *

# COMMAND ----------

#| export

from enum import Enum  

# COMMAND ----------

#| export

class ID_Type(Enum):
    """
    Types of universal IDs / Accession numbers used in this library
    """
    
    pmid = 0
    doi = 1
    s2ag = 2
