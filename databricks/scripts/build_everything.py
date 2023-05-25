# Databricks notebook source
# MAGIC %pip install nbdev

# COMMAND ----------

# MAGIC %sh rm ../../czLandscapingTk/*

# COMMAND ----------

# MAGIC %sh python ../../db2nb/convert_databricks_to_jupyter.py

# COMMAND ----------

# MAGIC %sh ls -l /Workspace/Repos/gully.burns@chanzuckerberg.com/czLandscapingTk/databricks/utils/01_general_utils

# COMMAND ----------

# MAGIC %sh ls -lh /Workspace/Repos/gully.burns@chanzuckerberg.com/czLandscapingTk/nbdev/utils

# COMMAND ----------

# MAGIC %sh ls ../../czLandscapingTk

# COMMAND ----------

# MAGIC %sh nbdev_prepare
