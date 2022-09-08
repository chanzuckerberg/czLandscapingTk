# Databricks notebook source
# MAGIC %pip install git+https://github.com/GullyBurns/czLandscapingTk.git

# COMMAND ----------

# MAGIC %md # Databricks script for building basic database from Airtable spreadsheet
# MAGIC 
# MAGIC You can run this script from any Databricks notebook with the following command (fill in parameters as needed):
# MAGIC 
# MAGIC ```
# MAGIC %run /path/to/this/repo/czLandscapingTk/databricks/scripts/DRSM/00_build_dashdb_from_airtable_queries 
# MAGIC   $airtable_api_key="apikey" 
# MAGIC   $airtable_file="filecode" 
# MAGIC   $airtable_table="tablecode" 
# MAGIC   $airtable_id_column="ID" 
# MAGIC   $airtable_query_column="QUERY" 
# MAGIC   $prefix="RARE_DISEASE_DB1_"
# MAGIC   $secret_scope="my-scope"
# MAGIC   $warehouse="DEV_WAREHOUSE"
# MAGIC   $database="DEV_DB"
# MAGIC   $schema="SKE"
# MAGIC   $role="ARST_TEAM"
# MAGIC   $loc="/dbfs/FileStore/user/gully/'"
# MAGIC   $pubmed_api_key="MY-PUBMED-API-KEY"
# MAGIC   $delete_database="True"
# MAGIC   $pubmed_include="True"
# MAGIC   $solr_include="True"  
# MAGIC   $empc_include="True"
# MAGIC ```

# COMMAND ----------

from czLandscapingTk.airtableUtils import AirtableUtils
from czLandscapingTk.dashdbUtils import DashboardDb
from czLandscapingTk.generalUtils import dump_data_to_disk
import re

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text('airtable_api_key','')
dbutils.widgets.text('airtable_file','')
dbutils.widgets.text('airtable_table','')
dbutils.widgets.text('airtable_id_column','')
dbutils.widgets.text('airtable_query_column','')
dbutils.widgets.text('airtable_subsets_table','')
dbutils.widgets.text('airtable_subsets_column','')
dbutils.widgets.text('prefix','')
dbutils.widgets.text('secret_scope','')
dbutils.widgets.text('warehouse','')
dbutils.widgets.text('database','')
dbutils.widgets.text('schema','')
dbutils.widgets.text('role','')
dbutils.widgets.text('loc','')
dbutils.widgets.text('pubmed_api_key','')

dbutils.widgets.dropdown('delete_database', "True", ['True','False'])
dbutils.widgets.dropdown('pm_include', "True", ['True','False'])
dbutils.widgets.dropdown('sf_include', "False", ['True','False'])
dbutils.widgets.dropdown('epmc_include', "False", ['True','False'])

airtable_api_key = dbutils.widgets.get('airtable_api_key')
airtable_file = dbutils.widgets.get('airtable_file')
airtable_table = dbutils.widgets.get('airtable_table')
airtable_id_column = dbutils.widgets.get('airtable_id_column')
airtable_query_column = dbutils.widgets.get('airtable_query_column')
airtable_subsets_table = dbutils.widgets.get('airtable_subsets_table')
airtable_subsets_column = dbutils.widgets.get('airtable_subsets_column')
delete_database = dbutils.widgets.get('delete_database')

prefix = dbutils.widgets.get('prefix')

secret_scope = dbutils.widgets.get('secret_scope')
warehouse = dbutils.widgets.get('warehouse')
database = dbutils.widgets.get('database')
schema = dbutils.widgets.get('schema')
role = dbutils.widgets.get('role')
loc = dbutils.widgets.get('loc')
pubmed_api_key = dbutils.widgets.get('pubmed_api_key')

pm_include = dbutils.widgets.get('pm_include') == 'True'
epmc_include = dbutils.widgets.get('epmc_include') == 'True'
sf_include = dbutils.widgets.get('sf_include') == 'True' 

delete_database = dbutils.widgets.get('delete_database') == 'True'

# COMMAND ----------

atu = AirtableUtils(airtable_api_key)
queries_df = atu.read_airtable(airtable_file, airtable_table)

# COMMAND ----------

user = dbutils.secrets.get(scope=secret_scope, key="SNOWFLAKE_SERVICE_USERNAME")
pem = dbutils.secrets.get(scope=secret_scope, key="SNOWFLAKE_SERVICE_PRIVATE_KEY")
pwd = dbutils.secrets.get(scope=secret_scope, key="SNOWFLAKE_SERVICE_PASSPHRASE")

dashdb = DashboardDb(prefix, user, pem, pwd, warehouse, database, schema, role, loc)
dashdb.build_database_from_queries(pubmed_api_key, queries_df, airtable_id_column, airtable_query_column, sf_include=False, pm_include=True, epmc_include=False)
