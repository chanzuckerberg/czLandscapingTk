# Databricks notebook source
# default_exp dashdbUtils
from nbdev import *

# COMMAND ----------

# MAGIC %md # Dashboard Database Utilities
# MAGIC 
# MAGIC > Simple query classes that allows contruction of a SQL database in Snowflake for science literature dashboard applications. Note that this implementation is intended primarily for internal CZI use.

# COMMAND ----------

#export

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import dsa
from cryptography.hazmat.primitives import serialization
import io
import os
import snowflake.connector
import pandas as pd
from enum import Enum
import re

class Snowflake():
   '''
    Class to provide simple access to Snowflake from within CZI
    
    Attributes (note - store `user`, `pem`, and `pwd` as `dbutils.secret` data ):
    * user: Snowflake username 
    * pem: SSH key 
    * pwd: Password for SSH key 
    * warehouse: name of the SNOWFLAKE warehouse
    * database: name of the SNOWFLAKE database
    * schema: name of the SNOWFLAKE schema
    * role: name of the SNOWFLAKE role with correct permissions to execute database editing
    '''

  def __init__(self, user, pem, pwd, warehouse, database, schema, role):
    self.user = user
    self.pem = pem
    self.pwd = pwd
    self.warehouse = warehouse
    self.database = database
    self.schema = schema
    self.role = role

    #string_private_key = f"-----BEGIN ENCRYPTED PRIVATE KEY-----\n{pem.strip()}\n-----END ENCRYPTED PRIVATE KEY-----"
    string_private_key = f"{pem.strip()}"

    p_key = serialization.load_pem_private_key(
        io.BytesIO(string_private_key.encode()).read(),
        password=pwd.strip().encode(),
        backend=default_backend())

    pkb = p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption())

    self.ctx = snowflake.connector.connect(
        user=user.strip(),
        private_key=pkb,
        account='lr02922')

    cur = self.ctx.cursor()
    cur.execute("select current_date;")
    print(cur.fetchone()[0])

  def cursor(self):
    self.cs = self.ctx.cursor()
    return self.cs

  def get_cursor(self):
    '''
    Gets an active cursor for use within the database 
    '''
    cs = self.cursor()
    cs.execute('USE ROLE '+self.role)
    cs.execute('USE WAREHOUSE ' + self.warehouse)
    print('USE SCHEMA '+self.database+'.'+self.schema)
    cs.execute('USE SCHEMA '+self.database+'.'+self.schema)

    return cs

  def execute_query(self, cs, sql, columns):
    '''
    Executes an SQL query with a list of column names and returns a Pandas DataFrame
    '''    
    cs.execute(sql)
    df = pd.DataFrame(cs.fetchall(), columns=columns)
    df = df.replace('\n', ' ', regex=True)
    return df

  def run_query_in_spark(self, user, query):
    string_private_key = f"{self.pem.strip()}"

    p_key = serialization.load_pem_private_key(
        io.BytesIO(string_private_key.encode()).read(),
        password=self.pwd.strip().encode(),
        backend=default_backend())

    pkb = p_key.private_bytes(
      encoding = serialization.Encoding.PEM,
      format = serialization.PrivateFormat.PKCS8,
      encryption_algorithm = serialization.NoEncryption()
      )

    pkb = pkb.decode("UTF-8")
    pkb = re.sub("-*(BEGIN|END) PRIVATE KEY-*\n","",pkb).replace("\n","")

    # snowflake connection options
    options = dict(sfUrl="https://lr02922.snowflakecomputing.com/",
                   sfUser=self.user.strip(),
                   pem_private_key=pkb,
                   sfRole="ARST_TEAM",
                   sfDatabase=g_database,
                   sfSchema=g_schema,
                   sfWarehouse="DEV_WAREHOUSE")

    sdf = spark.read \
          .format("snowflake") \
          .options(**options) \
          .option("query", query) \
          .load()

    return sdf

# COMMAND ----------

show_doc(Snowflake.get_cursor)

# COMMAND ----------

show_doc(Snowflake.execute_query)

# COMMAND ----------


#export

from pathlib import Path
from czLandscapingTk.searchEngineUtils import ESearchQuery, EuroPMCQuery
from czLandscapingTk.queryTranslator import QueryTranslator, QueryType
import czLandscapingTk.dashdbQueries

from datetime import datetime
import requests
import json
from tqdm import tqdm

class DashboardDb:
  """This class permits the construction of a database of resources generated from combining a list of queries with a list of subqueries on multiple online repositories.<BR>
  Functionality includes:
    * Define a spreadsheet with a column of queries expressed in boolean logic
    * Optional: Define a secondary spreadsheet with a column of subqueries expressed in boolean logic
    * Iterate over different sources (Pubmed + European Pubmed) to execute all combinations of queries and subqueries
    * Store extended records for all papers - including full text where available from CZI's internal data repo. 
    
  Attributes (note - store `user`, `pem`, and `pwd` as `dbutils.secret` data ):
    * prefix: a string that will be used as the prefix for each table in the database 
    * user: Snowflake username 
    * pem: SSH key 
    * pwd: Password for SSH key 
    * warehouse: name of the SNOWFLAKE warehouse
    * database: name of the SNOWFLAKE database
    * schema: name of the SNOWFLAKE schema
    * role: name of the SNOWFLAKE role with correct permissions to execute database editing
    * loc: local disk location for files 
  """

  def __init__(self, prefix, user, pem, pwd, warehouse, database, schema, role, loc):
    self.sf = Snowflake(user, pem, pwd, warehouse, database, schema, role)
    self.database = database
    self.schema = schema
    self.loc = loc
    self.prefix = prefix

    if os.path.exists(loc) is False:
      os.mkdir(loc)

    log_path = '%s/sf_log.txt' % (loc)
    if os.path.exists(log_path) is False:
      Path(log_path).touch()

    self.temp_annotations_path = '%s/TMP_SO.txt' % (loc)
    if os.path.exists(self.temp_annotations_path) is False:
      Path(self.temp_annotations_path).touch()

    self.temp_documents_path = '%s/TMP_DOC.txt' % (loc)
    if os.path.exists(self.temp_documents_path) is False:
      Path(self.temp_documents_path).touch()
      
  def get_cursor(self):
    cs = self.sf.get_cursor()
    return cs

  def execute_query(self, sql, columns):
    cs = self.get_cursor()
    cs.execute(sql)
    df = pd.DataFrame(cs.fetchall(), columns=columns)
    df = df.replace('\n', ' ', regex=True)
    return df

  def upload_wb(self, df2, table_name, cs=None):
    if cs is None: 
      cs = self.get_cursor()
    table_name = re.sub('PREFIX_', self.prefix, 'PREFIX_'+table_name)
    df = df2.replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n',  ' ', regex=True)
    df.to_csv(self.loc+'/'+table_name+'.tsv', index=False, header=False, sep='\t')
    cs.execute('DROP TABLE IF EXISTS '+table_name+';')
    cols = [re.sub(' ','_',c).lower() for c in df.columns if c is not None]
    cols = [c+' INT AUTOINCREMENT' if c=='ID' else c+' TEXT' for c in cols]
    cs.execute('CREATE TABLE '+table_name+'('+', '.join(cols)+');')
    print(self.loc +'/'+table_name+'.tsv')
    cs.execute('put file://' + self.loc +'/'+table_name+'.tsv' + ' @%'+table_name+';')
    cs.execute("copy into "+table_name+" from @%"+table_name+" FILE_FORMAT=(TYPE=CSV FIELD_DELIMITER=\'\\t\')")

  def clear_corpus_to_paper_table(self, cs=None):
    if cs is None: 
      cs = self.get_cursor()
    table_name = re.sub('PREFIX_', self.prefix, 'PREFIX_CORPUS_TO_PAPER')
    cs.execute('DROP TABLE IF EXISTS ' + table_name + ';')

  def build_lookup_table(self, cs=None, delete_existing=False):
    if cs is None: 
      cs = self.get_cursor()
    COLS = ['PMID', 'FIRST_AUTHOR', 'YEAR', 'VOLUME', 'PAGE']
    SQL = '''
      SELECT p.PMID as PMID, REGEXP_SUBSTR(a.NAME, '\\\\S*$') as FIRST_AUTHOR, 
          p.YEAR as YEAR, p.VOLUME as VOLUME, REGEXP_SUBSTR(p.PAGINATION, '^\\\\d+') as PAGE
      FROM FIVETRAN.KG_RDS_CORE_DB.PAPER as p
        JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER_TO_AUTHOR_v2 as pa on (p.ID=pa.ID_PAPER)
        JOIN FIVETRAN.KG_RDS_CORE_DB.AUTHOR_v2 as a on (a.ID=pa.ID_AUTHOR)
      WHERE pa.AUTHOR_INDEX=0
      ORDER BY p.PMID DESC
    '''
    BUILD_PMID_LOOKUP_SQL = "create table if not exists PMID_LOOKUP as " + SQL
    if deleted_existing: 
      cs.execute('drop table if exists PMID_LOOKUP')
    cs.execute(BUILD_PMID_LOOKUP_SQL)    
    
  def build_core_tables_from_pmids(self, cs=None):
    if cs is None: 
      cs = self.get_cursor()
    print('PAPER_NOTES')
    cs.execute("DROP TABLE IF EXISTS " + self.prefix + "PAPER_NOTES")
    cs.execute(re.sub('PREFIX_', self.prefix, czLandscapingTk.dashdbQueries.BUILD_DASHBOARD_PAPER_NOTES))
    print('PAPER_OPEN_ACCESS')
    cs.execute("DROP TABLE IF EXISTS " + self.prefix + "PAPER_OPEN_ACCESS")
    cs.execute(re.sub('PREFIX_', self.prefix, czLandscapingTk.dashdbQueries.BUILD_DASHBOARD_PAPER_OPEN_ACCESS))
    print('COLLABORATIONS')
    cs.execute("DROP TABLE IF EXISTS " + self.prefix + "COLLABORATIONS")
    cs.execute(re.sub('PREFIX_', self.prefix, czLandscapingTk.dashdbQueries.BUILD_DASHBOARD_COLLABORATIONS))
    print('ALL KNOWN AUTHOR LOCATIONS')
    cs.execute("DROP TABLE IF EXISTS " + self.prefix + "AUTHOR_LOCATION")
    cs.execute(re.sub('PREFIX_', self.prefix, czLandscapingTk.dashdbQueries.BUILD_DASHBOARD_AUTHOR_LOCATION))
    print('CITATION COUNTS')
    cs.execute("DROP TABLE IF EXISTS " + self.prefix + "CITATION_COUNTS")
    cs.execute(re.sub('PREFIX_', self.prefix, czLandscapingTk.dashdbQueries.BUILD_DASHBOARD_CITATION_COUNTS))

  def drop_database(self, cs=None):
    if cs is None: 
      cs = self.get_cursor()
    cs.execute("BEGIN")
    cs.execute("DROP TABLE IF EXISTS " + self.prefix + "AUTHOR_LOCATION")
    cs.execute("DROP TABLE IF EXISTS " + self.prefix + "CITATION_COUNTS")
    cs.execute("DROP TABLE IF EXISTS " + self.prefix + "COLLABORATIONS")
    cs.execute("DROP TABLE IF EXISTS " + self.prefix + "PAPER_NOTES")
    cs.execute("DROP TABLE IF EXISTS " + self.prefix + "CORPUS")
    cs.execute("DROP TABLE IF EXISTS " + self.prefix + "CORPUS_TO_PAPER")
    cs.execute("COMMIT")

  def build_database_from_queries(self, pubmed_api_key, query_df, id_col, q_col, subquery_df=None, subq_col=None, delete_db=True, pm_include=True, epmc_inlcude=True):
    '''
    Function to generate a snowflake database of scientific papers based on a list of queries listed in a dataframe 
    (and possibly faceted by a second set of queries in a second dataframe). This system will (optionally) 
    execute queries on the REST services of Pubmed and European PMC to build the database.    
    
    Attributes:
    * pubmed_api_key: see https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/
    * query_df: a Pandas Dataframe of corpora where one column specifies the query
    * id_col: ID column used to identify the corpus  
    * q_col: column for the query (expressed using '&' for AND and '|' for OR)
    * subquery_df: an optional Pandas Dataframe of subqueries
    * subq_col: an optional column for the subquery (expressed using '&' for AND and '|' for OR)
    * delete_db (default=True): delete the existing database?
    * pm_include (default=True): Run corpus construction queries on Pubmed 
    * epmc_include (default=True): Run corpus construction queries on European PMC
    '''
    qt = QueryTranslator(query_df, id_col, q_col)
    if subquery_df:
      qt2 = QueryTranslator(subquery_df, id_col, subq_col)
    else:
      qt2 = None
    corpus_paper_list = []
    
    if pm_include:
      (corpus_ids, pubmed_queries) = qt.generate_queries(QueryType.pubmed)
      if qt2:
        (subset_ids, pubmed_subset_queries) = qt2.generate_queries(QueryType.pubmed)
      else: 
        (subset_ids, pubmed_subset_queries) = ([0],[''])
      for (i, q) in zip(corpus_ids, pubmed_queries):
        for (j, sq) in zip(subset_ids, pubmed_subset_queries):
          if len(sq) > 0:
            q = '(%s) AND (%s)'%(q,sq) 
          q = re.sub('\s+','+',q)
          esq = ESearchQuery(pubmed_api_key)
          pubmed_pmids = esq.execute_query(q)
          print(len(pubmed_pmids))
          for pmid in pubmed_pmids:
            corpus_paper_list.append((pmid, i, 'pubmed', j))

    if epmc_inlcude:
      (corpus_ids, epmc_queries) = qt.generate_queries(QueryType.closed)
      if qt2:
        (subset_ids, epmc_subset_queries) = qt2.generate_queries(QueryType.closed)
      else: 
        (subset_ids, epmc_subset_queries) = ([0],[''])
      for (i, q) in zip(corpus_ids, epmc_queries):
        for (j, sq) in zip(subset_ids, epmc_subset_queries):
          query = q
          if len(sq) > 0:
            query = '(%s) AND (%s)'%(q, sq) 
          epmcq = EuroPMCQuery()
          numFound, epmc_pmids, other_ids = epmcq.run_empc_query(query)
          for id in tqdm(epmc_pmids):
            corpus_paper_list.append((id, i, 'epmc', j))
    
    corpus_paper_df = pd.DataFrame(corpus_paper_list, columns=['ID_PAPER', 'ID_CORPUS', 'SOURCE', 'SUBSET_CODE'])
  
    cs = self.get_cursor()
    cs.execute("BEGIN")
    if delete_db:
      self.drop_database(cs=cs)
    self.upload_wb(df, 'CORPUS', cs=cs)
    self.upload_wb(corpus_paper_df, 'CORPUS_TO_PAPER', cs=cs)
    self.build_core_tables_from_pmids(cs=cs)
    cs.execute('COMMIT')

# COMMAND ----------

show_doc(DashboardDb.build_database_from_queries)
