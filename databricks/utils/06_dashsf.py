# Databricks notebook source
# MAGIC %md # Dashboard Database Utilities
# MAGIC
# MAGIC > Simple query classes that allows contruction of a SQL database in Snowflake for science literature dashboard applications. Note that this implementation is intended primarily for internal CZI use.

# COMMAND ----------

#| default_exp dashdbUtils

# COMMAND ----------

#| hide
from nbdev import *

# COMMAND ----------

# MAGIC %md ![Schema for Dashboard Database](https://lucid.app/publicSegments/view/0388e058-e5f8-4914-9536-f718edf21d47/image.jpeg)
# MAGIC
# MAGIC Image source on LucidDraw: [Link](https://lucid.app/lucidchart/29f3e2c7-cd56-46fa-a6ce-0dda18d819e1/edit?viewport_loc=-2670%2C-1547%2C3099%2C1648%2CoxaLRZ4JBiatT&invitationId=inv_64fde248-ce31-40b5-85d5-2f3317b5f876#)

# COMMAND ----------

# MAGIC %md Use this class to run queries from a spreadsheet across various online academic graph systems and generate a database based on the data from those queries. 
# MAGIC
# MAGIC If we had a dataframe `query_df` where one of the columns described a literature query expressed in Boolean Logic:
# MAGIC
# MAGIC | ID | DISEASE NAME | MONDO_ID | QUERY  | 
# MAGIC |----|--------------|----------|--------|
# MAGIC | 1 | Adult Polyglucosan Body Disease | MONDO:0009897 | adult polyglucosan body disease \| adult polyglucosan body neuropathy
# MAGIC | 2 | Creatine transporter deficiency | MONDO:0010305 |creatine transporter deficiency \| guanidinoacetate methyltransferase deficiency \| AGAT deficiency \| cerebral creatine deficiency syndrome 1 \| X-linked creatine deficiency syndrome \| Cerebral Creatine Deficiency Syndromes \| creatine transporter defect \| SLC6A8 deficiency \| X-linked creatine transporter deficiency \| X-linked creatine deficiency \| X-linked creatine deficiency syndrome \| guanidinoacetate methyltransferase deficiency \| guanidinoacetate N-methyltransferase activity disease \| GAMT deficiency \| glycine amidinotransferase activity disease \| arginine:glycine amidinotransferase deficiency \| AGAT deficiency \| GATM deficiency              
# MAGIC | 3 | AGAT deficiency | MONDO:0012996 |  "GATM deficiency" \| "AGAT deficiency" \| "arginine:glycine amidinotransferase deficiency" \| "L-arginine:glycine amidinotransferase deficiency"
# MAGIC | 4 | Guanidinoacetate methyltransferase deficiency | MONDO:0012999 |  "guanidinoacetate methyltransferase deficiency" \| "GAMT deficiency"
# MAGIC | 5 | CLOVES Syndrome | MONDO:0013038 | "CLOVES syndrome \| (congenital lipomatous overgrowth) & (vascular malformation epidermal) & (nevi-spinal) & syndrome \| (congenital lipomatous overgrowth) & (vascular malformations) & (Epidermal nevi) & ((skeletal\|spinal) & abnormalities) \| CLOVE syndrome \| (congenital lipomatous overgrowth) & (vascular malformation) & (epidermal nevi)
# MAGIC
# MAGIC
# MAGIC It is straightforward to build a database of all corpora listed in the spreadsheet from the search queries expressed in the `QUERY` column:
# MAGIC
# MAGIC ```
# MAGIC from czLandscapingTk.airtableUtils import AirtableUtils
# MAGIC from czLandscapingTk.dashdbUtils import DashboardDb
# MAGIC from czLandscapingTk.generalUtils import dump_data_to_disk
# MAGIC import re
# MAGIC
# MAGIC at_ID_column = 'ID'
# MAGIC at_query_column = 'QUERY'
# MAGIC
# MAGIC # this will be substituted into the tables above instead of 'PREFIX_'
# MAGIC prefix = 'MY_AMAZING_DATABASE_' 
# MAGIC
# MAGIC # Databricks secret management
# MAGIC secret_scope = 'secret-scope' 
# MAGIC
# MAGIC # Location of data in SNOWFLAKE
# MAGIC warehouse = 'DEV_WAREHOUSE'
# MAGIC database = 'DEV_DB' 
# MAGIC schema = 'SKE'
# MAGIC
# MAGIC # SNOWFLAKE role for permissions
# MAGIC role = 'ARST_TEAM'
# MAGIC
# MAGIC # Location of temp files in Databricks file storage
# MAGIC loc = '/dbfs/FileStore/user/gully/'
# MAGIC
# MAGIC # See https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/
# MAGIC pubmed_api_key = 'blahblahblahblah' 
# MAGIC
# MAGIC # SNOWFLAKE Login credentials to be stored in secrets
# MAGIC user = dbutils.secrets.get(scope=secret_scope, key="SNOWFLAKE_SERVICE_USERNAME")
# MAGIC pem = dbutils.secrets.get(scope=secret_scope, key="SNOWFLAKE_SERVICE_PRIVATE_KEY")
# MAGIC pwd = dbutils.secrets.get(scope=secret_scope, key="SNOWFLAKE_SERVICE_PASSPHRASE")
# MAGIC
# MAGIC # Execution of the query and generation of the dashboard database
# MAGIC dashdb = DashboardDb(prefix, user, pem, pwd, warehouse, database, schema, role, loc)
# MAGIC corpus_paper_df = dashdb.run_remote_paper_queries(pubmed_api_key, queries_df, at_ID_column, at_query_column, 
# MAGIC     sf_include=False, pm_include=True, epmc_include=False)
# MAGIC ```
# MAGIC
# MAGIC The parameters `sf_include`, `pm_include`, and `empc_include` denote whether the Boolean queries listed will be run on (A) our own internal SNOWFLAKE database; (B) Pubmed; and (C) European PMC. Records for each of these databases are differentiated based on the `CORPUS` 

# COMMAND ----------

#| export

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

# COMMAND ----------

#| export

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

  def fetch_pandas(self, sql, batch=50000):
    cs = self.get_cursor()
    cs.execute(sql)
    rows = 0
    df = pd.DataFrame()
    while True:
        dat = cs.fetchmany(batch)
        if not dat:
          break
        cols = [desc[0] for desc in cs.description]
        df = pd.concat([df,pd.DataFrame(dat, columns=cols)])
        rows += df.shape[0]
    print(rows)
    return df

  def execute_query(self, cs, sql, columns):
    '''
    Executes an SQL query with a list of column names and returns a Pandas DataFrame
    '''    
    cs.execute(sql)
    df = pd.DataFrame(cs.fetchall(), columns=columns)
    df = df.replace('\n', ' ', regex=True)
    return df

  def run_query_in_spark(self, query):
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
                   sfDatabase=self.database,
                   sfSchema=self.schema,
                   sfWarehouse="DEV_WAREHOUSE")

    sdf = spark.read \
          .format("snowflake") \
          .options(**options) \
          .option("query", query) \
          .load()

    return sdf

# COMMAND ----------

#| export

from pathlib import Path
from czLandscapingTk.searchEngineUtils import ESearchQuery, EuroPMCQuery
from czLandscapingTk.queryTranslator import QueryTranslator, QueryType
from czLandscapingTk.dashdbQueries import BUILD_DASHBOARD_PAPER_NOTES, BUILD_DASHBOARD_PAPER_OPEN_ACCESS, \
        BUILD_DASHBOARD_COLLABORATIONS, BUILD_DASHBOARD_AUTHOR_LOCATION, BUILD_DASHBOARD_AUTHOR_LOCATION, \
        BUILD_DASHBOARD_CITATION_COUNTS

from datetime import datetime
from time import time,sleep

import requests
import json
from tqdm import tqdm

# COMMAND ----------

#| export

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

  def execute_query(self, sql, columns, cs=None):
    if cs is None: 
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
    if delete_existing:
      cs.execute('drop table if exists PMID_LOOKUP')
    cs.execute(BUILD_PMID_LOOKUP_SQL)    
    
  def build_core_tables_from_pmids(self, cs=None):
    if cs is None: 
      cs = self.get_cursor()
    print('PAPER_NOTES')
    cs.execute("DROP TABLE IF EXISTS " + self.prefix + "PAPER_NOTES")
    cs.execute(re.sub('PREFIX_', self.prefix, BUILD_DASHBOARD_PAPER_NOTES))
    print('PAPER_OPEN_ACCESS')
    cs.execute("DROP TABLE IF EXISTS " + self.prefix + "PAPER_OPEN_ACCESS")
    cs.execute(re.sub('PREFIX_', self.prefix, BUILD_DASHBOARD_PAPER_OPEN_ACCESS))
    print('COLLABORATIONS')
    cs.execute("DROP TABLE IF EXISTS " + self.prefix + "COLLABORATIONS")
    cs.execute(re.sub('PREFIX_', self.prefix, BUILD_DASHBOARD_COLLABORATIONS))
    print('ALL KNOWN AUTHOR LOCATIONS')
    cs.execute("DROP TABLE IF EXISTS " + self.prefix + "AUTHOR_LOCATION")
    cs.execute(re.sub('PREFIX_', self.prefix, BUILD_DASHBOARD_AUTHOR_LOCATION))
    print('CITATION COUNTS')
    cs.execute("DROP TABLE IF EXISTS " + self.prefix + "CITATION_COUNTS")
    print(re.sub('PREFIX_', self.prefix, BUILD_DASHBOARD_CITATION_COUNTS))
    cs.execute(re.sub('PREFIX_', self.prefix, BUILD_DASHBOARD_CITATION_COUNTS))

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


  def run_remote_paper_queries(self, pubmed_api_key, query_df, id_col, q_col, 
                                  subquery_df=None, subq_col=None, 
                                  delete_db=True, pm_include=True, 
                                  epmc_include=True, sf_include=True):
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
    if subquery_df is not None:
      qt2 = QueryTranslator(subquery_df, id_col, subq_col)
    else:
      qt2 = None

    corpus_paper_list = []
    if pm_include:
      pubmed_corpus_paper_list, pubmed_errors = self.execute_pubmed_queries(qt, qt2)
      corpus_paper_list.extend(pubmed_corpus_paper_list)

    epmc_errors = []        
    if epmc_include:
      epmc_corpus_paper_list, epmc_errors = self.execute_epmc_queries(qt, qt2)
      corpus_paper_list.extend(epmc_corpus_paper_list) 
      
    sf_errors = []
    if sf_include: 
      sf_corpus_paper_list, sf_errors_errors = self.execute_epmc_queries(qt, qt2)
      corpus_paper_list.extend(sf_corpus_paper_list) 

    corpus_paper_df = pd.DataFrame(corpus_paper_list, columns=['ID_PAPER', 'ID_CORPUS', 'SOURCE', 'SUBSET_CODE'])

    return corpus_paper_df   
    
  def execute_pubmed_queries(self, qt, qt2): 
    corpus_paper_list = []
    pubmed_errors = []
    (corpus_ids, pubmed_queries) = qt.generate_queries(QueryType.pubmed, skipErrors=False)
    if qt2:
      (subset_ids, pubmed_subset_queries) = qt2.generate_queries(QueryType.pubmed)
    else: 
      (subset_ids, pubmed_subset_queries) = ([0],[''])
    for (i, q) in zip(corpus_ids, pubmed_queries):
      for (j, sq) in zip(subset_ids, pubmed_subset_queries):
        query = q
        print(query)
        if query=='nan' or len(query)==0: 
          pubmed_errors.append((i, j, query))
          continue
        if len(sq) > 0:
          query = '(%s) AND (%s)'%(q,sq) 
        print(query)
        esq = ESearchQuery(pubmed_api_key)
        try: 
          pubmed_pmids = esq.execute_query(query)
        except:
          pubmed_errors.append((i, j, query))
          pummed_pmids = []
        print(len(pubmed_pmids))
        for pmid in pubmed_pmids:
          corpus_paper_list.append((pmid, i, 'pubmed', j))
    return corpus_paper_list, pubmed_errors
  
  def execute_epmc_queries(self, qt, qt2):
    corpus_paper_list = []
    epmc_errors = []
    (corpus_ids, epmc_queries) = qt.generate_queries(QueryType.closed)
    if qt2:
      (subset_ids, epmc_subset_queries) = qt2.generate_queries(QueryType.closed)
    else: 
      (subset_ids, epmc_subset_queries) = ([0],[''])
    for (i, q) in zip(corpus_ids, epmc_queries):
      for (j, sq) in zip(subset_ids, epmc_subset_queries):
        query = q
        if query=='nan' or len(query)==0: 
          epmc_errors.append((i, j, query))
          continue
        if len(sq) > 0:
          query = '(%s) AND (%s)'%(q, sq) 
        epmcq = EuroPMCQuery()
        try: 
          numFound, epmc_pmids = epmcq.run_empc_query(query)
          for id, doi in tqdm(epmc_pmids):
            corpus_paper_list.append((id, i, 'epmc', j, doi))
        except:
          epmc_errors.append((id, i, j, query))
    return corpus_paper_list, epmc_errors
  
  def execute_pubmed_queries_on_sections(self, qt, qt2, api_key='', sections=['tiab']):
    corpus_paper_list = []
    errors = []
    (corpus_ids, pubmed_queries) = qt.generate_queries(QueryType.pubmed, sections=sections)
    if qt2:
      (subset_ids, pubmed_subset_queries) = qt2.generate_queries(QueryType.pubmed, sections=sections)
    else: 
      (subset_ids, pubmed_subset_queries) = ([0],[''])
    for (i, q) in zip(corpus_ids, pubmed_queries):
      #if i != 851:
      #  continue
      for (j, sq) in zip(subset_ids, pubmed_subset_queries):
        query = q
        if query=='nan' or len(query)==0: 
          errors.append((i, j, query))
          continue
        if len(sq) > 0:
          query = '(%s) AND (%s)'%(q, sq) 
        pmq = ESearchQuery(api_key=api_key)
        num_found = pmq.execute_count_query(query)
        print(num_found)
        if num_found>0:
          pmids = pmq.execute_query(query)
          sleep(0.5) # Sleep for half a second
          for id in tqdm(pmids):
            corpus_paper_list.append((id, i, 'pubmed', j))
    return corpus_paper_list

  def execute_epmc_queries_on_sections(self, qt, qt2, sections=['paper_title', 'ABSTRACT']):
    corpus_paper_list = []
    epmc_errors = []
    (corpus_ids, epmc_queries) = qt.generate_queries(QueryType.epmc, sections=sections)
    if qt2:
      (subset_ids, epmc_subset_queries) = qt2.generate_queries(QueryType.epmc, sections=sections)
    else: 
      (subset_ids, epmc_subset_queries) = ([0],[''])
    for (i, q) in zip(corpus_ids, epmc_queries):
      for (j, sq) in zip(subset_ids, epmc_subset_queries):
        query = q
        if query=='nan' or len(query)==0: 
          continue
        if len(sq) > 0:
          query = '(%s) AND (%s)'%(q, sq) 
        epmcq = EuroPMCQuery()
        try:
          numFound, epmc_pmids = epmcq.run_empc_query(query)
          for id, doi in tqdm(epmc_pmids):
            corpus_paper_list.append((id, i, 'epmc', j, doi))
        except:
          epmc_errors.append((id, i, j, query))
    return corpus_paper_list, epmc_errors

  def execute_sf_queries(self, qt, qt2):
    corpus_paper_list = []
    sf_errors = []
    (corpus_ids, sf_queries) = qt.generate_queries(QueryType.snowflake)
    if qt2:
      (subset_ids, sf_subset_queries) = qt2.generate_queries(QueryType.snowflake)
    else: 
      (subset_ids, sf_subset_queries) = ([0],[''])
    cs = self.get_cursor()
    for (i, q) in zip(corpus_ids, sf_queries):
      for (j, sq) in zip(subset_ids, sf_subset_queries):
        stem = 'SELECT p.ID FROM FIVETRAN.KG_RDS_CORE_DB.PAPER as p WHERE '
        query = stem + q
        if query=='nan' or len(query)==0: 
          sf_errors.append((i, j, query))
          continue
        if len(sq) > 0:
          query = stem + '(%s) AND (%s)'%(q, sq) 
        print(query)
        try: 
          df = self.execute_query(query, ['ID'], cs)
          numFound = len(df)
          print(i, q, numFound)
          sf_ids = df.ID.to_list()
          for id in tqdm(sf_ids):
            corpus_paper_list.append((id, i, 'czkg', j))
        except:
          sf_errors.append((i, j, query))
    return corpus_paper_list, sf_errors
  
  
  def build_db(self, query_df, corpus_paper_df, subquery_df=None, delete_db=True):
    cs = self.get_cursor()
    #cs.execute("BEGIN")
    if delete_db:
      self.drop_database(cs=cs)
    self.upload_wb(query_df, 'CORPUS', cs=cs)
    self.upload_wb(corpus_paper_df, 'CORPUS_TO_PAPER_BASE', cs=cs)
    fix_sql = '''create table PREFIX_CORPUS_TO_PAPER as 
      SELECT DISTINCT cpp.ID_PAPER, cpp.ID_CORPUS, cpp.SOURCE, cpp.SUBSET_CODE, cpp.DOI
      FROM (
        SELECT DISTINCT p.ID as ID_PAPER, cp.ID_CORPUS, cp.SOURCE, cp.SUBSET_CODE, p.DOI   
        FROM PREFIX_CORPUS_TO_PAPER_BASE as cp 
          JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p ON (cp.ID_PAPER=p.ID)
        UNION 
        SELECT DISTINCT p.ID as ID_PAPER, cp.ID_CORPUS, cp.SOURCE, cp.SUBSET_CODE, p.DOI   
        FROM PREFIX_CORPUS_TO_PAPER_BASE as cp 
          JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p ON (cp.DOI=p.DOI)
      ) as cpp 
    '''
    cs.execute(re.sub('PREFIX_', self.prefix, fix_sql))
    cs.execute(re.sub('PREFIX_', self.prefix, "DROP TABLE PREFIX_CORPUS_TO_PAPER_BASE;"))
    if subquery_df is not None:
      self.upload_wb(subquery_df, 'SUB_CORPUS', cs=cs)
    self.build_core_tables_from_pmids(cs=cs)
    #cs.execute('COMMIT')
