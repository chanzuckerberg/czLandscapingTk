# Databricks notebook source
# MAGIC %md # Dashboard Database Utilities
# MAGIC
# MAGIC > Simple query classes that allows contruction of a SQL database in Snowflake for science literature dashboard applications. Note that this implementation is intended primarily for internal CZI use.

# COMMAND ----------

#| default_exp dashDatabricks

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

from pathlib import Path
from czLandscapingTk.searchEngineUtils import ESearchQuery, EuroPMCQuery
from czLandscapingTk.queryTranslator import QueryTranslator, QueryType
import czLandscapingTk.dashdbQueries

from datetime import datetime
from time import time,sleep

import requests
import json
from tqdm import tqdm
import os

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

  def __init__(self, catalog, database, loc):
    self.catalog = catalog
    self.database = database
    self.loc = loc

    if os.path.exists(loc) is False:
      os.mkdir(loc)

    log_path = '%s/db_log.txt' % (loc)
    if os.path.exists(log_path) is False:
      Path(log_path).touch()
      
  def execute_query(self, sql):
    sdf = spark.sql(sql)
    return sdf.toPandas()

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

  def execute_epmc_queries_on_sections(self, qt, qt2, sections=['paper_title', 'ABSTRACT'], extra_columns=[]):
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
        if query is None or query=='nan' or len(query)==0: 
          continue
        if len(sq) > 0:
          query = '(%s) AND (%s)'%(q, sq) 
        epmcq = EuroPMCQuery()
        #try:
        numFound, epmc_pmids = epmcq.run_empc_query(query, extra_columns=extra_columns)
        for row in tqdm(epmc_pmids):
            tup = [row[0], i, 'epmc', j, row[1]]
            if len(row)>2:
                tup.extend(row[2:])
            corpus_paper_list.append(tup)
        #except Exception as e:
        #  epmc_errors.append((i, j, query, e))
    return corpus_paper_list, epmc_errors

  def check_query_terms(self, qt, qt2=None, pubmed_api_key=''):
    pmq = ESearchQuery(api_key=pubmed_api_key)
    terms = set()
    for t in qt.terms2id.keys():
        terms.add(t)
    if qt2 is not None:
        for t2 in qt2.terms2id.keys():
            terms.add(t2)
    check_table = {} 
    for t in tqdm(terms):
        (is_ok, t2, c) = pmq._check_query_phrase(t)
        check_table[t] = (is_ok, c)
    return check_table

def run_empc_query(self, q, page_size=1000, timeout=60, extra_columns=[]):
    EMPC_API_URL = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=JSON&pageSize='+str(page_size)+'&synonym=TRUE'
    if len(extra_columns)>0:
        EMPC_API_URL += '&resultType=core'
    url = EMPC_API_URL + '&query=' + q
    r = requests.get(url, timeout=timeout)
    data = json.loads(r.text)
    numFound = data['hitCount']
    print(url + ', ' + str(numFound) + ' European PMC PAPERS FOUND')
    results = []
    cursorMark = '*'
    for i in tqdm(range(0, numFound, page_size)):
        url = EMPC_API_URL + '&cursorMark=' + cursorMark + '&query=' + q
        r = requests.get(url, timeout=timeout)
        data = json.loads(r.text)
        #print(data)
        if data.get('nextCursorMark'):
            cursorMark = data['nextCursorMark']
            for d in data['resultList']['result']:
                results.append(d)
    results = sorted(list(results), key = lambda x: x['id'])
    print(' Returning '+str(len(results)))
    return (numFound, results)

def execute_epmc_queries(self, qt, qt2, sections=['paper_title', 'ABSTRACT']):
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
            if query is None or query=='nan' or len(query)==0: 
                continue
            if len(sq) > 0:
                query = '(%s) AND (%s)'%(q, sq) 
            #try:
            numFound, tuples = run_empc_query(query)
            for tup in tqdm(tuples):
                corpus_paper_list.append(tup)
            #except Exception as e:
            #  epmc_errors.append((i, j, query, e))
    return corpus_paper_list, epmc_errors
    
def airtable_to_corpus_dataframes(self, at_key, at_sheets):
    atu = AirtableUtils(at_key)
    df1 = pd.DataFrame()
    for sn, id_col, query_col, col_map, sections in at_sheets: 
        cdf = atu.read_airtable(at_file, sn)
        cdf = cdf.rename(columns={id_col:'ID', query_col:'QUERY'})
        cdf = cdf.rename(columns=col_map)
        cdf = cdf.fillna('').rename(
            columns={c:re.sub('[\s\(\)]','_', c.upper()) for c in cdf.columns}
            )
        cdf.QUERY = [re.sub('^http[s]*://', '', r.QUERY) if r.QUERY[:4]=='http' else r.QUERY 
                     for i,r in cdf.iterrows()] 
        cdf.QUERY = [re.sub('/$', '', r.QUERY.strip())  
                        for i,r in cdf.iterrows()] 
        df1 = pd.concat([df1, cdf])
        qt = QueryTranslator(cdf, 'ID', 'QUERY')
        paper_list, errors = self.execute_epmc_queries(qt, None, sections=sections)
        l = [tup if re.match('\d',tup[0]) else (-1, tup[1], tup[2], tup[3], tup[4]) for tup in paper_list]
        cols = ['ID_PAPER', 'ID_CORPUS', 'SOURCE', 'SUBSET_CODE', 'DOI']
        cols.extend(extra_columns)
        temp = pd.DataFrame(paper_list, columns=cols)
        df2 = pd.concat([df2, temp.drop(columns=extra_columns).drop_duplicates()])
        if len(extra_columns)>0:
            df3 = pd.concat([df3, temp.drop(columns=['ID_CORPUS','SOURCE', 'SUBSET_CODE']).drop_duplicates()])

    df1.fillna('', inplace=True)
    df1 = df1.reset_index(drop=True)
    df2.fillna('', inplace=True)
    df2 = df2.reset_index(drop=True)
    df3.fillna('', inplace=True)
    df3 = df3.reset_index(drop=True).drop_duplicates()
    
    return df1, df2, df3
