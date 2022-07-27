# Databricks notebook source
# MAGIC %run ../../global_variables

# COMMAND ----------

# MAGIC %run ../../utils/dashboard_utils

# COMMAND ----------

# MAGIC %run ../../utils/nlm_eutils

# COMMAND ----------

# MAGIC %run ../../utils/solr

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text('airtable_file','')
dbutils.widgets.text('airtable_table','')
dbutils.widgets.text('prefix','')
dbutils.widgets.dropdown('pm_interface', "eutils", ['eutils','website'])
dbutils.widgets.dropdown('pm_order', "best_match", ['best_match','date'])
dbutils.widgets.dropdown('delete_database', "True", ['True','False'])
dbutils.widgets.dropdown('pm_include', "True", ['True','False'])
dbutils.widgets.dropdown('solr_include', "False", ['True','False'])
dbutils.widgets.dropdown('epmc_include', "False", ['True','False'])
dbutils.widgets.text('airtable_subsets_table','')
dbutils.widgets.text('airtable_subsets_column','Query')

airtable_file = dbutils.widgets.get('airtable_file')
airtable_table = dbutils.widgets.get('airtable_table')

prefix = dbutils.widgets.get('prefix')

pm_interface = dbutils.widgets.get('pm_interface')
pm_order = dbutils.widgets.get('pm_order')

pm_include = dbutils.widgets.get('pm_include')
solr_include = dbutils.widgets.get('solr_include')
epmc_include = dbutils.widgets.get('epmc_include')

# Any additional conditions for creating this corpus?
airtable_subsets_table = dbutils.widgets.get('airtable_subsets_table')
delete_database = dbutils.widgets.get('delete_database')

# COMMAND ----------

#airtable_file = 'appkqgTmweMqvQ7CW'
#airtable_table = 'Cycle1'
#prefix = 'FULLTEXTPILOT_PM_'
mcdash = RstContentDashboard(g_database, g_schema, g_loc, prefix)

# COMMAND ----------

import numpy as np
cs = get_dash_cursor(sf, mcdash)
#corpus_id='0'
#sql2 = '''
#    select cl.CLUSTER_ID as id, l.DRSM_LABEL as label, COUNT(l.ID_PAPER) as count
#    from PREFIX_CLUSTERS as cl
#        join PREFIX_SENTENCE_CLUSTERS as scl on (cl.cluster_id=scl.CLUSTER_ASSIGNMENTS and cl.corpus_id=scl.id_corpus)
#        join PREFIX_DRSM as l on (scl.ID_PAPER=l.ID_PAPER)
#    where cl.corpus_id = <<corpus_id>>
#    group by cl.CLUSTER_ID, l.DRSM_LABEL
#    order by cl.CLUSTER_ID+0;'''
#sql2 = re.sub('PREFIX_', prefix, sql2)
#sql2 = re.sub('<<corpus_id>>', corpus_id, sql2)
#cols2 = ['id', 'label', 'count']
#df2 = sf.execute_query(cs, sql2, cols2)
#df2_piv1 = pd.pivot_table(df2, values='count', index=['id'],
#                       columns=['label'], aggfunc=np.sum).fillna(0)
#labels = df2_piv1.columns.to_list()
#df2_piv1['drsm_counts'] = [json.dumps({y: row[1][y] for y in labels}) for row in df2_piv1.iterrows()]
#df2_piv1 = df2_piv1.reset_index()
#df2_piv1['id'] = df2_piv1.id.astype('int64', copy=False)
#df2_piv2 = df2_piv1.drop(columns=labels) 
#displayHTML(df2_piv2.to_html())

# COMMAND ----------

airtable_api_url = 'https://api.airtable.com/v0/%s/%s?api_key=%s'%(airtable_file, airtable_table, g_airtable_api_key) 
df = read_airtable(airtable_api_url)
displayHTML(df.to_html())

# COMMAND ----------

subsets_df = None
if len(airtable_subsets_table)>0: 
  airtable_subsets_api_url = 'https://api.airtable.com/v0/%s/%s?api_key=%s'%(airtable_file, airtable_subsets_table, g_airtable_api_key) 
  subsets_df = read_airtable(airtable_subsets_api_url)
  subsets_df = subsets_df.fillna('')
  subsets_df.reset_index(inplace=True, drop=True)
  displayHTML(subsets_df.to_html())
else: 
  subsets_df = pd.DataFrame([{"ID":0, "Subset_Name":"None", "Query":""}])
no_subsets_df = pd.DataFrame([{"ID":0, "Subset_Name":"None", "Query":""}])

# COMMAND ----------

# USE PYEDA TO PROCESS AND REPURPOSE QUERIES AS LOGICAL EXPRESSIONS FOR SEARCHING.
import re
import pprint
from pyeda.inter import *
from pyeda.boolalg.expr import Literal,AndOp,OrOp
from enum import Enum
import unicodedata

class QueryType(Enum):
  open = 1
  closed = 2
  solr = 3
  epmc = 4
  pubmed = 5
  andPlusOrPipe = 6
  pubmed_no_types = 7

class QueryTranslator(): 
  def __init__(self, df, query_col):
    pp = pprint.PrettyPrinter(indent=4)
    def fix_errors(expr_string):
      q = re.sub('\s+(AND)\s+',' & ',expr_string)
      q = re.sub('\s+(OR)\s+',' | ',q)
      q = re.sub('[\"\n]','',q)
      q = re.sub('\[(ti|ab|ft|tiab)\]',r'_\g<1>', q).strip()
      return q

    self.id2terms = {}
    self.terms2id = {}
    for tt in df[query_col]:
      redq = fix_errors(tt.strip())
      for t in re.split('[\&\|\(\)]', redq):
        t = re.sub('[\(\)]','', t).strip()
        #t = re.sub('\[(ti|ab|ft|tiab)\]',r'\g<1>', t).strip()
        if len(t)==0:
          continue
        if self.terms2id.get(t) is None:
          id = 't'+str(len(self.terms2id))
          self.id2terms[id] = unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode('ascii') # convert to ascii for searches via API 
          self.terms2id[t] = id

    ordered_names = sorted(self.terms2id.keys(), key=len, reverse=True)
    self.redq_list = []
    for row in df.iterrows():
      tt = row[1][query_col]
      row_id = row[1]['ID']
      redq = fix_errors(tt.strip())
      for t in ordered_names:
        id = self.terms2id[t]
        redq = re.sub('\\b'+t+'\\b', id, redq)
      self.redq_list.append((row_id, redq))

  def generate_queries(self, query_type:QueryType):
    queries = []
    IDs = []
    for ID, t in tqdm(self.redq_list):
      if t:
        print(t)
        ex = expr(t)
        queries.append(self._expand_expr(ex, query_type))
      else: 
        queries.append('')
      IDs.append(ID)
    return (IDs, queries)
    
  def _expand_expr(self, ex, query_type:QueryType):
    if query_type == QueryType.open:
      return self._simple(ex)
    elif query_type == QueryType.closed:
      return self._closed_quote(ex)
    elif query_type == QueryType.solr:
      return self._solr(ex)
    elif query_type == QueryType.epmc:
      return self._epmc(ex)
    elif query_type == QueryType.pubmed:
      return self._pubmed(ex)
    elif query_type == QueryType.andPlusOrPipe:
      return self._plusPipe(ex)
    elif query_type == QueryType.pubmed_no_types:
      return self._pubmed_no_types(ex)

  # expand the query as is with AND/OR linkagage, no extension. 
  # drop search fields
  def _simple(self, ex):
    if isinstance(ex, Literal):
      term = re.sub('_(ti|ab|ft|tiab)', '', self.id2terms[ex.name])
      return term
    elif isinstance(ex, AndOp):
      return '('+' AND '.join([self._simple(x) for x in ex.xs])+')'
    elif isinstance(ex, OrOp):
      return '('+' OR '.join([self._simple(x) for x in ex.xs])+')'

  def _closed_quote(self, ex):
    if isinstance(ex, Literal):
      term = re.sub('_(ti|ab|ft|tiab)', '', self.id2terms[ex.name])
      return '"'+term+'"'
    elif isinstance(ex, AndOp):
      return '('+' AND '.join([self._closed_quote(x) for x in ex.xs])+')'
    elif isinstance(ex, OrOp):
      return '('+' OR '.join([self._closed_quote(x) for x in ex.xs])+')'
  
  def _solr(self, ex):
    if isinstance(ex, Literal):
      p = re.compile('^(.*)_(ti|ab|ft|tiab)')
      m = p.match( self.id2terms[ex.name] )
      if m:
        t = m.group(1)
        f = m.group(2)
        if f == 'ti':
          return '(paper_title:"%s")'%(t)
        elif f == 'ab':
          return '(paper_abstract:"%s")'%(t)
        elif f == 'tiab':
          return '(paper_title:"%s" OR paper_abstract:"%s")'%(t,t)
        elif f == 'ft':
          return '(paper_title:"%s" OR paper_abstract:"%s")'%(t,t)
        else :
          raise Exception("Incorrect field specification, must be 'ti', 'ab', 'tiab', or 'ft': " + self.id2terms[ex.name] )
      else:              
        t = self.id2terms[ex.name]
        return '(paper_title:"%s" OR paper_abstract:"%s")'%(t,t)
    elif isinstance(ex, AndOp):
      return '('+' AND '.join([self._solr(x) for x in ex.xs])+')'
    elif isinstance(ex, OrOp):
      return '('+' OR '.join([self._solr(x) for x in ex.xs])+')'

  def _epmc(self, ex):
    if isinstance(ex, Literal):
      p = re.compile('^(.*)_(ti|ab|ft|tiab)')
      m = p.match( self.id2terms[ex.name] )
      if m:
        t = m.group(1)
        f = m.group(2)
        if f == 'ti':
          return '(TITLE:"%s")'%(t)
        elif f == 'ab':
          return '(ABSTRACT:"%s")'%(t)
        elif f == 'tiab':
          return '(TITLE:"%s" OR ABSTRACT:"%s")'%(t,t)
        elif f == 'ft':
          return '"%s"'%(t)
        else:
          raise Exception("Incorrect field specification, must be 'ti', 'ab', 'tiab', or 'ft': " + self.id2terms[ex.name] )
      else:              
        t = self.id2terms[ex.name]
        return '(paper_title:"%s" OR ABSTRACT:"%s")'%(t,t)
    elif isinstance(ex, AndOp):
      return '('+' AND '.join([self._epmc(x) for x in ex.xs])+')'
    elif isinstance(ex, OrOp):
      return '('+' OR '.join([self._epmc(x) for x in ex.xs])+')'

  def _pubmed(self, ex):
    if isinstance(ex, Literal):
      p = re.compile('^(.*)_(ti|ab|ft|tiab)$')
      m = p.match( self.id2terms[ex.name] )
      #print(m)
      if m:
        t = m.group(1)
        f = m.group(2)
        if f == 'ti':
          return '("%s"[ti])'%(t)
        elif f == 'ab':
          return '("%s"[ab])'%(t)
        elif f == 'tiab':
          return '("%s"[tiab])'%(t)
        elif f == 'ft':
          raise Exception("Can't run full text query on pubmed currently: " + self.id2terms[ex.name] )
        else:
          raise Exception("Incorrect field specification, must be 'ti', 'ab', 'tiab', or 'ft': " + self.id2terms[ex.name] )
      else:              
        t = self.id2terms[ex.name]
        return '()"%s"[tiab])'%(t,t)
    elif isinstance(ex, AndOp):
      return '('+' AND '.join([self._pubmed(x) for x in ex.xs])+')'
    elif isinstance(ex, OrOp):
      return '('+' OR '.join([self._pubmed(x) for x in ex.xs])+')'
    
  def _plusPipe(self, ex):
    if isinstance(ex, Literal):
      return '"%s"'%(self.id2terms[ex.name]) 
    elif isinstance(ex, AndOp):
      return '('+'+'.join([self._pubmed(x) for x in ex.xs])+')'
    elif isinstance(ex, OrOp):
      return '('+'|'.join([self._pubmed(x) for x in ex.xs])+')'

qt = QueryTranslator(df, 'TERMS')
qt2 = QueryTranslator(subsets_df, 'Query')

# COMMAND ----------

# MAGIC %md ## Pubmed Queries

# COMMAND ----------

qt.id2terms

# COMMAND ----------

(corpus_ids, pubmed_queries) = qt.generate_queries(QueryType.pubmed)
(subset_ids, pubmed_subset_queries) = qt2.generate_queries(QueryType.pubmed)
pubmed_queries

# COMMAND ----------

sbt_list = []
if pm_include == 'True':
  if pm_interface == 'eutils':
    for (i, q) in enumerate(pubmed_queries):
      for (j, sq) in zip(subset_ids, ['']):
        if len(sq)>0:
          q = '(%s) AND (%s)'%(q,sq) 
        q = re.sub('\s+','+',q)
        esq = ESearchQuery(g_pubmed_api_key)
        pubmed_pmids = esq.execute_query(q)
        print(len(pubmed_pmids))
        for pmid in pubmed_pmids:
          sbt_list.append((pmid, i, 'eutils', j))
          #print(pmid)
  else:
    for (i, q) in enumerate(pubmed_queries):
      for (j, sq) in zip(subset_ids, ['']):
        query = 'https://pubmed.ncbi.nlm.nih.gov/?format=pmid&size=10&term='+re.sub('\s+','+',q)
        if pm_order == 'date':
          query += '&sort=date'
        #print(query)
        #query = quote_plus(query)
        if len(sq)>0:
          query = '(%s) AND (%s)'%(sq) 
        response = urlopen(query)
        data = response.read().decode('utf-8')
        soup = BeautifulSoup(data, "lxml-xml")
        pmids = re.split('\s+', soup.find('body').text.strip())
        for pmid in pmids:
          sbt_list.append((int(pmid), i, 'pubmed', 0))
else:
  print('Skip Pubmed')
  sbt_list = []
pubmed_df = DataFrame(sbt_list, columns=['ID_PAPER', 'ID_CORPUS', 'SOURCE', 'SUBSET_CODE'])

# COMMAND ----------

# MAGIC %md ## SOLR Queries (less accurate but very fast)

# COMMAND ----------

from tqdm import tqdm
import requests
import os
import json
import re
import pprint
pp = pprint.PrettyPrinter(indent=4)

BASE_URL = 'http://aps-solr-http.staging.meta-infra.org:80/v1/solr/paper/select'
THIS_YEAR = 'AND paper_pub_date: [NOW-12MONTHS TO *]'

(corpus_ids, solr_queries) = qt.generate_queries(QueryType.solr)
(subset_ids, solr_subset_queries) = qt2.generate_queries(QueryType.solr)
subset_ids = [0]
solr_subset_queries = ['']

def exec_query_with_timeout_and_repeat(url, post_data_hash):
  r = requests.post(url, data=post_data_hash, timeout=10)
  data = json.loads(r.text)
  #print(data)
  if data.get('error') is not None:
      raise Exception("SOLR Error: " + data.get('error')['msg'])
  i = 0
  while(data.get('response') is None and i<10):
    r = requests.get(url, timeout=10)
    data = json.loads(r.text)
    i += 1
  if data.get('response') is None:
    return []
  return data

def run_solr_query(q, page_size=1000):   
  url = BASE_URL + '?wt=python&fl=id&rows=1&start=0&q='+q
  print(url)
  r = requests.get(url, timeout=10)
  #print(r.text)
  data = json.loads(r.text)
  #print(data)
  numFound = data['response']['numFound']
  print(q + ', ' + str(numFound) + ' SOLR PAPERS FOUND')
  pmids_from_q = set()
  for i in tqdm(range(0, numFound, page_size)):
      post_data_hash = {
          'wt': 'python',
          'fl': 'id,doi',
          'wt': 'python',
          'rows': str(page_size),
          'start': i,
          'q': '(' + q + ')'
      }
      #url = BASE_URL + '?wt=python&fl=id&rows='+str(page_size)+'&start='+str(i)+'&q='+q
      #r = requests.get(url)
      #print(r.text)
      data = json.loads(r.text)
      for d in data['response']['docs']:
        pmids_from_q.add(str(d['id']))
      data = exec_query_with_timeout_and_repeat(BASE_URL, post_data_hash)  
      for d in data['response']['docs']:
        pmids_from_q.add(str(d['id']))
      #pp.pprint(data)
      #break
  return (numFound, list(pmids_from_q))

META_API_URL = 'https://api.meta.org/work/'
if solr_include == 'True':
  dataset_list = []            
  sbt_list = []
  for (i, q) in enumerate(solr_queries):
    for (j, sq) in zip(subset_ids, solr_subset_queries):
      if len(sq)>0:
        q = '(%s) AND (%s)'%(q,sq) 
      numFound, solr_ids = run_solr_query(q)
      for id in tqdm(solr_ids):
        if '-' not in id:
          sbt_list.append((id, i, 'solr', j))
        elif 'Datafile-' in id: 
          m = re.search('^Datafile-(\d+)$', id)
          if m:
            dfid = m.group(1)
            r = requests.get(META_API_URL+'Dataset:'+dfid)
            dataset_list.append(json.loads(r.text))
      print("%d datasets found in KG"%(len(dataset_list)))
      
  solr_df = DataFrame(sbt_list, columns=['ID_PAPER', 'ID_CORPUS', 'SOURCE', 'SUBSET_CODE'])
  print("%d papers found in SOLR"%(len(sbt_list)))
else:
  print('Skipping SOLR.')
  solr_df = pd.DataFrame()

solr_df

# COMMAND ----------

# MAGIC %md ## European PMC queries

# COMMAND ----------

(corpus_ids, epmc_queries) = qt.generate_queries(QueryType.epmc)
(subset_ids, epmc_subset_queries) = qt2.generate_queries(QueryType.closed)

# COMMAND ----------

from tqdm import tqdm
import requests
import os
import json
import re
import pprint
pp = pprint.PrettyPrinter(indent=4)

(corpus_ids, epmc_queries) = qt.generate_queries(QueryType.epmc)
(subset_ids, epmc_subset_queries) = qt2.generate_queries(QueryType.closed)

def run_empc_query(q, page_size=1000):   
  EMPC_API_URL = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?resultType=idlist&format=JSON&pageSize='+str(page_size)+'&synonym=TRUE'
  url = EMPC_API_URL + '&query=' + q
  r = requests.get(url, timeout=10)
  data = json.loads(r.text)
  numFound = data['hitCount']
  print(q + ', ' + str(numFound) + ' European PMC PAPERS FOUND')
  pmids_from_q = set()
  otherIds_from_q = set()
  cursorMark = '*'
  for i in tqdm(range(0, numFound, page_size)):
    url = EMPC_API_URL + '&cursorMark='+cursorMark+'&query=' + q
    r = requests.get(url)
    data = json.loads(r.text)
    #print(data.keys())
    if data.get('nextCursorMark'):
      cursorMark = data['nextCursorMark']
    for d in data['resultList']['result']:
      if d.get('pmid'):
        pmids_from_q.add(str(d['pmid']))
      else: 
        otherIds_from_q.add(str(d['id']))
    #pp.pprint(data)
    #break
  return (numFound, list(pmids_from_q))

if epmc_include == 'True':
  epmc_sbt_list = []
  for (i, q) in enumerate(epmc_queries):
    for (j, sq) in zip(subset_ids, epmc_subset_queries):
      query = q
      if len(sq)>0:
        query = '(%s) AND (%s)'%(q,sq) 
      numFound, pmid_ids = run_empc_query(query)
      for id in tqdm(pmid_ids):
        epmc_sbt_list.append((id, i, 'epmc', j))
  epmc_df = DataFrame(epmc_sbt_list, columns=['ID_PAPER', 'ID_CORPUS', 'SOURCE', 'SUBSET_CODE'])
  print("%d papers found in EPMC"%(len(epmc_sbt_list)))
else:
  print('Skipping EPMC.')
  epmc_df = pd.DataFrame()

epmc_df

# COMMAND ----------

# MAGIC %md ## DataCite queries

# COMMAND ----------

from tqdm import tqdm
import requests
import os
import json
import re
import pprint
pp = pprint.PrettyPrinter(indent=4)

(corpus_ids, datacite_queries) = qt.generate_queries(QueryType.closed)
(subset_ids, datacite_subset_queries) = qt2.generate_queries(QueryType.closed)

def run_datacite_query(q, page_size=1000):   
  DATACITE_API_URL = 'https://api.datacite.org/dois?'
  url = DATACITE_API_URL + 'page[size]=1&query=' + q
  r = requests.get(url, timeout=10)
  data = json.loads(r.text)
  numFound = data.get('meta').get('total')
  #print(q + ', ' + str(numFound) + ' Datacite records found ')
  print('\n'+url+'\n'+str(numFound) + ' Datacite records found ')
  df = pd.DataFrame()
  for i in tqdm(range(0, numFound, page_size)):
    url = DATACITE_API_URL + 'page[size]='+str(page_size)+'&page[number]='+str(i)+'&query=' + q
    #print(url)
    r = requests.get(url)
    data = json.loads(r.text)
    df = df.append(pd.DataFrame.from_dict(data['data']))
    #for d in data['data']:
    #  records.add((d['attributes']['doi'], d['attributes'].get('types',{}).get('resourceType','')))
  #print(records)
  return (numFound, df)

datacite_include = 'True'
if datacite_include == 'True':
  rdf = pd.DataFrame()
  for (i, q) in enumerate(datacite_queries):
    for (j, sq) in zip(subset_ids, datacite_subset_queries):
      query = q
      if len(sq)>0:
        query = '(%s) AND (%s)'%(q,sq) 
      numFound, df = run_datacite_query(query)  
      rdf = rdf.append(df)
  print("%d papers found in DataCite"%(len(rdf)))
else:
  print('Skipping DataCite.')
  rdf = pd.DataFrame()

rdf['resourceType'] = [row.attributes['types'].get('resourceType','') for row in rdf.itertuples()]
rdf['title'] = [row.attributes['titles'][0].get('title','') for row in rdf.itertuples()]
rdf['description'] = ['\n'.join([desc['description'] for desc in row.attributes['descriptions']]) for row in rdf.itertuples()]
rdf['url'] = [row.attributes.get('url','') for row in rdf.itertuples()]

new_rdf = rdf.drop(columns=['attributes','relationships']).reset_index(drop=True)
new_rdf

# COMMAND ----------

# MAGIC %md ## CROSSREF Queries

# COMMAND ----------

from tqdm import tqdm
import requests
import os
import json
import re
import pprint
pp = pprint.PrettyPrinter(indent=4)

(corpus_ids, crossref_queries) = qt.generate_queries(QueryType.closed)
(subset_ids, crossref_subset_queries) = qt2.generate_queries(QueryType.closed)

def run_crossref_query(q, page_size=1000):   
  CROSSREF_API_URL = 'https://api.crossref.org/works??'
  url = CROSSREF_API_URL + 'page[size]=1&query=' + q
  r = requests.get(url, timeout=10)
  data = json.loads(r.text)
  numFound = data.get('meta').get('total')
  #print(q + ', ' + str(numFound) + ' Datacite records found ')
  print(str(numFound) + ' Datacite records found ')
  records = set()
  for i in tqdm(range(0, numFound, page_size)):
    url = DATACITE_API_URL + 'page[size]='+str(page_size)+'&page[number]='+str(i)+'&query=' + q
    r = requests.get(url)
    data = json.loads(r.text)
    for d in data['data']:
      records.add((d['attributes']['doi'], d['attributes'].get('types',{}).get('resourceType','')))
  print(records)
  return (numFound, list(records))

datacite_include = 'True'
if datacite_include == 'True':
  rlist = []
  for (i, q) in enumerate(datacite_queries):
    for (j, sq) in zip(subset_ids, datacite_subset_queries):
      query = q
      if len(sq)>0:
        query = '(%s) AND (%s)'%(q,sq) 
      numFound, records = run_datacite_query(query)  
      rlist.extend(records)
  print(rlist)
  rdf = DataFrame(rlist, columns=['DOI', 'SCHEMA.ORG'])
  print("%d papers found in DataCite"%(len(rdf)))
else:
  print('Skipping DataCite.')
  rdf = pd.DataFrame()

rdf

# COMMAND ----------

# MAGIC %md ## Add additional data to the landscape analysis

# COMMAND ----------

sbt_df = pubmed_df
sbt_df = sbt_df.append(solr_df)
sbt_df = sbt_df.append(epmc_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dashboard Table Creation within SNOWFLAKE. 
# MAGIC 
# MAGIC This section performs several subtasks: 
# MAGIC 1. Set up the RstContentDashboard instance
# MAGIC 1. Drop any previous instances of the tables in SNOWFLAKE - NOTE WE ALWAYS REBUILD THE WHOL
# MAGIC 1. Load data from Airtable to describe corpora to be mapped
# MAGIC 1. Rerun queries to build a subset of derived tables in SNOWFLAKE

# COMMAND ----------

if delete_database == 'True':
  mcdash.drop_database(get_dash_cursor(sf, mcdash))

# COMMAND ----------

cs = get_dash_cursor(sf, mcdash)

# COMMAND ----------

# HACK TO ADD ANOTHER SUBSET CODE TO THE DATABASE
#pubmed_df[pubmed_df.SUBSET_CODE==5]
#existing_data_df = execute_query(cs, 'SELECT * FROM RARE_CYCLE1_CORPUS_TO_PAPER;', ['ID_PAPER','ID_CORPUS','SOURCE','SUBSET_CODE'])
#data_to_upload = existing_data_df.append(pubmed_df[pubmed_df.SUBSET_CODE==5])
#table_name = re.sub('PREFIX_', mcdash.prefix, 'PREFIX_CORPUS_TO_PAPER')
#data_to_upload.to_csv(g_loc + '/' + table_name + '.tsv', index=False, header=False, sep='\t')
#print(g_loc + '/' + table_name + '.tsv')

# COMMAND ----------

df

# COMMAND ----------

cs.execute("BEGIN")
mcdash.upload_wb(cs, df, 'CORPUS')

# COMMAND ----------

table_name = re.sub('PREFIX_', mcdash.prefix, 'PREFIX_CORPUS_TO_PAPER')
sbt_df.to_csv(g_loc + table_name + '.tsv', index=False, header=False, sep='\t')
print(g_loc + table_name + '.tsv')
cs.execute('DROP TABLE IF EXISTS ' + table_name + ';')
cs.execute('CREATE TABLE ' + table_name + ' IF NOT EXISTS (ID_PAPER INT,ID_CORPUS INT,SOURCE TEXT, SUBSET_CODE INT);')
cs.execute('put file://' + g_loc + '/' + table_name + '.tsv' + ' @%' + table_name + ';')
cs.execute(
    "copy into " + table_name + " from @%" + table_name + " FILE_FORMAT=(TYPE=CSV FIELD_DELIMITER=\'\\t\')"
)  

# COMMAND ----------

cs = get_dash_cursor(sf, mcdash)
cs.execute("BEGIN")
if subsets_df is not None:
  table_name = re.sub('PREFIX_', mcdash.prefix, 'PREFIX_CORPUS_TO_PAPER')
  cs.execute('DROP TABLE IF EXISTS ' + table_name + ';')
  mcdash.upload_wb(cs, subsets_df, 'SUBSETS')

# COMMAND ----------

# UPLOAD PREFIX_CORPUS_TO_PAPER TO SNOWFLAKE
table_name = re.sub('PREFIX_', mcdash.prefix, 'PREFIX_CORPUS_TO_PAPER')
sbt_df.to_csv(g_loc + table_name + '.tsv', index=False, header=False, sep='\t')
cs.execute('CREATE TABLE ' + table_name + ' IF NOT EXISTS (ID_PAPER INT,ID_CORPUS INT,SOURCE TEXT, SUBSET_CODE INT);')
cs.execute('put file://' + g_loc + '/' + table_name + '.tsv' + ' @%' + table_name + ';')
cs.execute(
    "copy into " + table_name + " from @%" + table_name + " FILE_FORMAT=(TYPE=CSV FIELD_DELIMITER=\'\\t\')"
)  
mcdash.build_core_tables_from_pmids(cs)
cs.execute("COMMIT")
