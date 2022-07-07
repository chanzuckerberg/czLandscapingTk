# Databricks notebook source
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
        return '"%s"[tiab])'%(t)
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


# COMMAND ----------

# MAGIC %md ## Pubmed Queries

# COMMAND ----------

import pandas as pd
from tqdm import tqdm

df = pd.DataFrame([{'ID':0,'query':"Primary Ciliary Dyskinesia|Kartenger\'s Syndrome"}, {'ID':1,'query':'Alzheimer\'s Disease & machine learning'}])
qt = QueryTranslator(df, "query")
print(qt.id2terms)
print(qt.terms2id)
(corpus_ids, pubmed_queries) = qt.generate_queries(QueryType.pubmed)
pubmed_queries

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
