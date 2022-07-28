# Databricks notebook source
# default_exp queryTranslator
from nbdev import *

# COMMAND ----------

# MAGIC %md # Query Translation Tools  
# MAGIC 
# MAGIC > A library permits translation of complex boolean AND/OR queries between online APIs. 

# COMMAND ----------

#export
# USE PYEDA TO PROCESS AND REPURPOSE QUERIES AS LOGICAL EXPRESSIONS FOR SEARCHING.
import re
import pprint
from pyeda.inter import *
from pyeda.boolalg.expr import Literal,AndOp,OrOp
from enum import Enum
import unicodedata
from tqdm import tqdm

class QueryType(Enum):
  """
  An enumeration that permits conversion of complex boolean queries to different formats
  """
  open = 1
  closed = 2
  solr = 3
  epmc = 4
  pubmed = 5
  andPlusOrPipe = 6
  pubmed_no_types = 7

class QueryTranslator(): 
  def __init__(self, df, id_col, query_col):
    """This class allows a user to define a set of logical boolean queries in a Pandas dataframe and then convert them to a variety of formats for use on various online API systems.<BR>
    Functionality includes:
      * Specify queries as a table using '|' and '&' symbols
      * generate search strings to be used in API calls for PMID, SOLR, and European PMC

    Attributes:
      * df: The dataframe of queries to be processed (note: this dataframe must have a numerical ID column specified)
      * query_col: the column in the data frame where the query is specified
    """
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
      row_id = row[1][id_col]
      redq = fix_errors(tt.strip())
      for t in ordered_names:
        id = self.terms2id[t]
        redq = re.sub('\\b'+t+'\\b', id, redq)
      self.redq_list.append((row_id, redq))

  def generate_queries(self, query_type:QueryType):
    """
    Use this command to covert the queries to the different forms specified by the QueryType enumeration
    """
    queries = []
    IDs = []
    for ID, t in self.redq_list:
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
        return '("%s")[tiab])'%(t)
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

show_doc(QueryTranslator.__init__)

# COMMAND ----------

show_doc(QueryTranslator.generate_queries)
