# Databricks notebook source
# MAGIC %md # Search Engine Tools  
# MAGIC
# MAGIC > A library of classes that provide query access to a number of online academic search services including (Pubmed, PMC, and European PMC). 

# COMMAND ----------

#| default_exp searchEngineUtils

# COMMAND ----------

#| hide
from nbdev import *

# COMMAND ----------

# MAGIC %md ## NCBI Tools

# COMMAND ----------

#|export
import requests
import json
import datetime
from enum import Enum
from urllib.request import urlopen
from urllib.parse import quote_plus, quote, unquote
from urllib.error import URLError
from time import time,sleep
import re
from tqdm import tqdm
from bs4 import BeautifulSoup,Tag,Comment,NavigableString
import pandas as pd

# COMMAND ----------

#|export

PAGE_SIZE = 10000
TIME_THRESHOLD = 0.3333334

class NCBI_Database_Type(Enum):
  """
  Simple enumeration of the different NCBI databases supported by this tool
  """
  pubmed = 'pubmed'
  PMC = 'PMC'

class ESearchQuery:
  """
  Class to provide query interface for ESearch (i.e., query terms in elaborate ways, return a list of ids)
  Each instance of this class executes queries of a given type
  """

  def __init__(self, api_key=None, oa=False, db='pubmed'):
    """
    :param api_key: API Key for NCBI EUtil Services 
    :param oa: Is this query searching for open access papers? 
    :param db: The database being queried
    """
    self.api_key = api_key
    self.idPrefix = ''
    self.oa = oa
    self.db = db
  
  def build_query_tuples(self, df, check_threshold, name_col, terms_col, sep):
    '''
    Given a dataframe defining a set of 'OR' queries (i.e., entirely broken into | clauses), 
    check each indvidual term in Pubmed and return complete queries with problematic terms 
    stripped.
    '''
    query_tuples = []
    phrase_counts = []
    for ind in df.index:
      search_l = []
      terms_to_check = [df[name_col][ind]]
      terms_to_check.extend(df[terms_col][ind].split('|'))
      for s in terms_to_check: 
        go_no_go, phrase, count = _check_query_phrase(self, s.strip())
        print(go_no_go, phrase, count)
        if go_no_go:
          search_l.append(phrase)
          sleep(0.10)
          if count>check_threshold:
            phrase_counts.append((phrase, count))
      query_tuples.append( (ind, df[name_col][ind], ' OR '.join(search_l)) )
    return query_tuples, phrase_counts
  
  def _check_query_phrase(self, phrase):
    """
    Checks whether a phrase would work on Pubmed or would be expanded (which can lead to unpredictable errors). 
    Use this as a check for synonyms.   
    """
    idPrefix = ''
    m1 = re.match('^[a-zA-Z0-9]{1,5}$', phrase)
    if m1 is not None:
      return False, 'Abbreviation', 0

    m2 = re.search('[(\)]', phrase)
    if m2 is not None:
      return False, 'Brackets', 0

    m3 = re.search('[\,\;]', phrase)
    if m3 is not None:
      phrase = '("' + '" AND "'.join(re.split('[\,\;]', phrase.strip()))+'")'

    if self.api_key is not None: 
      esearch_stem = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?api_key='+self.api_key+'&db=' + self.db + '&term='
    else:
      esearch_stem = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db='+self.db + '&term='
    url =  esearch_stem + quote('"'+phrase+'"')

    esearch_response = urlopen(url)
    esearch_data = esearch_response.read().decode('utf-8')
    esearch_soup = BeautifulSoup(esearch_data, "lxml-xml")
    count = int(esearch_soup.find('Count').string)
    #n_translations = len(esearch_soup.find('TranslationStack').findAll('TermSet'))
    phrase_not_found = esearch_soup.find('PhraseNotFound')
    quoted_phrase_not_found = esearch_soup.find('QuotedPhraseNotFound')
    if phrase_not_found is not None or quoted_phrase_not_found is not None:
      return False, '"'+phrase+'" not found', 0
    if count == 0:
      return False, phrase, count
    return True, phrase, count        

  def execute_count_query(self, query):
    """
    Executes a query on the target database and returns a count of papers 
    """
    idPrefix = ''
    if self.oa:
      if self.db == NCBI_Database_Type.PMC:
        query = '"open access"[filter] AND (' + query + ')'
        idPrefix = 'PMC'
      elif self.db == NCBI_Database_Type.pubmed:
        query = '"loattrfree full text"[sb] AND (' + query + ')'
    if self.api_key: 
      esearch_stem = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?api_key='+self.api_key+'&db=' + self.db + '&term='
    else:
      esearch_stem = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db='+self.db + '&term='
    query = quote_plus(query)
    esearch_response = urlopen(esearch_stem + query)
    esearch_data = esearch_response.read().decode('utf-8')
    esearch_soup = BeautifulSoup(esearch_data, "lxml-xml")
    count_tag = esearch_soup.find('Count')
    if count_tag is None:
      raise Exception('No Data returned from "' + self.query + '"')
    return int(count_tag.string)

  def execute_query_on_website(self, q, pm_order='relevance'):
    """
    Executes a query on the Pubmed database and returns papers in order of relevance or date.
    This is important to determine accuracy of complex queries are based on the composition of the first page of results.  
    """
    query = 'https://pubmed.ncbi.nlm.nih.gov/?format=pmid&size=10&term='+re.sub('\s+','+',q)
    if pm_order == 'date':
      query += '&sort=date'
    response = urlopen(query)
    data = response.read().decode('utf-8')
    soup = BeautifulSoup(data, "lxml-xml")
    pmids = re.split('\s+', soup.find('body').text.strip())
    return pmids

  def execute_query(self, query):
    """
    Executes a query on the eutils service and returns data as a Pandas Dataframe
    """
    idPrefix = ''
    if self.oa:
      if self.db == NCBI_Database_Type.PMC:
        query = '"open access"[filter] AND (' + query + ')'
        idPrefix = 'PMC'
      elif self.db == NCBI_Database_Type.pubmed:
        query = '"loattrfree full text"[sb] AND (' + query + ')'

    if self.api_key: 
      esearch_stem = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?api_key='+self.api_key+'&db=' + self.db + '&term='
    else: 
      esearch_stem = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=' + self.db + '&term='

    query = quote_plus(query)
    print(esearch_stem + query)
    esearch_response = urlopen(esearch_stem + query)
    esearch_data = esearch_response.read().decode('utf-8')
    esearch_soup = BeautifulSoup(esearch_data, "lxml-xml")
    count_tag = esearch_soup.find('Count')
    if count_tag is None:
        raise Exception('No Data returned from "' + query + '"')
    count = int(count_tag.string)

    latest_time = time()

    ids = []
    for i in tqdm(range(0,count,PAGE_SIZE)):
      full_query = esearch_stem + query + '&retstart=' + str(i)+ '&retmax=' + str(PAGE_SIZE)
      esearch_response = urlopen(full_query)
      esearch_data = esearch_response.read().decode('utf-8')
      esearch_soup = BeautifulSoup(esearch_data, "lxml-xml")
      for pmid_tag in esearch_soup.find_all('Id') :
        ids.append(self.idPrefix + pmid_tag.text)
      delta_time = time() - latest_time
      if delta_time < TIME_THRESHOLD :
        sleep(TIME_THRESHOLD - delta_time)
        
    return ids

# COMMAND ----------

#|export
class EFetchQuery:
    """
    Class to provide query interface for EFetch (i.e., query based on a list of ids)
    Each instance of this class executes queries of a given type
    """

    def __init__(self, api_key=None, db='pubmed'):
        """
        :param api_key: API Key for NCBI EUtil Services 
        :param oa: Is this query searching for open access papers? 
        :param db: The database being queried
        """
        self.api_key = api_key
        self.db = db

    def execute_efetch(self, id):
        """
        Executes a query for a single specific identifier
        """
        if self.api_key:
          efetch_stem = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?api_key='+self.api_key+'&db=pubmed&retmode=xml&id='
        else:
          efetch_stem = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id='
        efetch_response = urlopen(efetch_stem + str(id))
        return self._generate_rows_from_medline_records(efetch_response.read().decode('utf-8'))

    def generate_data_frame_from_id_list(self, id_list):
        """
        Executes a query for a list of ID values
        """
        if self.api_key:
          efetch_stem = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?api_key='+self.api_key+'&db=pubmed&retmode=xml&id='
        else:
          efetch_stem = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id='
        page_size = 100
        i = 0
        url = efetch_stem
        efetch_df = pd.DataFrame()
        for line in tqdm(id_list):
            try:
                l = re.split('\s+', str(line))
                pmid = l[0]
                i += 1
                if i >= page_size:
                    efetch_response = urlopen(url)
                    df = self._generate_rows_from_medline_records(efetch_response.read().decode('utf-8'))
                    efetch_df = efetch_df.append(df)
                    url = efetch_stem
                    i = 0

                if re.search('\d$', url):
                    url += ','
                url += pmid.strip()
            except URLError as e:
                sleep(10)
                print("URLError({0}): {1}".format(e.errno, e.strerror))
            except TypeError as e2:
                pause = 1

        if url != efetch_stem:
            efetch_response = urlopen(url)
            df = self._generate_rows_from_medline_records(efetch_response.read().decode('utf-8'))
            efetch_df = efetch_df.append(df)
        return efetch_df

    def generate_mesh_data_frame_from_id_list(self, id_list):
        """
        Executes a query for MeSH data from a list of ID values
        """
        url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
        if self.api_key:
          payload = 'api_key='+self.api_key+'&db=pubmed&retmode=xml&id='
        else:
          payload = 'db=pubmed&retmode=xml&id='
          
        headers = {'content-type': 'application/xml'}

        page_size = 10000
        i = 0
        efetch_df = pd.DataFrame()
        for line in tqdm(id_list):
            try:
                l = re.split('\s+', line)
                pmid = l[0]
                i += 1
                if i >= page_size:
                    print('   running query')
                    r = requests.post(url, data=payload)
                    efetch_data = r.content.decode('utf-8')
                    df = self._generate_mesh_rows_from_medline_records(efetch_data)
                    efetch_df = efetch_df.append(df)
                    payload = 'db=pubmed&retmode=xml&id='
                    i = 0

                if re.search('\d$', payload):
                    payload += ','
                payload += pmid.strip()
            except URLError as e:
                sleep(10)
                print("URLError({0}): {1}".format(e.errno, e.strerror))

        if payload[-1] != '=':
            r = requests.post(url, data=payload)
            efetch_data = r.content.decode('utf-8')
            df = self._generate_mesh_rows_from_medline_records(efetch_data)
            efetch_df = efetch_df.append(df)

        return efetch_df

    def _generate_mesh_rows_from_medline_records(self, record):

        soup2 = BeautifulSoup(record, "lxml-xml")

        rows = []
        cols = ['PMID','MESH']
        for citation_tag in tqdm(soup2.find_all('MedlineCitation')):

            pmid_tag = citation_tag.find('PMID')

            mesh_labels = []

            if pmid_tag is None:
                continue

            for meshTag in citation_tag.findAll('MeshHeading'):
                desc = meshTag.find('DescriptorName')
                qual_list = meshTag.findAll('QualifierName')
                if len(qual_list)>0:
                    mesh_labels.append('%s/%s'%(desc.text.replace('\n', ' '),'/'.join(q.text.replace('\n', ' ') for q in qual_list)))
                else:
                    mesh_labels.append(desc.text.replace('\n', ' '))
            mesh_data = ",".join(mesh_labels)

            rows.append((pmid_tag.text, mesh_data))

        df = pd.DataFrame(data=rows, columns=cols)
        return df

    def _generate_rows_from_medline_records(self, record):

        soup2 = BeautifulSoup(record, "lxml-xml")

        rows = []
        cols = ['PMID', 'YEAR', 'PUBLICATION_TYPE', 'TITLE', 'ABSTRACT','MESH','KEYWORDS']
        for citation_tag in soup2.find_all('MedlineCitation'):

            pmid_tag = citation_tag.find('PMID')
            title_tag = citation_tag.find('ArticleTitle')
            abstract_txt = ''
            for x in citation_tag.findAll('AbstractText'):
                if x.label is not None:
                    abstract_txt += ' ' + x.label + ': '
                abstract_txt += x.text

            mesh_labels = []
            for meshTag in citation_tag.findAll('MeshHeading'):
                desc = meshTag.find('DescriptorName')
                qual_list = meshTag.findAll('QualifierName')
                if len(qual_list)>0:
                    mesh_labels.append('%s/%s'%(desc.text.replace('\n', ' '),'/'.join(q.text.replace('\n', ' ') for q in qual_list)))
                else:
                    mesh_labels.append(desc.text.replace('\n', ' '))
            mesh_data = ",".join(mesh_labels)

            if pmid_tag is None or title_tag is None or abstract_txt == '':
                continue

            year_tag = citation_tag.find('PubDate').find('Year')

            year = ''
            if year_tag is not None:
                year = year_tag.text

            is_review = '|'.join([x.text for x in citation_tag.findAll('PublicationType')])

            keyword_data = '|'.join([meshTag.text.replace('\n', ' ') for meshTag in citation_tag.findAll('Keyword')])

            rows.append((pmid_tag.text, year, is_review, title_tag.text, abstract_txt, mesh_data, keyword_data))

        df = pd.DataFrame(data=rows, columns=cols)
        return df


# COMMAND ----------

# MAGIC %md ## EuroPMCQuery

# COMMAND ----------

#|export
class EuroPMCQuery():
    """
    A class that executes search queries on the European PMC API 
    """
    def __init__(self, oa=False, db='pubmed'):
        """
        Initialization of the class
        :param oa:
        """
        self.oa = oa

    def run_empc_query(self, q, page_size=1000, timeout=60, extra_columns=[]):
        EMPC_API_URL = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=JSON&pageSize='+str(page_size)+'&synonym=TRUE'
        if len(extra_columns)>0:
            EMPC_API_URL += '&resultType=core'
        url = EMPC_API_URL + '&query=' + q
        r = requests.get(url, timeout=timeout)
        data = json.loads(r.text)
        numFound = data['hitCount']
        print(url + ', ' + str(numFound) + ' European PMC PAPERS FOUND')
        ids_from_q = []
        cursorMark = '*'
        for i in tqdm(range(0, numFound, page_size)):
            url = EMPC_API_URL + '&cursorMark=' + cursorMark + '&query=' + q
            r = requests.get(url, timeout=timeout)
            data = json.loads(r.text)
            #print(data)
            if data.get('nextCursorMark'):
                cursorMark = data['nextCursorMark']
            for d in data['resultList']['result']:
                if d.get('pubType','') == 'patent':
                    continue
                tup = [d.get('id',-1), d.get('doi','')]
                for c in extra_columns:
                    tup.append(d.get(c,'')) 
                ids_from_q.append(tup)
        ids_from_q = sorted(list(ids_from_q), key = lambda x: x[0])
        print(' Returning '+str(len(ids_from_q)))
        return (numFound, ids_from_q)
