# Databricks notebook source
#|default_exp drsm
from nbdev import *

# COMMAND ----------

# MAGIC %md # Disease Research State Model  
# MAGIC
# MAGIC > Classes and functions to execute functionality for generating and analyzing the state of research into one or many identified diseases

# COMMAND ----------

#|export

import activesoup
from bs4 import BeautifulSoup,Tag,Comment,NavigableString
import datetime
from enum import Enum
import json 
import matplotlib.pyplot as plt
import numpy
from owlready2 import *
import pandas as pd
from prophet.serialize import model_to_json, model_from_json
import re
import requests
from scipy.spatial.distance import cdist
import seaborn as sns
from time import time,sleep
from tqdm import tqdm
from urllib.request import urlopen
from urllib.parse import quote_plus, quote, unquote
from urllib.error import URLError

# COMMAND ----------

#|export

class DRSMCollection():
  """This class generates and supports analysis the research landscape over a collection of diseases. 
  """
  def __init__(self, study_name, corpora_df, name_col='CORPUS_NAME', mondo_col='MONDO_CURI', query_col='QUERY'):
    '''
    Initializes the DRSM Collection object.
    '''
    self.study_name = name
    self.corpora_df = corpora_df
    self.name_col = name_col
    self.query_col = query_col
    self.mondo_col = mondo_col
    
  def check_query_phrase(self, phrase):
    """
    Checks whether a single phrase would work on Pubmed or would be expanded (which can lead to unpredictable errors). 
    Use this as a check for synonyms.   
    """
    idPrefix = ''
    phrase = re.sub('"','',phrase)
    m1 = re.match('^[a-zA-Z0-9]{1,5}$', phrase)
    if m1 is not None:
      return False, phrase + ': Abbreviation', 0

    m2 = re.search('[(\)]', phrase)
    if m2 is not None:
      return False, phrase+': Brackets', 0

    m3 = re.search('[\,\;]', phrase)
    if m3 is not None:
      phrase = '("' + '" AND "'.join(re.split('[\,\;]', phrase.strip()))+'")'

    if self.api_key is not None: 
      esearch_stem = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?api_key='+self.api_key+'&db=' + self.db + '&term='
    else:
      esearch_stem = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db='+self.db + '&term='
    url =  esearch_stem + quote('"'+phrase+'"')

    #print(url)
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

  def build_query_tuples(self, df, check_threshold, name_col, terms_col, sep):
    '''
    '''
    query_tuples = []
    phrase_counts = []
    for ind in df.index:
      search_l = []
      terms_to_check = [df[name_col][ind]]
      terms_to_check.extend(df[terms_col][ind].split('|'))
      for s in terms_to_check: 
        go_no_go, phrase, count = check_query_phrase(esq, s.strip())
        print(go_no_go, phrase, count)
        if go_no_go:
          search_l.append(phrase)
          sleep(0.10)
          if count>check_threshold:
            phrase_counts.append((phrase, count))
      query_tuples.append( (ind, df[name_col][ind], ' OR '.join(search_l)) )
    return query_tuples, phrase_counts    


class DRSM():
  """This class provides a model of the state of research into a particular disease ('Disease Research State Model') based on broad subtypes of research article present in the literature. It makes use of functionality within the CZ Landscaping Toolkit to search online sources, classify the data it finds, and run analyses over that data. 
  
  This version is based on some assumptions: (1) data pertaining to a single disease is linked to an entry in a PREFIX_CORPUS table; (2) papers for that disease/corpus are indexed in the PREFIX_CORPUS_PAPERS table; (3) Codes denoting the type of each paper are stored in the PREFIX_DRSM table.  
  
  Note that the time series computation is also simply the difference between matched curves over the publishing timeframe of the analysis. 
  """
  
  def __init__(self, dashdb, corpus_id, name, mondo_id, event_lines=[]):
    '''
    Initializes the DRSM object.
    '''
    self.name = name
    self.dashdb = dashdb
    self.corpus_id = corpus_id
    self.mondo_id = mondo_id
    self.event_lines = event_lines
    
  def build_trend_dataset(self):
    '''
    Computes trend data from existing an underlying corpus of papers annotated for study categories 
    '''
    sql = '''SELECT DISTINCT count(DISTINCT p.id) AS paper_count, d.ID, p.YEAR, p.MONTH, drsm.DRSM_LABEL
          FROM PREFIX_CORPUS as d
              JOIN PREFIX_CORPUS_TO_PAPER as dp on (d.ID=dp.ID_CORPUS)
              JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p on (p.ID=dp.ID_PAPER)
              JOIN PREFIX_DRSM as drsm on (p.ID=drsm.ID_PAPER)
          WHERE d.ID='''+str(self.corpus_id)+'''
          GROUP BY drsm.DRSM_LABEL, YEAR, MONTH, d.ID
          ORDER BY drsm.DRSM_LABEL, YEAR, MONTH'''
    cols = ['paper_count', 'CORPUS_ID', 'YEAR', 'MONTH', 'DRSM_LABEL']
    sql = re.sub('PREFIX_', self.dashdb.prefix, sql)
    df = self.dashdb.execute_query(sql, cols)
    df = df.fillna(1).loc[df.YEAR>0]
    df['date'] = [datetime.date(int(row.YEAR), int(row.MONTH), 1).isoformat() for row in df.itertuples()]
    df = df.drop(columns=['YEAR', 'MONTH'])
    df = df.replace('irrelevant','reviews')
    df['date'] = df['date'].astype({'date': 'datetime64[ns]'})
    
    l = []
    for cat in df.DRSM_LABEL.unique():
      for d in pd.date_range(min(df['date']), max(df['date']), freq='MS'):
        idx = (df.DRSM_LABEL==cat) & (df.date==d)
        if any(idx):
          l.append((cat, d, df[idx].paper_count.values[0]))
        else:
          l.append((cat, d, 0))
    ts_df = pd.DataFrame(l, columns=['drsm','date','paper_count']) 
    self.raw_df = ts_df
    
    ts_piv_df = ts_df.pivot(index='date',columns='drsm', values='paper_count')
    for c in ['clinical characteristics or disease pathology', 'therapeutics in the clinic']:
      if c not in ts_piv_df.columns:
        ts_piv_df[c] = 0      
    ts_piv_df['clinical'] = ts_piv_df['clinical characteristics or disease pathology'] + ts_piv_df['therapeutics in the clinic']
    #ts_piv_df = ts_piv_df.set_index(pd.DatetimeIndex(ts_piv_df['date']))
    self.cols=['clinical', 'disease mechanism', 'patient-based therapeutics']
    ts_piv_df = ts_piv_df.drop(columns=[c for c in ts_piv_df.columns if c not in self.cols])
    
    prophet_models = []
    threshold = 0.01
    trends = {}
    changepoints = {}
    for i,c in enumerate(self.cols):
      if c not in ts_piv_df.columns:
        continue
      df1 = ts_piv_df.reset_index().rename(columns={'date':'ds', c:'y'}).drop(columns=[cc for cc in self.cols if cc!=c and cc in ts_piv_df.columns])  
      model = Prophet(seasonality_mode='additive', changepoint_range=0.99)
      model.fit(df1)
      future = model.make_future_dataframe(periods=12 * 3, freq='MS')
      forecast = model.predict(future)
      if trends.get('ds') is None: 
        trends['ds'] = forecast['ds']
      trends[c] = forecast['trend']
      cps = model.changepoints[ # Note - derived from how changepoints are computed in Prophet
            np.abs(np.nanmean(model.params['delta'], axis=0)) >= threshold
        ] if len(model.changepoints) > 0 else [] 
      changepoints[c] = [c for c in cps]
      prophet_models.append(json.loads(model_to_json(model)))
    
    self.prophet_models = prophet_models
    self.trends_df = pd.DataFrame(trends)
    self.changepoints = changepoints

  def plot_raw(self, w=10, h=5):
    '''
    Plots a line graph of raw monthly publication counts within the corpus for each study category 
    '''
    ig, ax = plt.subplots()
    plt.rcParams["figure.figsize"] = [w, 5]
    plt.rcParams["figure.autolayout"] = True
    ax = sns.lineplot (x = "date", y = "paper_count", data = self.raw_df, hue='drsm')
    ax.tick_params (rotation = 60)
    plt.show()
    
  def plot_prophet_models(self):
    '''
    Shows full plots of the Prophet models for each study category
    '''
    for i,c in enumerate(self.cols):
      model = model_from_json(json.dumps(self.prophet_models[i]))
      future = model.make_future_dataframe(periods=12 * 3, freq='MS')
      forecast = model.predict(future)
      fig = model.plot(forecast)
      add_changepoints_to_plot(fig.gca(), model, forecast)
      plt.title(c)
      plt.figure(i)
      plt.show()
      
  def plot_trends(self):
    long_df = self.trends_df.melt(id_vars=['ds'], value_vars=self.cols, var_name='drsm', value_name='monthly_publication_count')
    sns.set_theme(style="darkgrid")
    g = sns.lineplot(data=long_df, x="ds", y="monthly_publication_count", hue="drsm")
    g.set(title=self.name)
    if len(self.event_lines) > 0:
      for rl in self.event_lines:
        g.axvline(rl, color="red")

  def compute_history_euclidean_distance(self, that):
    if isinstance(that, DRSM) is False:
      raise Exception("Can only complare DRSM instances, not "+type(that))
    d = []
    for i,c in enumerate(self.cols):
      s_y1 = self.trends_df[c].to_numpy()
      cp1 = self.get_index_of_first_changepoint(c)
      l1 = s_y1.shape[0]
      s_y2 = that.trends_df[c].to_numpy()
      cp2 = that.get_index_of_first_changepoint(c)
      l2 = s_y2.shape[0]
      if l2 > l1:
        s_y1 = numpy.insert(s_y1, 0, [s_y1[0]] * (l2-l1))
        cp1 += l2-l1
      if l1 > l2:
        s_y2 = numpy.insert(s_y2, 0, [s_y2[0]] * (l1-l2))  
        cp2 += l1-l2
      denominator = s_y1.shape[0] - min(cp1, cp2)
      d.append(np.sqrt(sum(s_y1*s_y1 + s_y2*s_y2)) / denominator)  
    return d
  
  def get_index_of_first_changepoint(self, c):
    date = self.changepoints[c][0]
    index = self.trends_df[self.trends_df.ds == date].index[0]
    return index
  
  def scrape_fda_website_for_drug_approvals(self):
    d = activesoup.Driver()
    page = d.get("https://www.accessdata.fda.gov/scripts/opdlisting/oopd/")
    form1, form2 = page.find_all('form')
    r = form2.submit({"Designation": '%'+self.name+'%',
                    "Designation_Start_Date": "01/01/1983",
                    "Designation_End_Date": "07/25/2022", 
                    "Search_param": "DESDATE",
                    "Output_Format": "Short",
                    "Sort_order": "GENERIC_NAME",
                    "RecordsPerPage": 25})
    bsoup = BeautifulSoup(r._raw_response.text, "html.parser")
    if bsoup.find('th') is None:
      return None, None
    df = pd.read_html(str(bsoup.find('th').parent.parent), extract_links="all")[0]
    designations_df = pd.read_html(str(bsoup.find('th').parent.parent))[0]
    approvals_dflist = []
    for row in df.itertuples():
      name, url = row[2]
      designation, nuttin = row[5]
      if 'Approved' in designation:
        print(name)
        m = re.match('.*\((\d+)\).*', url)
        if(m):
          id = m.group(1)
          print()
          d = activesoup.Driver()
          r2 = d.get('https://www.accessdata.fda.gov/scripts/opdlisting/oopd/detailedIndex.cfm?cfgridkey='+id)
          bsoup2 = BeautifulSoup(r2._raw_response.text, "html.parser")
          table_list = bsoup2.find_all("table", {"class": "resultstable"})
          for j,t in enumerate(table_list):
            if j==0:
              continue
            df2 = pd.read_html(str(t))[0]
            # important data is generic name (r0c2) / trade name (r1c2) / approval date (r2c2) / Approved Labeled Indication (r3c2)
            generic_name = df2.iloc[0,2]
            trade_name = df2.iloc[1,2]
            approval_date = df2.iloc[2,2]
            approved_labeled_indication = df2.iloc[3,2]
            approvals_dflist.append((generic_name, trade_name, approval_date, approved_labeled_indication))
    approvals_df =  pd.DataFrame(approvals_dflist, columns=['generic_name', 'trade_name', 'approval_date', 'approved_labeled_indication'])
    return designations_df, approvals_df
  
  
