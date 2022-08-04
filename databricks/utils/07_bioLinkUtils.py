# Databricks notebook source
# default_exp bioLinkUtils
from nbdev import *

# COMMAND ----------

# MAGIC %md # BioLink Query Tools 
# MAGIC 
# MAGIC > Tools to query and analyze data from the Monarch Initiative's BioLink interface. This provides a live queryable interface for disease-based knowledge derived from Monarch's KG. Access to the service is through [https://api.monarchinitiative.org/api/](https://api.monarchinitiative.org/api/). More detailed information about the Biolink model is available from their [GitHub page](https://github.com/biolink/biolink-model). 

# COMMAND ----------

parent_sparql = '''PREFIX obo: <http://purl.obolibrary.org/obo/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
SELECT DISTINCT ?parent_id ?parent_name
WHERE {
	?mondo_id rdf:type owl:Class .
    ?mondo_id rdfs:label ?name .
    ?mondo_id rdfs:subClassOf ?parent_id .
    ?parent_id rdfs:label ?parent_name .
}'''

child_sparql = '''PREFIX obo: <http://purl.obolibrary.org/obo/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
SELECT DISTINCT ?child_id ?child_name
WHERE {
	?mondo_id rdf:type owl:Class .
    ?mondo_id rdfs:label ?name .
    ?child_id rdfs:subClassOf ?mondo_id .
    ?child_id rdfs:label ?child_name .
}'''

siblings_sparql = '''PREFIX obo: <http://purl.obolibrary.org/obo/>
SELECT DISTINCT ?d2 ?d2Name
WHERE {
  ?mondo_id rdf:type owl:Class .
  ?mondo_id rdfs:label ?mondo_Name .
  ?mondo_id rdfs:subClassOf ?p .
  ?d2 rdf:type owl:Class .
  ?d2 rdfs:label ?d2Name .
  ?d2 rdfs:subClassOf ?p .
}'''

descendents_sparql = '''
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
SELECT DISTINCT ?name ?parent_id ?parent_name ?descendent_id ?descendent_name
WHERE {
    ?mondo_id rdfs:label ?name .
    ?descendent_id rdfs:subClassOf+ ?mondo_id .
    ?descendent_id rdfs:label ?descendent_name .
    ?descendent_id rdfs:subClassOf ?parent_id .
	?parent_id rdfs:label ?parent_name .
} '''

# COMMAND ----------

#export

import json
import requests
import pandas as pd
from owlready2 import *
from tqdm import tqdm
import os
import urllib.request 

MONDO_LATEST_URL = 'https://github.com/monarch-initiative/mondo/releases/download/v2022-08-01/mondo.owl'

class BioLinkUtils: 
  descendents_lookup = {}
  '''
  Interactions with the BioLink KG developed by the Monarch initiaive. 
  Currently, this system can query for MONDO disease Ids and compute similar diseases based on phenotype overlap. 
  '''
  def __init__(self, local_files = None):
    self.local_files = local_files
    if local_files is not None:
      self.mondo_path = local_files+'/mondo.owl'
      if os.path.exists(local_files+'/mondo.owl') is False:
        print('Downloading latest version of MONDO')
        urllib.request.urlretrieve(MONDO_LATEST_URL, self.mondo_path)
    else:
      self.mondo_path = None
      
  def build_descendents_lookup(self, disease_ids): 
    mondo = get_ontology(self.mondo_path).load()
    descendents_df = self._run_substituted_sparql_over_mondo_ids(descendents_sparql, mondo_ids)
    self.descendents_lookup = {}
    for row in descendents_df.itertuples():
      d = row.descendent_id[-13:].replace('_', ':')
      m = row.mondo_id[-13:].replace('_', ':')
      if descendents_lookup.get(m) is None:
        self.descendents_lookup[m] = [d]
      else:
        self.descendents_lookup.get(m).append(d)

  def _run_substituted_sparql_over_mondo_ids(self, sparql, mondo_ids):
    df = pd.DataFrame()
    for mondo_id in mondo_ids:
      print(mondo_id)
      ldf = self._run_substituted_mondo_sparql(sparql, mondo_id)
      df = df.append(ldf)
    return df

  def _run_substituted_mondo_sparql(self, sparql, mondo_id):
    if mondo_id == 'obo:':
      return pd.DataFrame()
    obo = get_namespace("http://purl.obolibrary.org/obo/")
    sparql = re.sub('\\?mondo_id', mondo_id, sparql)
    m = re.search('SELECT DISTINCT (.*)\n', sparql)
    if m is not None:
      col_headings = m.group(1).split(' ')
    else:
      raise Exception("Can't read column headings in " + sparql )  
    l = [i for i in default_world.sparql(sparql)]
    df = pd.DataFrame(l, columns=col_headings)
    return df
  
  def query_diseases(self, disease_ids):
    BIOLINK_STEM = "https://api.monarchinitiative.org/api/bioentity/"
    recs = []
    for id in disease_ids:
      url = BIOLINK_STEM + 'disease/'+id 
      print(url)
      r = requests.get(url)
      d = r.content.decode('utf-8')
      recs.append(json.loads(d))
    return recs

  def compute_disease_similarity_across_disease_list(self, disease_ids, disease_names, metric='phenodigm', taxon=9606, limit=50, threshold=0.7):
    '''
    Iterates over a set of MONDO URIs to identify similar diseases based on phenotypic overlap.  
    '''
    df = pd.DataFrame()
    for (d_id, d_name) in zip(disease_ids, disease_names):
      m = re.match("^(MONDO\:\d{7})", d_id)
      if m is not None:
        df = df.append(self.compute_disease_similarity(m.group(1), d_name, metric=metric, taxon=taxon, limit=limit, threshold=threshold))
      else:
        print(d_id)
    return df  
  
  def compute_disease_similarity(self, disease_id, disease_name, descendents_df=None, metric='phenodigm', taxon=9606, limit=50, threshold=0.7):
    '''
    Computes similar diesases (with scores) for a single MONDO URI based on a phenotypic overlap metric (e.g., phenodigm). 
    Analysis is performed remotely. 
    '''
   
    BIOLINK_STEM = "https://api.monarchinitiative.org/api/sim/search?is_feature_set=false&"
    url = BIOLINK_STEM + 'metric='+metric+'&id='+disease_id+'&limit=100&taxon='+str(taxon)
    r = requests.get(url)
    d = r.content.decode('utf-8')
    sim_data = json.loads(d)
    l = []
    print(disease_name + ': ' + str(len(sim_data.get('matches'))))
    for match in sim_data.get('matches'):
      t_id = match.get('id')
      if self.descendents_lookup.get(disease_id) is not None and t_id in self.descendents_lookup.get(disease_id):
        continue
      if t_id != disease_id and \
          match.get('score') > threshold and \
          len(l)<limit:
        pl = [(pm.get('reference').get('IC'), pm.get('reference').get('id'), pm.get('reference').get('label')) for pm in match.get('pairwise_match')]
        l.append((disease_id, disease_name, match.get('type'), match.get('rank'), match.get('score'), match.get('label'), match.get('id'), pl))
    df = pd.DataFrame(l, columns=['source_disease_id','source_disease_name','type','rank','score','label','target_id', 'match'])
    return df 
