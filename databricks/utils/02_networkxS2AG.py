# Databricks notebook source
# default_exp kolsOnS2AG
from nbdev import *

# COMMAND ----------

# MAGIC %md # Key Opinion Leader Analysis
# MAGIC 
# MAGIC > API Details here
# MAGIC 
# MAGIC System to try to unpack key opinion leaders + close associates / prominent researchers based on co-authorship and co-citation within the same network of papers. 
# MAGIC 
# MAGIC 1. List key opinion leaders (KOL)
# MAGIC 2. Disambiguate each KOL based on clustering within bioinformatics + machine learning field
# MAGIC 3. List co-authors + referenced authors + citing authors
# MAGIC 4. Build paper / author networks 
# MAGIC 5. Derive co-citation graphs of authors
# MAGIC 6. Perform Author-based Eigenfactor on citation networks to list most influential authors
# MAGIC 7. Provide links to the KOL that generated the link to each influential author

# COMMAND ----------

#export

import pandas as pd
from datetime import datetime
import requests
import json
import os.path
from urllib.parse import quote_plus
import re
from tqdm import tqdm
import networkx as nx
from networkx.algorithms import bipartite
import pickle 
from tqdm import tqdm
from datetime import datetime
from collections import deque
import numpy as np
from scipy.sparse import dok_matrix
from scipy import linalg

class KOLsOnS2AG:
  """This class permits the construction of a local NetworkX graph that copies the basic organization of S2AG data.<BR>
  Functionality includes:
    * query all papers, references, and cited papers of a single individual
    * build an author-to-author citation graph
    * run eigenfactor analysis over the author graph
  
  Attributes:
    * x_api_key: an API key obtained from Semantic Scholar (https://www.semanticscholar.org/product/api)
    * author_stem_url, paper_stem_url: urls for API endpoints in S2AG
    * g: the networkx graph representing the citation / authorship network 
    * added_papers: papers that have had all citations and references added to graph
  """
  
  def __init__(self, x_api_key):
    """ Initialize the interface with an API key. 
    """
    self.author_search_url = 'https://api.semanticscholar.org/graph/v1/author/search'
    self.author_stem_url = 'https://api.semanticscholar.org/graph/v1/author/'
    self.paper_stem_url = 'https://api.semanticscholar.org/graph/v1/paper/'
    self.x_api_key = x_api_key
    self.added_papers = set()
    self.g = nx.DiGraph()

  def print_basic_stats(self):
    """ Prints out basic stats about the current graph stored in memory.
    """
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print("# Papers: %d"%(len(self.search_nodes('paper'))))
    print("# Authors: %d"%(len(self.search_nodes('author'))))
    print("# Authorship: %d"%(len(self.search_edges('wrote'))))
    print("# Citations: %d"%(len(self.search_edges('cites'))))
    infCites = [(e1, e2) for e1,e2,attrs in self.search_edges('cites') if attrs.get('isInfluential')]
    print("# Influential Citations: %d"%(len(infCites)))
    print("SCC: %d"%(nx.number_strongly_connected_components(self.g)))
    print("WCC: %d"%(nx.number_weakly_connected_components(self.g)))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

  # ~~~~~~~~~~ HIGH-LEVEL API ~~~~~~~~~~ 
  def search_for_disambiguated_author(self, author_name):
    """ Searches for an author and returns 10 most influential papers for each disambiguated example. 
    """
    search_url = self.author_search_url+'?query='+author_name+'&fields=name,paperCount,hIndex,papers.paperId,papers.title,papers.influentialCitationCount'
    r = requests.get(search_url, headers={"x-api-key":self.x_api_key})   
    author_data = json.loads(r.content.decode(r.encoding))
    n_found = author_data.get('total')
    df2 = pd.DataFrame(author_data.get('data'))
    #df2 = df2[df2.paperCount > n_papers_threshold]
    paper_titles = []
    for row2 in df2.itertuples():
      p_text_list = '     |     '.join([p.get('title') for p in sorted(row2.papers, key=lambda d: d['influentialCitationCount'], reverse=True)[:10]])
      paper_titles.append(p_text_list)
      #print(json.dumps(p, indent=4, sort_keys=True))  
      #  paperIds.add(p.get('paperId'))
    #df2 = df2.drop(columns=['papers'])
    df2['Top 10 Pubs'] = paper_titles
    return df2

  def build_author_citation_graph(self, authorId, pkl_file=None):
    """This builds a complete graph for a given individual based on 
    their papers, and 'highly influential' references + citations of those papers.
    See https://www.nature.com/articles/547032a. The system will then fill in 
    all cross-references of papers in this graph that cite each other. 
    """
    # Build the S2 author / paper graph
    self.addKeyOpinionLeader(authorId)
    inf_edges = [(n1,n2) for n1,n2,attrs in self.search_edges('cites') if attrs.get('isInfluential')]
    inf_papers = set([n1 for n1,n2 in inf_edges]).union(set([n2 for n1,n2 in inf_edges]))
    g2 = self.get_influential_graph()
    self.g = g2
    self.addCitationsOrReferencesToGraph(inf_papers, 'references', True, pkl_file)
    for p in inf_papers:
      self.added_papers.add(p)
  
  def run_thresholded_centrality_analysis(self, min_pub_count=3, top_n=100 ):
    """The system will analyse all authors within the graph that have total 
    number of publications above `min_pub_count` by peforming an author-based
    eigenfactor calculation (see [West et al 2013](https://jevinwest.org/papers/West2013JASIST.pdf)) 
    and then return a pandas data fram of the `top_n` most central authors 
    in the graph.
    """
    thresholded_authors, counts  = self.threshold_authors_by_pubcount(min_pub_count)
    author_eigfacs_df = self.compute_author_eigenfactors(thresholded_authors, verbose=True)
    top_n_df = author_eigfacs_df.sort_values('f',ascending=False)[0:top_n]
    authorIds = [row.id for row in top_n_df.itertuples()]
    topn_author_metadata_df = self.query_authors_metadata(authorIds)
    return topn_author_metadata_df.set_index('authorId').join(top_n_df.set_index('id'))
    
  # ~~~~~~~~~~ BUILDING THE GRAPH FROM S2AG ~~~~~~~~~~    
  def executeSemScholAuthorPapersQueryWithOffset(self, authorId, offset, verbose=True):
    fields = [
        'paperId',
        'authors',
        'referenceCount'
      ]
    url = '%s%d/papers?fields=%s&limit=1000&offset=%d'%(self.author_stem_url,authorId,','.join(fields),offset)
    
    if verbose:
      print('AUTHOR_ID: %d'%(authorId))
      print(url)
    
    r = requests.get(url, headers={"x-api-key":self.x_api_key}, timeout=20)   
    author_response = json.loads(r.content.decode(r.encoding))
    rdata = author_response.get('data')
    
    if rdata is None:
      return []
    
    #print(json.dumps(rdata, indent=4, sort_keys=True))

    paperTuples = list(set([(p_hash.get('paperId'), 
                             len(p_hash.get('authors')), 
                             p_hash.get('referenceCount'))
                         for p_hash in rdata if p_hash.get('referenceCount') is not None]))
    if verbose:
      print('Adding papers:'+str(len(paperTuples)))

    authorIds = list(set([a_hash.get('authorId') 
                          for p_hash in rdata 
                          for a_hash in p_hash.get('authors')
                          if a_hash.get('authorId') is not None]))
    if verbose:
      print('Adding authors:'+str(len(authorIds)))
    
    authorEdges1 = list(set([(a_hash.get('authorId'),p_hash.get('paperId')) 
                             for p_hash in rdata 
                             for a_hash in p_hash.get('authors')
                             if a_hash.get('authorId') is not None]))
    authorEdges2 = [(e2,e1) for (e1,e2) in authorEdges1]
    if verbose:
      print('Adding author edges:'+str(len(authorEdges1)))

    for tup in paperTuples:
      self.g.add_node(tup[0], label='paper', nAuthors=int(tup[1]), nRefs=int(tup[2]))
    self.g.add_nodes_from(authorIds, label='author' )
    self.g.add_edges_from(authorEdges1, label='wrote')
    self.g.add_edges_from(authorEdges2, label='was_written_by')
    
    return rdata

  # structure of data 
  # kolId = [{paperId,
  #                authors:[{authorId,name}], 
  #                citations:[{paperId,authors:[{authorId,name}]}], 
  #                references:[{paperId,authors:[{authorId,name}]}]}]
  def runSemScholAuthorPapersQuery(self, authorId, verbose=False):
    offset = 0
    rdata = []
    while len(rdata)%1000 == 0: 
      rdata = self.executeSemScholAuthorPapersQueryWithOffset(authorId, offset, verbose)
      offset += 1000
  
  def addKeyOpinionLeader(self, kolId, pkl_file=None, verbose=False):
    '''Given an author `kol`, add all papers published by `kol` to `g`. 
    Then, add all citations and references of those papers to `g`, 
    and add them to `added papers`.
    '''
    self.runSemScholAuthorPapersQuery(kolId, verbose=verbose)
    kol_papers = [e2 for e1, e2, attrs in self.g.out_edges(str(kolId), data=True) if attrs.get('label') == 'wrote']
    self.addCitationsOrReferencesToGraph(kol_papers, 'citations', False, pkl_file)
    self.addCitationsOrReferencesToGraph(kol_papers, 'references', False, pkl_file)
    for p in kol_papers:
      self.added_papers.add(p)
    
  def addCitationsOrReferencesWithOffset(self, paperId, citref, offset, isClosed, verbose=False):
    if citref == 'citations':
      citing_cited = 'citingPaper'
    elif citref == 'references':
      citing_cited = 'citedPaper' 
    else:
      raise Exception('error with citref: '+citref)

    paper_stem_url = 'https://api.semanticscholar.org/graph/v1/paper/'
    fields = [
        'paperId',
        'authors',
        'isInfluential',
        'referenceCount',
        'year'
      ]
    url = '%s%s/%s?fields=%s&limit=1000&offset=%d'%(paper_stem_url, paperId, citref, ','.join(fields), offset)
    r = requests.get(url, headers={"x-api-key":self.x_api_key}, timeout=20)   
    paper_response = json.loads(r.content.decode(r.encoding))
    rdata = paper_response.get('data')
    
    if verbose:
      print('\n'+str(paperId))
      print(url)
    #print(json.dumps(rdata, indent=4, sort_keys=True))
  
    try:
      
      paperTuples = list(set([(p_hash.get(citing_cited).get('paperId'), 
                                len(p_hash.get(citing_cited).get('authors')), 
                                p_hash.get(citing_cited).get('referenceCount'),
                                p_hash.get(citing_cited).get('year'))
                           for p_hash in rdata
                           if p_hash.get(citing_cited).get('paperId') is not None 
                              and p_hash.get(citing_cited).get('referenceCount') is not None]))
      if verbose:
        print('Adding papers:'+str(len(paperTuples)))
            
      authorIds = list(set([a_hash.get('authorId') 
                            for p_hash in rdata 
                            for a_hash in p_hash.get(citing_cited).get('authors')
                            if a_hash.get('authorId') is not None]))
      if verbose:
        print('Adding authors:'+str(len(authorIds)))
      
      authorEdges1 = list(set([(a_hash.get('authorId'), p_hash.get(citing_cited).get('paperId')) 
                               for p_hash in rdata 
                               for a_hash in p_hash.get(citing_cited).get('authors')
                               if a_hash.get('authorId') is not None]))
      authorEdges2 = [(e2,e1) for (e1,e2) in authorEdges1]
      if verbose:
        print('Adding author edges:'+str(len(authorEdges1)))
            
      if citref == 'citations':
        citEdges = list(set([(p_hash.get(citing_cited).get('paperId'), paperId, p_hash.get('isInfluential')) 
                                    for p_hash in rdata 
                                    if p_hash.get(citing_cited).get('paperId') is not None]))
      else: 
        citEdges = list(set([(paperId, p_hash.get(citing_cited).get('paperId'), p_hash.get('isInfluential')) 
                                    for p_hash in rdata 
                                    if p_hash.get(citing_cited).get('paperId') is not None]))
        
    except:
      print('Error in adding citations from paper: '+paperId)
      return []

    if verbose:
      print('Adding citations:'+str(len(citEdges))+'\n')
    
    if isClosed:
      checked_edge_list = [(e1,e2,sig) for e1,e2,sig in citEdges if e1 in self.g.nodes and e2 in self.g.nodes]
      for e1, e2, isInf in checked_edge_list:
        self.g.add_edge(e1, e2, label='cites', isInfluential=isInf)        
    else:
      for tup in paperTuples:
        self.g.add_node(tup[0], label='paper', nAuthors=int(tup[1]), nRefs=int(tup[2]), year=tup[3])
        #print(tup)
      self.g.add_nodes_from(authorIds, label='author' )
      self.g.add_edges_from(authorEdges1, label='wrote')
      self.g.add_edges_from(authorEdges2, label='was_written_by')
      for e1, e2, isInf in citEdges:
        self.g.add_edge(e1, e2, label='cites', isInfluential=isInf)        
    return rdata
  
  def addCitationsOrReferencesToGraph(self, paperIds, citref, isClosed, pklpath=None):
    
    for i, paperId in tqdm(enumerate(paperIds), total=len(paperIds)):
      
      if pklpath and i%1000==0: # checkpoint save
        with open(pklpath, 'wb') as f:
          pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)    
      
      if paperId in self.added_papers:
        continue
      
      offset = 0
      rdata = []
      while len(rdata)%1000 == 0: 
        rdata = self.addCitationsOrReferencesWithOffset(paperId, citref, offset, isClosed)
        if len(rdata) == 0:
          break      
        offset += 1000 
        
    # Final save
    if pklpath:
      with open(pklpath, 'wb') as f:
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        
  # ~~~~~~~~~~ S2 METADATA QUERIES ~~~~~~~~~~  
        
  def executeSemScholAuthorQuery(self, authorId):
    fields = [
      'authorId',
      'name',
      'paperCount',
      'citationCount',
      'hIndex',
      'papers.paperId',
      'papers.title',
      'papers.influentialCitationCount'
      ]
    url = '%s%d?fields=%s'%(self.author_stem_url, authorId,','.join(fields))
    #print(url)
    r = requests.get(url, headers={"x-api-key":self.x_api_key}, timeout=20)   
    author_response = json.loads(r.content.decode(r.encoding))

    p_text_list = '     |     '.join([p.get('title') for p in sorted(author_response['papers'], key=lambda d: d['influentialCitationCount'], reverse=True)[:10]])
    author_response['Top_10_Pubs'] = p_text_list
    author_response['authorId'] = int(author_response['authorId'])
    author_response.pop('papers')
    #print(json.dumps(author_response, indent=4, sort_keys=True))

    return author_response

  def query_authors_metadata(self, authorIds):
    extras = []
    for authorId in tqdm(authorIds):
      rdata = self.executeSemScholAuthorQuery(int(authorId))
      extras.append(rdata)
    extras_df = pd.DataFrame(extras)
    extras_df.set_index('authorId')
    return extras_df
  
  # ~~~~~~~~~~ INFLUENTIAL GRAPH FUNCTIONS ~~~~~~~~~~  

  def get_influential_graph(self):
    g = self.g
    g2 = nx.DiGraph()
    inf_edges = [(n1,n2) for n1,n2,attrs in self.search_edges('cites') if attrs.get('isInfluential')]
    inf_papers =  set([n1 for n1,n2 in inf_edges]).union(set([n2 for n1,n2 in inf_edges]))
    for paperId in tqdm(sorted(list(inf_papers))):
      nAuthors = g.nodes[paperId].get('nAuthors')
      nRefs = g.nodes[paperId].get('nRefs')
      g2.add_node(paperId, label='paper', nAuthors=nAuthors, nRefs=nRefs)        
      for e1, e2, attrs in g.out_edges(paperId, data=True): 
        if attrs.get('label') == 'was_written_by': 
          g2.add_node(e2, label='author')
          g2.add_edge(e1, e2, label='was_written_by')
          g2.add_edge(e2, e1, label='wrote')
        elif attrs.get('label') == 'cites' and e2 in inf_papers: 
          if e2 not in g2.nodes: 
            nAuthors = g.nodes[e2].get('nAuthors')
            nRefs = g.nodes[e2].get('nRefs')
            g2.add_node(e2, label='paper', nAuthors=nAuthors, nRefs=nRefs)
          isInf = attrs.get('isInfluential')
          g2.add_edge(e1, e2, label='cites', isInfluential=isInf)
    return g2

  # ~~~~~~~~~~ AUTHOR-INFLUENCE-GRAPH FUNCTIONS ~~~~~~~~~~  

  def threshold_authors_by_pubcount(self, min_pub_count):
    authors = sorted([int(a) for a,attrs in self.g.nodes.data() if attrs.get('label')=='author'])
    authors_to_id = {a:i for i,a in enumerate(authors)}


    n_authors = len(authors)
    thresholded_authors = []
    counts = []
    for i,a in tqdm(enumerate(sorted(authors)), total=n_authors):
      n_authors_articles = len([(e1,e2) for e1,e2 in self.g.out_edges(str(a)) if self.g.edges[e1,e2].get('label')=='wrote'])
      counts.append(n_authors_articles)
      if n_authors_articles > min_pub_count:
        thresholded_authors.append(a)
    return thresholded_authors, counts
  
  def compute_author_eigenfactors(self, thresholded_authors, alpha=0.99, verbose=False):
    thresholded_authors_df = pd.DataFrame(thresholded_authors, columns=['id'])
    n_thresholded_authors = len(thresholded_authors)
    thresholded_authors_to_id = {a:i for i,a in enumerate(thresholded_authors)}
    
    if verbose:
      print("Computing Z for %d authors"%(n_thresholded_authors))
    Z = dok_matrix((n_thresholded_authors, n_thresholded_authors), dtype=np.float64)
    for i, authorId in tqdm(enumerate(thresholded_authors), total=len(thresholded_authors)):
      edgeMap = self.compute_edges_for_author(authorId)
      for t in edgeMap.keys():  
        if int(t) in thresholded_authors:
          j = thresholded_authors_to_id[int(t)]
          Z[i,j] = edgeMap[t]
    
    if verbose:
      print("Computing M = Z / Z_colsum")
    Z_colsum = np.sum(Z, axis=0) 
    M = Z / Z_colsum  
    
    if verbose:
      print("Computing A = teleport probability")
    v = []
    n_all_articles = len([nid for nid, attrs in self.g.nodes.data() if attrs.get('label')=='paper'])
    for i,a in tqdm(enumerate(thresholded_authors), total=n_thresholded_authors):
      n_authors_articles = len([(e1,e2) for e1,e2 in self.g.out_edges(str(a)) if self.g.edges[e1,e2].get('label')=='wrote'])
      v.append(n_authors_articles / n_all_articles)
    A = np.array([v]).T @ np.ones((1,n_thresholded_authors))
    
    if verbose:
      print("Computing P = alpha * M + (1-alpha) * A (alpha=%f)"%(alpha))
    P = alpha * M + (1-alpha) * A
    
    if verbose:
      print("Computing Eigenfactors + Eigenvectors")
    PP = P + P.T
    eigf, eigv = linalg.eig(np.nan_to_num(PP))

    if verbose:
      print("Done")
    
    leading_eigenvector_index = np.argmax(eigf)
    f = eigv[:,leading_eigenvector_index].real#.reshape(P.shape[0],1)
    thresholded_authors_df['f'] = f 
    
    return thresholded_authors_df
  
  # modified from ./networkx/algorithms/traversal/breadth_first_search.py
  def search_for_reference_author_pathways(self, source_author):
    depth_limit = 3
    out = []
    queue = deque([('', source_author, depth_limit, self.g.successors(source_author))])
    while queue:
      route, parent, depth_now, children = queue[0]
      for child in children:
        if (self.g.nodes[child].get('label')=='paper' and depth_now>1) or (self.g.nodes[child].get('label')=='author' and depth_now==1):    
          out.append(route+'|'+parent+'|'+child)
          if depth_now > 1:
            queue.append((route+'|'+parent, child, depth_now - 1, self.g.successors(child)))
      queue.popleft()
    return out


  # modified from ./networkx/algorithms/traversal/breadth_first_search.py
  def search_for_citation_author_pathways(self, source_author):
    depth_limit = 3
    out = []
    queue = deque([('', source_author, depth_limit, self.g.predecessors(source_author))])
    while queue:
      route, parent, depth_now, children = queue[0]
      for child in children:
        if (self.g.nodes[child].get('label')=='paper' and depth_now>1) or (self.g.nodes[child].get('label')=='author' and depth_now==1):    
          out.append(route+'|'+parent+'|'+child)
          if depth_now > 1:
            queue.append((route+'|'+parent, child, depth_now - 1, self.g.predecessors(child)))
      queue.popleft()
    return out

  def compute_edges_for_author(self, a, forward=True):
    weights = {}
    if forward:
      traversals = self.search_for_citation_author_pathways(str(a))
    else:
      traversals = self.search_for_reference_author_pathways(str(a))
    for routes_string in traversals:
      #routes_string = r+'|'+p+'|'+c
      l = re.split('\|', routes_string)
      if len(l) == 5:
        xxx, a1, p1, p2, a2 = l
        p1_nAuthors = self.g.nodes[p1]['nAuthors'] if self.g.nodes[p1]['nAuthors']>0 else 5      
        p1_nRefs = self.g.nodes[p1]['nRefs'] if self.g.nodes[p1]['nRefs']>0 else 100
        p2_nAuthors = self.g.nodes[p2]['nAuthors'] if self.g.nodes[p2]['nAuthors']>0 else 100
        x = 1 / (p1_nAuthors*p1_nRefs*p2_nAuthors) 
        a2 = int(l[4])
        if weights.get(l[4]) is None:
          weights[l[4]] = x
        else: 
          weights[l[4]] += x
    return weights
  
  
  # ~~~~~~~~~~ UTILITIES ~~~~~~~~~~
  def clone(self):
    copy = KOLsOnS2AG(self.x_api_key)
    copy.added_papers = self.added_papers
    copy.g = self.g.copy()
    copy.cit_g = self.cit_g.copy()
    return copy
    
  def search_nodes(self, label):
    nids = [(nid,attrs) for nid, attrs in self.g.nodes.data() if attrs.get('label')==label]
    return nids

  def search_edges(self, label):
    edges = [(e1,e2,attrs) for e1,e2,attrs in self.g.edges.data() if attrs.get('label')==label]
    return edges
  
  def load_from_pickle(self, file):
    with open(file, 'rb') as f:
      loaded_copy = pickle.load(f) 
    self.added_papers = loaded_copy.added_papers
    self.g = loaded_copy.g

  def save_to_pickle(self, file):
    with open(file, 'wb') as f:
      pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    
  def check_before_adding_edges_from(self, edge_list, label):
    checked_edge_list = [(e1,e2) for e1,e2 in edge_list if e1 in self.g.nodes and e2 in self.g.nodes]
    if len(checked_edge_list) > 0: 
      print('adding %d new edges'%(len(checked_edge_list)))
      self.g.add_edges_from(checked_edge_list, label=label)
      


# COMMAND ----------

show_doc(KOLsOnS2AG.print_basic_stats)

# COMMAND ----------

show_doc(KOLsOnS2AG.search_for_disambiguated_author)

# COMMAND ----------

show_doc(KOLsOnS2AG.build_author_citation_graph)

# COMMAND ----------

show_doc(KOLsOnS2AG.run_thresholded_centrality_analysis)
