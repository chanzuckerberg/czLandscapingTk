# Databricks notebook source
#hide
from nbdev import *
#from your_lib.core import *

# COMMAND ----------

# MAGIC %md # Chan Zuckerberg Landscaping Toolkit
# MAGIC 
# MAGIC > This is a public-facing library of components designed to support and facilitate 'scientific knowledge landscaping' within the Chan Zuckerberg Initiative's Science Program. It consists of several utility libraries to help build and analyze corpora of scientific knowledge expressed as natural language and structured data. This system is built on the excellent [`nbdev`](https://nbdev.fast.ai/) package 
# MAGIC 
# MAGIC CZI adheres to the Contributor Covenant [code of conduct](https://github.com/chanzuckerberg/.github/blob/master/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to opensource@chanzuckerberg.com.

# COMMAND ----------

# MAGIC %md ## Install

# COMMAND ----------

# MAGIC %md `pip install git+https://github.com/GullyBurns/czLandscapingTk.git`

# COMMAND ----------

# MAGIC %md ## How to use:

# COMMAND ----------

# MAGIC %md This libray is built on [databricks_to_nbdev_template](https://github.com/GullyBurns/databricks_to_nbdev_template), which is modified version of [nbdev_template](https://github.com/fastai/nbdev_template) tailored to work with databricks notebooks.
# MAGIC 
# MAGIC The steps to using this are: 
# MAGIC 1. Use the basic template to clone your repository and access it via databricks. 
# MAGIC 2. Fill in your `settings.ini` file (especially with any `requirements` that would need to be built to run your code).
# MAGIC 3. Place your scripts and utility notebooks in subdirectories of the `databricks` folder in the file hierarchy.
# MAGIC 4. Any databricks notebooks that contain the text: `from nbdev import *` will be automatically converted to Jupyter notebooks that live at the root level of the repository.
# MAGIC 5. When you push this repository to Github from Databricks, Jupyter notebooks will be built, added to the repo and then processed by nbdev to generate modules and documentation (refer to https://nbdev.fast.ai/ for full documentation on how to do this). Note that pushing code to Github will add and commit *more* code to github, requiring you to perform another `git pull` to load and refer to the latest changes in your code.  

# COMMAND ----------

# MAGIC %md ## Instructions for how to use Toolkit Classes:

# COMMAND ----------

# MAGIC %md ### AirtableUtils Class

# COMMAND ----------

# MAGIC %md  
# MAGIC Load the class and instantiate it with the API-KEY from Airtable:
# MAGIC ```
# MAGIC from czLandscapingTk.airtableUtils import AirtableUtils
# MAGIC atu = AirtableUtils('keyXYZXYZXYZYXZY')
# MAGIC ```
# MAGIC 
# MAGIC Read a complete table into a pandas dataframe: 
# MAGIC ```
# MAGIC # atu.read_airtable(<notebook id>, <table id>)
# MAGIC atu.read_airtable('appXYZXYZXYZXYZ', 'tblXYZXYZXYZXYZ')
# MAGIC ```
# MAGIC 
# MAGIC Write a dataframe to an Airtable (note, column names of Airtable must match the columns of the dataframe and must be instantiated manually ahead of time): 
# MAGIC ```
# MAGIC # atu.send_df_to_airtable(<notebook id>, <table id>, df):
# MAGIC df = <YOUR DATA>
# MAGIC atu.send_df_to_airtable('appXYZXYZXYZXYZ', 'tblXYZXYZXYZXYZ', df)
# MAGIC ```

# COMMAND ----------

# MAGIC %md ### NetworkxS2AG Class

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Instantiate the class using an api key you should obtain from the S2AG team to permit more than 100 request calls per 5 minutes. This script will burn through that limit immediately. Obtain API keys here: https://www.semanticscholar.org/product/api#Partner-Form
# MAGIC 
# MAGIC ```
# MAGIC from czLandscapingTk.networkxS2AG import NetworkxS2AG
# MAGIC kolsGraph = NetworkxS2AG('<API-KEY-FROM-S2AG-TEAM>')
# MAGIC ```
# MAGIC 
# MAGIC Maybe start by searching for a reseracher by name. e.g. [Daphne Koller](https://api.semanticscholar.org/graph/v1/author/search?query=Daphne+Koller) 
# MAGIC 
# MAGIC ```
# MAGIC kolsGraph.search_for_disambiguated_author('Daphne Koller')
# MAGIC ```
# MAGIC 
# MAGIC Generating the following output: 
# MAGIC <table border="1" class="dataframe">  <thead>    <tr style="text-align: right;"><th></th>      <th>authorId</th>      <th>name</th>      <th>paperCount</th>      <th>hIndex</th>      <th>Top 10 Pubs</th>    </tr>  </thead>  <tbody>    <tr>      <th>0</th>      <td>1736370</td>      <td>D. Koller</td>      <td>351</td>      <td>130</td>      <td>Probabilistic Graphical Models - Principles and Techniques     |     The Genotype-Tissue Expression (GTEx) project     |     FastSLAM: a factored solution to the simultaneous localization and mapping problem     |     Support Vector Machine Active Learning with Applications to Text Classification     |     Max-Margin Markov Networks     |     SCAPE: shape completion and animation of people     |     Self-Paced Learning for Latent Variable Models     |     The Genotype-Tissue Expression (GTEx) pilot analysis: Multitissue gene regulation in humans     |     Decomposing a scene into geometric and semantically consistent regions     |     Toward Optimal Feature Selection</td>    </tr>    <tr>      <th>1</th>      <td>2081968396</td>      <td>D. Koller</td>      <td>5</td>      <td>1</td>      <td>Systematic Analysis of Breast Cancer Morphology Uncovers Stromal Features Associated with Survival     |     [Relevance of health geographic research for dermatology].     |     Convolutional neural networks of H&amp;E-stained biopsy images accurately quantify histologic features of non-alcoholic steatohepatitis     |     IDENTIFYING GENETIC DRIVERS OF CANCER MORPHOLOGY     |     Features Associated with Survival Systematic Analysis of Breast Cancer Morphology Uncovers Stromal</td>    </tr>    <tr>      <th>2</th>      <td>50678963</td>      <td>D. Koller</td>      <td>1</td>      <td>0</td>      <td>½º Äääöòòòò Èöóóóóóðð×øø Êêððøøóòòð Åóð×</td>    </tr>    <tr>      <th>3</th>      <td>2049948919</td>      <td>Daphne Koller</td>      <td>1</td>      <td>1</td>      <td>Team-Maxmin Equilibria☆</td>    </tr>    <tr>      <th>4</th>      <td>2081968988</td>      <td>Daphne Koller</td>      <td>1</td>      <td>0</td>      <td>Í××òò Øùöö Àààööö Blockin× Ò Ý×××ò Aeaeøûóöö Äääöòòòò´´üøøòòòò ×øöö Blockinøµ</td>    </tr>    <tr>      <th>5</th>      <td>2081968959</td>      <td>Daphne Koller</td>      <td>3</td>      <td>1</td>      <td>Abstract 1883: Large scale viability screening with PRISM underscores non-inhibitory properties of small molecules     |     Strategic and Tactical Decision-Making Under Uncertainty     |     2 . 1 Pursuit / Evader in the UAV / UGV domain</td>    </tr>    <tr>      <th>6</th>      <td>1753668669</td>      <td>Daphne Koller</td>      <td>4</td>      <td>1</td>      <td>A Data-Driven Lens to Understand Human Biology: An Interview with Daphne Koller     |     Conservation and divergence in modules of the transcriptional programs of the human and mouse immune systems [preprint]     |     ImmGen at 15     |     Speaker-specific terms and resources</td>    </tr>    <tr>      <th>7</th>      <td>46193831</td>      <td>D. Stanford</td>      <td>3</td>      <td>0</td>      <td>Unmanned Aircraft Systems     |     Inference : Exploiting Local Structure     |     Learning : Parameter Estimation</td>    </tr>  </tbody></table>
# MAGIC 
# MAGIC Then generate an author+paper graph based on her ID:`1736370` 
# MAGIC 
# MAGIC ```
# MAGIC kolsGraph.build_author_citation_graph(1736370)
# MAGIC kolsGraph.print_basic_stats()
# MAGIC ```
# MAGIC 
# MAGIC This command performs the following steps: 
# MAGIC 
# MAGIC * Retrieve all her papers indexed in S2AG add those papers and all authors to the graph
# MAGIC * Iterate through those papers and add any papers that either she cites 'meaninfully' or that cite her 'meaningfully' (for a definition of what constitutes a 'meaningful' citation, see [Valenzuela et al 2015](https://ai2-website.s3.amazonaws.com/publications/ValenzuelaHaMeaningfulCitations.pdf)). 
# MAGIC * Add or link authors to these papers. 
# MAGIC * Iterate over all papers in this extended set and add all citations / references between them.
# MAGIC * Print out the results

# COMMAND ----------

# MAGIC %md ### QueryTranslator class

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC This class processes a Pandas Dataframe where one of the columns describes a Boolean Query that could be issued on an online scientific database written as a string (e.g., this query searches for various terms denoting neurodegenerative diseases and then links them to the phrase "Machine Learning": `'("Neurodegeneration" | "Neurodegenerative disease" | "Alzheimers Disease" | "Parkinsons Disease") & "Machine Learning"'`. This dataframe must also contain a numerical ID column to identify each query. 
# MAGIC 
# MAGIC Load the class and instantiate it with dataframe:
# MAGIC ```
# MAGIC from czLandscapingTk.queryTranslator import QueryType, QueryTranslator
# MAGIC df = pd.DataFrame({'ID':0, 'query':'("Neurodegeneration" | "Neurodegenerative disease" |
# MAGIC     "Alzheimers Disease" | "Parkinsons Disease") & "Machine Learning"'})
# MAGIC qt = QueryTranslator(df, 'query')
# MAGIC ```
# MAGIC 
# MAGIC Generate a list of queries that work on Pubmed:
# MAGIC ```
# MAGIC (corpus_ids, pubmed_queries) = qt.generate_queries(QueryType.pubmed)
# MAGIC query = [{'ID':0, 'query': '("Neurodegeneration" | "Neurodegenerative disease" | "Alzheimers Disease" | "Parkinsons Disease") & "Machine Learning"'}]
# MAGIC df = pd.DataFrame(query)
# MAGIC qt = QueryTranslator(df, 'query')
# MAGIC print(qt.generate_queries(QueryType.pubmed))
# MAGIC ```
# MAGIC 
# MAGIC This gives you the following output: 
# MAGIC ```
# MAGIC (t0 | t1 | t2 | t3) & t4
# MAGIC ([0], ['(("Machine Learning")[tiab]) AND (("Neurodegeneration")[tiab]) OR ("Neurodegenerative disease")[tiab]) OR ("Alzheimers Disease")[tiab]) OR ("Parkinsons Disease")[tiab])))'])
# MAGIC ```
# MAGIC 
# MAGIC Generate a list of queries that work on European PMC:
# MAGIC ```
# MAGIC (corpus_ids, epmcs_queries) = qt.generate_queries(QueryType.epmc)
# MAGIC ```
# MAGIC This will give you:
# MAGIC ```
# MAGIC (t0 | t1 | t2 | t3) & t4
# MAGIC ([0], ['((paper_title:"Machine Learning" OR ABSTRACT:"Machine Learning") AND ((paper_title:"Neurodegeneration" OR ABSTRACT:"Neurodegeneration") OR (paper_title:"Neurodegenerative disease" OR ABSTRACT:"Neurodegenerative disease") OR (paper_title:"Alzheimers Disease" OR ABSTRACT:"Alzheimers Disease") OR (paper_title:"Parkinsons Disease" OR ABSTRACT:"Parkinsons Disease")))'])
# MAGIC ```
# MAGIC 
# MAGIC Thus the same query can be executed easily on different APIs.

# COMMAND ----------

# MAGIC %md ### ESearchQuery / EFetchQuery

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC These classes provides an interface for performing queries on NCBI Etuils. This is designed to work in conjunction with the `QueryTranslator` class. 
# MAGIC 
# MAGIC ```
# MAGIC from czLandscapingTk.searchEngineUtils import ESearchQuery, EFetchQuery
# MAGIC 
# MAGIC import urllib.parse 
# MAGIC from time import time, sleep
# MAGIC 
# MAGIC esq = ESearchQuery()
# MAGIC pcd_search = urllib.parse.quote("Primary Ciliary Dyskinesia")
# MAGIC print(esq.execute_count_query(pcd_search))
# MAGIC sleep(3) # Sleep for 3 seconds
# MAGIC esq.execute_query(pcd_search)
# MAGIC 
# MAGIC efq = EFetchQuery()
# MAGIC sleep(3) # Sleep for 3 seconds
# MAGIC efq.execute_efetch(35777446)
# MAGIC ```

# COMMAND ----------

# MAGIC %md ### EuroPMCQuery

# COMMAND ----------


