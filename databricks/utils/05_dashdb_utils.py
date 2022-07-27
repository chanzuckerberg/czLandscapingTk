# Databricks notebook source
# default_exp dashdbUtils
from nbdev import *

# COMMAND ----------

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import dsa
from cryptography.hazmat.primitives import serialization
import io
import snowflake.connector
import pandas as pd
from enum import Enum

class Platform(Enum):
  GC = 'Google Cloud'
  DB = 'Databricks'

class SecretManager():
  def __init__(self, platform):
    self.platform = platform
  
  def get_creds(self, key):
    # databricks credentials
    if self.platform == Platform.DB:
      user = dbutils.secrets.get(scope=key, key="SNOWFLAKE_SERVICE_USERNAME")
      pem = dbutils.secrets.get(scope=key, key="SNOWFLAKE_SERVICE_PRIVATE_KEY")
      pwd = dbutils.secrets.get(scope=key, key="SNOWFLAKE_SERVICE_PASSPHRASE")
    else:
      raise Exception("Platform not set")
    return (user, pem, pwd)

class Snowflake():

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
    cs = self.cursor()
    cs.execute('USE WAREHOUSE ' + self.warehouse)
    cs.execute('USE SCHEMA '+self.database+'.'+self.schema)
    cs.execute('USE ROLE '+self.role)
    print('USE SCHEMA '+self.database+'.'+self.schema)
    return cs

  def execute_query(self, cs, sql, columns):
    cs.execute(sql)
    df = pd.DataFrame(cs.fetchall(), columns=columns)
    df = df.replace('\n', ' ', regex=True)
    return df

  def run_query_in_spark(self, user, query):

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
                   sfDatabase=g_database,
                   sfSchema=g_schema,
                   sfWarehouse="DEV_WAREHOUSE")

    sdf = spark.read \
          .format("snowflake") \
          .options(**options) \
          .option("query", query) \
          .load()

    return sdf

# COMMAND ----------

# test 
sec = SecretManager(Platform.DB)
(user, pem, pwd) = sec.get_creds('gully-scope')

# COMMAND ----------

# SET UP LOCAL VARIABLES FOR DASHBOARD DATABASE CREATION HERE

from datetime import datetime
import requests
import json

# COMMAND ----------

def get_dash_cursor(sf, mcdash):
  cs = sf.cursor()
  cs.execute('USE WAREHOUSE DEV_WAREHOUSE')
  cs.execute('USE SCHEMA '+mcdash.database+'.'+mcdash.schema)
  cs.execute('USE ROLE ARST_TEAM')
  print('USE SCHEMA '+mcdash.database+'.'+mcdash.schema)
  return cs

def execute_query(cs, sql, columns):
  cs.execute(sql)
  df = pd.DataFrame(cs.fetchall(), columns=columns)
  df = df.replace('\n', ' ', regex=True)
  return df

# COMMAND ----------

def build_lookup_table(cs, delete_existing=False):
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
  if deleted_existing: 
    cs.execute('drop table if exists PMID_LOOKUP')
  cs.execute(BUILD_PMID_LOOKUP_SQL)

# COMMAND ----------

# metaDash.py - GENERATE DASHBOARD TABLES IN SNOWFLAKE

import os.path
from itertools import islice
from pandas import DataFrame
from urllib.parse import quote_plus
import re

DASHBOARD_CORPUS = """
SELECT d.* 
FROM PREFIX_CORPUS as d
"""

DASHBOARD_CORPUS_TO_PAPER = """
SELECT dp.* 
FROM PREFIX_CORPUS as d
        INNER JOIN PREFIX_CORPUS_TO_PAPER as dp on (d.ID=dp.ID_CORPUS)
"""

DASHBOARD_PAPER = """
SELECT DISTINCT * FROM (
    SELECT DISTINCT p.id AS ID, p.DOI, p.TITLE, p.ABSTRACT, p.YEAR, p.MONTH, 
        p.DAY, p.VOLUME, p.ISSUE, p.PAGINATION, p.SOURCE, p.ISO as ISO_ABBREVIATION,
        p.JOURNAL_NAME_RAW as JOURNAL_TITLE, p.EIGENFACTOR, p.TYPE as ARTICLE_TYPE    
    FROM PREFIX_CORPUS as d
        INNER JOIN PREFIX_CORPUS_TO_PAPER as dp on (d.ID=dp.ID_CORPUS)
        INNER JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p on (p.ID=dp.ID_PAPER)
  ) as P2 
  LEFT JOIN (SELECT PAPER_ID, F1000, FACEBOOK, MENDELEY, NEWS, REDDIT, TWITTER, WIKIPEDIA
      FROM PROD_DB.CORE_RAW.ALTMETRIC_IMPORT) V on P2.ID=V.PAPER_ID
"""

DASHBOARD_BASIC_PAPER_COLUMNS = ['ID', 'DOI', 'TITLE', 'ABSTRACT', 
                                 'YEAR', 'MONTH', 'VOLUME', 
                                 'ISSUE', 'MESH', 'PAGINATION', 
                                 'JOURNAL_TITLE', 'ARTICLE_TYPE']
DASHBOARD_BASIC_PAPER = '''
SELECT DISTINCT p.id AS ID, p.DOI, p.TITLE, p.ABSTRACT, p.YEAR, p.MONTH, 
        p.VOLUME, p.ISSUE, p.MESH_TERMS_RAW as MESH, p.PAGINATION, 
        p.JOURNAL_NAME_RAW as JOURNAL_TITLE, p.TYPE as ARTICLE_TYPE  
FROM PREFIX_CORPUS as d
      JOIN PREFIX_CORPUS_TO_PAPER as dp on (d.ID=dp.ID_CORPUS)
      JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p on (p.ID=dp.ID_PAPER)
'''

DASHBOARD_PAPER_TO_AUTHOR = """
SELECT DISTINCT pa.*
FROM PREFIX_CORPUS_TO_PAPER as dp 
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p ON (p.ID=dp.ID_PAPER)
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER_TO_AUTHOR_V2 as pa ON (p.ID=pa.ID_PAPER)
"""

DASHBOARD_AUTHOR = """
SELECT DISTINCT a.id as ID, a.id_orcid as ID_ORCID, a.NAME as NAME, a.SOURCE as SOURCE, 
    zeroifnull(sum(alt.MENDELEY)) as MENDELEY, zeroifnull(SUM(alt.TWITTER)) as TWITTER
FROM (SELECT DISTINCT ID_PAPER FROM PREFIX_CORPUS_TO_PAPER) as dp 
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p ON (p.ID=dp.ID_PAPER)
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER_TO_AUTHOR_V2 as pa ON (p.ID=pa.ID_PAPER)
    JOIN FIVETRAN.KG_RDS_CORE_DB.AUTHOR_V2 as a ON (a.ID=pa.ID_AUTHOR) 
    LEFT JOIN (SELECT PAPER_ID, F1000, FACEBOOK, MENDELEY, NEWS, REDDIT, TWITTER, WIKIPEDIA
      FROM PROD_DB.CORE_RAW.ALTMETRIC_IMPORT) alt on p.ID=alt.PAPER_ID
GROUP BY a.ID, a.id_orcid, a.NAME, a.SOURCE
"""

DASHBOARD_PAPER_NOTES = """
SELECT p.id as PMID,
  LISTAGG(a.name, ', ') WITHIN GROUP (order by author_index) as AUTHOR_STRING,
  COUNT(a.name) as AUTHOR_COUNT 
FROM (SELECT DISTINCT ID_PAPER FROM PREFIX_CORPUS_TO_PAPER) as dp 
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p ON (p.ID=dp.ID_PAPER)
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER_TO_AUTHOR_V2 as pa ON (p.ID=pa.ID_PAPER)
    JOIN FIVETRAN.KG_RDS_CORE_DB.AUTHOR_V2 as a ON (a.ID=pa.ID_AUTHOR) 
GROUP BY p.id ;
"""
BUILD_DASHBOARD_PAPER_NOTES = "create table PREFIX_PAPER_NOTES as " + DASHBOARD_PAPER_NOTES

DASHBOARD_PAPER_OPEN_ACCESS = """
SELECT p.id as PMID, uu.license, uu.open_access 
FROM (SELECT DISTINCT ID_PAPER FROM PREFIX_CORPUS_TO_PAPER) as dp 
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p ON (p.ID=dp.ID_PAPER)
    JOIN FIVETRAN.KG_RDS_CORE_DB.UNPAYWALL_URLS as uu on (p.DOI=uu.DOI)
GROUP BY p.id, uu.license, uu.open_access;
"""
BUILD_DASHBOARD_PAPER_OPEN_ACCESS = "create table PREFIX_PAPER_OPEN_ACCESS as " + DASHBOARD_PAPER_OPEN_ACCESS


DASHBOARD_AFFILIATION = """
SELECT aff.* 
FROM PREFIX_CORPUS_TO_PAPER as dp 
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p ON (p.ID=dp.ID_PAPER)
    JOIN FIVETRAN.KG_RDS_CORE_DB.AFFILIATION aff ON (p.ID = aff.ID_PAPER)
"""

DASHBOARD_INSTITUTION = """
select i.*, r.PLACE_LATITUDE as latitude, r.PLACE_LONGITUDE as longitude 
from PREFIX_CORPUS_TO_PAPER as dp 
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p ON (p.ID=dp.ID_PAPER)
    JOIN FIVETRAN.KG_RDS_CORE_DB.AFFILIATION aff ON (p.ID = aff.ID_PAPER)
    JOIN FIVETRAN.KG_RDS_CORE_DB.INSTITUTION i ON (i.ID = aff.ID_INSTITUTION)
    JOIN DEV_DB.SKE.RINGGOLD_LAT_LONG as r ON (i.id_source=r.ringgold_id )
"""

DASHBOARD_PAPER_TO_CONCEPT = """
SELECT pc.* 
FROM PREFIX_CORPUS_TO_PAPER as dp 
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p ON (p.ID=dp.ID_PAPER)
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER_TO_CONCEPT pc ON (p.ID=pc.ID_PAPER)
"""

DASHBOARD_CONCEPT = """
SELECT c.* 
FROM PREFIX_CORPUS_TO_PAPER as dp 
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p ON (p.ID=dp.ID_PAPER)
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER_TO_CONCEPT pc ON (p.ID=pc.ID_PAPER)
    JOIN FIVETRAN.KG_RDS_CORE_DB.CONCEPT c ON (c.ID=pc.ID_CONCEPT)
"""

DASHBOARD_CONCEPT_TO_SEMANTIC_TYPE = """
SELECT cst.* 
FROM PREFIX_CORPUS_TO_PAPER as dp 
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p ON (p.ID=dp.ID_PAPER)
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER_TO_CONCEPT pc ON (p.ID=pc.ID_PAPER)
    JOIN FIVETRAN.KG_RDS_CORE_DB.CONCEPT_TO_SEMANTIC_TYPE cst ON (pc.ID_CONCEPT=cst.ID_CONCEPT)
"""

DASHBOARD_SEMANTIC_TYPE = """
SELECT st.* 
FROM DEV_DB.SKE.SEMANTIC_TYPES_CATEGORIES as st
"""

DASHBOARD_COLLABORATIONS = """
SELECT DISTINCT id_author_a, id_author_b, id_paper as id_paper, author_count as author_count 
FROM 
    (select distinct a.id_author as id_author_a, b.id_author_b, a.id_paper, b.author_count 
    from RARE_CORPUS_TO_PAPER as cp 
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER_TO_AUTHOR_V2 a on (cp.ID_PAPER=a.ID_PAPER)
    left join 
        (select pa1.id_author as id_author_b, pa1.id_paper as id_paper, COUNT(pa2.id_author) as author_count 
        from RARE_CORPUS_TO_PAPER as cp 
        JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER_TO_AUTHOR_V2 pa1 on (cp.ID_PAPER=pa1.ID_PAPER)
        JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER_TO_AUTHOR_V2 pa2 on (pa1.ID_PAPER=pa2.ID_PAPER)
        GROUP BY pa1.id_author, pa1.id_paper) b on a.id_paper=b.id_paper 
    where id_author_a > id_author_b 
    order by a.id_paper, id_author_a) 
"""
BUILD_DASHBOARD_COLLABORATIONS = "create table PREFIX_COLLABORATIONS as " + DASHBOARD_COLLABORATIONS

DASHBOARD_AUTHOR_LOCATION = '''
SELECT DISTINCT
    a.id as author_id, a.name as author_name, ordered_locations.REV_ORDER, ordered_locations.institution, 
    ordered_locations.city, ordered_locations.country, ordered_locations.lat, ordered_locations.long
FROM PREFIX_CORPUS_TO_PAPER as cp 
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER_TO_AUTHOR_V2 pa on (cp.ID_PAPER=pa.ID_PAPER)
    JOIN FIVETRAN.KG_RDS_CORE_DB.AUTHOR_V2 AS a on (pa.ID_AUTHOR=a.ID)
        JOIN (
          SELECT p2a.ID_AUTHOR as ID_AUTHOR, 
                p.YEAR as YEAR, 
                rank() over (partition by p2a.ID_AUTHOR order by p.PMID desc, i.name) as REV_ORDER,
                i.name as institution, i.city as city, i.country as country, rll.PLACE_LATITUDE as lat, rll.PLACE_LONGITUDE as long
          FROM PREFIX_CORPUS_TO_PAPER as cp 
            JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER_TO_AUTHOR_V2 p2a on (cp.ID_PAPER=p2a.ID_PAPER)
            JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER AS p ON (p2a.ID_PAPER = p.ID)
            JOIN FIVETRAN.KG_RDS_CORE_DB.AFFILIATION AS aff ON (p2a.id_paper=aff.id_paper AND p2a.author_index=aff.index_author)
            JOIN FIVETRAN.KG_RDS_CORE_DB.INSTITUTION AS i ON (aff.id_institution = i.id)
            JOIN DEV_DB.SKE.RINGGOLD_LAT_LONG AS rll ON (rll.ringgold_id = i.id_source)
         ORDER BY id_author, REV_ORDER
        ) AS ordered_locations on (ordered_locations.ID_AUTHOR=a.id)
WHERE ordered_locations.REV_ORDER=1
GROUP BY author_id, author_name, REV_ORDER, year, institution, lat, long, city, country
ORDER BY author_name, REV_ORDER;
'''
BUILD_DASHBOARD_AUTHOR_LOCATION = "create table PREFIX_AUTHOR_LOCATION as " + DASHBOARD_AUTHOR_LOCATION

DASHBOARD_CITATION_COUNTS = '''
        select p.id as ID, cit.DOI_TO_PAPER as DOI, COUNT(cit.ID) as CITATION_COUNT 
        from PREFIX_CORPUS_TO_PAPER as cp 
            JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p ON (cp.ID_PAPER=p.ID)
            JOIN FIVETRAN.KG_RDS_CORE_DB.CITATION as cit ON (cit.DOI_TO_PAPER=p.DOI) 
        where DOI != ''
        group by p.id, cit.DOI_TO_PAPER;
'''
BUILD_DASHBOARD_CITATION_COUNTS = 'create table PREFIX_CITATION_COUNTS as ' + DASHBOARD_CITATION_COUNTS

ADD_SEMANTIC_TYPES_TO_FILTER_CONCEPTS = """
"""

DASHBOARD_CURATED_DATA = '''
        select p.id as ID_PAPER, cit.DOI_TO_PAPER as DOI, COUNT(cit.ID) as CITATION_COUNT 
        from PREFIX_CORPUS_TO_PAPER as cp 
            JOIN PREFIX_FIVETRAN.KG_RDS_CORE_DB.PAPER as p ON (cp.ID_PAPER=p.ID)
            JOIN FIVETRAN.KG_RDS_CORE_DB.CITATION as cit ON (cit.DOI_TO_PAPER=p.DOI) 
        where DOI != ''
        group by p.id, cit.DOI_TO_PAPER;
'''
BUILD_DASHBOARD_CITATION_COUNTS = 'create table PREFIX_CITATION_COUNTS as ' + DASHBOARD_CITATION_COUNTS


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PRIMARILY NOTES TO ALLOW US TO RECORD HOW QUERIES WERE PERFORMED ON THE DASHBOARD

DASH_CORPUS_SQL = '''SELECT CORPUS_NAME 
FROM PREFIX_CORPUS as d
'''

DASH_PAPER_COUNT_SQL = '''SELECT DISTINCT COUNT(p.id) AS PAPER_COUNT, dp.SUBSET_CODE AS SUBSET, CORPUS_NAME, d.ID
FROM PREFIX_CORPUS as d
    JOIN PREFIX_CORPUS_TO_PAPER as dp on (d.ID=dp.ID_CORPUS)
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p on (p.ID=dp.ID_PAPER)
GROUP BY CORPUS_NAME, SUBSET, d.ID 
'''
CREATE_DASH_PAPER_COUNT = 'create table PREFIX_DASH_PAPER_COUNT as select * from \n(%s)'%(DASH_PAPER_COUNT_SQL)

DASH_OA_PAPER_COUNT_SQL = '''SELECT DISTINCT COUNT(DISTINCT p.id) AS OA_PAPER_COUNT, dp.SUBSET_CODE AS SUBSET, CORPUS_NAME, d.ID
FROM PREFIX_CORPUS as d
    JOIN PREFIX_CORPUS_TO_PAPER as dp on (d.ID=dp.ID_CORPUS)
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p on (p.ID=dp.ID_PAPER)
    JOIN FIVETRAN.KG_RDS_CORE_DB.UNPAYWALL_URLS as uu on (p.DOI=uu.DOI)
WHERE uu.open_access is not null
GROUP BY CORPUS_NAME, SUBSET, d.ID  
'''
CREATE_DASH_OA_PAPER_COUNT = 'create table PREFIX_DASH_OA_PAPER_COUNT as select * from \n(%s)'%(DASH_OA_PAPER_COUNT_SQL)

DASH_AUTHOR_COUNT_SQL = '''SELECT count(DISTINCT a.id) AS AUTHOR_COUNT, dp.SUBSET_CODE AS SUBSET, CORPUS_NAME, d.ID
FROM PREFIX_CORPUS as d
    INNER JOIN PREFIX_CORPUS_TO_PAPER as dp on (d.ID=dp.ID_CORPUS)  
    INNER JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p on (p.ID=dp.ID_PAPER)
    INNER JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER_TO_AUTHOR_V2 as pa on (p.ID=pa.ID_PAPER)  
    INNER JOIN FIVETRAN.KG_RDS_CORE_DB.AUTHOR_V2 as a on (a.ID=pa.ID_AUTHOR)
GROUP BY CORPUS_NAME, SUBSET, d.ID 
'''
CREATE_DASH_AUTHOR_COUNT_SQL = 'create table PREFIX_DASH_AUTHOR_COUNT as select * from \n(%s)'%(DASH_AUTHOR_COUNT_SQL)

DASH_AUTHOR_SQL = '''select distinct a.id, 
        a.id_orcid, 
        a.name as author_name, 
        sum(ZEROIFNULL(cc.citation_count)/(2021-p.year)) as normalized_citation_count,  
        zeroifnull(sum(alt.MENDELEY)) as Mendeley, 
        zeroifnull(SUM(alt.TWITTER)) as Twitter,
        loc.institution as Institution,
        loc.city as City,
        loc.country as Country,
        DENSE_RANK() OVER (PARTITION BY CORPUS_NAME ORDER BY normalized_citation_count DESC, a.id, a.id_orcid, a.name, loc.institution DESC) as RANK,
        CORPUS_NAME,
        dop.SUBSET_CODE as SUBSET
    from PREFIX_CORPUS_TO_PAPER as dop 
        JOIN PREFIX_CORPUS as do on (dop.id_corpus=do.id)
        JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER_TO_AUTHOR_V2 as pa on (dop.id_paper=pa.id_paper)
        JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p on (pa.id_paper=p.id)
        JOIN PROD_DB.CORE_RAW.ALTMETRIC_IMPORT AS alt on (p.id=alt.paper_id)
        JOIN FIVETRAN.KG_RDS_CORE_DB.AUTHOR_V2 as a on (pa.id_author=a.id)
        JOIN PREFIX_CITATION_COUNTS as cc on (dop.ID_PAPER=cc.id)
        LEFT JOIN PREFIX_AUTHOR_LOCATION as loc on (a.id=loc.author_id)    
    group by a.id, a.id_orcid, a.name, alt.Mendeley, alt.Twitter, loc.Institution, loc.City, loc.Country, CORPUS_NAME, SUBSET
    order by Twitter desc
'''
CREATE_DASH_AUTHOR_SQL = 'create table PREFIX_DASH_AUTHOR as select * from \n(%s)'%(DASH_AUTHOR_SQL)

DASH_PAPERS_SQL = '''    
    SELECT DISTINCT p.id as PMID, 
        ZEROIFNULL(cc.CITATION_COUNT) as N_CITATIONS,
        alt.MENDELEY,
        alt.TWITTER,
        pas.author_string as AUTHORS,
        pas.open_access as open_access,
        YEAR AS YEAR,
        TITLE, 
        ABSTRACT,
        CONCAT(JOURNAL_NAME_RAW, ' ', p.VOLUME ,':',p.PAGINATION) as Journal_Ref,
        REPLACE(REGEXP_REPLACE( p.TYPE, 'Research Support,.*?($|\|)', '\\1' ), '|', '; ') AS Type,  
        DENSE_RANK() OVER (PARTITION BY CORPUS_NAME ORDER BY N_CITATIONS DESC) AS RANK,
        CORPUS_NAME,
        dp.SUBSET_CODE as SUBSET
    FROM PREFIX_CORPUS as d 
        JOIN PREFIX_CORPUS_TO_PAPER as dp on (d.ID=dp.ID_CORPUS) 
        JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p on (p.ID=dp.ID_PAPER)
        LEFT JOIN PROD_DB.CORE_RAW.ALTMETRIC_IMPORT AS alt on (p.id=alt.paper_id)
        LEFT JOIN PREFIX_CITATION_COUNTS as cc on (p.ID=cc.id)
        JOIN PREFIX_PAPER_NOTES as pas on (p.ID=pas.PMID)
        JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER_TO_AUTHOR_V2 as pa on p.ID=pa.ID_PAPER
        JOIN FIVETRAN.KG_RDS_CORE_DB.AUTHOR_V2 as a on (a.ID=pa.ID_AUTHOR)
    ORDER BY CORPUS_NAME, SUBSET, N_CITATIONS DESC
'''
CREATE_DASH_PAPERS_SQL = 'create table PREFIX_DASH_PAPERS as select * from \n(%s)'%(DASH_PAPERS_SQL)

DASH_CONCEPTS_COLUMNS = ['CONCEPT', 'PAPER_COUNT', 'SEMTYPES', 'CORPUS_NAME']
DASH_CONCEPTS_SQL = '''SELECT e.NAME AS CONCEPT, 
    count(distinct c.ID) AS PAPER_COUNT, 
    f.semtypes AS SEMTYPES,
    CORPUS_NAME,
    b.SUBSET_CODE as SUBSET
FROM PREFIX_CORPUS as a
    JOIN PREFIX_CORPUS_TO_PAPER as b on (a.ID = b.ID_CORPUS)
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as c on (b.ID_PAPER = c.ID)
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER_TO_CONCEPT as d on (c.ID = d.ID_PAPER) 
    JOIN FIVETRAN.KG_RDS_CORE_DB.CONCEPT as e on (d.ID_CONCEPT = e.ID)  
    JOIN (
      SELECT concept_id, LISTAGG(semtype, '; ') as semtypes 
      FROM (
        SELECT DISTINCT x.ID_CONCEPT as concept_id, y.NAME as semtype 
          FROM FIVETRAN.KG_RDS_CORE_DB.CONCEPT_TO_SEMANTIC_TYPE as x
            JOIN FIVETRAN.KG_RDS_CORE_DB.SEMANTIC_TYPE as y on (x.ID_SEMANTIC_TYPE = y.ID)
      )
      GROUP BY concept_id) as f on (e.ID = f.concept_id)
GROUP BY e.NAME, f.semtypes, CORPUS_NAME, SUBSET
ORDER BY count(distinct c.ID) DESC
'''
CREATE_DASH_CONCEPTS_SQL = 'create table PREFIX_DASH_CONCEPTS as select * from \n(%s)'%(DASH_CONCEPTS_SQL)

DASH_GEO_MAP_SQL = '''select distinct 
        institute.NAME as NAME,
        institute.CITY as CITY,        
        institute.COUNTRY as COUNTRY,
        institute.LATITUDE as LATITUDE,
        institute.LONGITUDE as LONGITUDE,
        sum(institute.C_COUNT) as C_COUNT, 
        CONCAT('Count of Citations: ', sum(institute.C_COUNT)) as CITATION_COUNT, 
        LISTAGG(institute.author_list, '; ') as LIST, 
        CORPUS_NAME,
        dop.SUBSET_CODE as SUBSET,
        DENSE_RANK() OVER (PARTITION BY CORPUS_NAME ORDER BY "C_COUNT" DESC) AS "Rank"
    from 
        (SELECT loc.institution as NAME,
              loc.city as CITY,        
              loc.country as COUNTRY,
              loc.lat as LATITUDE,
              loc.long as LONGITUDE,
              ZEROIFNULL( cc.citation_count) as C_COUNT,
              p.id as id_paper,
              LISTAGG(DISTINCT a.name, '; ') as author_list
          FROM PREFIX_AUTHOR_LOCATION as loc 
              JOIN FIVETRAN.KG_RDS_CORE_DB.AUTHOR_V2 as a on (a.id=loc.author_id) 
              JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER_TO_AUTHOR_V2 as pa on (pa.id_author=a.id) 
              JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p on (pa.id_paper=p.id)
              JOIN PREFIX_CITATION_COUNTS as cc on (p.id=cc.id)
          group by loc.institution, loc.CITY, loc.COUNTRY, LATITUDE, LONGITUDE, p.id, C_COUNT
        ) as institute 
        JOIN PREFIX_CORPUS_TO_PAPER as dop on (dop.id_paper=institute.id_paper)
        JOIN PREFIX_CORPUS as do on (dop.id_corpus=do.id)
    GROUP BY institute.NAME, institute.CITY, institute.COUNTRY, institute.LATITUDE, institute.LONGITUDE, C_COUNT, CORPUS_NAME, SUBSET
    ORDER BY "Rank"
'''
CREATE_DASH_GEO_MAP_SQL = 'create table PREFIX_DASH_GEO_MAP as select * from \n(%s)'%(DASH_GEO_MAP_SQL)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CONTENT_CORPUS_SQL = '''SELECT DISTINCT p.id as PMID, 
        YEAR AS YEAR,
        TITLE, 
        ABSTRACT, 
        JOURNAL_NAME_RAW as Journal,
        CORPUS_NAME
    FROM PREFIX_CORPUS as d 
        JOIN PREFIX_CORPUS_TO_PAPER as dp on (d.ID=dp.ID_CORPUS) 
        JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p on (p.ID=dp.ID_PAPER)
    WHERE dp.SUBSET_CODE=NULL
    ORDER BY N_CITATIONS DESC
'''
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MONDO_SEARCH_TERMS = '''
'''

# COMMAND ----------

CREATE_FULL_TEXT_DOCUMENT_TABLE = """
    CREATE TABLE FULLTEXT_DOCUMENT (ID INT IDENTITY, 
        ID_FTD TEXT, 
        PLAIN_TEXT TEXT,
        ENCODING TEXT, 
        PROVENANCE TEXT );
"""

CREATE_FULL_TEXT_ANNOTATION_TABLE = """
    CREATE TABLE FULLTEXT_ANNOTATION (
        ID INT IDENTITY, 
        LOCAL_ID TEXT,
        ID_FTD TEXT, 
        START_SPAN INT, 
        END_SPAN INT,
        TYPE TEXT, 
        TAG TEXT, 
        ATTR TEXT,
        SANITY_CHECK TEXT);
"""


# COMMAND ----------

from pathlib import Path

DASHBOARD_PAPER_NOTES = """
SELECT p.id as PMID,
  LISTAGG(a.name, ', ') WITHIN GROUP (order by author_index) as AUTHOR_STRING,
  COUNT(a.name) as AUTHOR_COUNT 
FROM (SELECT DISTINCT ID_PAPER FROM PREFIX_CORPUS_TO_PAPER) as dp 
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER as p ON (p.ID=dp.ID_PAPER)
    JOIN FIVETRAN.KG_RDS_CORE_DB.PAPER_TO_AUTHOR_V2 as pa ON (p.ID=pa.ID_PAPER)
    JOIN FIVETRAN.KG_RDS_CORE_DB.AUTHOR_V2 as a ON (a.ID=pa.ID_AUTHOR) 
GROUP BY p.id ;
"""
BUILD_DASHBOARD_PAPER_NOTES = "create table PREFIX_PAPER_NOTES as " + DASHBOARD_PAPER_NOTES

class RstContentDashboard:

    def __init__(self, database, schema, loc, prefix='PREFIX_'):
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

    def upload_wb(self, cs, df2, table_name):
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

    def clear_corpus_to_paper_table(self, cs):
        table_name = re.sub('PREFIX_', self.prefix, 'PREFIX_CORPUS_TO_PAPER')
        cs.execute('DROP TABLE IF EXISTS ' + table_name + ';')
        
    def build_core_tables_from_pmids(self, cs):
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
        cs.execute(re.sub('PREFIX_', self.prefix, BUILD_DASHBOARD_CITATION_COUNTS))

    def run_build_pipeline(self, cs, df, api_key):
        cs.execute("BEGIN")
        self.upload_wb(cs, df)
        self.execute_pubmed_queries(cs, df, api_key)
        self.build_core_tables_from_pmids(cs)
        cs.execute("COMMIT")
              
    def drop_database(self, cs):
        try:
            cs.execute("BEGIN")
            cs.execute("DROP TABLE IF EXISTS " + self.prefix + "AUTHOR_LOCATION")
            cs.execute("DROP TABLE IF EXISTS " + self.prefix + "CITATION_COUNTS")
            cs.execute("DROP TABLE IF EXISTS " + self.prefix + "COLLABORATIONS")
            cs.execute("DROP TABLE IF EXISTS " + self.prefix + "PAPER_NOTES")
            cs.execute("DROP TABLE IF EXISTS " + self.prefix + "CORPUS")
            cs.execute("DROP TABLE IF EXISTS " + self.prefix + "CORPUS_TO_PAPER")
            cs.execute("COMMIT")

        finally:
            cs.close()
         

# COMMAND ----------

import os
import re
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import dsa
from cryptography.hazmat.primitives import serialization
import io

def port_from_snowflake_to_databricks(database, schema, table_names, table_sql):
  
  # Use secret manager to get the login name and password for the Snowflake user
  user = dbutils.secrets.get(scope=g_scope, key="SNOWFLAKE_SERVICE_USERNAME")
  pem = dbutils.secrets.get(scope=g_scope, key="SNOWFLAKE_SERVICE_PRIVATE_KEY")
  pwd = dbutils.secrets.get(scope=g_scope, key="SNOWFLAKE_SERVICE_PASSPHRASE")

  string_private_key = f"{pem.strip()}"

  p_key = serialization.load_pem_private_key(
      io.BytesIO(string_private_key.encode()).read(),
      password=pwd.strip().encode(),
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
                 sfUser=user.strip(),
                 pem_private_key=pkb,
                 sfRole="ARST_TEAM",
                 sfDatabase=database,
                 sfSchema=schema,
                 sfWarehouse="DEV_WAREHOUSE")

  table_dict = {}
  for name, sql in zip(table_names, table_sql):
      table_dict[name] = re.sub('PREFIX_', prefix, sql)     

  sqlContext.sql('CREATE DATABASE IF NOT EXISTS ' + prefix )
  sqlContext.sql('USE '+prefix )
  for t in table_names:
    sqlContext.sql('DROP TABLE IF EXISTS ' + t)
  for t in table_names:
    sdf = spark.read \
        .format("snowflake") \
        .options(**options) \
        .option("query", table_dict[t]) \
        .load()
    print(t)
    print(table_dict[t])
    sdf.write.saveAsTable(t)
