# Databricks notebook source
# default_exp dashdbQueries
from nbdev import *

# COMMAND ----------

# MAGIC %md # Dashboard Database Queries
# MAGIC 
# MAGIC > Simple queries in SQL provided for use by `czLandscapingTk.dashdbUtils.DashboardDb` class.

# COMMAND ----------

#export

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
# NONFUNCTIONAL SQL LOGIC FROM OLD DATABASE TO COMPUTE H-INDEX FROM DATA. 
# KEEPING HERE AS A RECORD. 
H_INDEX_SQL = '''
  select do_id, a_id, a_name, count(*) as h_index
        from (
          select do.id as do_id, 
              a.id as a_id, 
              a.name as a_name, 
              p.id as p_id, 
              cc.CITATION_COUNT as citation_count, 
              rank() over (partition by a.id order by cc.CITATION_COUNT desc) as ranking
          from MFP_CITATION_COUNTS as cc 
              JOIN MFP_PAPER as p ON (p.id=cc.id) 
              JOIN MFP_PAPER_TO_AUTHOR as pa ON (p.id=pa.ID_PAPER) 
              JOIN MFP_AUTHOR as a ON (pa.ID_AUTHOR=a.ID)
              JOIN MFP_PATIENT_ORGANIZATION_TO_PAPER as dop on (p.id=dop.id_paper) 
              JOIN MFP_PATIENT_ORGANIZATION as do on (dop.id_org=do.id) 
          order by a_name, citation_count desc 
        ) t
        where ranking <= citation_count
        group by do_id, a_id, a_name
        order by a_name;
'''

MONDO_SEARCH_TERMS = '''
'''

# COMMAND ----------

#export

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

