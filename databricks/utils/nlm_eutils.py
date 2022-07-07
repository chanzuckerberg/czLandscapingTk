# Databricks notebook source
"""
A package designed for interactions with the EUtils tools from the National Library of Medicine
"""
import requests
import json
import datetime
from enum import Enum
from urllib.request import urlopen
from urllib.parse import quote_plus
from urllib.error import URLError
from time import time,sleep
import re
from tqdm import tqdm
from bs4 import BeautifulSoup,Tag,Comment,NavigableString
import pandas as pd

PAGE_SIZE = 10000
TIME_THRESHOLD = 0.3333334

class NCBI_Database_Type(Enum):
    """
    Supported NCBI databases within this API
    """
    pubmed = 'pubmed'
    PMC = 'PMC'

class ClinicalCategory(Enum):
    """
    Simple enumeration of types of clinical query
    """
    NoCategory = 0
    Diagnosis = 1
    Therapy = 2
    Etiology = 3
    Prognosis = 4
    ClinicalPrediction = 5

class QueryBreadth(Enum):
    """
    Simple enumeration of broad/narrow types of clinical query
    """
    Unspecified = 0
    Broad = 1
    Narrow = 2

#
# Added Filters for Clinical Queries
# see https://www.ncbi.nlm.nih.gov/books/NBK3827/#pubmedhelp.Clinical_Queries_Filters
#
BROAD_THERAPY_FILTER = '((clinical[Title/Abstract] AND trial[Title/Abstract]) OR clinical trials as topic[MeSH Terms] OR ' \
                'clinical trial[Publication Type] OR random*[Title/Abstract] OR random allocation[MeSH Terms] OR ' \
                'therapeutic use[MeSH Subheading]) AND '
NARROW_THERAPY_FILTER = '(randomized controlled trial[Publication Type] OR (randomized[Title/Abstract] AND ' \
                 'controlled[Title/Abstract] AND trial[Title/Abstract])) AND '
BROAD_DIAGNOSIS_FILTER = '(sensitiv*[Title/Abstract] OR sensitivity and specificity[MeSH Terms] OR diagnose[Title/Abstract] ' \
                  'OR diagnosed[Title/Abstract] OR diagnoses[Title/Abstract] OR diagnosing[Title/Abstract] OR ' \
                  'diagnosis[Title/Abstract] OR diagnostic[Title/Abstract] OR diagnosis[MeSH:noexp] OR diagnostic ' \
                  '* [MeSH:noexp] OR diagnosis,differential[MeSH:noexp] OR diagnosis[Subheading:noexp]) AND '
NARROW_DIAGNOSIS_FILTER = '(specificity[Title/Abstract]) AND '
BROAD_ETIOLOGY_FILTER = '(risk*[Title/Abstract] OR risk*[MeSH:noexp] OR risk *[MeSH:noexp] OR cohort studies[MeSH Terms] ' \
                 'OR group[Text Word] OR groups[Text Word] OR grouped [Text Word]) AND '
NARROW_ETIOLOGY_FILTER = '((relative[Title/Abstract] AND risk*[Title/Abstract]) OR (relative risk[Text Word]) OR ' \
                  'risks[Text Word] OR cohort studies[MeSH:noexp] OR (cohort[Title/Abstract] AND ' \
                  'study[Title/Abstract]) OR (cohort[Title/Abstract] AND studies[Title/Abstract])) AND '
BROAD_PROGNOSIS_FILTER = '(incidence[MeSH:noexp] OR mortality[MeSH Terms] OR follow up studies[MeSH:noexp] OR ' \
                  'prognos*[Text Word] OR predict*[Text Word] OR course*[Text Word]) AND '
NARROW_PROGNOSIS_FILTER = '(prognos*[Title/Abstract] OR (first[Title/Abstract] AND episode[Title/Abstract]) OR ' \
                   'cohort[Title/Abstract]) AND '
BROAD_CLINICAL_PREDICTION_FILTER = '(predict*[tiab] OR predictive value of tests[mh] OR score[tiab] OR scores[tiab] OR ' \
                            'scoring system[tiab] OR scoring systems[tiab] OR observ*[tiab] OR observer variation[mh])'
NARROW_CLINICAL_PREDICTION_FILTER = '(validation[tiab] OR validate[tiab]) AND '

class ESearchQuery:
    """
    Class to provide query interface for ESearch (i.e., query terms in elaborate ways, return a list of ids)
    Each instance of this class executes queries of a given type
    """

    def __init__(self, api_key, oa=False, db='pubmed', category=ClinicalCategory.NoCategory, scope=QueryBreadth.Unspecified):
        """
        Initialization of the class
        :param query:
        :param oa:
        :param db:
        :param category:
        :param scope:
        """
        self.api_key = api_key
        self.category = category
        self.scope = scope
        self.idPrefix = ''
        self.oa = oa
        self.db = db

    def execute_count_query(self, query):

        if self.category == ClinicalCategory.Diagnosis and self.scope == QueryBreadth.Broad:
            query = BROAD_DIAGNOSIS_FILTER + '(' + query + ')'
        elif self.category == ClinicalCategory.Diagnosis and self.scope == QueryBreadth.Narrow:
            query = NARROW_DIAGNOSIS_FILTER + '(' + query + ')'
        elif self.category == ClinicalCategory.Therapy and self.scope == QueryBreadth.Broad:
            query = BROAD_THERAPY_FILTER + '(' + query + ')'
        elif self.category == ClinicalCategory.Therapy and self.scope == QueryBreadth.Narrow:
            query = NARROW_THERAPY_FILTER + '(' + query + ')'
        elif self.category == ClinicalCategory.Etiology and self.scope == QueryBreadth.Broad:
            query = BROAD_ETIOLOGY_FILTER + '(' + query + ')'
        elif self.category == ClinicalCategory.Etiology and self.scope == QueryBreadth.Narrow:
            query = NARROW_ETIOLOGY_FILTER + '(' + query + ')'
        elif self.category == ClinicalCategory.Prognosis and self.scope == QueryBreadth.Broad:
            query = BROAD_PROGNOSIS_FILTER + '(' + query + ')'
        elif self.category == ClinicalCategory.Prognosis and self.scope == QueryBreadth.Narrow:
            query = NARROW_PROGNOSIS_FILTER + '(' + query + ')'
        elif self.category == ClinicalCategory.ClinicalPrediction and self.scope == QueryBreadth.Broad:
            query = BROAD_CLINICAL_PREDICTION_FILTER + '(' + query + ')'
        elif self.category == ClinicalCategory.ClinicalPrediction and self.scope == QueryBreadth.Narrow:
            query = NARROW_CLINICAL_PREDICTION_FILTER + '(' + query + ')'

        idPrefix = ''
        if self.oa:
            if self.db == NCBI_Database_Type.PMC:
                query = '"open access"[filter] AND (' + query + ')'
                idPrefix = 'PMC'
            elif self.db == NCBI_Database_Type.pubmed:
                query = '"loattrfree full text"[sb] AND (' + query + ')'
        esearch_stem = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?api_key='+self.api_key+'&db=' + self.db + '&term='
        
        esearch_response = urlopen(esearch_stem + query)
        esearch_data = esearch_response.read().decode('utf-8')
        esearch_soup = BeautifulSoup(esearch_data, "lxml-xml")
        count_tag = esearch_soup.find('Count')
        if count_tag is None:
            raise Exception('No Data returned from "' + self.query + '"')
        return int(count_tag.string)

    def find_max_min_of_pmid_range_on_given_day(self, dd, min_flag=True):
        d = dd.strftime("%Y/%m/%d")
        esearch = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?api_key='+self.api_key+'&db=pubmed&mindate=%s&maxdate=%s&datetype=edat&retmax=1'%(d,d)
        esearch_response = urlopen(esearch)
        esearch_data = esearch_response.read().decode('utf-8')
        esearch_soup = BeautifulSoup(esearch_data, "lxml-xml")
        count_tag = esearch_soup.find('Count')

        if count_tag is None:
            raise Exception('No Data returned from "' + self.query + '"')
        count = int(count_tag.string)
        esearch = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?api_key='+self.api_key+'&db=pubmed&mindate=%s&maxdate=%s&datetype=edat&retmax=%d'%(d,d,count)
        esearch_response = urlopen(esearch)
        esearch_data = esearch_response.read().decode('utf-8')
        esearch_soup = BeautifulSoup(esearch_data, "lxml-xml")

        pmids = sorted([int(id_tag.string) for id_tag in esearch_soup.findAll('Id')])

        # find biggest block of broadly contiguous IDs with a list of lists
        pmid_blocks = []
        current_block = [pmids[0]]
        pmid_blocks.append(current_block)
        for i in range(len(pmids)-1):
            if pmids[i+1] > pmids[i] + 10:
                current_block = []
                pmid_blocks.append(current_block)
            current_block.append(pmids[i+1])
        longest_block_length = max([len(i) for i in pmid_blocks])
        max_min = [min(i) if min_flag else max(i) for i in pmid_blocks if len(i)==longest_block_length][0]

        return max_min

    def execute_query(self, query):

        if self.category == ClinicalCategory.Diagnosis and self.scope == QueryBreadth.Broad:
            query = BROAD_DIAGNOSIS_FILTER + '(' + query + ')'
        elif self.category == ClinicalCategory.Diagnosis and self.scope == QueryBreadth.Narrow:
            query = NARROW_DIAGNOSIS_FILTER + '(' + query + ')'
        elif self.category == ClinicalCategory.Therapy and self.scope == QueryBreadth.Broad:
            query = BROAD_THERAPY_FILTER + '(' + query + ')'
        elif self.category == ClinicalCategory.Therapy and self.scope == QueryBreadth.Narrow:
            query = NARROW_THERAPY_FILTER + '(' + query + ')'
        elif self.category == ClinicalCategory.Etiology and self.scope == QueryBreadth.Broad:
            query = BROAD_ETIOLOGY_FILTER + '(' + query + ')'
        elif self.category == ClinicalCategory.Etiology and self.scope == QueryBreadth.Narrow:
            query = NARROW_ETIOLOGY_FILTER + '(' + query + ')'
        elif self.category == ClinicalCategory.Prognosis and self.scope == QueryBreadth.Broad:
            query = BROAD_PROGNOSIS_FILTER + '(' + query + ')'
        elif self.category == ClinicalCategory.Prognosis and self.scope == QueryBreadth.Narrow:
            query = NARROW_PROGNOSIS_FILTER + '(' + query + ')'
        elif self.category == ClinicalCategory.ClinicalPrediction and self.scope == QueryBreadth.Broad:
            query = BROAD_CLINICAL_PREDICTION_FILTER + '(' + query + ')'
        elif self.category == ClinicalCategory.ClinicalPrediction and self.scope == QueryBreadth.Narrow:
            query = NARROW_CLINICAL_PREDICTION_FILTER + '(' + query + ')'

        idPrefix = ''
        if self.oa:
            if self.db == NCBI_Database_Type.PMC:
                query = '"open access"[filter] AND (' + query + ')'
                idPrefix = 'PMC'
            elif self.db == NCBI_Database_Type.pubmed:
                query = '"loattrfree full text"[sb] AND (' + query + ')'
        esearch_stem = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?api_key='+self.api_key+'&db=' + self.db + '&term='
        #query = quote_plus(query)
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

class EFetchQuery:
    """
    Class to provide query interface for ESearch (i.e., query terms in elaborate ways, return a list of ids)
    Each instance of this class executes queries of a given type
    """

    def __init__(self, oa=False, db='pubmed'):
        """
        Initialization of the class
        :param query:
        :param oa:
        :param db:
        """
        self.oa = oa
        self.db = db

    def execute_efetch(self, pmid):
        efetch_stem = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id='
        efetch_response = urlopen(efetch_stem + str(pmid))
        return self._generate_rows_from_medline_records(efetch_response.read().decode('utf-8'))

    def generate_data_frame_from_id_list(self, id_list):

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

        url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
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

