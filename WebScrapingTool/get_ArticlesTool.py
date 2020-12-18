import requests
import json
import urllib3
from requests import get
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as n
from time import sleep
from openpyxl import load_workbook

# ignore warning messages
import warnings
warnings.filterwarnings('ignore')

### setting output display options
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

# desktop user-agent
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Chrome/80.0.3987.149 Safari/601.3.9"

# mobile user-agent
MOBILE_USER_AGENT = "Mozilla/5.0 (Linux; Android 7.0; SM-G930V Build/NRD90M) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.125 Mobile Safari/537.36"

# creating header for request
headers = {'User-Agent': USER_AGENT}

# Microsoft API Key
ms_api = '06ddca37db8f4f028f6a9f1e32003a95'

# CORE API key
core_api = 'TSYp9xWZK7dm3XBz6r8RnalOGyAvjEFg'

# Scopus API Key
scp_api = '6c6b7dff8f437772ee8c0135cd70e102'

# ScienceDirect API Keys
# 1. Search All API Key
sd1_api = '7f59af901d2d86f78a1fd60c1bf9426a'
# 2. Search Article attributes API Key
sd2_api = '6c6b7dff8f437772ee8c0135cd70e102'

# Springer API KEY
spr_api = '9771722066583fa9990238afde4495f1'

# search keywords
# search_query = "Python"
search_query = str(input("Enter your name to search:")).capitalize().strip()
#search_query='Probabilistic inference on uncertain semantic link network and its application in event identification'

# create dictioanry object for output
data = []


# function for search engines
def search_engines(engine_name, query):
    # Search all engines
    if engine_name is None:
        try:
             ##uncomment the search engine baesd your requiremnt
             #search_pubMed(query)
             #search_PlosOne(query)
             #search_academia(query)
             #search_msAcademic(query)
             #search_googleScholar(query)
             search_sciDirect(query)
             #search_scopus(query)
             #search_core(query)
             #search_springer(query)

        except Exception as e:  # raise e
            pass  # print('error:', e)


# 1. Google Scholar engine
def search_googleScholar(query):
    # request url
    url = 'https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=' + query + '&btnG='

    # response object
    # response = requests.get(url, headers=headers)
    response = requests.get(url, headers={'User-agent': 'your bot 0.1'})
    soup = BeautifulSoup(response.content, 'lxml')

    ######## Find required attributes in the response object by checking tag [data-lid]'))
    for item in soup.select('[data-lid]'):
        try:
            resp_obj = {"entities": {"Search Engine": "Google Scholar",
                                     "items": [
                                         {"DOI": ['No information found'],
                                          "Title": item.select('h3')[0].get_text(),
                                          "URLs": item.select('a')[0]['href'],
                                          "Authors": item.select('.gs_a')[0].get_text(),
                                          "Publication Name": ['No information found'],
                                          "ISSN": ['No information found'],
                                          #"Cited count": str(item.select('.gs_fl')[0].get_text()).split(' ', 1)[4],
                                           "Cited count": ['No information found'],
                                          "Affiliation": ['No information found '],
                                          "Type": ['No information found'],
                                         " Published date": 'No Information found',
                                         # "Published date":str(item.select('.gs_a')[0].get_text()).split('-', 1)[1].split('-', 1)[0],
                                          "Abstract": item.select('.gs_rs')[0].get_text()
                                          }
                                     ]}}
            # append dict object data
            data.append(resp_obj)

        except Exception as e:  # raise e
            pass
            # print('error Google:', e)


### 2. PubMed Search Engine
def search_pubMed(query):
    url = 'https://pubmed.ncbi.nlm.nih.gov/?term=' + query

    # response object
    response = requests.get(url, headers=headers, timeout=30)
    soup = BeautifulSoup(response.content, 'lxml')

    for item in soup.select('article'):
        try:
            resp_obj = {"entities": {"Search Engine": "PubMed Engine",
                                     "items": [
                                         {"DOI": str(item.find_all('span', class_='docsum-journal-citation')[
                                                         0].get_text()).split('doi:', 1)[1],
                                          "Title": str(item.find_all('a', class_='docsum-title')[0].get_text()).strip(),
                                          "URLs": 'https://pubmed.ncbi.nlm.nih.gov' +
                                                  item.find_all('a', class_='docsum-title')[0]['href'],
                                          "Authors": str(
                                              item.find_all('span', class_='docsum-authors')[0].get_text()).strip(),
                                          "Publication Name": ['No information found'],
                                          "ISSN": ['No information found'],
                                          "Cited count": ['No information found'],
                                          "Affiliation": ['No information found '],
                                          "Type": ['article'],
                                          " Published date": 'No Information found',
                                          "Published date": str(item.find_all('span', class_='docsum-journal-citation')[
                                                                    0].get_text()).split(';', -1)[0].split('.', 1)[1],
                                          "Abstract": str(
                                              item.find_all('div', class_='full-view-snippet')[0].get_text()).strip()
                                          }
                                     ]}}
            data.append(resp_obj)

        except Exception as e:  # raise e
            pass
        # print('error pubmed:', e)


###3. Academia Engine
def search_academia(query):
    q = query.title().replace(' ', '_')
    url = 'https://www.academia.edu/Documents/in/' + q

    # response object
    response = requests.get(url, headers=headers)

    if response.status_code == 200:  # check for ok response
        soup = BeautifulSoup(response.content, 'html.parser')

        ######## Find required attributes in the response object
        for item in soup.find_all('div', class_='u-borderBottom1'):
            try:
                try:
                    # few records doesnt have summary attribute so check them
                    if bool(item.select('.summarized')[0].get_text()):
                        abs = item.select('.summarized')[0].get_text()
                except Exception as e:  # raise e
                    abs = item.select('.summary')[0].get_text()

                resp_obj = {"entities": {"Search Engine": "Academia Search Engine",
                                         "items": [
                                             {"DOI": ['No information found'],
                                              "Title": item.select('a')[0].get_text(),
                                              "URLs": item.select('a')[0]['href'],
                                              "Authors": item.select('.u-fw700')[0].get_text(),
                                              "Publication Name": ['No information found'],
                                              "ISSN": ['No information found'],
                                              "Cited count": ['No information found'],
                                              "Affiliation": ['No information found '],
                                              "Type": ['No information found'],
                                              "Published date": ['No information found'],
                                              "Abstract": abs
                                              }
                                         ]}}
                # append dict object data
                data.append(resp_obj)
            except Exception as e:  # raise e
               # pass
             print('error acad:', e)
    else:
        pass


### 4. CORE Search Engine
def search_core(query):
    url = url1 = 'https://core.ac.uk:443/api-v2/search/' + query + '?page=1&pageSize=10&apiKey=' + core_api

    # response object
    response = requests.get(url, headers=headers, timeout=30)
    soup = BeautifulSoup(response.content, 'lxml')

    # convert soup object into json
    obj = json.loads(soup.text)

    ######## Find required attributes in the response object
    for item in obj['data']:
        try:
            resp_obj = {"entities": {"Search Engine": "CORE Search Engine",
                                     "items": [{"DOI": item['_source']['doi'],
                                                "Title": item['_source']['title'],
                                                "URLs": item['_source']['urls'],
                                                "Authors": item['_source']['authors'],
                                                "Publication Name": item['_source']['publisher'],
                                                "ISSN": item['_source']['issn'],
                                                "Cited count": item['_source']['citationCount'],
                                                "Affiliation": ['No Information'],
                                                "Type": item['_type'],
                                                # "Keywords": item['topics'],
                                                "Published Date": item['_source']['datePublished'],
                                                "Abstract": item['_source']['description']
                                                }]}}

            # append dict object data
            data.append(resp_obj)
        except Exception as e:  # raise e
            pass
            # print('error core:', e)


### 5. Elseiver Scopous Engine - building url

def search_scopus(query):
    url = 'https://api.elsevier.com/content/search/scopus?query=' + query + '&apiKey=' + scp_api

    # response object
    response = requests.get(url, headers=headers, timeout=30)
    soup = BeautifulSoup(response.content, 'lxml')

    # convert resonse into josn
    obj = json.loads(soup.text)

    ######## Find required attributes in the response object
    for item in obj['search-results']['entry']:
        try:
            if "prism:Issn" and "prism:issn" not in obj:
                issn = item['prism:eIssn']
            else:
                issn = item['prism:issn']

            resp_obj = {"entities": {"Search Engine": "Elsevier SCOPUS Search Engine",
                                     "items": [
                                         {"DOI": item['prism:doi'],
                                          "Title": item['dc:title'],
                                          "URLs": item['prism:url'],
                                          "Authors": item['dc:creator'],
                                          "Publication Name": item['prism:publicationName'],
                                          "ISSN": issn,
                                          "Cited count": item['citedby-count'],
                                          "Affiliation": item['affiliation'][0]['affilname'],
                                          "Type": item['subtypeDescription'],
                                          "Published date": item['prism:coverDate'],
                                          "Abstract": item['prism:publicationName']
                                          }
                                     ]}}
            # append dict object data
            data.append(resp_obj)
        except Exception as e:  # raise e
            pass
            # print('error scopus:', e)


# 6. ScienceDirect  Engine - building url

def search_sciDirect(query):
    #query = "Probabilistic inference on uncertain semantic link network and its application in event identification"
    url = 'https://api.elsevier.com/content/search/sciencedirect?&apiKey='+ sd1_api + '&query=' + '"'+ query +'"'

    # response object
    response = requests.get(url, headers={'User-agent': 'your bot 0.1'})
    soup = BeautifulSoup(response.content, 'lxml')
    obj = json.loads(soup.text)

    ######## Find required attributes in the response object
    for item in obj['search-results']['entry']:
        try:
            publish_date = str(item['load-date']).split('T', -1)[0]

            # get document ID from the result first
            doi = item['prism:doi']

            # call again api with DOI to the get the attriutes
            url2 = 'https://api.elsevier.com/content/article/doi/' + doi + '?apiKey=' + sd2_api
            response1 = requests.get(url2, headers=headers)
            soup1 = BeautifulSoup(response1.content, 'lxml')

            ######## Find required attributes in the response object
            for item in soup1.find_all('coredata'):
                resp_obj = {"entities": {"Search Engine": "Science Direct Search Engine",
                                         "items": [
                                             {"DOI": item.find_all('prism:doi')[0].get_text(),
                                              "Title": item.find_all('dc:title')[0].get_text().strip(),
                                              "URLs": item.find_all('prism:url')[0].get_text(),
                                              "Authors": item.find_all('dc:creator')[0].get_text(),
                                              "Publication Name": item.find_all('prism:publicationname')[0].get_text(),
                                              "ISSN": item.find_all('prism:issn')[0].get_text(),
                                              "Cited count": ['No information found'],
                                              "Affiliation": ['No information found '],
                                              "Type": item.find_all('document-type'),
                                              "Published date": publish_date,
                                              "Abstract": item.find_all('dc:description')[0].get_text().strip()
                                              }
                                         ]}}
                # append dict object data
                data.append(resp_obj)
        except Exception as e:  # raise e
            pass
            # print('error:', e)


### 7. PLOS API- used by the PLOS journal websites

def search_PlosOne(query):
    url = 'https://journals.plos.org/plosone/dynamicSearch?filterJournals=PLoSONE&q=' + query + '%20papers&page=1'
    # response object
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    obj = json.loads(soup.text)

    ######## Find required attributes in the response object
    for item in obj['searchResults']['docs']:
        try:

            resp_obj = {"entities": {"Search Engine": "PLOS Engine",
                                     "items": [
                                         {"DOI": item['id'],
                                          "Title": item['title_display'],
                                          "URLs": 'https://doi.org/' + item['id'],
                                          "Authors": item['author_display'],
                                          "Publication Name": ['No information found'],
                                          "ISSN": item['eissn'],
                                          "Cited count": item['alm_scopusCiteCount'],
                                          "Affiliation": ['No information found '],
                                          "Type": item['article_type'],
                                          "Published date": str(item['publication_date']).split('T', -1)[0],
                                          "Abstract": item['figure_table_caption']
                                          }
                                     ]}}
            # append dict object data
            data.append(resp_obj)
        except Exception as e:  # raise e
            # pass
           pass
            # print('error plos:', e)


### 8. Microsoft Academy engine
def search_msAcademic(query):
    q = query.lower()
    url1 = 'https://api.labs.cognitive.microsoft.com/academic/v1.0/evaluate?expr=Composite(F.FN=%27' + q + '%27)&model=latest&count=10&offset=0&attributes=DOI,Ti,Y,BT,D,W,PB,CC,AA.AuN,AA.AuId,AA.DAfN,AA.AfN,S,AW&subscription-key=' + ms_api

    # response object
    response = requests.get(url1, headers=headers)
    soup = BeautifulSoup(response.content, 'lxml')
    obj = json.loads(soup.text)

    ######## Find required attributes in the response object
    for item in obj['entities']:
        try:
            # extract abstract keywords from the response as it doesnt have a spefcific abstract attribute
            abs_str = str(item['AW'])
            abs_new = abs_str.replace(',', '').replace("'", '')
            if item['BT'] == 'a':
                type = 'Journal/Article'
            elif item['BT'] == 'b':
                type = 'Book'
            elif item['BT'] == 'p':
                type = 'Conference Paper'
            else:
                type = ''
            if 'DOI' not in obj:
                doi = ['No information found']
            else:
                doi = item['DOI']
            if 'PB' not in obj:
                pb = ['No information found']
            else:
                pb = item['PB']

            resp_obj = {"entities": {"Search Engine": "Microsoft Academy",
                                     "items": [
                                         {"DOI": doi,
                                          "Title": item['Ti'],
                                          "URLs": item['S'][0]['U'],
                                          "Authors": item['AA'][0]['AuN'],
                                          "Publication Name": pb,
                                          "ISSN": [''],
                                          "Cited count": item['CC'],
                                          "Affiliation": item['AA'][0]['DAfN'],
                                          "Type": type,
                                          "Published date": item['D'],
                                          "Abstract": abs_new

                                          }
                                     ]}}
            # append dict object data
            data.append(resp_obj)
        except Exception as e:  # raise e
            pass
            #print('error MS:', e)


### 9. Springer Search Engine
def search_springer(query):
    url = 'http://api.springernature.com/meta/v2/json?q=' + query + '&s=1&p=10&api_Key=' + spr_api
    # http://api.springernature.com/meta/v2/json?q=python&api_key=9771722066583fa9990238afde4495f1

    # response object
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'lxml')
    obj = json.loads(soup.text)

    ######## Find required attributes in the response object
    for item in obj['records']:
        try:
            resp_obj = {"entities": {"Search Engine": "Springer Search Enginer",
                                     "items": [
                                         {"DOI": item['identifier'],
                                          "Title": item['title'],
                                          "URLs": item['url'][0]['value'],
                                          "Authors": item['creators'][0]['creator'],
                                          "Publication Name": item['publicationName'],
                                          "ISSN": item['issn'],
                                          "Cited count": ['No Information found'],
                                          "Affiliation": ['No information found'],
                                          "Type": item['contentType'],
                                          "Published date": item['onlineDate'],
                                          "Abstract": item['abstract']
                                          }
                                     ]}}
            # append dict object data
            data.append(resp_obj)
        except Exception as e:  # raise e
            pass  # print('error:', e)


### call the search fucntion
search_engines(None, search_query)

# print the dict output
print(data)

# check if the output received or not then create further dataframe
if bool(data):
    # convert dict object into JSON:
    json_output = json.dumps(data, indent=2)
    # print(json_output)

    # convert json into datafrme
    df = pd.json_normalize(data)

    #####-----creating final output------#####
    # drop nested columns and keep 1st attribute
    df.drop(["entities.items"], axis=1, inplace=True)

    # create required temp objets
    d1 = pd.DataFrame([])
    result = pd.DataFrame([])

    # split nested attributes into separate columns and stored output in a temp object d1
    i = 0
    for i in range(0, len(data)):
        d = pd.json_normalize(data[i]['entities']['items'])
        d1 = d1.append(d, True)

        # concatenate both dataframes into one
        result = pd.concat([df, d1], axis=1)

    # save final output to csv
    result.to_excel('search_results.xlsx', index=False)
    print('Spreadsheet saved.')
    print(result)

else:
    print("No record found!")
