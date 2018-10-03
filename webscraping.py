from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException

from httplib import BadStatusLine

# utilizing beautifulsoup
from bs4 import BeautifulSoup
import requests
from lxml import html
from requests.exceptions import MissingSchema, ConnectionError

import socket
from process import *

# for uzing regexp
import re
import time

# to help with pickling html files from BeautifulSoup
import sys
sys.setrecursionlimit(40000)


url_list_1 = []
# opening url txt file and parsing
with open('source_files/NewsUrls.txt') as f:

    # reading first line from file
    line = f.readline()

    # looping through rest of lines in file
    while line != '':
        # removing newline at end of each line
        url_list_1.append(line.replace('\n', '').replace('\r', ''))
        line = f.readline()

# filtering out empty lines in url list
url_list_1 = filter(lambda x: x != '', url_list_1)

url_list_2 = []
# opening url txt file and parsing
with open('source_files/NewsUrls2.txt') as f:

    # reading first line from file
    line = f.readline()

    # looping through rest of lines in file
    while line != '':
        # removing newline at end of each line
        url_list_2.append(line.replace('\n', '').replace('\r', ''))
        line = f.readline()

# filtering out empty lines in url list
url_list_2 = filter(lambda x: x != '', url_list_2)


def search_url(url):
    '''
    Method to search and extract the text information from a Census Bureau url
    @param url - the url to access and from which to access information
    @return - the text from the html document
    '''

    # try catch block to check for errors in accessing web
    # driver = webdriver.Chrome("/home/dsam99/Downloads/chromedriver")

    error_urls = []
    corpus = []
    html_corpus = []

    # tuple list showing the result for each linked url and the corresponding
    # links
    results = []

    try:
        # using request module to access html
        if url.startswith("https://", 0) or url.startswith("http://", 0):
            result = requests.get(url)

            print("Accessed webpage")

            # checking if valid request
            if result.status_code == 200:

                print("Extracted html information")

                soup = BeautifulSoup(result.content, 'html.parser')
                links = soup.findAll('a')
                # return links

                pages = []

                # getting links for other pages
                for link in links:
                    if link.text == "View Clip":
                        pages.append(link.attrs['href'])

                print("Extracted all " + str(len(pages)) + " links")

                # searching up other pages
                for page in pages:
                    doc = []
                    try:
                        # checking if all schema is present
                        if page.startswith("https://", 0) or page.startswith("http://", 0):

                            inner_page = requests.get(page)

                            # checking if valid request
                            if inner_page.status_code == 200:

                                print("Valid page")
                                page_text = BeautifulSoup(
                                    inner_page.content, 'html.parser')

                                # results and html code for whole page
                                results.append((page, 200))
                                html_corpus.append(page_text)

                                # iterating through each page to find p tags
                                # for text
                                for p in page_text.findAll('p'):
                                    doc.append(p)
                            else:
                                print("Invalid linked URL")
                                error_urls.append(page)
                                results.append(
                                    (page, inner_page.status_code))
                                html_corpus.append(None)
                        else:
                            print(
                                "Invalid linked URL, does not start with correct Schema")
                            error_urls.append(page)
                            results.append((page, None))
                            html_corpus.append(None)

                        print("Finished page")
                        corpus.append(doc)

                    except socket.error:
                        print("Error with connecting sockets")
                        error_urls.append(page)
                        results.append((page, None))
                    except BadStatusLine(line):
                        print("Error in status response")
                        error_urls.append(page)
                        results.append((page, None))
                    except MissingSchema(error):
                        print("Missing Schema Error")
                        results.append((page, None))
                        error_urls.append(page)
                    except WebDriverException:
                        print("Error in web driver")
                    except TimeoutException():
                        print("Page did not load in given time")

            else:
                print("Invalid URI")
                error_urls.append(url)
                results.append(None)

        else:
            print("Invalid URI")
            error_urls.append(url)
            results.append(None)

    # except block to catch TimeoutException
    except TimeoutException():
        print("Page did not load in given time")
    except socket.error:
        print("Error with connecting sockets")
    except BadStatusLine(line):
        print("Error in status response")
    except WebDriverException:
        print("Error in web driver")

    # finally block to close web driver
    finally:
        print("reached finally block")
        return corpus, error_urls, results, html_corpus


def webscrape(url_list):
    '''
    Method to automate the webscraping of a webpage
    '''

    # a list of text lists containing word from the documents
    corpus = []
    error_links = []
    result_dict = {}
    html_corpus = []

    # looping through urls to look up through webdriver
    for url in url_list:
        url_docs = []
        # pulling information <p> tags from each webpage
        raw_data, errors, results, htmls = search_url(url)
        # appending all text information
        for doc in raw_data:
            url_docs.append([i.text for i in doc])

        corpus.append(url_docs)

        # appending all error links
        for error in errors:
            error_links.append(error)

        result_dict[url] = results
        html_corpus.append(htmls)

    return corpus, error_links, result_dict, html_corpus


def store_webscraping():
    '''
    Method to automate the running of the webscraping and pickling of the files
    '''

    webscraping_data, error_links, result_dict, html_corpus = webscrape(
        url_list_2)

    # Storing files for later use
    pickle.dump(webscraping_data, open(
        "source_files/webscraping_data_2.p", "wb"))

    with open('source_files/error_urls_2.txt', 'wb') as f:
        for error in error_links:
            f.write(error + '\n')

    pickle.dump(result_dict, open("source_files/result_dictionary_2.p", "wb"))
    pickle.dump(html_corpus, open("source_files/html_corpus_2.p", "wb"))


def process_web_data(filepath):
    '''
    Method to process web data that has been pickled
    '''

    webscraping_data = pickle.load(
        open(filepath, "rb"))

    print("opened pickle")
    text_list = []

    # formatting words to be analyzed for both sklearn and gensim
    for uri in webscraping_data:
        # processing each document data
        for doc in uri:
            for p in doc:
                text_list.append(process_string(p)[1])

            # text_list.append(word_list)

    print("processed html text")

    # filtering out empty strings
    text_list = filter(lambda x: len(x) != 0, text_list)

    # creating bigrams to run model on
    bigram_model = create_bigram_model(text_list)
    bigram_word_list = list(bigram_model[text_list])

    return bigram_word_list


def extract_html(filepath):
    '''
    Method to extract information from the pickled html strings
    @param filepath - the path to the pickled html object list
    '''

    # opening pickled html_corpus
    html_corpus = pickle.load(open(filepath, 'rb'))
    corpus = []

    # extracting information by iterating through uri stored info
    for uri in html_corpus:
        # doc = []
        # extracting info from each url in the uris
        for html_doc in uri:
            # creating doc to hold text
            if html_doc != None:
                for p in html_doc.findAll('p'):
                    corpus.append(process_string(p.text)[1])

        # corpus.append(doc)

    print("processed html text")

    # filtering out empty strings
    word_list = filter(lambda x: len(x) != 0, corpus)

    # creating bigrams to run model on
    bigram_model = create_bigram_model(word_list)
    bigram_word_list = list(bigram_model[word_list])

    return bigram_word_list

    # return corpus

if __name__ == "__main__":
    store_webscraping()
