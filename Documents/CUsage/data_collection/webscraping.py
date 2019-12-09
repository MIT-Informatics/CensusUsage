import re
import time
import socket
import sys

# packages to help with extracting web data
# from httplib import BadStatusLine
from bs4 import BeautifulSoup
import requests
from lxml import html
from requests.exceptions import MissingSchema, ConnectionError

# importing file to help processing data
from process import *

# to help with pickling html files from BeautifulSoup
sys.setrecursionlimit(40000)

# setup url_list1
url_list_1 = []
with open('../source_files/uris/NewsUrls.txt') as f:
    line = f.readline()
    while line != '':
        # removing newline at end of each line
        url_list_1.append(line.replace('\n', '').replace('\r', ''))
        line = f.readline()
url_list_1 = filter(lambda x: x != '', url_list_1)

# setup url_list2
url_list_2 = []
with open('../source_files/uris/NewsUrls2.txt') as f:
    line = f.readline()
    while line != '':
        # removing newline at end of each line
        url_list_2.append(line.replace('\n', '').replace('\r', ''))
        line = f.readline()
url_list_2 = filter(lambda x: x != '', url_list_2)

# setup url_list_3
url_list_3 = []
with open('../source_files/uris/NewsUrls3.txt') as f:
    line = f.readline()
    while line != '':
        # removing newline at end of each line
        url_list_3.append(line.replace('\n', '').replace('\r', ''))
        line = f.readline()
url_list_3 = filter(lambda x: x != '', url_list_3)


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
                    # except BadStatusLine(line):
                    #     print("Error in status response")
                    #     error_urls.append(page)
                    #     results.append((page, None))
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
    # except BadStatusLine(line):
    #     print("Error in status response")
    except WebDriverException:
        print("Error in web driver")

    # finally block to close web driver
    finally:
        print("reached finally block")
        return corpus, error_urls, results, html_corpus


def store_webscraping(webscraping_data, error_links, result_dict, html_corpus):
    '''
    Method to store the result of webscraping

    Keyword Args:
    webscraping_data - the text data from webscraping
    error_links - a list containing URLs that threw errors
    result_dict - a dictionary containing URI -> URL -> http response
    html_corpus - a list of html files

    :Return:
    None

    '''

    # Storing files for later use
    pickle.dump(webscraping_data, open(
        "../source_files/uris/news_urls_3/webscraping_data_3_300-400.p", "wb"))

    with open('../source_files/uris/news_urls_3/error_urls_3_400-400.txt', 'w') as f:
        for error in error_links:
            f.write(error + '\n')

    pickle.dump(result_dict, open("../source_files/uris/news_urls_3/result_dictionary_3_300-400.p", "wb"))
    pickle.dump(html_corpus, open("../source_files/uris/news_urls_3/html_corpus_3_300-400.p", "wb"))


def process_web_data(webscraping_data):
    '''
    Method to process the text data from webscraping

    Keyword Args:
    webscraping_data - the text data from webscraping

    :Return:
    bigram_word_list - the cleaned text data with the bigram model applied
    '''

    text_list = []

    # formatting words to be analyzed for both sklearn and gensim
    for uri in webscraping_data:
        # processing each document data
        for doc in uri:
            for p in doc:
                # print(process_string(p))
                text_list.append(process_string(p)[1])

    print("processed html text")

    # filtering out empty strings
    text_list = filter(lambda x: len(x) != 0, text_list)

    # creating bigrams to run model on
    bigram_model = create_bigram_model(text_list)
    bigram_word_list = list(bigram_model[text_list])

    return bigram_word_list


def extract_web_data(filepath):
    '''
    Method to extract information from the pickled web text data strings

    Keyword Args:
    filepath - the string filepath to the web text data

    :Return:
    bigram_word_list - the cleaned text data with the bigram model applied
    '''

    web_data = pickle.load(open(filepath, 'rb'))
    bigram_word_list = process_web_data(web_data)

    return bigram_word_list


def extract_html(filepath):
    '''
    Method to extract information from the pickled html strings

    Keyword Args:
    filepath - the string filepath to the pickled html files

    :Return:
    bigram_word_list - the cleaned text data with the bigram model applied

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

    print("processed html text")

    # filtering out empty strings
    word_list = filter(lambda x: len(x) != 0, corpus)

    # creating bigrams to run model on
    bigram_model = create_bigram_model(word_list)
    bigram_word_list = list(bigram_model[word_list])

    return bigram_word_list


def webscrape(url_list):
    '''
    Method to extract the information from webscraping the list of URIs

    Keyword Args:
    url_list - the list of URIs that links to URLs

    :Return:
    webscraping_data - the text data from webscraping
    error_links - a list containing URLs that threw errors
    result_dict - a dictionary containing URI -> URL -> http response
    html_corpus - a list of html files

    '''

    # a list of text lists containing word from the documents
    corpus = []
    error_links = []
    result_dict = {}
    html_corpus = []

    count = 0

    # looping through urls to look up through webdriver
    for url in url_list[300:400]:
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

        print("[" + str(count) +  " of " + str(100) + " urls read]")
        count += 1

    store_webscraping(corpus, error_links, result_dict, html_corpus)
    bigram_word_list = process_web_data(corpus)

    return bigram_word_list

if __name__ == "__main__":
    webscrape(url_list_3)
