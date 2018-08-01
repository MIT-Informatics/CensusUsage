#importing selenium packages
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException

from httplib import BadStatusLine

#utilizing beautifulsoup
from bs4 import BeautifulSoup
import requests
from requests.exceptions import MissingSchema, ConnectionError

from process import *
import socket

#for uzing regexp
import re
import time

url_list = []
#opening url txt file and parsing
with open('source_files/NewsUrls.txt') as f:

	#reading first line from file
	line = f.readline()

	#looping through rest of lines in file
	while line != '':
		#removing newline at end of each line
		url_list.append(line.replace('\n', '').replace('\r', ''))
		line = f.readline()

#filtering out empty lines in url list
url_list = filter(lambda x: x != '', url_list)

def search_url(url):
	'''
	Method to search and extract the text information from a Census Bureau url
	@param url - the url to access and from which to access information
	@return - the text from the html document
	'''

	#try catch block to check for errors in accessing web
	# driver = webdriver.Chrome("/home/dsam99/Downloads/chromedriver")

	error_urls = []
	corpus = []
	html_corpus = []

	#tuple list showing the result for each linked url and the corresponding links
	results = []

	try:
		#using request module to access html
		if url.startswith("https://", 0) or url.startswith("http://", 0):
			result = requests.get(url)

			print("Accessed webpage")

			#checking if valid request
			if result.status_code == 200:

				print("Extracted html information")

				soup = BeautifulSoup(result.content, 'html.parser')
				links = soup.findAll('a')
				# return links

				pages = []

				#getting links for other pages
				for link in links:
					if link.text == "View Clip":
						pages.append(link.attrs['href'])

				print("Extracted all " + str(len(links)) + " links")

				#searching up other pages
				for page in pages:
					try:
						#checking if all schema is present
						if page.startswith("https://", 0) or page.startswith("http://", 0):

							inner_page = requests.get(page)

							#checking if valid request
							if inner_page.status_code == 200:	
								
								print("Valid page")
								page_text = BeautifulSoup(inner_page.content, 'html.parser')

								#results and html code for whole page
								results.append((page, 200))
								html_corpus.append(page_text)

								#iterating through each page to find p tags for text
								for p in page_text.findAll('p'):
									corpus.append(p)
							else:
								print("Invalid linked URL")
								error_urls.append(page)
								results.append((page, inner_page.status_code))
								html_corpus.append(None)
						else:
							print("Invalid linked URL, does not start with correct Schema")
							error_urls.append(page)
							results.append((page, None))
							html_corpus.append(None)


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
				print("Finished page")
		else:
			print("Invalid URI")
			error_urls.append(url)
			results.append(None)

	#except block to catch TimeoutException
	except TimeoutException():
		print("Page did not load in given time")
	except socket.error:
		print("Error with connecting sockets")
	except BadStatusLine(line):
		print("Error in status response")
	except WebDriverException:
		print("Error in web driver")

	#finally block to close web driver
	finally:	
		print("reached finally block")
		return corpus, error_urls, results, html_corpus

def webscrape():

	'''
	Method to automate the webscraping of a webpage
	'''
	
	#a list of text lists containing word from the documents
	corpus = []
	error_links = []
	result_dict = {}
	html_corpus = []
	
	#looping through urls to look up through webdriver
	for url in url_list: 
		#pulling information <p> tags from each webpage
		raw_data, errors, results, htmls = search_url(url)
		#appending all text information
		if raw_data != None:
			for i in raw_data:
				corpus.append(i.text)
		#appending all error links
		for error in errors:
			error_links.append(error)
		
		result_dict[url] = results
		html_corpus.append(htmls)

	return corpus, error_links, result_dict, html_corpus

if __name__ == "__main__":
	print(webscrape()[0])
