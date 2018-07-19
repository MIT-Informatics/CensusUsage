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
from requests.exceptions import MissingSchema

from process import *
import socket

#for uzing regexp
import re
import time

url_list = []
#opening url txt file and parsing
with open('NewsUrls.txt') as f:

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

	try:
		#creating webdriver through Chrome
		# driver.get(url)
		# page = driver.page_source

		#using request module to access html
		if url.startswith("https://", 0):
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

				print("Extracted all links")

				#searching up other pages
				for page in pages:
					try:
						# driver.get(page)
						#checking if all schema is present
						if page.startswith("https://", 0) or page.startswith("http://", 0):
							inner_page = requests.get(page)

							#checking if valid request
							if inner_page.status_code == 200:		
								page_text = BeautifulSoup(inner_page.content, 'html.parser')
								for p in page_text.findAll('p'):
									corpus.append(p)
							else:
								error_urls.append(page)
						else:
							error_urls.append(page)

					except socket.error:
						print("Error with connecting sockets")
						error_urls.append(page)

					except BadStatusLine(line):
						print("Error in status response")
						error_urls.append(page)

					except MissingSchema(error):
						print("Missing Schema Error")
						error_urls.append(page)

					except WebDriverException:
						print("Error in web driver")
					except TimeoutException():
						print("Page did not load in given time")


				print("Finished page")
				
		else:
			error_urls.append(url)

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
		# driver.quit()
		print("reached finally block")
		return corpus, error_urls


def webscrape():

	'''
	Method to automate the webscraping of a webpage
	'''
	
	corpus = []
	error_links = []

	#looping through urls to look up through webdriver
	for url in url_list[0:1]: 
		#pulling information <p> tags from each webpage
		raw_data, errors = search_url(url)
		#appending all text information
		if raw_data != None:
			for i in raw_data:
				corpus.append(i.text)
		#appending all error links
		for error in errors:
			error_links.append(error)
	return corpus, error_links

if __name__ == "__main__":
	print(webscrape()[0])
