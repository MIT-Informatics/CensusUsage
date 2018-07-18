#importing selenium packages
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.common.exceptions import TimeoutException

#utilizing beautifulsoup
from bs4 import BeautifulSoup

from process import *

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
	try:
		#creating webdriver through Chrome
		driver = webdriver.Chrome("/home/dsam99/Downloads/chromedriver")

		driver.get(url)
		page = driver.page_source
		soup = BeautifulSoup(page, 'html.parser')
		return soup.findAll('p')

		# #trying access hidden record_main div tag with javascript
		# record_main = driver.find_element_by_xpath("//div[@class='controllers-contents']")

		# driver.execute_script("arguments[0].click();",record_main)

		# driver.execute_script("arguments[0].setAttribute('style','visibility:visible;');",record_main)

		# time.sleep(5) # seconds

		# # Wait for the dynamically loaded elements to show up
		# WebDriverWait(driver, 400)

	#except block to catch TimeoutException
	except TimeoutException():
		print("Page did not load in given time")
	#finally block to close web driver
	finally:	
		driver.quit()

def webscrape():

	'''
	Method to automate the webscraping of a webpage
	'''
	
	corpus = []

	#looping through urls to look up through webdriver
	for url in url_list[:5]: 
		#pulling information <p> tags from each webpage
		raw_data = search_url(url)
		for i in raw_data:
			corpus.append(i.text)

	return corpus

if __name__ == "__main__":
	print(webscrape())
