import pickle
import numpy as np

result_dictionary = pickle.load(open('../source_files/uris/news_urls_1/result_dictionary_1.p', 'rb'))

def print_results(result_dict):

	'''
	Method to better visualize the results of webscraping
	@param result_dict - the dictinoary of uri -> (url -> status code) mappings
	'''

	total_uri = len(result_dict)
	total_urls = 0
	success_urls = 0

	for key in result_dict.keys():
		#iterating through linked urls from each uri page
		for url_status_pair in result_dict[key]:
			total_urls += 1
			if url_status_pair[1] == 200:
				success_urls += 1

	return total_uri, total_urls, success_urls


def errors_search():

	'''
	Method to handle the error_urls from the webscrape method
	@param error_urls - a list containing the errant urls from the webscrape method 
	@return - a quintuple containing: list of 200 responses,list of 403 errors, list of 404 errors, list of 410 errors, and a list of errors caught by try catches
	'''

	with open('../source_files/uris/news_urls_1/error_urls_1.txt', 'rb') as f:
		error_urls = f.readlines()

	#filtering out newline characters
	error_urls = [i.replace('\n', '') for i in error_urls]

	list_200 = []
	list_403 = []
	list_404 = []
	list_410 = []
	other_list = []

	for url in error_urls:
		try:
			result = requests.get(url)
			if result.status_code == 200:
				list_200.append(url)
			elif result.status_code == 403:
				list_403.append(url)
			elif result.status_code == 404:
				list_404.append(url)
			elif result.status_code == 410:
				list_410.append(url)
			else:
				other_list.append(url)
		except ConnectionError as e:
			print(str(e) + " ConnectionError")
			other_list.append(url)

	return list_200, list_403, list_404, list_410, other_list