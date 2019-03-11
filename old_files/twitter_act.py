#importing twitter packages
from twitter import *

#Reading access keys and tokens from OAuth.txt file

with open('OAuth.txt') as file:
	consumer_key = file.readline().replace('\n','')
	consumer_secret = file.readline().replace('\n','')
	access_token = file.readline().replace('\n','')
	access_secret = file.readline().replace('\n','')

class TwitterSearch(object):

	'''
	Class to search twitter from @dylanjsam99 account and the CensusUsageApp
	Utilizes the Python Twitter Tools package to integrate the Search API
	'''

	def __init__(self, consumer_key, consumer_secret, access_token, access_secret):

		'''
		Constructor for TwitterSearch object
		Takes in consumer key, consumer secret, access token, and access secret to create a twitter object
		'''

		#API tokens and secret
		self.consumer_key = consumer_key
		self.consumer_secret = consumer_secret
		self.access_token = access_token
		self.access_secret = access_secret

		#creating Twitter object
		self.t = Twitter(auth=OAuth(self.access_token, self.access_secret, self.consumer_key, self.consumer_secret))

	def searchHashtag(self, keyword):

		'''
		Method to use the Search API for a keyword
		@param - keyword, the keyword to search up in the twitter tweets files
		@return - the list of tweets  
		'''

		tweets = self.t.search.tweets(q=keyword)

		#Dictionary containing information about search results
		metadata = tweets['search_metadata']

		#List containing tweet objects 
		statuses = tweets['statuses']
		return statuses

def main():

	'''
	Main method for the twitter_act script
	- creating the TwitterSearch object 
	'''

	t = TwitterSearch(consumer_key, consumer_secret, access_token, access_secret)
	return t

if __name__ == '__main__':
	t = main()
	print(t.searchHashtag('#ACS'))