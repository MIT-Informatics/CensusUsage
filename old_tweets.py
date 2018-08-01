#importing package from github
from GetOldTweets import got
import pickle

def search_old_tweets(query, num_tweets):

	'''
	Method to search tweets with the Get Old Tweets package
	@param query - the word to query for in the package
	@return - the list of resulting tweets
	'''

	tweetCriteria = got.manager.TweetCriteria().setQuerySearch(query).setSince("2015-05-01").setUntil("2018-07-31").setMaxTweets(num_tweets)
	tweet_list = got.manager.TweetManager.getTweets(tweetCriteria)

	return tweet_list

#when running script
if __name__ = '__main__':
	tweets = search_old_tweets('american census survey', 1000)
