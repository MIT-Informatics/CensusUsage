# importing package from github to extract tweets
from GetOldTweets import got

def extract_tweet_data(query, num_tweets):
    '''
    Method to search tweets with the Get Old Tweets package

    Keyword Args:
    query - the word to query for tweets
    num_tweets - the list number of tweets to look for

    :Return:
    tweet_list - a list of tweet objects
    '''

    # creating criteria object based on query and other parameters
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(query).setSince(
        "2013-05-01").setUntil("2018-07-31").setMaxTweets(num_tweets)
    # searching tweets based on tweet criteria
    tweet_list = got.manager.TweetManager.getTweets(tweetCriteria)

    return tweet_list