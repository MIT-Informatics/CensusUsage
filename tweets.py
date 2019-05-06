import pickle
import re
import time
import json

# importing file to help parsing text data
from process import *

# requires python 2 for models library
# from extract_tweets import *

def check_tweet(tweet_object):
    '''
    Method to check validity of tweet recieved from GetOldTweets

    Keyword Args:
    tweet_object - the tweet object to parse and extract info

    :Return:
    text_data - the extracted text information
    '''

    # creating regex to extract
    extracted_text = re.search(
        r'((^|\s*)american community survey)|((^|\s*)american community survey)|((^|\s*)#acssurvey)|((^|\s*)Americancommunitysurvey)|((^|\s*)#ACS survey)|((^|\s*)ACS survey)|((^|\s*) ACS survey)"', tweet_object.text, re.IGNORECASE)
    # checking for matches
    if not extracted_text == None:
        return extracted_text.group(0)
    else:
        return None


def filter_tweets(tweet_list):
    """
    Method to filter out tweets that do not contain any of the regexs

    Keyword Args:
    tweet_list - a list of tweet objects

    :Return:
    filtered_list - the filtered list of tweets
    """

    filtered_data = {}

    count = 0
    for tweet in tweet_list:
        # checking for match
        match = check_tweet(tweet)
        if match != None:
            # creating JSON structure to store data of tweet
            data = {
                "ID": tweet.id,
                "username": tweet.username,
                "text": tweet.text.encode('utf-8'),
                "date": str(tweet.date),
                "retweets": tweet.retweets,
                "favorites": tweet.favorites,
                "mentions": tweet.mentions,
                "hashtags": tweet.hashtags,
                "geo": tweet.geo
            }

            filtered_data[count] = data
            count += 1

    return filtered_data


def store_data(tweets_data, filename):
    """
    Method to store extracted information as pickle files

    Keyword Args:
    tweets_data - a JSON object of historical twitter data
    filename - the name of the file

    :Return:
    None
    """

    # writing text to file with newline in between
    with open("source_files/twitter/" + filename, "w") as f:
        json.dump(tweets_data, f)

    print("JSON object written to file")
    return


def clean_tweets(json_data):
    """
    Method to clean an input of full tweet texts

    Keword Args:
    text_list - a list of text to process

    :Returns:
    cleaned - a list of tokenized and cleaned text in bow format
    """

    cleaned = [process_tweet(json_data[key]["text"])[0] for key in json_data]
    return cleaned


def create_tweet_corpuses():
    """
    Method to automize the querying
    """

    start_time = time.time()

    # list to hold corpuses
    corpuses = []

    # list of terms to query for in historical tweets package
    search_terms = [
        "american community survey",
        "#acssurvey",
        "Americancommunitysurvey",
        "#ACS survey",
        "ACS survey",
        " ACS survey"
    ]

    # # extracting tweet information
    # tweets1 = extract_tweet_data(search_terms[0], 10000)
    # corpuses.append(filter_tweets(tweets1))

    # print("Extracted 1st term after " +
    #       str((time.time() - start_time)) + " seconds")

    # # extracting tweet information
    # tweets2 = extract_tweet_data(search_terms[1], 10000)
    # corpuses.append(filter_tweets(tweets2))

    # print("Extracted 2nd term after " +
    #       str((time.time() - start_time)) + " seconds")

    # # extracting tweet information
    # tweets3 = extract_tweet_data(search_terms[2], 10000)
    # corpuses.append(filter_tweets(tweets3))

    # print("Extracted 3rd term after " +
    #       str((time.time() - start_time)) + " seconds")

    # # extracting tweet information
    # tweets4 = extract_tweet_data(search_terms[3], 10000)
    # corpuses.append(filter_tweets(tweets4))

    # print("Extracted 4th term after " +
    #       str((time.time() - start_time)) + " seconds")

    # # extracting tweet information
    # tweets5 = extract_tweet_data(search_terms[4], 10000)
    # corpuses.append(filter_tweets(tweets5))

    # print("Extracted 5th term after " +
    #       str((time.time() - start_time)) + " seconds")

    # # extracting tweet information
    # tweets6 = extract_tweet_data(search_terms[5], 10000)
    # corpuses.append(filter_tweets(tweets6))

    # print("Extracted 6th term after " +
    #       str((time.time() - start_time)) + " seconds")

    # store_data(text_list, "tweets.p")
    return corpuses


def create_clean_corpuses():

    # querying api for corpuses for all search terms
    corpus1, corpus2, corpus3, corpus4, corpus5, corpus6 = create_tweet_corpuses()

    # storing text of the data
    # store_data(corpus1, "american_community_survey.json")
    # store_data(corpus2, "#acssurvey.json")
    # store_data(corpus3, "Americancommunitysurvey.json")
    # store_data(corpus4, "#ACS_survey.json")
    # store_data(corpus5, "ACS_survey.json")
    # store_data(corpus6, "_ACS_survey.json")

    cleaned_corpuses = []

    # cleaning corpuses to use in model
    cleaned_corpuses.append(clean_tweets(corpus1))
    cleaned_corpuses.append(clean_tweets(corpus2))
    cleaned_corpuses.append(clean_tweets(corpus3))
    cleaned_corpuses.append(clean_tweets(corpus4))
    cleaned_corpuses.append(clean_tweets(corpus5))
    cleaned_corpuses.append(clean_tweets(corpus6))

    return cleaned_corpuses

def load_twitter_corpus():
	'''
	Method to load twitter corpuses
	'''

	return pickle.load(open("source_files/twitter/twitter_corpus.p", "rb"))


# when running script
if __name__ == '__main__':
	main()
