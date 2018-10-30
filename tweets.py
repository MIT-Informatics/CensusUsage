import pickle
import re
import time

# importing file to help parsing text data
from process import *

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

    filtered_list = []

    for tweet in tweet_list:
        # checking for match
        match = check_tweet(tweet)
        if match != None:
            data = (tweet.id, tweet.username, tweet.text.encode('utf-8'))
            filtered_list.append(data)

    return filtered_list


def store_data(list_of_text, filename):
    """
    Method to store extracted information as pickle files

    Keyword Args:
    list_of_text - a list of tweets texts from historical twitter data

    :Return:
    None
    """

    # writing text to file with newline in between
    with open("source_files/twitter/" + filename, "w") as f:
        for data in list_of_text:
            f.write(data[0] + "," + data[1] + "\n")
            f.write(data[2] + "\n")

    print(str(len(list_of_text)) + " tweets written to file")
    pass


def clean_tweets(text_list):
    """
    Method to clean an input of full tweet texts

    Keword Args:
    text_list - a list of text to process

    :Returns:
    cleaned - a list of tokenized and cleaned text in bow format
    """

    cleaned = [process_string(text[2])[1] for text in text_list]
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

    # extracting tweet information
    tweets1 = extract_tweet_data(search_terms[0], 10000)
    corpuses.append(filter_tweets(tweets1))

    print("Extracted 1st term after " +
          str((time.time() - start_time)) + " seconds")

    # extracting tweet information
    tweets2 = extract_tweet_data(search_terms[1], 10000)
    corpuses.append(filter_tweets(tweets2))

    print("Extracted 2nd term after " +
          str((time.time() - start_time)) + " seconds")

    # extracting tweet information
    tweets3 = extract_tweet_data(search_terms[2], 10000)
    corpuses.append(filter_tweets(tweets3))

    print("Extracted 3rd term after " +
          str((time.time() - start_time)) + " seconds")

    # extracting tweet information
    tweets4 = extract_tweet_data(search_terms[3], 10000)
    corpuses.append(filter_tweets(tweets4))

    print("Extracted 4th term after " +
          str((time.time() - start_time)) + " seconds")

    # extracting tweet information
    tweets5 = extract_tweet_data(search_terms[4], 10000)
    corpuses.append(filter_tweets(tweets5))

    print("Extracted 5th term after " +
          str((time.time() - start_time)) + " seconds")

    # extracting tweet information
    tweets6 = extract_tweet_data(search_terms[5], 10000)
    corpuses.append(filter_tweets(tweets6))

    print("Extracted 6th term after " +
          str((time.time() - start_time)) + " seconds")

    # store_data(text_list, "tweets.p")
    return corpuses


def main():

    # querying api for corpuses for all search terms
    corpus1, corpus2, corpus3, corpus4, corpus5, corpus6 = create_tweet_corpuses()

    # storing text of the data
    store_data(corpus1, "american_community_survey.txt")
    store_data(corpus2, "#acssurvey.txt")
    store_data(corpus3, "Americancommunitysurvey.txt")
    store_data(corpus4, "#ACS_survey.txt")
    store_data(corpus5, "ACS_survey.txt")
    store_data(corpus6, "_ACS_survey.txt")

    cleaned_corpuses = []

    # cleaning corpuses to use in model
    cleaned_corpuses.append(clean_tweets(corpus1))
    cleaned_corpuses.append(clean_tweets(corpus2))
    cleaned_corpuses.append(clean_tweets(corpus3))
    cleaned_corpuses.append(clean_tweets(corpus4))
    cleaned_corpuses.append(clean_tweets(corpus5))
    cleaned_corpuses.append(clean_tweets(corpus6))


# when running script
if __name__ == '__main__':
    main()