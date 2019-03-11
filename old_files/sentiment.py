from textblob import TextBlob

from nltk.sentiment.sentiment_analyzer import SentimentAnalyzer
from nltk.sentiment import vader

#for using the multiple sci-kit learn classifier algos
from nltk.classify.scikitlearn import SklearnClassifier

from twitter_act import *
from process import *

#creating sentiment analyzer object with function parameter
SA = SentimentAnalyzer()
v = vader.SentimentIntensityAnalyzer()

def polarize_tweets(tweets):

	'''
	Method to separate tweets objects to positive and negative tweets
	@param tweets - a list of tweet objects
	@return - a tuple containing a list tuples of positive (tweet text, polariity score) and a list of negative tweets with (tweet text, polariity score)
	'''
	
	tweets_text = []
	#extracting string text from the tweet objects
	for tweet in tweets:
		tweets_text.append(tweet['text'])

	pos_tweets = []
	neg_tweets = []

	#splitting text based on sentiment analysis
	for text in tweets_text:
		score = v.polarity_scores(text)
		if score['compound'] >= 0:
			pos_tweets.append((text, score))
		else:
			neg_tweets.append((text, score))

	return pos_tweets, neg_tweets

if __name__ == '__main__':
	#using twitter_act file to access Twitter API
	t = main()
	#searching for tweets with #ACS hashtag
	tweets_objects = t.searchHashtag('uses census data')

	tweets_text = []

	for item in tweets_objects:
		tweets_text.append(process_string(item['text']))
	
	for string in tweets_text:
		print("Tweet:")
		print(string)
		print("Polarity scores: ")
		print(v.polarity_scores(string))
		print("=================================================================")

	#training sets to run algorithm on
	training_set = []
	testing_set = []

	#training NLTK NaiveBayesClassifier
	# classifier = nltk.NaiveBayesClassifier.train(training_set)
	# print(nltk.classify.accuracy(classifier, testing_set))*100
	# classifier.show_most_informative_features(10)

	pos, neg = polarize_tweets(tweets_objects)