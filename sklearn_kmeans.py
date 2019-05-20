# coding=utf-8
import sys
import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# importing data storage/extraction files
from tweets import *
from webscraping import *
from jstor import *


class kmeans_model():
	'''
	Class for kmeans clustering for text corpus for ACS data
	using tfidf vectorizer

	'''

	def __init__(self, k):
		'''
		Constructor for kmeans implementation in sklearn 
		'''

		self.k = k
		self.model = KMeans(n_clusters=k, 
					   init='k-means++', 
					   max_iter=100, 
					   n_init=1)
		self.vectorizer = TfidfVectorizer(stop_words='english')


	def train(self, corpus):
		'''
		Method to train the kmeans model

		Args:
		corpus - the documents to train the model on

		Return:
		:None:
		'''

		# vectorizing documents
		X = self.vectorizer.fit_transform(corpus)

		# fitting model to the corpus
		self.model.fit(X)

		return


	def predict(self, document):
		'''
		Method to predict the cluster based on the input corpus
		
		Args:
		document - a document to predict the cluster of
	
		Return:
		:None:
		'''

		Y = self.vectorizer.transform(["chrome browser to open."])
		prediction = model.predict(Y)
		print(prediction)


	def print_topics(self, num):
		'''
		Method to print out the topics in the clusters
		'''

		print("Top terms per cluster:")
		order_centroids = self.model.cluster_centers_.argsort()[:, ::-1]
		terms = self.vectorizer.get_feature_names()

		for i in range(self.k):
		    print("Cluster %d:" % i),
		    for ind in order_centroids[i, :]:
		        print(' %s' % terms[ind])

def main():
	model = kmeans_model(5)

	corpus = pickle.load(open("source_files/twitter/twitter_corpus.p", "rb"))
	corpus = [x for y in corpus for x in y]

	model.train(corpus)
	model.print_topics(10)

if __name__ == "__main__":
	main()
