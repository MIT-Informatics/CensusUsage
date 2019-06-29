# coding=utf-8
import sys
import pickle
import numpy as np

# packages to help with NLP
import gensim
from gensim import corpora
from gensim.models import ldamodel, CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import Text8Corpus, Word2Vec
from gensim.corpora import dictionary

sys.path.append("../data_collection")

# importing data storage/extraction files
from tweets import *
from webscraping import *
from jstor import *

import operator

# to help with pickling html files from BeautifulSoup
sys.setrecursionlimit(40000)

# seed words for web data
seed_words = [
	("research", 0), 
	("policy", 1), 
	("commercial", 2),
	("education", 3)
]

# seed words for jstor data titles
seed_words_jstor = [
	("analysis", 0),
	("regression", 0),
	("machine learning", 0),
	("artificial intelligence", 0),
	("statistics", 0),
	("graphs", 0),
	("monte carlo", 0),
	("deep learning", 0),
	("neural network", 0),
	("correlation", 0),
	("data", 0),
	("analytical", 0),
	("studies", 0),
	("measurement", 0),
	("technical", 0),
	("methods", 0)
]


class LDAModel():

	def __init__(self, X, num_topics):
		'''
		Constructor for a LDA topic model

		Keyword args:
		X - the input text to construct the model around
		'''

		self.X = X
		self.prior = None
		self.model = None
		self.id2word = None
		self.corpus = None

		# number of topics to print eventually
		self.num_topics = num_topics

	def change_priors(self, word, topic_number, eta, weight):
		'''
		Helper method to change the value of a weight inside of an eta priors list

		Keyword args:
		word - the word to change the value of
		topic_number - the topic which to scale the probability by
		eta - the list of topic to word probability values
		weight - the amount to scale the value of word by

		:Return:
		the updated version of eta
		'''

		# looking up id of word in dictionary
		for i in self.id2word.keys():
			if word == self.id2word[i]:
				word_id = self.id2word.token2id[word]

				# multiplying the value of the current prob in eta
				eta[topic_number][word_id] *= weight
				print(word + " found in word vectors")

				# zeroing probability of prob in other topics
				for i in range(len(eta)):
					if i != topic_number:
						eta[i][word_id] *= 0

		return eta

	def create_priors(self):
		'''
		Method to create the eta matrix that will be used in the lda model as a prior based from topic/word probability
		'''

		eta = []

		# initializing each topic with values for each word
		for topic in range(self.num_topics):
			eta.append(np.ones(len(self.id2word)) * 0.01)

		# iterating through seed words and scaling eta
		for (word, topic_num) in seed_words:
			eta = self.change_priors(word, topic_num, eta, 50)

		self.prior = eta

	def gensim_topic_analysis(self, seeding=False):
		'''
		Method to model the topics of an input text list using gensim
		
		Keyword Args:
		seeding - a boolean representing if seeding should be used in model, default to false
		'''

		# formatting dictionary to use in LDA model
		self.id2word = corpora.Dictionary(self.X)
		self.corpus = [self.id2word.doc2bow(word) for word in self.X]

		# checking if seeding parameter is passed -> default to false
		if not seeding:
			print("Using unseeded model")
			self.eta = []
			per_word_topics = True

			# creating LDA model with parameterized topics and training passes
			lda = gensim.models.ldamodel.LdaModel(
				corpus=self.corpus,
				id2word=self.id2word,
				num_topics=self.num_topics,
				passes=5,  # 5 passes through corpus
				alpha='auto',  # learning prior from corpus
				per_word_topics=per_word_topics,
			)

		else:
			print("Using seeded model")
			self.create_priors()
			per_word_topics = False

			print(self.prior[0][self.id2word.token2id["research"]])
			print(self.prior[0][self.id2word.token2id["commercial"]])


			# creating LDA model with parameterized topics and training passes
			lda = gensim.models.ldamodel.LdaModel(
				corpus=self.corpus,
				id2word=self.id2word,
				num_topics=self.num_topics,
				passes=5,  # 5 passes through corpus
				alpha='auto',  # learning prior from corpus
				eta=self.prior,  # optional seeding parameter
				per_word_topics=per_word_topics,
			)

		topics = lda.print_topics()
		for i in range(len(topics)):
			print("Topic #" + str(i))
			print(topics[i])

		self.model = lda

	def evaluate_gensim_lda(self):
		'''
		Method to evaluate the lda model provided by gensim
		'''

		# held out likelihood score - simplified idea as predictablity of model
		print("Perplexity: ", self.model.log_perplexity(self.corpus))
		coherence = CoherenceModel(
			model=self.model, texts=self.X, dictionary=self.id2word, coherence='c_v')

		# multiple pipeline score evaluated quality of topics
		score = coherence.get_coherence()
		print("Coherence", score)

	def show_pyldavis(self):
		'''
		Method to create a pyLdavis visualization of the topic model
		'''

		import pyLDAvis.gensim

		prepared_data = pyLDAvis.gensim.prepare(
			self.model, self.corpus, self.id2word)
		pyLDAvis.show(prepared_data)

	def prediction(self, document):
		'''
		Method to apply the trained lda model to a subset document

		Keyword Args:
		document - the document in the form of a list of words to apply the model on

		Returns:
		the probability vector that is represents the probability that the document falls within each topic of the model
		'''

		doc = dictionary.doc2bow(document)

		# creating the probability vector of the document in the model
		vector = self.model[doc][0]
		return vector

	def split_documents(self, text_list):
		'''
		Method to split documents by the topic that they fall in
		@param model - the trained lda topic model
		@param all_docmuents - the total documents that the model was trained on
		@return a dictionary from topic number to the corresponding list of documents that are most likely to fall in the topic
		'''

		topic_dict = {i: [] for i in range(9)}

		for doc in text_list:
			vector = apply_model(doc)

			# finding the max probability tuple based on second element
			max_prob = max(vector, key=lambda x: x[1])

			topic_dict[max_prob[0]].append(doc)

		return topic_dict


def train_model(corpus_data):
	'''
	Method to automate training of the lda model on the an input textlist
	@param text_list - an input list of list of words in bow format
	'''

	corpus_data = list(filter(lambda x: x != [], corpus_data))

	# creating instance of lda model
	model = LDAModel(corpus_data, 4)
	model.gensim_topic_analysis(seeding=True)

	print("finished training model")

	# storing model
	pickle.dump((model, corpus, id_word, text_list),
				open("../source_files/model_result_2.p", "wb"))

def analyze_topics(model):
	'''
	Method to analyze the different topics produced by the lda model

	Keyword Args:
	model - the lda model object
	'''

	topics = model.model.get_topics()
	topic_words = []
	for t in topics:
		words_for_topic = {}

		# taxing max from comparing to other topics
		for ot in topics:
			if not np.array_equal(ot, t):
				diff = t - ot

				# updting what max for topic is
				for i in range(len(diff)):
					if diff[i] in words_for_topic:
						if diff[i] < words_for_topic[i]:
							words_for_topic[i] = diff[i]
						else:
							pass
					else:
						words_for_topic[i] = diff[i]
		topic_words.append(words_for_topic)



	for t in topic_words:
		print("Topic:")
		sorts = sorted(t.items(), key=lambda x: x[1] * -1)
		# print(sorts)
		for i in range(20):
			print((model.id2word[sorts[i][0]], sorts[i][1]))

		print("\n")
	return topic_words

if __name__ == "__main__":

	########################################################################
	# jstor data
	# x = load_jstor_corpus()

	# # removing other language documents skewing our results
	# conditional = lambda x: not ("del" in x or "der" in x or "sich" in x or "qui" in x or "usepackage" in x)
	# x = list(filter(conditional, x))
	# print("JSTOR corpus extracted")


	########################################################################
	# web data
	# web_data = pickle.load(
	# 	open('source_files/uris/news_urls_2/webscraping_data_2.p', "rb"))
	# x = process_web_data(web_data)
	# print("web corpus extracted")


	########################################################################
	# Twitter data
	# flat_list = [item for sublist in l for item in sublist]

	x = [x for y in load_twitter_corpus() for x in y]
	x = [y.split() for y in x]
	# print("twitter corpus extracted")

	model = LDAModel(x, 5)
	model.gensim_topic_analysis(seeding=True)
	# model.evaluate_gensim_lda()
	analyze_topics(model)

	pickle.dump(model, open("../source_files/twitter/model_result.p", "wb"))
	# show_pyldavis(model, corpus, id_word)

