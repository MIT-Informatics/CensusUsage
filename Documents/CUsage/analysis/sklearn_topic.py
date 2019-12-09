# coding=utf-8
import sys
import pickle
import numpy as np

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append("../data_collection")

# importing data storage/extraction files
from tweets import *
from webscraping import *
from jstor import *

class lda_topic_model():
    '''
    Class for a sci-kit learn based implementation of a LDA topic model
    '''

    def __init__(self, num_topics, num_features, max_df=0.3, min_df=3, num_iter=5):
        '''
        Constructor for a lda_topic_model object

        Input:
        num_topics - a hyperparameter for the number of topics
        max_df - a hyperparameter representing the maximum document frequency
        min_df - a hyperparameter representing the number of min docs showed up in
        num_iter - a hyperparameter for the number of iterations
        '''
        self.num_topics = num_topics
        self.num_features = num_features
        self.model = None
        self.vectorizer = None
        self.max_df = max_df
        self.min_df = min_df
        self.num_iter = num_iter

    def train(self, corpus):
        '''
        Method to train a topic model using lda and the sci-kit learn package

        Input:
        corpus -  a list of texts representing documents
        '''

        # conerting corpus into vectors
        self.vectorizer = CountVectorizer(max_df=self.max_df,
                                        min_df=self.min_df,
                                        max_features=self.num_features,
                                        stop_words='english')

        tf = self.vectorizer.fit_transform(corpus)
        tf_feature_names = self.vectorizer.get_feature_names()

        self.model = LatentDirichletAllocation(n_topics=self.num_topics,
                                               max_iter=self.num_iter,
                                               learning_offset=50.,
                                               random_state=0).fit(tf)
        print("LDA Model Training Finished")


    def print_topics(self, top_n=10):
        for idx, topic in enumerate(self.model.components_):
            print("Topic %d:" % (idx))
            print([(self.vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])

def main():
    # loading twitter corpus
    corpus = pickle.load(open("../source_files/twitter/twitter_corpus.p", "rb"))

    model = lda_topic_model(5, 1000)

    corpus = [x for y in corpus for x in y]
    model.train(corpus)
    model.print_topics()


if __name__ == "__main__":
    main()
