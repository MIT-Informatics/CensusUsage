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
import pyLDAvis.gensim

#importing data storage/extraction files
from tweets import *
from webscraping import *
from jstor import *

# to help with pickling html files from BeautifulSoup
sys.setrecursionlimit(40000)

# number of topics to print eventually
number_topics = 5


def change_priors(word, topic_number, id2word, eta, weight):
    '''
    Method to change the value of a weight inside of an eta priors list

    Keyword args:
    word - the word to change the value of
    topic_number - the topic which to scale the probability by
    id2word - the id2word dictionary object
    eta - the list of topic to word probability values
    weight - the amount to scale the value of word by

    :Return:
    the updated version of eta

    '''

    # looking up id of word in dictionary
    
    for i in id2word.keys():
        if word == id2word[i]:
            word_id = id2word.token2id[word]
            # multiplying the value of the current prob in eta
            eta[topic_number][word_id] *= weight
            print(word + " found in word vectors")

    return eta


def create_priors(id2word):
    '''
    Method to create the eta matrix that will be used in the lda model as a prior based from topic/word probability
    @param id2word - the id to word lookup dictionary
    @returns - the eta matrix
    '''

    eta = []

    # initializing each topic with values for each word
    for topic in range(number_topics):
        eta.append(np.ones(len(id2word)) * 0.1)
        # aggressively seed the word 'system', in one of the
        # two topics, 10 times higher than the other words

    # values for seed words in each topic
    seed_words = [
        (u'model', 0), (u'statistics', 0), (u'graph', 0),
        (u'academic', 1), (u'academia', 1), (u'research', 1),
        (u'commercial', 2), (u'industry', 2), (u'town', 2),
        (u'people', 3), (u'place', 3),
    ]

    # iterating through seed words and scaling eta
    for (word, topic_num) in seed_words:
        eta = change_priors(word, topic_num, id2word, eta, 5)

    eta = change_priors('people', 3, id2word, eta, 1.5)

    return eta


def gensim_topic_analysis(text_list, seeding=False):
    '''
    Method to model the topics of an input text list using gensim
    @param text_list - a list of words to analyze and produce topics
    @param seeding - boolean representing if seeding should be used in model, default to false
    @return - the lda model produced
    '''

    # formatting dictionary to use in LDA model
    id_word = corpora.Dictionary(text_list)
    texts = text_list
    corpus = [id_word.doc2bow(text) for text in texts]

    # checking if seeding parameter is passed -> default to false
    if seeding:
        eta = None
        per_word_topics = True
    else:
        print("Using seeded model")
        eta = create_priors(id_word)
        per_word_topics = False

    # creating LDA model with parameterized topics and training passes
    # params - dictionary, number topics to print, seed to create random
    # number array, iterative learning through 1 doc per pass, 3 docs per
    # pass, 5 passes, normalized prior, bool to sort topics in order
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                          id2word=id_word,
                                          num_topics=number_topics,
                                          random_state=100,
                                          update_every=1,
                                          chunksize=3,
                                          passes=5,
                                          alpha='auto',
                                          eta=eta,
                                          per_word_topics=per_word_topics)

    print(lda.print_topics())
    return lda, corpus, id_word, text_list


def evaluate_gensim_lda(lda_model, corpus, id_word, text_list):
    '''
    Method to evaluate the lda model provided by gensim
    @param lda_model - the LDA model created by the gensim package
    @param corpus - the total corpus evaluated by the lda model
    @param id_word - the dictionary created by the corpus
    @param text_list - the initial B.O.W formatted text
    '''

    # held out likelihood score - simplified idea as predictablity of model
    print("Perplexity: ", lda_model.log_perplexity(corpus))
    coherence = CoherenceModel(
        model=lda_model, texts=text_list, dictionary=id_word, coherence='c_v')

    # multiple pipeline score evaluated quality of topics
    score = coherence.get_coherence()
    print("Coherence", score)


def show_pyldavis(model, corpus, dictionary):
    '''
    Method to create a pyLdavis visualization of the topic model
    @param model - the trained gensim model
    @param corpus - the total corpus the model was trained on
    @param dictionary - the dictionary created in the model
    '''

    prepared_data = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    pyLDAvis.show(prepared_data)


def apply_model(model, dictionary, document):
    '''
    Method to apply the trained lda model to a subset document
    @param model - the model to apply on the document
    @param document - the document in the form of a list of words to apply the model on
    @return - the probability vector that is represents the probability that the document falls within each topic of the model
    '''

    doc = dictionary.doc2bow(document)

    # creating the probability vector of the document in the model
    vector = model[doc][0]
    return vector


def split_documents(model, id_word, text_list):
    '''
    Method to split documents by the topic that they fall in
    @param model - the trained lda topic model
    @param all_docmuents - the total documents that the model was trained on
    @return a dictionary from topic number to the corresponding list of documents that are most likely to fall in the topic
    '''

    topic_dict = {i: [] for i in range(9)}

    for doc in text_list:
        vector = apply_model(model, id_word, doc)

        # finding the max probability tuple based on second element
        max_prob = max(vector, key=lambda x: x[1])

        topic_dict[max_prob[0]].append(doc)

        return topic_dict


def train_model(corpus_data):
    '''
    Method to automate training of the lda model on the an input textlist
    @param text_list - an input list of list of words in bow format
    '''

    # fitting gensim model
    gensim_model, corpus, id_word, text_list = gensim_topic_analysis(
        corpus_data, seeding=True)

    print("finished training model")

    # storing model
    # pickle.dump((gensim_model, corpus, id_word, text_list),
    # open("source_files/model_result_2.p", "wb"))


def evaluate_model():
    '''
    Method to evaluate the training of the lda model
    '''

    gensim_model, corpus, id_word, text_list = pickle.load(
        open('source_files/model_result_2.p', 'rb'))

    # evaluating perplexity and coherence of gensim lda model
    evaluate_gensim_lda(gensim_model, corpus, id_word, text_list)

    # create visualization of pyldavis
    show_pyldavis(gensim_model, corpus, id_word)


if __name__ == "__main__":

    x = process_web_data('source_files/webscraping_data_2.p')
    gensim_model, corpus, id_word, text_list = gensim_topic_analysis(
        x[:10], seeding=True)
    show_pyldavis(gensim_model, corpus, id_word)


    # print(create_priors(id_word))
