# coding=utf-8

import gensim
from gensim import corpora
from gensim.models import ldamodel, CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import Text8Corpus, Word2Vec


import numpy as np
from twitter_act import *
from process import *
from webscraping import *

import pickle

#using pyLDAvis to visualize
import pyLDAvis.gensim

#number of topics to print eventually
number_topics = 9

def gensim_topic_analysis(text_list):

    '''
    Method to model the topics of an input text list using gensim
    @param text_list - a list of words to analyze and produce topics
    @return - the lda model produced
    '''

    #formatting dictionary to use in LDA model
    id_word = corpora.Dictionary(text_list)
    texts = text_list
    corpus = [id_word.doc2bow(text) for text in texts]

    #creating LDA model with parameterized topics and training passes
    #params - dictionary, number topics to print, seed to create random number array, iterative learning through 1 doc per pass, 3 docs per pass, 5 passes, normalized prior, bool to sort topics in order
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                          id2word=id_word,
                                          num_topics=number_topics,
                                          random_state=100,
                                          update_every=1,
                                          chunksize=3,
                                          passes=5,
                                          alpha='auto',
                                          per_word_topics=True)

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

    #held out likelihood score - simplified idea as predictablity of model
    print("Perplexity: ", lda_model.log_perplexity(corpus))
    coherence = CoherenceModel(model=lda_model, texts=text_list, dictionary=id_word, coherence = 'c_v')

    #multiple pipeline score evaluated quality of topics
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

def apply_model(model, document, dictionary):

    '''
    Method to apply the trained lda model to a subset document
    @param model - the model to apply on the document
    @param document - the document in the form of a list of words to apply the model on
    @return - the probability vector that is represents the probability that the document falls within each topic of the model
    '''

    doc = dictionary.doc2bow(document)

    #creating the probability vector of the document in the model
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
        vector = apply_model(model, doc, id_word)

        #finding the max probability tuple based on second element
        max_prob = max(vector,key=lambda x:x[1])

        topic_dict[max_prob[0]].append(doc)

    return topic_dict

def run_webscraping():

    '''
    Method to automate the running of the webscraping and pickling of the files
    '''

    webscraping_data, error_links, result_dict, html_corpus = webscrape()

    # Storing files for later use
    pickle.dump(webscraping_data, open("webscraping_data.p", "wb"))

    with open('error_urls.txt', 'wb') as f:
        for error in error_links:
            f.write(error + '\n')

    pickle.dump(result_dict, open("result_dictionary.p", "wb"))
    pickle.dump(html_corpus, open("html_corpus.p", "wb"))

def train_model():

    '''
    Method to automate training of the lda model on the corpus data
    '''
    
    webscraping_data = pickle.load(open("webscraping_data.p", "rb"))

    word_list = []

    #formatting words to be analyzed for both sklearn and gensim
    for item in webscraping_data:
        word_list.append(process_string(item)[1])

    #filtering out empty strings
    word_list = filter(lambda x: len(x) != 0, word_list)

    #creating bigrams to run model on
    bigram_model = create_bigram_model(word_list)
    bigram_word_list = list(bigram_model[word_list])

    #fitting gensim model
    gensim_model, corpus, id_word, text_list = gensim_topic_analysis(bigram_word_list)

    #storing model
    pickle.dump((gensim_model, corpus, id_word, text_list), open("model_result_bigrams_count5_threshold2.p", "wb"))
    
def evaluate_model():

    '''
    Method to evaluate the training of the lda model
    '''

    gensim_model, corpus, id_word, text_list = pickle.load(open('model_result_bigrams.p', 'rb'))

    #evaluating perplexity and coherence of gensim lda model
    evaluate_gensim_lda(gensim_model, corpus, id_word, text_list)

    #create visualization of pyldavis
    show_pyldavis(gensim_model, corpus, id_word)

#when running script
if __name__ == "__main__":
    train_model()



