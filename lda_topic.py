from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

from sklearn.metrics import log_loss

import gensim
from gensim import corpora
from gensim.models import ldamodel, CoherenceModel

import numpy as np
from twitter_act import *
from process import *



#setting paramteres for training
#number of top features that appear in corpus to be considered
number_features = 1000
#number of topics to print eventually
number_topics = 5
#n-gram size to consider
n_size = 2

def print_topics(model, feature_names, n_top_words):
    
    '''
    Method to print the 'top' most important words of the features
    @param model - a trained model to evaluate the features
    @param feature_names - features of the document to evaluate
    @param n_top_words - the number of top words to print
    '''

    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)

def sk_topic_analysis(text_list):
    '''
    Method to model the topics of an input text list using sklearn
    @param text_list - a list of strings to analyze and produce the topics
    @return - the lda model produced
    '''

    #df determines which words to ignore with a document frequency between 2 and 0.95
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=number_features,
                                stop_words='english')
    
    tf = tf_vectorizer.fit_transform(text_list)

	#creating lda model
    lda = LatentDirichletAllocation(n_components=number_topics, max_iter=5,
                                learning_method='online',
                                random_state=0)
	#fitting model to Bag Of Words vector
    lda.fit(tf)

    tf_feature_names = tf_vectorizer.get_feature_names()

    print_topics(lda, tf_feature_names, n_size)
    return lda

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

def loglikelihood(true, model):
    
    '''
    Method to calculate the loglikelihood and evaluate goodness of topic model
    @param labels - the accepted values of the data
    @param preds - the values calculated by the model
    '''

    norm_prediction = model.components_ / model.components_.sum(axis=1)[:,np.newaxis]
    return log_loss(true, norm_prediction)

def evaluate_gensim_lda(lda_model, corpus, id_word, text_list):

    '''
    Method to evaluate the lda model provided by gensim
    @param lda_model - the LDA model created by the gensim package
    @param corpus - the total corpus evaluated by the lda model
    @param id_word - the dictionary created by the corpus
    @param text_list - the initial B.O.W formatted text
    '''

    print("Perplexity: ", lda_model.log_perplexity(corpus))
    coherence = CoherenceModel(model=lda_model, texts=text_list, dictionary=id_word, coherence = 'c_v')
    score = coherence.get_coherence()
    print("Coherence", score)

#when running script
if __name__ == "__main__":
    #using twitter_act file to access Twitter API
    t = main()
    #searching for tweets with #ACS hashtag
    # tweets_objects = t.searchHashtag('#ACS')

    # tweets_text = []

    # adding only text into tweet_text list
    # for item in tweets_objects:
    #     tweets_text.append(process_string(item['text']))

    # topic_analysis(tweets_text)
    words = [u'Hi what is your name.', u"Ever need to add some random or meaningless text into Microsoft Word to test a document, temporarily fill some space, or to see how some formatting looks? Luckily, Word provides a couple of quick and easy methods for entering random text into your document.", u"To do this, position the cursor at the beginning of a blank paragraph. Type the following and press Enter. It does not matter if you use lowercase, uppercase, or mixed case.", u"I'm a student and research assistant at Brown University, who is studying the fields of Computer Science and Applied Math. I am currently working in the Rubenstein Lab within the DARPA Molecular Informatics Project and as an intern in the MIT Program of Information Science. I am particularly interested in the ability to perceive information from data sets through analytical methods. I love research, data science, and machine learning. I am always looking for opportunities to learn new skills.", u"At school, I am on the organizing team of the Brown Data Science club and the Fintech at Brown, a member of the Brown Club Tennis Team, and a writer for the Ursa Sapiens Blog. In my free time, I love to play soccer and the viola."]

    word_list = []
    sentence_list = []
    for item in words:
        sentence_list.append(process_string(item)[0])
        word_list.append(process_string(item)[1])
    sklearn_model = sk_topic_analysis(sentence_list)
    gensim_model, corpus, id_word, text_list = gensim_topic_analysis(word_list)
    evaluate_gensim_lda(gensim_model, corpus, id_word, text_list)

    # print(loglikelihood(['Brown University', 'data science', 'computer science', 'MIT internship', 'DARPA Moleculear', 'research assistant', 'brown club'], model))