from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

import gensim
from gensim.models import ldamodel


from twitter_act import *
from process import *



#setting paramteres for training
#number of top features that appear in corpus to be considered
number_features = 1000
#number of topics to print eventually
number_topics = 20
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
    Method to model the topics of an input text list
    @param text_list - a list of strings to analyze and produce the topics
    '''
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
    for item in words:
        word_list.append(process_string(item))

    sk_topic_analysis(word_list)