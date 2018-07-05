from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_20newsgroups
from twitter_act import *
from process import *

#creating data set of 20 news articles
# dataset = fetch_20newsgroups(shuffle=True, random_state=1,
#                              remove=('headers', 'footers', 'quotes'))

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

def topic_analysis(text_list):

    '''
    Method to calculate the topics of an input text list
    @param text_list - the list of text to analyze
    '''

    #Creating vectorizer to take raw text data and transform into tf-idf matrix
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=number_features,
                                       stop_words='english')

    print("Fitting tfidf vectorizer to samples")
    tfidf = tfidf_vectorizer.fit_transform(text_list)

    nmf = NMF(n_components=number_topics, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)

    print("\nTopics in NMF model (Frobenius norm):")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print((tfidf_feature_names))
    print_topics(nmf, tfidf_feature_names, n_size)

if __name__ == '__main__':
    
    #using twitter_act file to access Twitter API
    t = main()
    #searching for tweets with #ACS hashtag
    # tweets_objects = t.searchHashtag('uses census data')

    # tweets_text = []

    #adding only text into tweet_text list
    # for item in tweets_objects:
    #     tweets_text.append(process_string(item['text']))

    # topic_analysis(tweets_text)
    # data_samples = dataset.data[:n_samples]

    words = [u'Hi what is your name.', u"Ever need to add some random or meaningless text into Microsoft Word to test a document, temporarily fill some space, or to see how some formatting looks? Luckily, Word provides a couple of quick and easy methods for entering random text into your document.", u"To do this, position the cursor at the beginning of a blank paragraph. Type the following and press Enter. It does not matter if you use lowercase, uppercase, or mixed case.", u"I'm a student and research assistant at Brown University, who is studying the fields of Computer Science and Applied Math. I am currently working in the Rubenstein Lab within the DARPA Molecular Informatics Project and as an intern in the MIT Program of Information Science. I am particularly interested in the ability to perceive information from data sets through analytical methods. I love research, data science, and machine learning. I am always looking for opportunities to learn new skills.", u"At school, I am on the organizing team of the Brown Data Science club and the Fintech at Brown, a member of the Brown Club Tennis Team, and a writer for the Ursa Sapiens Blog. In my free time, I love to play soccer and the viola."]

    word_list = []
    for item in words:
        word_list.append(process_string(item))

    topic_analysis(word_list)

    
