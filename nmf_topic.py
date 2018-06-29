from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_20newsgroups
from twitter_act import *
from process import *

#creating data set of 20 news articles
# dataset = fetch_20newsgroups(shuffle=True, random_state=1,
#                              remove=('headers', 'footers', 'quotes'))


#setting paramteres for training
features = 10
components = 10
number_topics = 5

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
                             for i in topic.argsort()[:-number_topics - 1:-1]])
        print(message)

def topic_analysis(text_list):

    '''
    Method to calculate the topics of an input text list
    @param text_list - the list of text to analyze
    '''

    #Creating vectorizer to take raw text data and transform into tf-idf matrix
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=features,
                                       stop_words='english')

    print("Fitting tfidf vectorizer to samples")
    tfidf = tfidf_vectorizer.fit_transform(text_list)


    # Fit the NMF model
    print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
          "n_samples=%d and n_features=%d..."
          % (len(text_list), features))

    nmf = NMF(n_components=components, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)

    print("\nTopics in NMF model (Frobenius norm):")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print((tfidf_feature_names))
    print_topics(nmf, tfidf_feature_names, number_topics)

if __name__ == '__main__':
    
    #using twitter_act file to access Twitter API
    t = main()
    #searching for tweets with #ACS hashtag
    tweets_objects = t.searchHashtag('uses census data')

    tweets_text = []

    #adding only text into tweet_text list
    for item in tweets_objects:
        tweets_text.append(process_string(item['text']))

    topic_analysis(tweets_text)
    # data_samples = dataset.data[:n_samples]

    
