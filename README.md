# CensusUsage

The CensusUsage project's goal is to better understand and catogorize the different usages of Census data by purpose and methodology.  

This repository contains the data mining/processing pipeline and exploratory data analysis via unsupervised machine learning and natural language processing. 

The files jstor.py, webscraping.py, and tweets.py/extract_tweets.py are the main files in the data mining pipeline. They extract data from three different sources: JSTOR's academic database, NewsURIs from the Census Burueau, and the GetOldTweets github package/Twitter API. 

After extracting these corpuses, the text is processed using the process.py file and its different methods. The methods include removing punctuation, stopwords, stemming, lemmatizing, etc. 

I have two different implementations of the LDA topic model, which is an unsupervised learning algorithm to cluster the documents into different topics based on word probability distributions. I first implemented the model using the gensim package, which creates a model object and takes in a text corpus. It takes in a corpus in a bag-of-words format. This implementation has the most functionality and has been run on all of the three different data sets. The Sci-kit Learn implementation is contained within sklearn.py and it takes in a corpus in document form, which is a list of strings. It creates a count vectorizer and puts it through the lda model. This implementation has only been run on the twitter data.


-------------------------------

Dylan Sam, Brown University '21