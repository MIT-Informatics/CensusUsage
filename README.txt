CensusUsage

@author - Dylan Sam (github.com/dsam99)
---------------------------------------
This repository is part of the CensusUsage project with the MIT Program of Information Science. 


*You must obtain a working consumer access token and secret access tokens.*

twitter_act.py
---------------
This is a file that contains the TwitterSearch class, which simply takes in the required access_tokens to interact with the Twitter Standard Search API.
The searchHashtag function simply queries the string parameter to the twitter API and recieves a list of tweets.

OAuth.txt
----------
A text file that you must update to contain consumer access tokens and secrets.

lda_topic.py
------------
A python script that contains information about running LDA topic modelling analysis on an input text corpus. The script utilizes the Gensim LDA package.

webscraping.py
--------------
A python script that webscrapes the information from multiple News sources from an input URI text file.

jstor.py 
--------
A python script for dealing with n-gram datasets from the JSTOR database.

error_analysis.py
-----------------
A python script for analyzing the results of webscraping. It takes in a list of URLs that had an error in parsing or have a poor status code.

