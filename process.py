import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import defaultdict

from gensim.test.utils import datapath
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import Text8Corpus

import string
import regex as re

from webscraping import *


#retrieving stopwords to remove from data analysis
stop_words = set(stopwords.words("english"))

stop_words.add('rt') #adding retweet to stopwords for Twitter API

#adding to stop words for newspaper
stop_words.add("footnotes")
stop_words.add("advertisement")
stop_words.add("ET")
stop_words.add("PM")
stop_words.add("AM")
stop_words.add("really")
stop_words.add("also")
stop_words.add("said")
stop_words.add("saying")
stop_words.add("say")
stop_words.add("jerry")
stop_words.add("baker")
stop_words.add("chronicle")
stop_words.add("would")
stop_words.add("story")
stop_words.add("here")
stop_words.add("I")


#creating stemmer to stem through tweets
ps = PorterStemmer()

#creating lemmatizer to sdafads synonyms of words to normalize
lemmatizer = WordNetLemmatizer()

#training bigram model on gensim corpus
sentences = Text8Corpus(datapath('testcorpus.txt'))
phrases = Phrases(sentences, min_count=1, threshold=1)  	
bigram = Phraser(phrases)

def convert_pos(pos_tag):
	
	'''
	Method to convert the pos_tag into a wordnet tag to use in lemmatizer
	@param pos_tag - a POS_tag from the nltk
	@return - the wordnet version of the POS tag
	'''

	if pos_tag.startswith('J'):
	    return wordnet.ADJ
	elif pos_tag.startswith('V'):
	    return wordnet.VERB
	elif pos_tag.startswith('N'):
	    return wordnet.NOUN
	elif pos_tag.startswith('R'):
	    return wordnet.ADV
	else:
		#as default for lemmatizer
	    return wordnet.NOUN

def clean_text(text):
    
    '''
    Method to remove the punctuation from a unicode text input
    @param text - the text input
    @return - the same input with all punctuation removed
    '''

    #removing any non ascii characters that cannot be read
    text = text.encode("ascii", "ignore")

    text = text.replace("  ", " ")

	#removing other special characters    
    text = re.sub(r'[-|\n|\t|\']','',text)

    #removing punctuation
    text = re.sub(r'[^\w\s]','',text)
    	
    #removing numbers
    text = re.sub(r'[0-9]','',text)

    return text

def process_string(text):

	'''
	Method to do preprocessing of an input text
	@param text - raw text to process
	@return - the processed version of the text
	'''

	#tokenizing string into words and POS tagging and removing all punctuation
	words = word_tokenize(clean_text(text))
	words = filter(lambda x: x != "", words)

	# applying model to words from corpus
	removed = list(filter(lambda x: x.lower() not in stop_words, bigram[words]))

	tags = nltk.pos_tag(removed)
	return_text = ''
	return_list = []

	#removing stop words in strings and lemmatizing
	for word, tag in tags:
		return_text += lemmatizer.lemmatize(word, convert_pos(tag)) + " "
		return_list.append(lemmatizer.lemmatize(word, convert_pos(tag)))
	return return_text, return_list


if __name__ == '__main__':
	corpus = webscrape()

	for doc in corpus:
		print(process_string(doc))