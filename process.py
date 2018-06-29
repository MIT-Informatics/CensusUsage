import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

import string
import regex as re


#retrieving stopwords to remove from data analysis
stop_words = set(stopwords.words("english"))
stop_words.add('rt') #adding retweet to stopwords

#creating stemmer to stem through tweets
ps = PorterStemmer()

#creating lemmatizer to choose synonyms of words to normalize
lemmatizer = WordNetLemmatizer()

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

def remove_punctuation(text):
    
    '''
    Method to remove the punctuation from a unicode text input
    @param text - the text input
    @return - the same input with all punctuation removed
    '''

    return re.sub(ur"\p{P}+", "", text)

def process_string(text):

	'''
	Method to do preprocessing of an input text
	@param text - raw text to process
	@return - the processed version of the text
	'''

	#tokenizing string into words and POS tagging and removing all punctuation
	words = word_tokenize(remove_punctuation(text))
	removed = list(filter(lambda x: x not in stop_words, words))

	tags = nltk.pos_tag(removed)
	return_text = ''

	#removing stop words in strings and lemmatizing
	for word, tag in tags:
		return_text += lemmatizer.lemmatize(word, convert_pos(tag)) + " "
	return return_text


if __name__ == '__main__':
	print(process_string("Hi what is your name? I'm so hungry, I want to eat some food."))