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
import pickle

# retrieving stopwords to remove from data analysis
stop_words = set(stopwords.words("english"))
bigram_filter_list = []

stop_words.add('rt')  # adding retweet to stopwords for Twitter API

# adding to stop words for newspaper
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
stop_words.add("would")
stop_words.add("here")
stop_words.add("I")

# from news articles bigram analysis
stop_words.add("inline")
stop_words.add("div")
stop_words.add("cta")
stop_words.add("bin")
stop_words.add("url")
stop_words.add("jQuery")
stop_words.add("css")
stop_words.add("module")
stop_words.add("is")
stop_words.add("sailthru")
stop_words.add("btn")
stop_words.add("bg")
stop_words.add("href")
stop_words.add("magazine")
stop_words.add("font")

# from JSTOR dataset
stop_words.add("from")
stop_words.add("also")
stop_words.add("which")
stop_words.add("can")
stop_words.add("were")
stop_words.add("has")
stop_words.add("been")
stop_words.add("some")
stop_words.add("than")
stop_words.add("however")
stop_words.add("would")
stop_words.add("given")
stop_words.add("have")


# creating stemmer to stem through tweets
ps = PorterStemmer()

# creating lemmatizer to sdafads synonyms of words to normalize
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
        # as default for lemmatizer
        return wordnet.NOUN


def clean_text(text):
    '''
    Method to remove the punctuation from a unicode text input
    @param text - the text input
    @return - the same input with all punctuation removed
    '''

    # removing any non ascii characters that cannot be read
    text = text.replace("  ", " ")

    # removing other special characters
    text = re.sub(r'[-|\n|\t|\']', '', text)

    # removing punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # removing numbers
    text = re.sub(r'[0-9]', '', text)

    return text.lower()


def create_bigram_model(corpus):
    '''
    Method to create a bigram model
    @param corpus - the corpus to train the model on
    @return - the trained bigram model
    '''

    phrases = Phrases(corpus, min_count=400, threshold=200)
    bigram = Phraser(phrases)
    return bigram


def remove_stopwords(words):
    '''
    Method to remove stopwords from a text corpus
    
    Keyword Args:
    words - the corpus input

    Returns:
    the corpus with all stopwords
    '''

    # applying model to words from corpus
    removed = list(filter(lambda x: x.lower() not in stop_words, words))
    return removed


def process_string(text):
    '''
    Method to do preprocessing of an input text
    @param text - raw text to process
    @return - the processed version of the text
    '''

    # tokenizing string into words and POS tagging and removing all punctuation
    words = word_tokenize(clean_text(text))
    words = filter(lambda x: x != "", words)

    removed = remove_stopwords(words)

    tags = nltk.pos_tag(removed)
    return_text = ''
    return_list = []

    # removing stop words in strings and lemmatizing
    for word, tag in tags:
        return_text += lemmatizer.lemmatize(word, convert_pos(tag)) + " "
        return_list.append(lemmatizer.lemmatize(word, convert_pos(tag)))

    return return_text, return_list


def process_bow(bow_doc):
    '''
    Method to do preprocessing of an input bag of words
    @param bow - a bag of words format input of strings
    @return - return text, return list
    '''

    # processing the bow documents
    words = [clean_text(i) for i in bow_doc]

    words = filter(lambda x: x != "", words)

    # applying model to words from corpus
    removed = remove_stopwords(words)

    tags = nltk.pos_tag(removed)
    return_text = ''
    return_list = []

    # removing stop words in strings and lemmatizing
    for word, tag in tags:
        return_text += lemmatizer.lemmatize(word, convert_pos(tag)) + " "
        return_list.append(lemmatizer.lemmatize(word, convert_pos(tag)))
    return return_text, return_list

if __name__ == '__main__':
    corpus = pickle.load(open('source_files/webscraping_data.p'))
    word_list = []
    for doc in corpus:
        word_list.append(process_string(doc)[1])

    bigram_model = create_bigram_model(word_list)
    bigram_word_list = list(bigram_model[word_list])
    print(bigram_word_list)

    count = 0

    for sentence in bigram_word_list:
        for word in sentence:
            if "_" in word:
                print(word)
                count += 1

    print(count)
