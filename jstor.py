import numpy as np
import pickle
from gensim.corpora import Dictionary
import os
import glob


def ngram_to_doc(filepath):
    '''
    Method to read a ngrams file into a usable BOW gensim file
    @param filepath - the path to the ngram file to read
    @return - the document to add to the dictionary
    '''

    # raw data read from n gram file
    n_gram_data = []

    # format data to add to corpus
    document = []

    # opening and reading file
    with open(filepath, "rb") as f:
        # reading first line of file
        line = f.readline()

        while line:
            tuple_data = line.split("\t")
            # creating triple of word, count, id
            n_gram_data.append(
                (tuple_data[0], tuple_data[1].replace('\n', '')))
            line = f.readline()

    # adding words to document
    for tup in n_gram_data:
        for i in range(int(tup[1])):
            document.append(tup[0])

    return document


def create_text_list(filepath_list):
    '''
    Method to create a full dictionary from the text in ngram files from JSTOR database
    @param filepath_list - a list of file paths to multiple ngram files from JSTOR
    @return - the dictionary containing all the documents
    '''

    text_list = []

    # reading all files
    for file in filepath_list:
        doc = ngram_to_doc(file)
        # adding document to total dictionary
        text_list.append(doc)

    return text_list

# an example path from the jstor database
example_path = "source_files/american_census_data_ngram/journal-article-10.5406_ethnomusicology.55.2.0200-ngram1.txt"

all_files = glob.glob("source_files/american_census_data_ngram/*")
