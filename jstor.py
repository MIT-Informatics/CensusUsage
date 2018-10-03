# encoding: utf-8

import numpy as np
import pickle
from gensim.corpora import Dictionary
import os
import glob
from process import *
from string import digits


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
    with open(filepath, "r") as f:
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
        word = tup[0].translate(digits)
        # word = word.decode("ascii", errors="ignore").encode('ascii')
        if not word in stop_words and len(word) > 2:
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


def run_jstor():
    '''
    Method to run the topic model on JSTOR datasets
    '''

    all_files = glob.glob("source_files/american_community_survey_ngram/*")
    text_list = create_text_list(all_files)

    cleaned_text = []
    for i in text_list:
        cleaned_text.append(process_bow(i)[0])

    return cleaned_text

# an example path from the jstor database
example_path = "source_files/american_census_data_ngram/journal-article-10.5406_ethnomusicology.55.2.0200-ngram1.txt"

all_files = glob.glob("source_files/american_community_survey_ngram/*")

if __name__ == "__main__":
    print(create_text_list(all_files))
