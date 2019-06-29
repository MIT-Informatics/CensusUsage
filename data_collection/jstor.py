# encoding: utf-8

import os
import numpy as np
import pickle
import re
import math

import xml.etree.ElementTree as ET
import glob
from gensim.corpora import Dictionary
from string import digits

from process import *


def ngram_to_doc(filepath):
    '''
    Method to read a ngrams file into a usable BOW gensim file
    @param filepath - the path to the ngram file to read
    @return - the document to add to the dictionary
    '''

    # format data to add to corpus
    document = []

    # opening and reading file
    with open(filepath, "r") as f:
        # reading first line of file
        line = f.readline()

        while line:
            tuple_data = line.split("\t")
            # reading word and frequency information
            word = tuple_data[0]
            count = int(tuple_data[1].replace("\n", ''))

            # processing text string
            word = clean_text(word)

            # adding word to document the number of times as the unigram count
            if not word in stop_words and len(word) > 2:
                for i in range(count):
                    document.append(word)

            line = f.readline()

    return document


def create_text_list(filepath_list):
    '''
    Method to create a full dictionary from the text in ngram files from JSTOR database
    @param filepath_list - a list of file paths to multiple ngram files from JSTOR
    @return - the dictionary containing all the documents
    '''

    text_list = []
    total = len(filepath_list)
    count = 0

    # reading all files
    for file in filepath_list:
        doc = ngram_to_doc(file)
        # adding document to total dictionary
        text_list.append(doc)

        if count % 50 == 0:
            print("[" + str(count) + " / " + str(total) + " files read]")

        count += 1

    return text_list


def create_jstor_corpus():
    '''
    Method to run the topic model on JSTOR datasets
    '''

    all_files = glob.glob(
        "source_files/jstor/american_community_survey_ngram/*")
    text_list = create_text_list(all_files)

    cleaned_texts = []

    total = len(text_list)
    count = 0

    for i in text_list:
        text = process_bow(i)[1]
        if text != []:
            cleaned_texts.append(text)
            if count % 50 == 0:
                print("[" + str(count) + " / " + str(total) + " files read]")
        count += 1

    return cleaned_texts


def load_jstor_corpus():
    '''
    Method to load the JSTOR corpus from a pickled file
    '''

    return pickle.load(open("source_files/jstor/jstor_corpus_bow_2.p", "rb"))


def read_journal_titles(directory):
    '''
    Method to extract the titles of the journals from the JSTOR database

    Keyword Args:
    directory - a list of filenames that contain metadata about the JSTOR journals

    Return:
    a list containing all the names of the journal entries
    '''

    titles = {}
    count = 0

    # counts for weird title names
    fm_count = 0
    bm_count = 0
    total_files = len(directory)

    pattern = r"<article-title>.*</article-title>|<book-title>.*</book-title>"

    for filepath in directory:

        if count % 2000 == 0:
            print("[ " + str(count) + " / " + str(total_files) + "] files read")

        with open(filepath) as f:
            text = f.read()

        match = re.findall(pattern, text)

        # checking number of titles
        if len(match) == 0:
            print("No titles in metadata")
        else:
            cleaned = match[0].replace("<article-title>", "")
            cleaned = cleaned.replace("</article-title>", "")
            cleaned = cleaned.replace("<book-title>", "")
            cleaned = cleaned.replace("</book-title>", "")

            # finding articles with front matter or back matter names
            if cleaned == "Front Matter":
                fm_count += 1
            elif cleaned == "Back Matter":
                bm_count += 1
            else:
                if cleaned in titles:
                    print(cleaned)
                else:
                    titles[cleaned] = filepath

        count += 1

    print(len(titles))
    print("Back Matter count: " + str(bm_count))
    print("Front Matter count: " + str(fm_count))
    return titles


def store_titles():
    '''
    Method to write all the files from the JSTOR database to a .txt file

    Keyword Args:
    void

    Return:
    void
    '''

    titles = read_journal_titles(all_metadata)
    titles_set = set(titles.keys())
    with open("source_files/jstor/jstor_titles.txt", "w") as f:
        for title in titles_set:
            f.write(title + "\n")


def convert_metadata_ngram(filepath):
    '''
    Method to convert a metadata filepath to an ngram filepath

    Keyword Args:
    filepath - the filepath to the metadata .xml file

    Returns:
    a filepath to the corresponding -ngrams.txt file
    '''

    toReturn = filepath.replace("metadata", "ngram")
    return toReturn.replace(".xml", "-ngram1.txt")


def split_journal_titles(titles, words):
    '''
    Method to split journal articles into two subsets based on a set of words

    Keyword Args:
    titles - a dictionary of (title -> filepath) mappings
    words - the set of words to split on

    Return:
    two lists, the first is all the titles that contain any word in words and the
    second list containing all other titles
    '''

    titles_with = []
    titles_without = []

    for title in titles:
        added = False
        for word in words:
            if word in title.lower():
                titles_with.append(
                    (title, convert_metadata_ngram(titles[title])))
                added = True
                break
        if not added:
            titles_without.append(
                (title, convert_metadata_ngram(titles[title])))

    return titles_with, titles_without


def calculate_word_frequencies(tuples_list):
    '''
    Method to calculate the different in word frequency of documents that are
    split by containing specific keywords

    Keyword Args:
    tuples_list - a list of tuples of (titles, filepath) to the ngram file

    Returns:
    a dictionaries containing (word -> frequency) mappings of the tuples_list
    and the number of articles
    '''

    doc2freq = {}

    # looping through files
    for title, file in tuples_list:
        document = ngram_to_doc(file)
        words = list(map(lambda x: x.lower(), document))

        # removing any duplicates from list
        words = set(words)

        # incrementing counts in dictionary
        for word in words:
            if word in doc2freq:
                doc2freq[word] += 1
            else:
                doc2freq[word] = 1

    return doc2freq, len(tuples_list)


def create_split_corpuses(words_with, words_without):
    '''
    Method to create corpuses from dictionaries of (title->filepath) dictionaries

    Keyword Args:
    words_with - a dictionary of (title->filepath) that has titles that contain the words
    words_without - a dictionary of (title->filepath) that has titles that doesn't contain the words

    Returns:
    two strings that contain all the words with and words without
    '''

    links_w = list(map(lambda x: x[1], words_with))
    links_wout = list(map(lambda x: x[1], words_without))

    texts_w = create_text_list(links_w)
    texts_wout = create_text_list(links_wout)

    string_w = flatten_string_list(texts_w)
    string_wout = flatten_string_list(texts_wout)

    return string_w, string_wout, texts_w, texts_wout


def flatten_string_list(bow):
    '''
    Method to flatten a string list by adding and placing a space in between

    Keyword Args:
    bow - a list of strings that represent each document strings

    Returns:
    a string containing all the strings concattenated together
    '''
    acc = ""
    for string_list in bow:
        for s in string_list:
            acc += s + " "
    return acc


def calculate_different_frequencies(directory, words_list):
    '''
    Method to calculate the different in word frequency of documents that are split by containing specific keywords

    Keyword Args:
    directory - a list of files that contain JSTOR metadata
    words_list - a list of keywords

    Returns:
    two dictionaries containing (word -> frequency) mappings for the articles with a keyword and articles without a keyword
    '''

    # creating titles dictionary
    titles = read_journal_titles(directory)

    # splitting dictionary into list of tuples
    titles_with, titles_without = split_journal_titles(titles, words_list)

    # calculating frequencies with and without word sets
    count_with, num_with = calculate_word_frequencies(titles_with)
    count_without, num_without = calculate_word_frequencies(titles_without)

    freq_with = {}
    freq_without = {}

    # converting frequencies to ratios
    for word in count_with:
        freq_with[word] = (float(count_with[word]) / num_with)

    for word in count_without:
        freq_without[word] = (float(count_without[word]) / num_without)

    print("Articles with: " + str(num_with) +
          " | Articles without: " + str(num_without))

    return freq_with, freq_without, count_with, count_without


def calculate_max_diff_freq(freq_with, freq_without):
    '''
    Method to sort words by difference in frequency of documents with titles
    containing the analytical word set vs documents with titles that do not
    contain the word set

    Keyword Args:
    freq_with - a dictionary of word to doc frequency of docs with titles
    freq_wihtout - a dictionary of word to doc frequency of docs without titles

    Returns:
    a sorted dictionary of word -> (freq_with, freq_without) sorted by max of
    |freq_with - freq_without|
    '''

    sorted_diff = {}

    # looping through words and populating dictionary with freq_without keys
    for word in freq_without:
        if word in freq_with:
            sorted_diff[word] = (freq_with[word], freq_without[word])
        else:
            sorted_diff[word] = (0.0, freq_without[word])

    # looping through remaining words in freq_with keys
    for word in freq_with:
        if word in freq_with:
            pass
        else:
            sorted_diff[word] = (freq_with[word], 0.0)

    sorted_diff = sorted(sorted_diff.items(), key=lambda x:
                         x[1][0] - x[1][1], reverse=True)

    return sorted_diff


def calc_relative_diff_freq(count_with, count_without):
    '''
    Method to sort words by difference in frequency of documents with titles
    containing the analytical word set vs documents with titles that do not
    contain the word set

    Keyword Args:
    freq_with - a dictionary of word to doc frequency of docs with titles
    freq_wihtout - a dictionary of word to doc frequency of docs without titles

    Returns:
    a sorted dictionary of word -> (freq_with, freq_without) sorted by max of 
    relative frequency difference
    '''

    rel_freq = {}

    for word in count_with:
        if word in count_without:
            rel_in = count_with[word] / \
                float(count_with[word] + count_without[word])
        else:
            rel_in = 1

        rel_freq[word] = rel_in

    # sorting rel_freq dictionary by key
    rel_freq = sorted(rel_freq.items(), key=lambda x: x[1], reverse=True)

    return rel_freq


# an example path from the jstor database
example_path = "source_files/american_census_data_ngram/journal-article-10.5406_ethnomusicology.55.2.0200-ngram1.txt"

# getting all files within respective data or metadata directories
all_ngrams = glob.glob("source_files/jstor/american_community_survey_ngram/*")
all_metadata = glob.glob(
    "source_files/jstor/american_community_survey_metadata/*")

# placeholder list of words to split on
split_words = [
    "analysis",
    "regression",
    "machine learning",
    "artificial intelligence",
    "statistics",
    "graphs",
    "monte carlo",
    "deep learning",
    "neural network",
    "correlation",
    "data",
    "analytical",
    "studies",
    "measurement",
    "technical",
    "methods"
]

if __name__ == "__main__":

    # creating titles dictionary
    # a, b, c, d = calculate_different_frequencies(all_metadata, split_words)

    # sorted_diff = calculate_max_diff_freq(a, b)
    # rel_freq = calc_relative_diff_freq(c, d)

    # count = 0

    # # creating enchant dictionary
    # import enchant
    # dictionary = enchant.Dict("en_US")

    # writing out to a file
    # with open("most_diff_words.txt", "w") as f:
    # 	for item in sorted_diff:
    # 		word = item[0]
    # 		if dictionary.check(word):
    # 			freq_w = item[1][0]
    # 			freq_wout = item[1][1]
    # 			f.write(word + " : " + str(freq_w) + " : " + str(freq_wout) + "\n")
    # 			count += 1
    # 		if count == 200:
    # 			break

    # count = 0

    # with open("rel_diff_words.txt", "w") as f:
    # 	for i in rel_freq:
    # 		word = i[0]
    # 		if dictionary.check(word):
    # 			f.write(i[0] + " : " + str(i[1]) + "\n")
    # 			count += 1
    # 		if count == 200:
    # 			break

    # storing JSTOR corpus
    corpus = create_jstor_corpus()
    pickle.dump(corpus, open("source_files/jstor/jstor_corpus_bow_py2.p", "wb"), protocol = 2)

    # sorted_a = sorted(a.items(), key=lambda x: x[1], reverse=True)
    # sorted_b = sorted(b.items(), key=lambda x: x[1], reverse=True)

    # print("Top 20 words in a")
    # for word in sorted_a[:40]:
    #     word = word[0]
    #     if word in b:
    #         print(word, "A: " + str(a[word]) + " | B: " + str(b[word]))
    #     else:
    #         print(word, "A: " + str(a[word]) + " | B: 0")

    # print("Top 20 words in b")
    # for word in sorted_b[:40]:
    #     word = word[0]
    #     if word in b:
    #         print(word, "A: " + str(a[word]) + " | B: " + str(b[word]))
    #     else:
    #         print(word, "A: 0" + " | B: " + str(b[word]))
