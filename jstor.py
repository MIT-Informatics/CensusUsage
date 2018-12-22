# encoding: utf-8

import os
import numpy as np
import pickle
import re

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


def create_jstor_corpus():
	'''
	Method to run the topic model on JSTOR datasets
	'''

	all_files = glob.glob("source_files/american_community_survey_ngram/*")
	text_list = create_text_list(all_files)

	cleaned_text = []
	for i in text_list:
		cleaned_text.append(process_bow(i)[0])

	return cleaned_text


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
	total_files = len(directory)

	pattern = r"<journal-title>.*</journal-title>|<book-title>.*</book-title>|<journal-title content-type=\"full\">.*</journal-title>|<article-title>.*</article-title>"

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
			cleaned = match[0].replace("<journal-title>", "")
			cleaned = cleaned.replace("</journal-title>", "")
			cleaned = cleaned.replace("<book-title>", "")
			cleaned = cleaned.replace("</book-title>", "")
			titles[cleaned] = filepath

		count += 1

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
	two lists, the first is all the titles that contain any word in words and the second list containing all other titles
	'''

	titles_with = []
	titles_without = []

	for title in titles:
		added = False
		for word in words:
			if word in title.lower():
				titles_with.append((title, convert_metadata_ngram(titles[title])))
				added = True
				break
		if not added:
			titles_without.append((title, convert_metadata_ngram(titles[title])))

	return titles_with, titles_without

def calculate_word_frequencies(tuples_list):
	'''
	Method to calculate the different in word frequency of documents that are split by containing specific keywords

	Keyword Args:
	tuples_list - a list of tuples of (titles, filepath) to the ngram file

	Returns:
	a dictionaries containing (word -> frequency) mappings of the tuples_list
	and the number of articles
	'''

	word2freq = {}

	# looping through files
	for title, file in tuples_list:
		document = ngram_to_doc(file)
		words = list(map(lambda x: x.lower(), document))

		# incrementing counts in dictionary
		for word in words:
			if word in word2freq:
				word2freq[word] += 1
			else:
				word2freq[word] = 1

	return word2freq, len(tuples_list)

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
	links_wout =  list(map(lambda x: x[1], words_without))

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
	freq_with, num_with = calculate_word_frequencies(titles_with)
	freq_without, num_without = calculate_word_frequencies(titles_without)

	# converting frequencies to ratios
	for word in freq_with:
		freq_with[word] /= num_with

	for word in freq_without:	
		freq_without[word] /= num_without

	print("Articles with: " + str(num_with) + " | Articles without: " + str(num_without))

	return freq_with, freq_without


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
	"graphs"
]


if __name__ == "__main__":

	# creating titles dictinoary
	a, b = calculate_different_frequencies(all_metadata, split_words)

	sorted_a = sorted(a.items(), key=lambda x: x[1], reverse=True)
	sorted_b = sorted(b.items(), key=lambda x: x[1], reverse=True)

	print("Top 20 words in a")
	for word in sorted_a[:40]:
		word = word[0]
		if word in b:
			print(word, a[word], b[word])
		else:
			print(word, a[word], 0)


	print("Top 20 words in b")
	for word in sorted_b[:40]:
		word = word[0]
		if word in a:
			print(word, a[word], b[word])
		else:
			print(word, 0, b[word])





