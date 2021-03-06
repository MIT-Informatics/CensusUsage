import sys
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter

sys.path.append("../data_collection")

from jstor import *


def plot_wc(corpus, title):
	'''
	Method to visualize an inputted string as a wordcloud

	Keyword Args:
	corpus - a string containing a corpus of JSTOR data

	Returns: 
	None
	'''

	# generating wordcloud object
	wordcloud = WordCloud(max_font_size=150,
						  width=1600,
						  height=800,
						  collocations=False,
						  background_color="grey").generate(corpus)

	# changing font size of plot
	plt.rcParams.update({'font.size': 32})

	# plotting wordcloud in matplotlib
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.title(title)

	plt.show()


def plot_freq_hist(corpus, title):
	'''
	Method to plot the corpus as a frequency distribution of the top 50 most 
	frequent words

	Keyword Args:
	corpus - the corpus of words in bow format
	title - the title for the plot

	Returns:
	None
	'''

	labels = []
	values = []

	# creating labels and values for labels
	for i in corpus:
		labels.append(i[0])
		values.append(i[1])

	# taking first 40 of each
	labels = labels[:40]
	values = values[:40]

	# enumerating locations for labels
	indexes = np.arange(len(labels))

	bar_width = 0.8

	plt.bar(indexes, values, width=bar_width)

	# formatting axes
	plt.xticks(indexes, labels)
	plt.ylabel("Frequency", {'fontsize': 32})
	plt.yticks(np.arange(0, 1.2, step=0.2))
	ax = plt.gca()

	# setting x axis format
	ax.tick_params(axis="x",
				   labelsize=20,
				   labelrotation=90)
	# setting y axis format
	ax.tick_params(axis="y",
				   labelsize=20)
	# formatting white space in bar chart
	ax.set_xlim(-0.6, len(labels) - 0.4)

	# changing font size of plot
	plt.rcParams.update({'font.size': 32})

	# adding title
	plt.title(title)

	plt.show()


def main():
	'''
	Script to visualize the jstor data based on split keywords
	'''

	# creating titles dictionary
	titles = read_journal_titles(all_metadata)


	# splitting dictionary into list of tuples
	titles_with, titles_without = split_journal_titles(titles, split_words)

	# creating text corpuses
	t_w, t_wout, bow_w, bow_wout = create_split_corpuses(titles_with,
														 titles_without)
	# plotting word clouds
	plot_wc(t_w, "Words in JSTOR Data containing word set")
	plot_wc(t_wout, "Words in JSTOR Data not containing word set")

	# creating frequency distributions
	freq_with, freq_without = calculate_different_frequencies(
		all_metadata, split_words)

	# sorting frequency distributions
	sorted_with = sorted(freq_with.items(), key=lambda x: x[1], reverse=True)
	sorted_without = sorted(freq_without.items(),
							key=lambda x: x[1], reverse=True)

	# for i in range(0, 1000):
		# print(sorted_with[i])
		# print(sorted_without[i])

	# # plotting frequency histograms
	plot_freq_hist(sorted_with,
	               "Word in Document Frequency of JSTOR Documents containing analytical word set")
	plot_freq_hist(sorted_without,
	               "Word in Document Frequency of JSTOR Documents not containing analytical word set")


if __name__ == "__main__":
	main()
