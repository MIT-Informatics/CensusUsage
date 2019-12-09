from wordcloud import WordCloud
import matplotlib.pyplot as plt

from webscraping import *

def create_wordcloud(text):
	
	'''
	Method to create a wordcloud visualization from input text
	@param text - input text to visualize
	@return - a wordcloud object
	'''
	
	wc = WordCloud(max_words = 100)
	wc.generate(text)

	return wc

def show_wordcloud(wc):

	'''
	Method to display a wordcloud object
	@param wc - a wordcloud object
	'''

	plt.imshow(wc, interpolation='bilinear')	
	plt.show()

if __name__ == "__main__":
	corpus = webscrape()

	all_text = ""
	for i in corpus:
		all_text += i

	w = create_wordcloud(all_text)

	show_wordcloud(w)