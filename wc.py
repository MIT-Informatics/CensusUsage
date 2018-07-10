from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
	w = create_wordcloud("You have choices about the information on your profile, such as your education, work experience, skills, photo, city or area and endorsements. Some Members may choose to complete a separate ProFinder profile. You dont have to provide additional information on your profile; however, profile information helps you to get more from our Services, including helping recruiters and business opportunities find you. Its your choice whether to include sensitive information on your profile and to make that sensitive information public. Please do not post or add personal data to your profile that you would not want to be publicly available.")

	show_wordcloud(w)