# coding=utf-8

import gensim
from gensim import corpora
from gensim.models import ldamodel, CoherenceModel

import numpy as np
from twitter_act import *
from process import *
from webscraping import *

#using pyLDAvis to visualize
import pyLDAvis.gensim

#number of topics to print eventually
number_topics = 9

def gensim_topic_analysis(text_list):

    '''
    Method to model the topics of an input text list using gensim
    @param text_list - a list of words to analyze and produce topics
    @return - the lda model produced
    '''

    #formatting dictionary to use in LDA model
    id_word = corpora.Dictionary(text_list)
    texts = text_list
    corpus = [id_word.doc2bow(text) for text in texts]

    #creating LDA model with parameterized topics and training passes
    #params - dictionary, number topics to print, seed to create random number array, iterative learning through 1 doc per pass, 3 docs per pass, 5 passes, normalized prior, bool to sort topics in order
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                          id2word=id_word,
                                          num_topics=number_topics,
                                          random_state=100,
                                          update_every=1,
                                          chunksize=3,
                                          passes=5,
                                          alpha='auto',
                                          per_word_topics=True)

    print(lda.print_topics())
    return lda, corpus, id_word, text_list

def evaluate_gensim_lda(lda_model, corpus, id_word, text_list):

    '''
    Method to evaluate the lda model provided by gensim
    @param lda_model - the LDA model created by the gensim package
    @param corpus - the total corpus evaluated by the lda model
    @param id_word - the dictionary created by the corpus
    @param text_list - the initial B.O.W formatted text
    '''

    #held out likelihood score - simplified idea as predictablity of model
    print("Perplexity: ", lda_model.log_perplexity(corpus))
    coherence = CoherenceModel(model=lda_model, texts=text_list, dictionary=id_word, coherence = 'c_v')

    #multiple pipeline score evaluated quality of topics
    score = coherence.get_coherence()
    print("Coherence", score)

def show_pyldavis(model, corpus, dictionary):

    '''
    Method to create a pyLdavis visualization of the topic model
    @param model - the trained gensim model
    @param corpus - the total corpus the model was trained on
    @param dictionary - the dictionary created in the model
    '''

    prepared_data = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    pyLDAvis.show(prepared_data)

#when running script
if __name__ == "__main__":
    #using twitter_act file to access Twitter API
    t = main()
    #searching for tweets with #ACS hashtag
    # tweets_objects = t.searchHashtag('#ACS')

    # tweets_text = []

    # adding only text into tweet_text list
    # for item in tweets_objects:
    #     tweets_text.append(process_string(item['text']))

    # topic_analysis(tweets_text)
    words = [u'Hi what is your name.', 
            u"Ever need to add some random or meaningless text into Microsoft Word to test a document, temporarily fill some space, or to see how some formatting looks? Luckily, Word provides a couple of quick and easy methods for entering random text into your document.", 
            u"To do this, position the cursor at the beginning of a blank paragraph. Type the following and press Enter. It does not matter if you use lowercase, uppercase, or mixed case.", 
            u"I'm a student and research assistant at Brown University, who is studying the fields of Computer Science and Applied Math. I am currently working in the Rubenstein Lab within the DARPA Molecular Informatics Project and as an intern in the MIT Program of Information Science. I am particularly interested in the ability to perceive information from data sets through analytical methods. I love research, data science, and machine learning. I am always looking for opportunities to learn new skills.", 
            u"At school, I am on the organizing team of the Brown Data Science club and the Fintech at Brown, a member of the Brown Club Tennis Team, and a writer for the Ursa Sapiens Blog. In my free time, I love to play soccer and the viola.", 
            u"The National Center for Health Statistics, part of the U.S. Centers for Disease Control, analyzed death reports since 1999 to determine rates of death due to drug overdoses. The map below shows that data for Pennsylvania counties. Tha map is shaded according to 2016 estimated death rates for each county. The estimated number of drug-related deaths was calculated using the ranges reported by the NCHS and annual population estimates from the U.S. {Census Bureau}. Click on a county to see its statistics going back to 2000. Scroll and zoom to see all Pennsylvania counties. SOURCE: U.S. Center for Health Statistics Copyright 2018, The Morning Call 2018-07-01",
            "State vital records show South Carolina isnt there just yet, but the state is inching closer to the national trend of more white people dying than are being born. Birth and death rates are key indicators in how a population is changing. The national rates show America is becoming more racially diverse. Department of Health and Environmental Control data shows South Carolina had almost 2,800 more births than deaths among its white population in 2016. York and Lancaster had more births than deaths. Chester County had more deaths than births. Our journalism takes a lot of time, effort, and hard work to produce. If you read and enjoy our journalism, please consider subscribing today. Those statistics come even as one national study claims South Carolinas white population already is in \"natural decrease\" -- a term the study uses to describe a point where there are more deaths than births in a population. A recent study lists South Carolina among a growing number of states where white deaths outpace births. Discrepancies aren't unusual, and can be based on labeling. The Applied Population Lab study, done at the University of Wisconsin-Madison, deals specifically with \"non-Hispanic whites.\" The state data lists births by mother's race alone, and only as \"white.\" \"We used data for non-Hispanic whites,\" said Kenneth M. Johnson, who co-authored the national study with Rogelio Sáenz. \"The difference between white and non-Hispanic whites is an important distinction. Also, individual states may use slightly different methods to classify births and deaths.\" Federal data categorizes by race (white, black or African-American, American Indian or Alaska Native, Asian, Native Hawaiian or other Pacific Islander) and by ethnicity (Hispanic or Latino, non-Hispanic). So, for example, a Rock Hill woman who might check \"white\" on hospital forms after giving birth would account for a birth in the \"white\" category at the state level. However, she may be listed differently in the federal study if she also selected \"Hispanic or Latino\" on a federal form. According to U.S. Census data, 4.7 percent of South Carolina's population - more than 236,000 people - are listed as both white and Hispanic. Birth and death totals at the state and county level are different too, based on where the birth or death happened. For example: a Tega Cay mother who gives birth at a Charlotte hospital would count in one data set but not the other. On a typical day in South Carolina in 2016, there were 157 births and 132 deaths. The white population in South Carolina and nationally still makes up a majority. The U.S. {Census Bureau} estimates there are a little more than 5 million people living in South Carolina. The white population accounts for 68.5 percent of them, with 27.3 percent black and 5.7 percent Hispanic or Latino. York County has a little more than 266,000 people. The {census bureau} estimates more than 75 percent of York County is white, compared to 19 percent black and 6 percent all other races. The county is almost 6 percent Hispanic or Latino. Lancaster County is 75 percent white and 22 percent black among its nearly 93,000 residents. More than 5 percent of residents are Hispanic. Chester County is 60 percent white compared to 37 percent, with 2 percent also listed as Hispanic or Latino. Chester County is about a third the size of Lancaster County. The Wisconsin study used data from the National Center for Health Statistics, not the state health department, bringing in discrepancies with federal and state listings for race and ethnicity. The question of whether more white people are dying than being born in South Carolina isn't an easy one. However, even with the state data, there is a trend toward white births and deaths converging. In 1997 the birth rate among whites was 12.8 per 1,000 residents. The death rate was 9 per 1,000 residents. By 2007 the birth rate was up to 13.3 while the death rate sat at 9.3. Then recession hit. The birth rate has declined or remained the same every year since 2008. The death rate has increased. In 2016 there was less than one more birth per 1,000 residents than death - 11.1 births to 10.3 deaths - among whites. Recession is a main reason the national study cites for birth decreases among whites. U.S. {Census Bureau} studies also relate economics with fertility rates, with a down economy or economic uncertainly typically tied to lower birth rates. \"The pace of decline in white births intensified from 2007 to 2016, due in part to the Great Recession's significant impact on U.S. fertility,\" reads the Applied Population Lab study. \"The recession, the greatest shock to the American economic system in nearly two generations, influenced both fertility and life-cycle decisions for many families.\" That impact isn't specific to whites. \"Some 500,000 fewer babies are being born annually now than had pre-recession fertility rates been sustained,\" study authors wrote. \"And, nearly 2.1 million more women of prime childbearing age are childless than would be expected. A significant share of those 500,000 annual births that are not occurring would have been white.\" Nationally, more white people died than were born in a year for the first time in 2016. Whites made up almost 78 percent of all U.S. deaths, but just 53 percent of all births. The national study indicates that trend is spreading to more states. In 2000 only four states had more deaths than births among the non-Hispanic white population. In 2015, the first year South Carolina made that list, there were 23 states. Three more states joined the group in 2016. It's more likely states will join the list in coming years than fail to make it again. \"Once an area begins to experience natural decrease, the trend is likely to continue,\" reads the study. White natural decrease, as the study terms it, has been happening for more than a decade in Florida, Pennsylvania, Rhode Island, West Virginia, California, New Mexico and Connecticut. More recent states to join South Carolina include North Carolina, Tennessee, Ohio, Michigan and others. States like Texas, Illinois, Wisconsin and Georgia are seeing slightly more white people born than dying. Alaska, Virginia and nine largely midwestern states have larger gaps where births still exceed deaths. The national decrease is largely a phenomenon of the white population. Only three states - West Virginia, Vermont, Maine - had more deaths than births among their entire population. Latino births outpaced deaths \"by a substantial margin\" in every state, according to the study. The path toward more white deaths than births has been a long but, at least of late, steady one. In 2000 there were 403,049 more white people born than died in the United States. In 2010 there were 192,490 more births than deaths. A tipping point came in 2015, when only 6,648 more people - about the population of Clover - were born than died. Data from 2016 shows 39,409 more white people died than were born nationally. The convergence comes from both ends. Deaths are up almost 10 percent from 2009 to 2016. Births dropped more than 11 percent from 2000 to 2016. Economics and people waiting longer to get married, if they marry at all, are listed among impacts on the birth end. There also are reasons why more white people are dying younger, upping the death rate. Combined with an aging white population, these factors put the birth-to-death rates for whites (.98) well below those of Latinos (4.9), Asians (3.9) and African-Americans (1.7) in 2016. \"Deaths of despair,\" as the study calls them, are up among whites age 30-59. \"Such deaths include drug-induced deaths, intentional suicide, accidental drug overdose and alcohol deaths,\" reads the study. \"Such deaths have increased sharply in recent years among whites.\" Those \"deaths of despair\" alone were the difference eight states had more deaths than births among whites in 2016. \"These deaths of despair are likely to accelerate the transition from natural increase to natural decrease in many other states in the near future,\" reads the study."]

    word_list = []

    webscraping_data = webscrape()

    #formatting words to be analyzed for both sklearn and gensim
    for item in webscraping_data:
        word_list.append(process_string(item)[1])
    
    #fitting gensim model
    gensim_model, corpus, id_word, text_list = gensim_topic_analysis(word_list)
    #evaluating perplexity and coherence of gensim lda model
    evaluate_gensim_lda(gensim_model, corpus, id_word, text_list)

    #create visualization of pyldavis
    show_pyldavis(gensim_model, corpus, id_word)
