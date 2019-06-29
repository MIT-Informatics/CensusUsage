import numpy as np
import guidedlda

# Guided LDA seed topics
seed_topic_list = [
    ['model', 'team', 'win', 'player', 'season', 'second', 'victory'],
    ['percent', 'company', 'market', 'price',
     'sell', 'business', 'stock', 'share'],
    ['music', 'write', 'art', 'book', 'world', 'film'],
    ['political', 'government', 'leader', 'official', 'state', 'country',
     'american', 'case', 'law', 'police', 'charge', 'officer', 'kill',
     'arrest', 'lawyer']

]

# creating the guidedlda model
model = guidedlda.GuidedLDA(n_topics=5,
                            n_iter=100,
                            random_state=7,
                            refresh=20)


# using NYT dataset from guidedlda
X = guidedlda.datasets.load_data(guidedlda.datasets.NYT)
vocab = guidedlda.datasets.load_vocab(guidedlda.datasets.NYT)
word2id = dict((v, idx) for idx, v in enumerate(vocab))

# fitting the model with the seeds
model.fit(X, seed_topics=seed_topic_list, seed_confidence=0.15)


# printing out the 10 words
n_top_words = 10
topic_word = model.topic_word_

print(X)

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][
        :-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
