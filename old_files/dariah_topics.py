from dariah_topics import preprocessing, modeling, postprocessing, visualization

#source code from the dariah topic example
pathlist = ['corpus/dickens_bleak.txt', 'corpus/thackeray_vanity.txt']
labels = ['dickens_bleak', 'thackeray_vanity']
corpus = preprocessing.read_files(pathlist)
tokens = [preprocessing.tokenize(document) for document in corpus]
matrix = preprocessing.create_document_term_matrix(tokens, labels)
stopwords = preprocessing.list_mfw(matrix)
clean_matrix = preprocessing.remove_features(stopwords, matrix)
vocabulary = clean_matrix.columns
model = modeling.lda(topics=10, iterations=1000, implementation='mallet')
topics = postprocessing.show_topics(model, vocabulary)
document_topics = postprocessing.show_document_topics(model, topics, labels)
PlotDocumentTopics = visualization.PlotDocumentTopics(document_topics)
static_heatmap = PlotDocumentTopics.static_heatmap()
static_heatmap.show()