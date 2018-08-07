import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

from KaggleWord2VecUtility import KaggleWord2VecUtility

def makeFeatureVector(words, model, num_features):
	'''Function to average  all of the word vectors in a given paragraph'''
	featureVec = np.zeros((num_features,),dtype = 'float32')
	nwords = 0.
	#Index2word is a list that contains the names of the words in
	# the model's vocabulary. convert it to a set for speed
	index2word_set = set(model.wv.index2word)
	
	#Loop over each word in the review and if it is in the model's vocabulary
	#then add its feature vector to the total
	
	for word in words:
		if word in index2word_set:
			nwords = nwords + 1.
			featureVec = np.add(featureVec, model[word])
	
	#Divide the result by the number of wordes to get the average
	featureVec = np.divide(featureVec, nwords)
	return featureVec
	
	
def getAvgFeatureVecs(reviews, model, num_features):
	#Given a set of reviews, calculate the average feature
	#vector for each one and return a 2D numpy array
		
	counter = 0.
	reviewFeatureVecs = np.zeros((len(reviews), num_features),dtype = 'float32')
	for review in reviews:
		if counter %1000. == 0. :
			print('review %d of %d',counter, len(reviews))
		reviewFeatureVecs[int(counter)] = makeFeatureVector(review, model, num_features)
		
		counter = counter + 1.
	return reviewFeatureVecs

def getCleanReviews(reviews):
	clean_reviews = []
	for review in reviews['review']:
		clean_reviews.append(KaggleWord2VecUtility.review_to_wordlist(list, remove_stopwords = True))
	return clean_reviews
	
if __name__ == '__main__':

		train = pd.read_csv('input/labeledTrainData.tsv', header = 0, delimiter = '\t', quoting = 3)
		test = pd.read_csv('input/testData.tsv', header = 0, delimiter = '\t', quoting = 3)
		
		unlabeled_train = pd.read_csv('input/unlabeledTrainData.tsv', header = 0, delimiter = '\t', quoting = 3)
		
		#verify the number of reviews that were read
		print('',train.shape, test.shape,unlabeled_train.shape)
		
		#Load the punkt tokenizer
		tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
		
		#split the labeled and unlabeled training sets into clean sentences
		sentences = []
		for review in train['review']:
			sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
		
		for review in unlabeled_train['review']:
			sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
			
		#Set parameters and the train the word2vec model
		#import built in logging module and configure it so that word2vec creates nice output messages
		logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
		
		#Set values for various parameters
		num_features = 300 # word vector dimensions
		min_word_count = 40 #Minimum word count
		num_workers = 4 #number of threads to run in parallel
		context = 10
		downsampling = 1e-3
		
		#Initialize and train the model
		model = Word2Vec(sentences, workers = num_workers, size = num_features,\
		min_count = min_word_count,\
		window = context, sample = downsampling, seed = 1)
			
		#init_sims will make the model memory efficient
		model.init_sims(replace = True)
		
		model_name = '300features_40minwords_10context'
		model.save(model_name)
		
		print(model.doesnt_match("man woman child kitchen".split()))
		print(model.doesnt_match("france england germany berlin".split()))
		print(model.doesnt_match("paris berlin london austria".split()))
		print(model.most_similar("man"))
		print(model.most_similar("queen"))
		print(model.most_similar("awful"))
		
		print('Creating average feature vecs for training reviews')
		
		trainDataVecs = getAvgFeatureVecs(getCleanReviews(train), model, num_features)
		
		print('Creaing Average feature vecs for test reviews')
		testDataVecs = getAvgFeatureVecs(getCleanReviews(test), model, num_features)
		
		# Fit a random forest to the training data, using 100 trees
		forest = RandomForestClassifier(n_estimators = 100)
		
		forest = forest.fit(trainDataVecs, train['sentiment'])
		
		result = forest.predict(testDataVecs)