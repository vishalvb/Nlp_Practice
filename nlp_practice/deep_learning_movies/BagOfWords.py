import os 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np

if __name__ == '__main__':
	train = pd.read_csv('input/labeledTrainData.tsv', header = 0,\
	delimiter = '\t', quoting = 3)
	test = pd.read_csv('input/testData.tsv', header = 0,\
	delimiter = '\t', quoting = 3)
	print ('the first review is', train["review"][0])
	
	#Initialize an empty list to hold the clean reivews
	clean_train_reviews = []
	
	#Loop over each review; create an index i that goes from 0 to the length
	#of the movie review list
	
	print('clearning and parsing the training set moview reviews\n')
	for i in range(0, len(train['review'])):
		clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train['review'][i], True)))
	
	print('\ncleaned review is')
	print(clean_train_reviews[0])
	print ('creating bag of words..\n')
	
	#Initialize CountVectorizer object
	vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, \
	preprocessor = None, stop_words = None,\
	max_features = 5000)
	
	#fit_transform() does to functions: first, it fits the model and learns the vocabulary;
	#second it transforms our training data into feature vectors. 
	#input should be list of strings
	
	train_data_features = vectorizer.fit_transform(clean_train_reviews)
	
	#numpy arrays are easy to work with, 
	np.asarray(train_data_features)
	
	#Training a random forest
	
	forest = RandomForestClassifier(n_estimators = 100)
	
	#fit the forest to the training set, using the bag of words as features
	#and the sentiment labels as the response variable
	
	forest = forest.fit(train_data_features, train['sentiment'])
	
	clean_test_reviews = []
	
	print('cleaning and parsing the test set movie reviews')
	for i in range(0, len(test['review'])):
		clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))
	
	test_data_features = vectorizer.tranform(clean_test_reviews)
	np.asarray(test_data_features)
	
	print("predicting test labels")
	result = forest.predict(test_data_features)
	
	#copy the result to a pandas dataframe 
	output = pd.DataFrame(data = {"id":test["id"], "sentiment":result})
	
	