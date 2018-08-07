import re
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords

class KaggleWord2VecUtility(object):
	''' utility class for processing raw html text into segments'''
	@staticmethod
	def review_to_wordlist(review, remove_stopwords = False):
		'''function to conver a document to a sequence of words.'''
		#1. remove html
		review_text = BeautifulSoup(review).get_text()
		#2. Remove non-letters
		review_text = re.sub("[^a-zA-Z]"," ", review_text)
		#3. convert words to lower case and split them
		words = review_text.lower().split()
		#4. Optionally remove stop words
		if remove_stopwords:
			stops = set(stopwords.words("english"))
			words = [w for w in words if not w in stops]
		#5. return the list of words
		return words
	
	@staticmethod
	def review_to_sentences(review, tokenizer, remove_stopwords = False):
		'''function to split a review into parsed sentences'''
		#1. Use nltk tokenizer to split the paragrah into sentences
		raw_sentences = tokenizer.tokenize(review.strip())
		sentences = []
		for raw_sentence in raw_sentences:
			if len(raw_sentence) > 0:
				sentences.append(KaggleWord2VecUtility.review_to_wordlist(raw_sentence, remove_stopwords))
		
		return sentences
		