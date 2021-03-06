#!\Python36\python.exe

import json
import re
import os
from nltk.stem import PorterStemmer
import PreProcessing as pp

ps = PorterStemmer()

def GenerateVocabularyData(stemming, vocabulary_path, train_data_path):
	with open(train_data_path) as tdf:
		train_data = json.load(tdf)
	
	vocabulary_dict = {}
	unigram_set = set()
	bigram_set = set()
	index = 0

	TotalPositiveWordCount = 0
	TotalNegativeWordCount = 0
	
	for key in train_data:
		for value in train_data.get(key):
			pp_value = pp.StopWordAndSpecialCharRemoval(value, stemming)
			if stemming == True:
				bigrams = [b for b in zip(pp.perform_stemming(re.split('\s+',pp_value)[:-1]), pp.perform_stemming((re.split('\s+',pp_value)[1:])))]
				unigrams = [ps.stem(word) for word in pp_value]
			else:
				bigrams = [b for b in zip(re.split('\s+',pp_value)[:-1], re.split('\s+',pp_value)[1:])]
				unigrams = [u for u in re.split('\s+',pp_value)]
			
			temp_bigram_set = set()
			for bigram in bigrams:
				temp_bigram_set.add(bigram)
			bigram_set = bigram_set | temp_bigram_set #to append sets
			
			for unigram in unigrams:
				unigram_set.add(unigram)
			if key == 'positive':
				TotalPositiveWordCount = len(set(unigrams)) + len(temp_bigram_set) + TotalPositiveWordCount
			else:
				TotalNegativeWordCount = len(set(unigrams)) + len(temp_bigram_set) + TotalNegativeWordCount
	vocabulary_dict.update({'unigram': list(unigram_set)})
	vocabulary_dict.update({'bigram': list(bigram_set)})
	if os.path.exists(vocabulary_path):
		os.remove(vocabulary_path)
	with open(vocabulary_path, 'w') as outfile:
		json.dump(vocabulary_dict, outfile, sort_keys=True, indent=4)
	return {'PositiveWordCount': TotalPositiveWordCount, 'NegativeWordCount' : TotalNegativeWordCount}
