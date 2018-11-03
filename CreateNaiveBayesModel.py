#!\Python35\python.exe
import json
import CreateVocabulary
import re
from collections import Counter
import joblib
import os
import numpy
import PreProcessing as pp
from time import gmtime, strftime

class NaiveBayesModel(object):
    stemming = False
    TrainDataPath = os.path.join('data', 'train data without stop words.json')
    file_name_suffix = 'WithoutStemming'
    model_name = ''
    ModelPath = ''
    VocabularyJsonDataPath = ''
    LikelihoodDebugJsonTable = ''

    LikelihoodDict = {}
    PositiveClassProbability = 0
    NegativeClassProbability = 0
    
    def __init__(self, stemming):
        self.stemming = stemming
        if stemming == True:
            self.file_name_suffix = 'WithStemming'
        self.model_name = 'NaiveBayesModel'+ self.file_name_suffix +'.pickle'
        self.ModelPath = os.path.join(os.path.join('output', self.file_name_suffix), self.model_name)
        self.VocabularyJsonDataPath = os.path.join(os.path.join('output', self.file_name_suffix), 'vocabulary.json')
        self.LikelihoodDebugJsonTable = os.path.join(os.path.join('output', self.file_name_suffix), 'likelihood_debug_json.json')
    
    def generateWordCount(self, train_data, stemming):
        word_and_word_count = {}
        document = ''
        for td in train_data:
            document = document + ' ' + td
        document = pp.StopWordAndSpecialCharRemoval(document, stemming)
        word_array = numpy.array(document.split())
        unique, counts = numpy.unique(word_array, return_counts=True)
        return dict(zip(unique, counts))
        
    def generateBigramCount(self, vacabulary_list, train_data, stemming):
        bigram_and_count = {}
        document = ''
        for td in train_data:
            document = document + ' ' + td
        document = pp.StopWordAndSpecialCharRemoval(document, stemming)
        TempBigrams = [b for b in zip(re.split('\s+',document)[:-1], re.split('\s+',document)[1:])]
        for word in vacabulary_list:
            FrequencyCount = 0
            if tuple(word) in TempBigrams:
                FrequencyCount  = TempBigrams.count(tuple(word))
            bigram_and_count[word[0]+','+word[1]] = FrequencyCount
        return bigram_and_count
    
    def calculateProbability(self, WordFrequency, TotalWordCount, VocabularySize):
        return float(WordFrequency+1)/float(TotalWordCount+VocabularySize)
    
    def log(self, msg):
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + msg)
        return
    
    def generateProbabilityTable(self):
        self.log(' : Generating vocuabulary data(both unigram and bigram)...')
        TotalWordCounts = CreateVocabulary.GenerateVocabularyData(self.stemming, self.VocabularyJsonDataPath, self.TrainDataPath)
        self.log(' : Vocabulary geneartion successful!')

        self.log(' : Generating frequency and likelihood table...')
        with open(self.VocabularyJsonDataPath) as VocabularyJSONData:    
            VocabularyData = json.load(VocabularyJSONData)
        with open(self.TrainDataPath) as TrainJSONData:    
            TrainingData = json.load(TrainJSONData)

        VocabularySize = len(VocabularyData['unigram']) + len(VocabularyData['bigram'])
        UnigramLikelihoodDict = {}
        BigramLikelihoodDict = {}
        
        positive_unigrams_and_its_count = self.generateWordCount(TrainingData['positive'], self.stemming)
        negative_unigrams_and_its_count = self.generateWordCount(TrainingData['negative'], self.stemming)
        self.log(' : Frequency table generation for unigram is finished!')
        
        positive_bigrams_and_its_count = self.generateBigramCount(VocabularyData['bigram'], TrainingData['positive'], self.stemming)
        negative_bigrams_and_its_count = self.generateBigramCount(VocabularyData['bigram'], TrainingData['negative'], self.stemming)
        self.log(' : Frequency table generation for bigram is finished!')

        for Word in VocabularyData['unigram']:
            PositiveFrequencyCount = 0 if (positive_unigrams_and_its_count.get(Word.lower()) is None) else positive_unigrams_and_its_count[Word.lower()]
            NegativeFrequencyCount = 0 if (negative_unigrams_and_its_count.get(Word.lower()) is None) else negative_unigrams_and_its_count[Word.lower()]
            PositiveProbability = self.calculateProbability(PositiveFrequencyCount, TotalWordCounts['PositiveWordCount'], VocabularySize)
            NegativeProbability = self.calculateProbability(NegativeFrequencyCount, TotalWordCounts['NegativeWordCount'], VocabularySize)
            UnigramLikelihoodDict[Word] = [PositiveProbability, NegativeProbability]
        self.log(' : Likelihood table generation for unigram is finished!')
        
        for Word in VocabularyData['bigram']:
            PositiveFrequencyCount = positive_bigrams_and_its_count[Word[0]+','+Word[1]]
            NegativeFrequencyCount = negative_bigrams_and_its_count[Word[0]+','+Word[1]]
            #probability of a word in a specific class  for creating likelihood table
            PositiveProbability = self.calculateProbability(PositiveFrequencyCount, TotalWordCounts['PositiveWordCount'], VocabularySize)
            NegativeProbability = self.calculateProbability(NegativeFrequencyCount, TotalWordCounts['NegativeWordCount'], VocabularySize)
            BigramLikelihoodDict[Word[0] + ',' + Word[1]] = [PositiveProbability , NegativeProbability]
        self.log(' : Likelihood table generation for bigram is finished!')
        
        self.LikelihoodDict['unigram'] = UnigramLikelihoodDict
        self.LikelihoodDict['bigram'] = BigramLikelihoodDict
        self.PositiveClassProbability = float(TotalWordCounts['PositiveWordCount'])/float(TotalWordCounts['PositiveWordCount']+TotalWordCounts['NegativeWordCount'])
        self.NegativeClassProbability = float(TotalWordCounts['NegativeWordCount'])/float(TotalWordCounts['PositiveWordCount']+TotalWordCounts['NegativeWordCount'])
            
        self.LikelihoodDict['PositiveClassProbability'] = self.PositiveClassProbability
        self.LikelihoodDict['NegativeClassProbability'] = self.NegativeClassProbability
        self.LikelihoodDict['TotalPositiveWordCount'] = TotalWordCounts['PositiveWordCount']
        self.LikelihoodDict['TotalNegativeWordCount'] = TotalWordCounts['NegativeWordCount']
        self.LikelihoodDict['VocabularySize'] = VocabularySize

        #DEBUG
        with open(self.LikelihoodDebugJsonTable, 'w') as outfile:
            json.dump(self.LikelihoodDict, outfile, sort_keys=True, indent=4)
        self.log(' : Table creation successful!')

        return self.LikelihoodDict
        
if __name__ == '__main__':
    print('Please enter 1 for model with stemming and 2 for model without stemming:')
    choice = input()
    if int(choice) == 1:
        choice = True
    else:
        choice = False
    NBM = NaiveBayesModel(choice)
    print(NBM.ModelPath)
    LikelihoodDict = NBM.generateProbabilityTable()
    joblib.dump(LikelihoodDict, NBM.ModelPath, compress=9)
