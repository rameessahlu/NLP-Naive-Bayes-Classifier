#!\Python35\python.exe
import json
import CreateVocabulary
import re
from collections import Counter
import joblib
import os

class NaiveBayesModel(object):
    stemming = False
    TrainDataPath = os.path.join('data', 'train data without stop words.json')
    VocabularyJsonDataPath = os.path.join('output', 'vocabulary.json')
    LikelihoodDebugJsonTable = os.path.join('output', 'likelihood_debug_json.json')
    file_name_suffix = 'WithoutStemming'
    model_name = 'NaiveBayesModel'+ file_name_suffix +'.pickle'
    ModelPath = os.path.join('output', model_name)

    LikelihoodDict = {}
    PositiveClassProbability = 0
    NegativeClassProbability = 0
    
    def __init__(self, stemming):
        self.stemming = stemming
        if stemming == True:
            file_name_suffix = 'WithStemming'
    
    def calculateProbability(self, WordFrequency, TotalWordCount, VocabularySize):
        return float(WordFrequency+1)/float(TotalWordCount+VocabularySize)

    def generateProbabilityTable(self):
        print('Generating vocuabulary data(both unigram and bigram)...')
        TotalWordCounts = CreateVocabulary.GenerateVocabularyData(self.stemming)
        print('Vocabulary geneartion successful!')

        print('Generating frequency and likelihood table...')
        with open(self.VocabularyJsonDataPath) as VocabularyJSONData:    
            VocabularyData = json.load(VocabularyJSONData)
        with open(self.TrainDataPath) as TrainJSONData:    
            TrainingData = json.load(TrainJSONData)

        VocabularySize = len(VocabularyData['unigram']) + len(VocabularyData['bigram'])
        UnigramLikelihoodDict = {}
        BigramLikelihoodDict = {}
        
        for Word in VocabularyData['unigram']:
            PositiveFrequencyCount = 0
            NegativeFrequencyCount = 0
            for TD in TrainingData['positive']:
                PositiveFrequencyCount = TD.split().count(Word) + PositiveFrequencyCount
            for TD in TrainingData['negative']:
                NegativeFrequencyCount = TD.split().count(Word) + NegativeFrequencyCount
            PositiveProbability = self.calculateProbability(PositiveFrequencyCount, TotalWordCounts['PositiveWordCount'], VocabularySize)
            NegativeProbability = self.calculateProbability(NegativeFrequencyCount, TotalWordCounts['NegativeWordCount'], VocabularySize)
            UnigramLikelihoodDict[Word] = [PositiveProbability, NegativeProbability]

        for Word in VocabularyData['bigram']:
            PositiveFrequencyCount = 0
            NegativeFrequencyCount = 0
            for TD in TrainingData['positive']:
                TempBigrams = [b for b in zip(re.split('\s+',TD)[:-1], re.split('\s+',TD)[1:])]
                TempCount = 0
                for TempBigram in TempBigrams:
                    if set(TempBigram) == set((Word[0],Word[1])):
                        TempCount  = TempCount + 1
                PositiveFrequencyCount = TempCount + PositiveFrequencyCount
            for TD in TrainingData['negative']:
                TempBigrams = [b for b in zip(re.split('\s+',TD)[:-1], re.split('\s+',TD)[1:])]
                TempCount = 0
                for TempBigram in TempBigrams:
                    if set(TempBigram) == set((Word[0],Word[1])):
                        TempCount  = TempCount + 1
                NegativeFrequencyCount = TempCount + NegativeFrequencyCount
            #probability of a word in a specific class  for creating likelihood table
            PositiveProbability = self.calculateProbability(PositiveFrequencyCount, TotalWordCounts['PositiveWordCount'], VocabularySize)
            NegativeProbability = self.calculateProbability(NegativeFrequencyCount, TotalWordCounts['NegativeWordCount'], VocabularySize)
            BigramLikelihoodDict[Word[0] + ',' + Word[1]] = [PositiveProbability , NegativeProbability]
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
        print('Table creation successful!')

        return self.LikelihoodDict
        
if __name__ == '__main__':
    print('Please enter 1 for model with stemming and 2 for model without stemming:')
    choice = input()
    if choice == 1:
        choice = True
    else:
        choice = False
    NBM = NaiveBayesModel(choice)
    LikelihoodDict = NBM.generateProbabilityTable()
    joblib.dump(LikelihoodDict, ModelPath, compress=9)
