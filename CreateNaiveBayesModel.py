#!\Python35\python.exe
import json
import CreateVocabulary
import re
from collections import Counter
import joblib

ModelPath = '.\\output\\NaiveBayesModel.pickle'

class NaiveBayesModel(object):
    TrainDataPath = '.\\data\\train data without stop words.json'
    VocabularyJsonDataPath = '.\\output\\vocabulary.json'
    LikelihoodDebugJsonTable = '.\\output\\likelihood_debug_json.json'

    LikelihoodDict = {}
    PositiveClassProbability = 0
    NegativeClassProbability = 0
    
    def calculateProbability(self, WordFrequency, TotalWordCount, VocabularySize):
        return float(WordFrequency+1)/float(TotalWordCount+VocabularySize)

    def generateProbabilityTable(self):
        print('Generating vocuabulary data(both unigram and bigram)...')
        TotalWordCounts = CreateVocabulary.GenerateVocabularyData()
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
            PositiveProbability = self.calculateProbability(PositiveFrequencyCount, TotalWordCounts['PositiveWordCount'], VocabularySize)
            NegativeProbability = self.calculateProbability(NegativeFrequencyCount, TotalWordCounts['NegativeWordCount'], VocabularySize)
            BigramLikelihoodDict[Word[0] + ',' + Word[1]] = [PositiveProbability , NegativeProbability]
        self.LikelihoodDict['unigram'] = UnigramLikelihoodDict
        self.LikelihoodDict['bigram'] = BigramLikelihoodDict
        self.PositiveClassProbability = float(TotalWordCounts['PositiveWordCount'])/float(TotalWordCounts['PositiveWordCount']+TotalWordCounts['NegativeWordCount'])
        self.NegativeClassProbability = float(TotalWordCounts['NegativeWordCount'])/float(TotalWordCounts['PositiveWordCount']+TotalWordCounts['NegativeWordCount'])
        
        #DEBUG
        with open(self.LikelihoodDebugJsonTable, 'w') as outfile:
            json.dump(self.LikelihoodDict, outfile, sort_keys=True, indent=4)
        print('Table creation successful!')
            
        self.LikelihoodDict['PositiveClassProbability'] = self.PositiveClassProbability
        self.LikelihoodDict['NegativeClassProbability'] = self.NegativeClassProbability
        self.LikelihoodDict['TotalPositiveWordCount'] = TotalWordCounts['PositiveWordCount']
        self.LikelihoodDict['TotalNegativeWordCount'] = TotalWordCounts['NegativeWordCount']
        self.LikelihoodDict['VocabularySize'] = VocabularySize
        return self.LikelihoodDict
        
if __name__ == '__main__':
    NBM = NaiveBayesModel()
    LikelihoodDict = NBM.generateProbabilityTable()
    joblib.dump(LikelihoodDict, ModelPath, compress=9)