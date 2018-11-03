import joblib
import re,os
from numpy import log
import json
from nltk.stem import PorterStemmer
import PreProcessing as pp

ps = PorterStemmer()

class TestNaiveBayesModel(object):
    def __init__(self, ModelObject):
        self.VocabularySize = ModelObject['VocabularySize']
        self.ModelObject = ModelObject
    
        self.PositiveClassProbability = float(ModelObject['PositiveClassProbability'])
        self.NegativeClassProbability = float(ModelObject['NegativeClassProbability'])
        
        self.TotalPositiveWordCount = ModelObject['TotalPositiveWordCount']
        self.TotalNegativeWordCount = ModelObject['TotalNegativeWordCount']
    
    def perform_stemming(self, tds):
        stemmed_td = []
        for td in tds:
            stemmed_td.append(ps.stem(td))
        return stemmed_td
    
    def calculateProbability(self, TargetOrClass, WordFrequency, TotalWordCount, ClassProbability):
        if WordFrequency is None:
            return float(1+1)/float(TotalWordCount+self.VocabularySize)
        else:
            if TargetOrClass == 'negative':
                return float(WordFrequency[1])/ float(ClassProbability)
            else:
                return float(WordFrequency[0])/ float(ClassProbability)
    
    def main(self, TestData, stemming):
        TestBigrams =[]
        
        if stemming == True:
            TestBigrams = [b for b in zip(self.perform_stemming(re.split('\s+',TestData)[:-1]), self.perform_stemming(re.split('\s+',TestData)[1:]))]
        else:
            TestBigrams = [b for b in zip(re.split('\s+',TestData)[:-1], re.split('\s+',TestData)[1:])]
    
        ProbabilityOfPositiveClassGivenTestData = 1.0 * self.PositiveClassProbability
        ProbabilityOfNegativeClassGivenTestData = 1.0 * self.NegativeClassProbability
    
    
        for TestBigram in TestBigrams:
            key = TestBigram[0]+ ',' + TestBigram[1]
            WordProbability = self.ModelObject['bigram'].get(key, None)
    
            if WordProbability is None:
                if TestBigram == TestBigrams[0]:
                    UnigramFirstWordProbabilities =  ModelObject['unigram'].get(TestBigram[0], None)
                    UnigramSecondWordProbabilities = ModelObject['unigram'].get(TestBigram[1], None)
                    ProbabilityOfPositiveClassGivenTestData = ProbabilityOfPositiveClassGivenTestData * self.calculateProbability('positive',UnigramFirstWordProbabilities, self.TotalPositiveWordCount, self.PositiveClassProbability) * self.calculateProbability('positive',UnigramSecondWordProbabilities, self.TotalPositiveWordCount, self.PositiveClassProbability)
                    ProbabilityOfNegativeClassGivenTestData = ProbabilityOfNegativeClassGivenTestData * self.calculateProbability('negative', UnigramFirstWordProbabilities, self.TotalNegativeWordCount, self.NegativeClassProbability) * self.calculateProbability('negative', UnigramSecondWordProbabilities, self.TotalNegativeWordCount, self.NegativeClassProbability)
                else:
                    UnigramSecondWordProbabilities = ModelObject['unigram'].get(TestBigram[1], None)
                    ProbabilityOfPositiveClassGivenTestData = ProbabilityOfPositiveClassGivenTestData * self.calculateProbability('positive',UnigramSecondWordProbabilities, self.TotalPositiveWordCount, self.PositiveClassProbability)
                    ProbabilityOfNegativeClassGivenTestData = ProbabilityOfNegativeClassGivenTestData * self.calculateProbability('negative',UnigramSecondWordProbabilities, self.TotalNegativeWordCount, self.PositiveClassProbability)
            else:
                ProbabilityOfPositiveClassGivenTestData = ProbabilityOfPositiveClassGivenTestData * (float(WordProbability[0]) / self.PositiveClassProbability)
                ProbabilityOfNegativeClassGivenTestData = ProbabilityOfNegativeClassGivenTestData * (float(WordProbability[1]) / self.NegativeClassProbability)
    
        #Posterior Probability
        #print("{0:.90f}".format(float(ProbabilityOfPositiveClassGivenTestData), 'f'))
        #print("{0:.90f}".format(float(ProbabilityOfNegativeClassGivenTestData), 'f'))
        if float(ProbabilityOfPositiveClassGivenTestData) > float(ProbabilityOfNegativeClassGivenTestData):
            #print('+')
            return 'positive'
        else:
            #print('-')
            return 'negative'
        #print(ProbabilityOfPositiveClassGivenTestData)
        #print(ProbabilityOfNegativeClassGivenTestData)
        #log(ProbabilityOfPositiveClassGivenTestData)

if __name__ == '__main__':
    print('Please enter 1 for single test data and 2 multiple test data: ')
    main_choice = input()
    print('Please enter 1 for stemming test data and 2 for not stemming:')
    second_choice = input()
    file_name_affix = 'WithStemming'
    stemming = True
    if int(second_choice) != 1:
        stemming = False
        file_name_affix = 'WithoutStemming'
    ModelObject = joblib.load(os.path.join(os.path.join('output', file_name_affix), "NaiveBayesModel{}.pickle".format(file_name_affix)))
    TNBM = TestNaiveBayesModel(ModelObject)
    if int(main_choice) == 1:
        print('Please enter a review of your choice for testing the model:')
        TestData = input()
        print(TNBM.main(pp.StopWordAndSpecialCharRemoval(TestData, stemming), stemming))
    else:
        test_data_path = os.path.join('data', r'test data without stop words.json')
        successful_prediction = 0
        failure_prediction = 0
        cont_dist = {}
        temp_p_count = 0
        count = 1
        with open(test_data_path) as tdf:
            test_data = json.load(tdf)
        for key in test_data:
            for value in test_data.get(key):
                if count % 500 == 0:
                    cont_dist[count] = (temp_p_count / 500)
                    temp_p_count = 0
                pp_value = pp.StopWordAndSpecialCharRemoval(value, stemming)
                result = TNBM.main(pp_value, stemming)
                #print(key + ' ' + result)
                if key == result:
                    successful_prediction = successful_prediction + 1
                    temp_p_count = temp_p_count + 1
                else:
                    failure_prediction = failure_prediction + 1
                count = count + 1
        print(r':) ' + str(successful_prediction))
        print(r':( ' + str(failure_prediction))
        print(cont_dist)