import joblib
import re,os
from numpy import log
import json
from nltk.stem import PorterStemmer

ModelObject = joblib.load(os.path.join("output", "NaiveBayesModel.pickle"))
ps = PorterStemmer()

StopWords = ['a','about','above','after','again','against','all','am','an','and','any','are','aren\'t','as','at','be','because','been','before','being','below','between','both','but','by','can\'t','cannot','could','couldn\'t','did','didn\'t','do','does','doesn\'t','doing','don\'t','down','during','each','few','for','from','further','had','hadn\'t','has','hasn\'t','have','haven\'t','having','he','he\'d','he\'ll','he\'s','her','here','here\'s','hers','herself','him','himself','his','how','how\'s','i','i\'d','i\'ll','i\'m','i\'ve','if','in','into','is','isn\'t','it','it\'s','its','itself','let\'s','me','more','most','mustn\'t','my','myself','no','nor','not','of','off','on','once','only','or','other','ought','our','ours', 'ourselves','out','over','own','same','shan\'t','she','she\'d','she\'ll','she\'s','should','shouldn\'t','so','some','such','than','that','that\'s','the','their','theirs','them','themselves','then','there','there\'s','these','they','they\'d','they\'ll','they\'re','they\'ve','this','those','through','to','too','under','until','up','very','was','wasn\'t','we','we\'d','we\'ll','we\'re','we\'ve','were','weren\'t','what','what\'s','when','when\'s','where','where\'s','which','while','who','who\'s','whom','why','why\'s','with','won\'t','would','wouldn\'t','you','you\'d','you\'ll','you\'re','you\'ve','your','yours','yourself','yourselves']

VocabularySize = ModelObject['VocabularySize']

PositiveClassProbability = float(ModelObject['PositiveClassProbability'])
NegativeClassProbability = float(ModelObject['NegativeClassProbability'])

TotalPositiveWordCount = ModelObject['TotalPositiveWordCount']
TotalNegativeWordCount = ModelObject['TotalNegativeWordCount']

def perform_stemming(tds):
    stemmed_td = []
    for td in tds:
        stemmed_td.append(ps.stem(td))
    return stemmed_td

def calculateProbability(TargetOrClass, WordFrequency, TotalWordCount, ClassProbability):
    if WordFrequency is None:
        return float(1+1)/float(TotalWordCount+VocabularySize)
    else:
        if TargetOrClass == 'negative':
            return float(WordFrequency[1])/ float(ClassProbability)
        else:
            return float(WordFrequency[0])/ float(ClassProbability)

def main(TestData, stemming):
    StopWordsRemovedTestData = ''
    for word in TestData.lower().split():
        if word not in StopWords:
            StopWordsRemovedTestData = StopWordsRemovedTestData + word + ' '
    StopWordsRemovedTestData = StopWordsRemovedTestData[:-1]
    StopWordsRemovedTestData = re.sub('[^A-Za-z0-9 ]+', '', StopWordsRemovedTestData)
    
    if stemming == True:
        TestBigrams = [b for b in zip(perform_stemming(re.split('\s+',StopWordsRemovedTestData)[:-1]), perform_stemming(re.split('\s+',StopWordsRemovedTestData)[1:]))]
    else:
        TestBigrams = [b for b in zip(re.split('\s+',StopWordsRemovedTestData)[:-1], re.split('\s+',StopWordsRemovedTestData)[1:])]

    ProbabilityOfPositiveClassGivenTestData = 1.0 * PositiveClassProbability
    ProbabilityOfNegativeClassGivenTestData = 1.0 * NegativeClassProbability


    for TestBigram in TestBigrams:
        key = TestBigram[0]+ ',' + TestBigram[1]
        WordProbability = ModelObject['bigram'].get(key, None)

        if WordProbability is None:
            if TestBigram == TestBigrams[0]:
                UnigramFirstWordProbabilities =  ModelObject['unigram'].get(TestBigram[0], None)
                UnigramSecondWordProbabilities = ModelObject['unigram'].get(TestBigram[1], None)
                ProbabilityOfPositiveClassGivenTestData = ProbabilityOfPositiveClassGivenTestData * calculateProbability('positive',UnigramFirstWordProbabilities, TotalPositiveWordCount, PositiveClassProbability) * calculateProbability('positive',UnigramSecondWordProbabilities, TotalPositiveWordCount, PositiveClassProbability)
                ProbabilityOfNegativeClassGivenTestData = ProbabilityOfNegativeClassGivenTestData * calculateProbability('negative', UnigramFirstWordProbabilities, TotalNegativeWordCount, NegativeClassProbability) * calculateProbability('negative', UnigramSecondWordProbabilities, TotalNegativeWordCount, NegativeClassProbability)
            else:
                UnigramSecondWordProbabilities = ModelObject['unigram'].get(TestBigram[1], None)
                ProbabilityOfPositiveClassGivenTestData = ProbabilityOfPositiveClassGivenTestData * calculateProbability('positive',UnigramSecondWordProbabilities, TotalPositiveWordCount, PositiveClassProbability)
                ProbabilityOfNegativeClassGivenTestData = ProbabilityOfNegativeClassGivenTestData * calculateProbability('negative',UnigramSecondWordProbabilities, TotalNegativeWordCount, PositiveClassProbability)
        else:
            ProbabilityOfPositiveClassGivenTestData = ProbabilityOfPositiveClassGivenTestData * (float(WordProbability[0]) / PositiveClassProbability)
            ProbabilityOfNegativeClassGivenTestData = ProbabilityOfNegativeClassGivenTestData * (float(WordProbability[1]) / NegativeClassProbability)

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
    #print('Please enter a review of your choice for testing the model:')
    #TestData = input()
    print('Please enter 1 for stemming test data and 2 for not stemming:')
    choice = input()
    if choice == 1:
        choice = True
    else:
        choice = False
    test_data_path = os.path.join('data', r'test data without stop words.json')
    successful_prediction = 0
    failure_prediction = 0
    cont_dist = {}
    temp_p_count = 0
    count = 1
    with open(test_data_path) as tdf:    
        train_data = json.load(tdf)
    for key in train_data:
        for value in train_data.get(key):
            if count % 400 == 0:
                cont_dist[count] = (temp_p_count / 400)
                temp_p_count = 0
            if key == main(value, choice):
                successful_prediction = successful_prediction + 1
                temp_p_count = temp_p_count + 1
            else:
                failure_prediction = failure_prediction + 1
            count = count + 1
    print(r':) ' + str(successful_prediction))
    print(r':( ' + str(failure_prediction))
    print(cont_dist)