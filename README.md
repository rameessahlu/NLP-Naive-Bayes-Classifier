# NLP-Naive-Bayes-Classifier
An implementation of Naive bayes classifier for sentiment analysis. [for learning]

Steps to Execute:
1. Execute GenerateSubsetOfDataset for generating finite dataset from huge dataset. 
2. Execute CreateVocabulary.py for feature selection. [unigram + bigram]
3. Execute CreateNaiveBayesModel.py for performing training and creation of the model. There are two options available: with stemming and without stemming.
4. Execute TestNaiveBayesModel.py for testing the model using the user entered input or a set of test data from file.
