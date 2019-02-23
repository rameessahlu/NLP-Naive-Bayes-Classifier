# NLP-Naive-Bayes-Classifier [Educational Purpose]
An implementation of Naive Bayes classifier for sentiment analysis. 

Steps to Execute:
1. Execute GenerateSubsetOfDataset.py for splitting the dataset into training and testing set. 
2. Execute CreateVocabulary.py for vocabulary creation and tokenization - unigram, bigram
3. Execute CreateNaiveBayesModel.py for performing training and creation of the model. There are two options available for model creation based on feature set: the first option is model creation using feature sets which underwent through stemming during the preprocessing stage and the second option is for model creation using feature sets without stemming.
4. Execute TestNaiveBayesModel.py for testing the model and there are two options, one is for classifying the data entered in the shell and the second one is for testing the model from a test data set.
