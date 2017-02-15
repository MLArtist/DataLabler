#!/bin/python
import re
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier



def DataLabler(train_file_name,querry_file_name):
#import the training data using pndas
	train=pd.read_csv(train_file_name, header=None, delimiter=",,, ", names=['question','lable'])
#remove the whilte spaces from labels
	train['lable']=[temp.strip() for temp in train['lable']]
#remove the punctuation and numbers
	train['question']=[re.sub("[^a-zA-Z]", " ", temp) for temp in train['question']]
	train['lable']=[re.sub("[^a-zA-Z]", " ", temp) for temp in train['lable']]
#Lables ['unknown', 'what', 'who', 'when', 'affirmation']
#convert into numpy array as they are easy to work with
	X_train =np.array(train['question'])
	Y_train =np.array(train['lable'])
#Get a bag of words for the test set, and convert to a numpy array
	vectorizer1 = CountVectorizer(analyzer = "word",tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 
#classifier						 
	classifier = Pipeline([('vectorizer', vectorizer1),('tfidf', TfidfTransformer()),('clf', OneVsRestClassifier(LinearSVC()))])
#train
	classifier.fit(X_train, Y_train)
#import the test file 
	test=pd.read_csv(querry_file_name, header=None,delimiter=":", names=["junk","questions"])
	test_ques=[temp.strip() for temp in test["questions"]]
#remove the punctuation and numbers
	test_ques=[re.sub("[^a-zA-Z]", " ", temp) for temp in test_ques]
#predict
	predicted = classifier.predict(test_ques)
#write the output using panads
	output = pd.DataFrame( data={"Question":test["questions"], "Type":predicted})
	output.to_csv( "response.csv", index=False)

def main():
	train_file_name="train.txt"
	querry_file_name="test.txt"
	DataLabler(train_file_name,querry_file_name)
	
if __name__=="__main__":
	main()
