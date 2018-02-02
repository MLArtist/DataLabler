#coding: utf-8
#!/bin/python
import re
import pandas as pd
import numpy as np
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC

np.random.seed(42)

def Stemmer(word):
    return PorterStemmer().stem(word)

def remove_SpecialCharactersNumbers(text):
    return re.findall(r'\w+', text)

def apply_text_processing(text):
    word_list = remove_SpecialCharactersNumbers(text.strip().lower()) 
    return ' '.join([Stemmer(word) for word in word_list if len(word)>1])

def visualize(dataframe):
    count_classes = pd.value_counts(dataframe['lable'], sort=True).sort_index()
    count_classes.plot(kind='bar')
    plt.title('Class Bars')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    print("Please close the plot window to proceed further")
    plt.show()
    
def data_labler(train_file_name, query_file_name):
    #import the training data using pandas
    train=pd.read_csv(train_file_name, header=None, delimiter=',,, ', names=['question','lable'], engine='python')
    
    print("Number of samples=%d"%len(train))
    #apply preprocessing
    train['question']=train['question'].apply(apply_text_processing)
    train['lable']=train['lable'].apply(lambda x: x.strip().lower())
    
    #visualize whether classess aare not highly unbalanced
    visualize(train)
    
    #shuffle the data 
    train=train.sample(frac=1)
    
    #Get a bag of words for the test set, and convert to a numpy array
    count_vect= CountVectorizer(analyzer = 'word', lowercase=False, max_features = 5000)
    X_train=count_vect.fit_transform(train['question'])
    
    #Apply TFIDF
    tfidf_transformer = TfidfTransformer()
    X_train = tfidf_transformer.fit_transform(X_train)
    
    y_train =np.array(train['lable'])
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2)
    
    #classifier
    cls=SVC(kernel='linear', C=1, class_weight='balanced')

    model=cls.fit(X_train, y_train)
    print ("Score:", model.score(X_test, y_test))
    
    #predict
    #import the test file 
    test=pd.read_csv(query_file_name, header=None, delimiter=':', names=['junk','question'])
    test['question_preprocessed']=test['question'].apply(apply_text_processing)
    X=count_vect.transform(test['question_preprocessed'])
    X=tfidf_transformer.transform(X)
    
    test['prediction']= model.predict(X)
    
    #write to csv file 
    test.to_csv('result.csv', columns=['question', 'prediction'], index=False)
    print("Prediction Completed")
    
def main():
    train_file_name='train.txt'
    query_file_name='test.txt'
    data_labler(train_file_name,query_file_name)
    
if __name__=='__main__':
    main()

