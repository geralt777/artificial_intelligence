import os
import sys
import math
import string
import numpy as np
from numpy import array
import re

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from nltk.stem import PorterStemmer
from nltk import word_tokenize

def stemming_tokenizer(text):
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    a=""
    for w in word_tokenize(text):
        a=a+stemmer.stem(w)+" "
    return a

#Get a list of data from all files in a folder
def collect_words(folder_name, sflag):
    num_files = 0
    final_data = []
    for file_name in os.listdir(folder_name):
        num_files+=1
        #get entire data from file
        file_data = (open(folder_name+file_name, 'r')).read()
        #Get each line
        file_data_lines = file_data.split("\n")
        #discard the header
        start_point = 8
        end_point = len(file_data_lines)
        file_data_lines = file_data_lines[start_point:end_point]
        #combine all the lines to get a single string to operate on
        data = ' '.join(file_data_lines)
        #extract words from the single string 
        data = re.sub("[^\w]", " ",  data)
        #data = re.sub('['+string.punctuation+']', '', data)
        #remove extra blank spaces
        data = re.sub(' +', ' ', data)
        if(sflag==1):
            f_data = stemming_tokenizer(data)
        if(sflag==0):
            f_data = data
        final_data.append(f_data)
        
    #print(len(final_data))
    #sys.exit()        
    return final_data, num_files

#get data all processed to be sent for training and evaluation
def get_train_test_data(train_folder, test_folder, vflag, sflag, wflag):
    
    train_data = []
    #get training data
    hockey_words, hockey_files = collect_words(train_folder+"\\rec.sport.hockey\\", sflag)
    train_data.extend(hockey_words)
    med_words, med_files = collect_words(train_folder+"\\sci.med\\", sflag)
    train_data.extend(med_words)
    christian_words, christian_files = collect_words(train_folder+"\\soc.religion.christian\\", sflag)
    train_data.extend(christian_words)
    misc_words, misc_files = collect_words(train_folder+"\\talk.religion.misc\\", sflag)
    train_data.extend(misc_words)
    
    test_data = []
    #get test data
    thockey_words, thockey_files = collect_words(test_folder+"\\rec.sport.hockey\\", sflag)
    test_data.extend(thockey_words)
    tmed_words, tmed_files = collect_words(test_folder+"\\sci.med\\", sflag)
    test_data.extend(tmed_words)
    tchristian_words, tchristian_files = collect_words(test_folder+"\\soc.religion.christian\\", sflag)
    test_data.extend(tchristian_words)
    tmisc_words, tmisc_files = collect_words(test_folder+"\\talk.religion.misc\\", sflag)
    test_data.extend(tmisc_words)
    
    #tfid or count vectorizer
    if(vflag==0):
        if(wflag==0):
            cvectorizer = CountVectorizer(analyzer="word", ngram_range =(1,1))
        if(wflag==1):
            cvectorizer = CountVectorizer(analyzer="word", stop_words=stopwords.words('english'), ngram_range =(1,1))
    if(vflag==1):
        if(wflag==0):
            cvectorizer = TfidfVectorizer(analyzer="word", ngram_range =(1,1))
        if(wflag==1):
            cvectorizer = TfidfVectorizer(analyzer="word", stop_words=stopwords.words('english'), ngram_range =(1,1))
    
    #Get training and test features
    temp1 = cvectorizer.fit_transform(train_data)
    train_features = temp1.toarray()
    #print(train_features.shape)    
    temp2 = cvectorizer.transform(test_data)
    test_features = temp2.toarray()
    #print(test_features.shape)
    
    #Get classes
    target_values = []
    target_values.extend([1]*hockey_files)
    target_values.extend([2]*med_files)
    target_values.extend([3]*christian_files)
    target_values.extend([4]*misc_files)
    train_classes = array(target_values)
    
    target_values1 = []
    target_values1.extend([1]*thockey_files)
    target_values1.extend([2]*tmed_files)
    target_values1.extend([3]*tchristian_files)
    target_values1.extend([4]*tmisc_files)
    test_classes = array(target_values1)
    
    # print train_data_features
    # print train_data_features.shape
    
    return train_features, test_features, train_classes, test_classes   
 
 
#run all classifications
def classification(train_features, test_features, train_classes, alpha_val):
    
    #Multinomial Naive Bayes
    mnb = MultinomialNB(alpha = alpha_val)  
    mnb_prediction = mnb.fit(train_features, train_classes).predict(test_features)
    
    return mnb_prediction
    
#The main function
if __name__ == '__main__':

    train_folder = sys.argv[1]
    test_folder = sys.argv[2]
    a_val = 0.01
    
    train_features, test_features, train_classes, test_classes = get_train_test_data(train_folder, test_folder, 1, 0, 1)
    nb_prediction = classification(train_features, test_features, train_classes, a_val)    
    nb_prediction_f1_score = f1_score(test_classes, nb_prediction, average='macro')
    print(nb_prediction_f1_score)
    