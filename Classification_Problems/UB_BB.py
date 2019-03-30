import os
import sys
import math
import string
import numpy as np
from numpy import array
import re

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
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
from sklearn.datasets import load_digits

#Get a list of data from all files in a folder
def collect_words(folder_name):
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
        
        final_data.append(data)
        
    #print(len(final_data))
    #sys.exit()        
    return final_data, num_files

#get data all processed to be sent for training and evaluation
def get_train_test_data(train_folder, test_folder, gram):
    
    train_data = []
    #get training data
    hockey_words, hockey_files = collect_words(train_folder+"\\rec.sport.hockey\\")
    train_data.extend(hockey_words)
    med_words, med_files = collect_words(train_folder+"\\sci.med\\")
    train_data.extend(med_words)
    christian_words, christian_files = collect_words(train_folder+"\\soc.religion.christian\\")
    train_data.extend(christian_words)
    misc_words, misc_files = collect_words(train_folder+"\\talk.religion.misc\\")
    train_data.extend(misc_words)
    
    test_data = []
    #get test data
    thockey_words, thockey_files = collect_words(test_folder+"\\rec.sport.hockey\\")
    test_data.extend(thockey_words)
    tmed_words, tmed_files = collect_words(test_folder+"\\sci.med\\")
    test_data.extend(tmed_words)
    tchristian_words, tchristian_files = collect_words(test_folder+"\\soc.religion.christian\\")
    test_data.extend(tchristian_words)
    tmisc_words, tmisc_files = collect_words(test_folder+"\\talk.religion.misc\\")
    test_data.extend(tmisc_words)
    
    #tfid or count vectorizer
    cvectorizer = CountVectorizer(analyzer="word", ngram_range =(gram,gram))
    
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
    train_classes = np.asarray(target_values)
    
    target_values1 = []
    target_values1.extend([1]*thockey_files)
    target_values1.extend([2]*tmed_files)
    target_values1.extend([3]*tchristian_files)
    target_values1.extend([4]*tmisc_files)
    test_classes = np.asarray(target_values1)
    
    # print train_data_features
    # print train_data_features.shape
    
    return train_features, test_features, train_classes, test_classes   
    
#run all classifications
def classification(train_features, test_features, train_classes):
    
    #Multinomial Naive Bayes
    mnb = MultinomialNB()    
    mnb_prediction = mnb.fit(train_features, train_classes).predict(test_features)
    
    #Logistic Regression
    logreg = LogisticRegression(C=1e5)
    logreg_prediction = logreg.fit(train_features, train_classes).predict(test_features)
    
    #SVM
    svmsvc = svm.LinearSVC()
    svm_prediction = svmsvc.fit(train_features, train_classes).predict(test_features)
    
    #Random Forest
    rfc = RandomForestClassifier()
    rfc_prediction = rfc.fit(train_features, train_classes).predict(test_features)

    return mnb, logreg, svmsvc, rfc, mnb_prediction, logreg_prediction, svm_prediction, rfc_prediction
    
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("F1 Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring="f1_macro")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt
    
def display_lc_function(train_features, train_classes):
    #print("Display_LC")    
    title = "Learning Curves (Naive Bayes)"
    cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
    estimator = MultinomialNB()
    plot_learning_curve(estimator, title, train_features, train_classes, ylim=(0.5, 1.01), cv=cv, n_jobs=4)
    title = "Learning Curves (SVM, Linear kernel)"
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
    estimator = svm.LinearSVC()
    plot_learning_curve(estimator, title, train_features, train_classes, ylim=(0.5, 1.01), cv=cv, n_jobs=4)
    estimator = LogisticRegression(C=1e5)
    title = "Learning Curves (Logistic Regression)"
    plot_learning_curve(estimator, "Logistic Regression", train_features, train_classes, ylim=(0.5, 1.01), cv=cv, n_jobs=4)
    estimator = RandomForestClassifier()
    plot_learning_curve(estimator, "Random Forest", train_features, train_classes, ylim=(0.5, 1.01), cv=cv, n_jobs=4)
    plt.show()

#The main function
if __name__ == '__main__':

    train_folder = sys.argv[1]
    test_folder = sys.argv[2]
    output_file = open(sys.argv[3], 'w')
    display_lc = sys.argv[4]
    
    train_features, test_features, train_classes, test_classes = get_train_test_data(train_folder, test_folder, 1)
    mnb, logreg, svmsvc, rfc, nb_prediction, logreg_prediction, svm_prediction, rfc_prediction = classification(train_features, test_features, train_classes)
    #print(np.mean(nb_prediction == test_classes))
    #print(np.mean(logreg_prediction == test_classes))
    #print(np.mean(svm_prediction == test_classes))
    #print(np.mean(rfc_prediction == test_classes))
    
    bi_train_features, bi_test_features, bi_train_classes, bi_test_classes = get_train_test_data(train_folder, test_folder, 2)
    bi_mnb, bi_logreg, bi_svmsvc, bi_rfc, bi_nb_prediction, bi_logreg_prediction, bi_svm_prediction, bi_rfc_prediction = classification(bi_train_features, bi_test_features, bi_train_classes)
    #print(np.mean(bi_nb_prediction == bi_test_classes))
    #print(np.mean(bi_logreg_prediction == bi_test_classes))
    #print(np.mean(bi_svm_prediction == bi_test_classes))
    #print(np.mean(bi_rfc_prediction == bi_test_classes))
    
    #Precision values
    nb_prediction_precision_score = precision_score(test_classes, nb_prediction, average='macro')
    logreg_prediction_precision_score = precision_score(test_classes, logreg_prediction, average='macro')
    svm_prediction_precision_score = precision_score(test_classes, svm_prediction, average='macro')
    rfc_prediction_precision_score = precision_score(test_classes, rfc_prediction, average='macro')
    bi_nb_prediction_precision_score = precision_score(test_classes, bi_nb_prediction, average='macro')
    bi_logreg_prediction_precision_score = precision_score(test_classes, bi_logreg_prediction, average='macro')
    bi_svm_prediction_precision_score = precision_score(test_classes, bi_svm_prediction, average='macro')
    bi_rfc_prediction_precision_score = precision_score(test_classes, bi_rfc_prediction, average='macro')
    
    #Recall Values
    nb_prediction_recall_score = recall_score(test_classes, nb_prediction, average='macro')
    logreg_prediction_recall_score = recall_score(test_classes, logreg_prediction, average='macro')
    svm_prediction_recall_score = recall_score(test_classes, svm_prediction, average='macro')
    rfc_prediction_recall_score = recall_score(test_classes, rfc_prediction, average='macro')
    bi_nb_prediction_recall_score = recall_score(test_classes, bi_nb_prediction, average='macro')
    bi_logreg_prediction_recall_score = recall_score(test_classes, bi_logreg_prediction, average='macro')
    bi_svm_prediction_recall_score = recall_score(test_classes, bi_svm_prediction, average='macro')
    bi_rfc_prediction_recall_score = recall_score(test_classes, bi_rfc_prediction, average='macro')
    
    #F1 Scores
    nb_prediction_f1_score = f1_score(test_classes, nb_prediction, average='macro')
    logreg_prediction_f1_score = f1_score(test_classes, logreg_prediction, average='macro')
    svm_prediction_f1_score = f1_score(test_classes, svm_prediction, average='macro')
    rfc_prediction_f1_score = f1_score(test_classes, rfc_prediction, average='macro')
    bi_nb_prediction_f1_score = f1_score(test_classes, bi_nb_prediction, average='macro')
    bi_logreg_prediction_f1_score = f1_score(test_classes, bi_logreg_prediction, average='macro')
    bi_svm_prediction_f1_score = f1_score(test_classes, bi_svm_prediction, average='macro')
    bi_rfc_prediction_f1_score = f1_score(test_classes, bi_rfc_prediction, average='macro')
    
    #print(str(nb_prediction_precision_score)+","+str(nb_prediction_recall_score)+","+str(nb_prediction_f1_score))
    
    output_file.write("NB,UB,"+format(nb_prediction_precision_score, '.3f')+","+format(nb_prediction_recall_score, '.3f')+","+format(nb_prediction_f1_score, '.3f'))
    output_file.write("\n")
    output_file.write("NB,BB,"+format(bi_nb_prediction_precision_score, '.3f')+","+format(bi_nb_prediction_recall_score, '.3f')+","+format(bi_nb_prediction_f1_score, '.3f'))
    output_file.write("\n")
    output_file.write("LR,UB,"+format(logreg_prediction_precision_score, '.3f')+","+format(logreg_prediction_recall_score, '.3f')+","+format(logreg_prediction_f1_score, '.3f'))
    output_file.write("\n")
    output_file.write("LR,BB,"+format(bi_logreg_prediction_precision_score, '.3f')+","+format(bi_logreg_prediction_recall_score, '.3f')+","+format(bi_logreg_prediction_f1_score, '.3f'))
    output_file.write("\n")
    output_file.write("SVM,UB,"+format(svm_prediction_precision_score, '.3f')+","+format(svm_prediction_recall_score, '.3f')+","+format(svm_prediction_f1_score, '.3f'))
    output_file.write("\n")
    output_file.write("SVM,BB,"+format(bi_svm_prediction_precision_score, '.3f')+","+format(bi_svm_prediction_recall_score, '.3f')+","+format(bi_svm_prediction_f1_score, '.3f'))
    output_file.write("\n")
    output_file.write("RF,UB,"+format(rfc_prediction_precision_score, '.3f')+","+format(rfc_prediction_recall_score, '.3f')+","+format(rfc_prediction_f1_score, '.3f'))
    output_file.write("\n")
    output_file.write("RF,BB,"+format(bi_rfc_prediction_precision_score, '.3f')+","+format(bi_rfc_prediction_recall_score, '.3f')+","+format(bi_rfc_prediction_f1_score, '.3f'))   
     
    #print("Done")
    
    #Display the curves
    if(display_lc=='1'):
        display_lc_function(train_features, train_classes)
    
    #print(train_classes.shape)
    #print(test_classes)
        
    