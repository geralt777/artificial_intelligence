import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier

data = pd.read_csv('./bank-full.csv', sep=';', header='infer')
#convert string values to integer for better performance
data['y']=data['y'].map({'yes':1,'no':0}) 
data['default']=data['default'].map({'yes':1,'no':0}) 
data['housing']=data['housing'].map({'yes':1,'no':0}) 
data['loan']=data['loan'].map({'yes':1,'no':0}) 
data['contact']=data['contact'].map({'telephone':1,'cellular':2, 'unknown':3}) 
data['month']=data['month'].map({'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}) 

#parsing age field
data['agecat'] = 0
data.loc[(data['age'] < 19) & (data['age'] >= 0), 'agecat'] = 1
data.loc[(data['age'] < 40) & (data['age'] >= 19), 'agecat'] = 2
data.loc[data['age'] >= 40, 'agecat'] = 3
data.drop('age', axis=1, inplace=True)

#parsing campain field
data.loc[data['campaign'] == 1, 'campaign'] = 1
data.loc[(data['campaign'] >= 2) & (data['campaign'] <= 3), 'campaign'] = 2
data.loc[data['campaign'] >= 4, 'campaign'] = 3

#parsing previous field
data['contacted'] = 0
data.loc[data['previous'] == 0, 'contacted'] = 0
data.loc[(data['previous'] >= 1) & (data['pdays'] < 30), 'contacted'] = 1
data.loc[(data['previous'] >= 30) & (data['pdays'] < 35), 'contacted'] = 2
data.loc[data['previous'] >= 35, 'contacted'] = 3
data.drop('previous', axis=1, inplace=True)

#parsing pdays field
data.loc[data['pdays'] == -1, 'pdays'] = 10000
data['recent_pdays'] = np.where(data['pdays'], 1 / data.pdays, 1 / data.pdays)
data.drop('pdays', axis=1, inplace=True)

#parsing balance field
data['balancecat'] = 0
data.loc[(data['balance'] < 500), 'balancecat'] = 1
data.loc[(data['balance'] >= 500) & (data['balance'] < 1000), 'balancecat'] = 2
data.loc[data['balance'] >= 1000, 'balancecat'] = 3
data.drop('balance', axis=1, inplace=True)

#parsing job field
data['job'] = data['job'].replace(['management', 'admin.', 'blue-collar'], 2)
data['job'] = data['job'].replace(['retired','services', 'housemaid','technician'], 1)
data['job'] = data['job'].replace(['unemployed','unknown'], 3)
data['job'] = data['job'].replace(['student', 'entrepreneur', 'self-employed'], 4)

#parsing marital field
data['marital']=data['marital'].map({'married':1,'single':2,'divorced':3})

#parsing poutcome field
data['poutcome']=data['poutcome'].map({'unknown':0,'failure':1,'success':3,'other':0})

#parsing education field
data['level1'] = 0
data['level2'] = 0
data['level3'] = 0
data['unknown'] = 0
data.loc[data['education'] == 'primary', 'level1'] = 1
data.loc[data['education'] == 'secondary', 'level1'] = 1
data.loc[data['education'] == 'tertiary', 'level3'] = 1
data.loc[data['education'] == 'unknown', 'unknown'] = 1
data.drop('education', axis=1, inplace=True)

data_y = pd.DataFrame(data['y'])
data_X = data.drop(['y', 'day'],axis=1)
log_cols = ["Classifier", "Accuracy","Precision Score","Recall Score","F1-Score","roc-auc_Score"]
log = pd.DataFrame(columns=log_cols)


import warnings
warnings.filterwarnings('ignore')
rs = ShuffleSplit(n_splits=2, test_size=0.2, random_state=0)
rs.get_n_splits(data_X, data_y)

#SVM
train_index, test_index = rs.split(data_X, data_y)
X, X_test = data_X.iloc[train_index[0]], data_X.iloc[test_index[0]]
y, y_test = data_y.iloc[train_index[0]], data_y.iloc[test_index[0]]
cls = SVC(kernel='linear',gamma=0.001,C=1000,max_iter=10000).fit(X, y)
y_out = cls.predict(X_test)
accuracy = accuracy_score(y_test, y_out)
precision = metrics.precision_score(y_test, y_out, average='macro')
recall = metrics.recall_score(y_test, y_out, average='macro')
f1_score = metrics.f1_score(y_test, y_out, average='macro')
roc_auc = roc_auc_score(y_out, y_test)

X, X_test = data_X.iloc[train_index[1]], data_X.iloc[test_index[1]]
y, y_test = data_y.iloc[train_index[1]], data_y.iloc[test_index[1]]
cls = SVC(kernel='linear',gamma=0.001,C=1000,max_iter=10000).fit(X, y)
y_out = cls.predict(X_test)
accuracy += accuracy_score(y_test, y_out)
precision += metrics.precision_score(y_test, y_out, average='macro')
recall += metrics.recall_score(y_test, y_out, average='macro')
f1_score += metrics.f1_score(y_test, y_out, average='macro')
roc_auc += roc_auc_score(y_out, y_test)
log_entry = pd.DataFrame([['SVM', accuracy/2, precision/2, recall/2, f1_score/2, roc_auc/2]], columns=log_cols)
log = log.append(log_entry)

#Logistic Regression
train_index, test_index = rs.split(data_X, data_y)
X, X_test = data_X.iloc[train_index[0]], data_X.iloc[test_index[0]]
y, y_test = data_y.iloc[train_index[0]], data_y.iloc[test_index[0]]
cls = LogisticRegression(intercept_scaling=.1).fit(X, y)
y_out = cls.predict(X_test)
accuracy = accuracy_score(y_test, y_out)
precision = metrics.precision_score(y_test, y_out, average='macro')
recall = metrics.recall_score(y_test, y_out, average='macro')
f1_score = metrics.f1_score(y_test, y_out, average='macro')
roc_auc = roc_auc_score(y_out, y_test)

X, X_test = data_X.iloc[train_index[1]], data_X.iloc[test_index[1]]
y, y_test = data_y.iloc[train_index[1]], data_y.iloc[test_index[1]]
cls = LogisticRegression(intercept_scaling=.1).fit(X, y)
y_out = cls.predict(X_test)
accuracy += accuracy_score(y_test, y_out)
precision += metrics.precision_score(y_test, y_out, average='macro')
recall += metrics.recall_score(y_test, y_out, average='macro')
f1_score += metrics.f1_score(y_test, y_out, average='macro')
roc_auc += roc_auc_score(y_out, y_test)
log_entry = pd.DataFrame([['LR', accuracy/2, precision/2, recall/2, f1_score/2, roc_auc/2]], columns=log_cols)
log = log.append(log_entry)

#K Nearest Neighbour
train_index, test_index = rs.split(data_X, data_y)
X, X_test = data_X.iloc[train_index[0]], data_X.iloc[test_index[0]]
y, y_test = data_y.iloc[train_index[0]], data_y.iloc[test_index[0]]
cls = KNeighborsClassifier(6).fit(X, y)
y_out = cls.predict(X_test)
accuracy = accuracy_score(y_test, y_out)
precision = metrics.precision_score(y_test, y_out, average='macro')
recall = metrics.recall_score(y_test, y_out, average='macro')
f1_score = metrics.f1_score(y_test, y_out, average='macro')
roc_auc = roc_auc_score(y_out, y_test)

X, X_test = data_X.iloc[train_index[1]], data_X.iloc[test_index[1]]
y, y_test = data_y.iloc[train_index[1]], data_y.iloc[test_index[1]]
cls = KNeighborsClassifier(6).fit(X, y)
y_out = cls.predict(X_test)
accuracy += accuracy_score(y_test, y_out)
precision += metrics.precision_score(y_test, y_out, average='macro')
recall += metrics.recall_score(y_test, y_out, average='macro')
f1_score += metrics.f1_score(y_test, y_out, average='macro')
roc_auc += roc_auc_score(y_out, y_test)
log_entry = pd.DataFrame([['k-means', accuracy/2, precision/2, recall/2, f1_score/2, roc_auc/2]], columns=log_cols)
log = log.append(log_entry)

#Decision Tree Classifier
train_index, test_index = rs.split(data_X, data_y)
X, X_test = data_X.iloc[train_index[0]], data_X.iloc[test_index[0]]
y, y_test = data_y.iloc[train_index[0]], data_y.iloc[test_index[0]]
cls = DecisionTreeClassifier(max_depth=7).fit(X, y)
y_out = cls.predict(X_test)
accuracy = accuracy_score(y_test, y_out)
precision = metrics.precision_score(y_test, y_out, average='macro')
recall = metrics.recall_score(y_test, y_out, average='macro')
f1_score = metrics.f1_score(y_test, y_out, average='macro')
roc_auc = roc_auc_score(y_out, y_test)

X, X_test = data_X.iloc[train_index[1]], data_X.iloc[test_index[1]]
y, y_test = data_y.iloc[train_index[1]], data_y.iloc[test_index[1]]
cls = DecisionTreeClassifier(max_depth=7).fit(X, y)
y_out = cls.predict(X_test)
accuracy += accuracy_score(y_test, y_out)
precision += metrics.precision_score(y_test, y_out, average='macro')
recall += metrics.recall_score(y_test, y_out, average='macro')
f1_score += metrics.f1_score(y_test, y_out, average='macro')
roc_auc += roc_auc_score(y_out, y_test)
log_entry = pd.DataFrame([['DTC', accuracy/2, precision/2, recall/2, f1_score/2, roc_auc/2]], columns=log_cols)
log = log.append(log_entry)

#Linear Discriminant Analysis
train_index, test_index = rs.split(data_X, data_y)
X, X_test = data_X.iloc[train_index[0]], data_X.iloc[test_index[0]]
y, y_test = data_y.iloc[train_index[0]], data_y.iloc[test_index[0]]
cls = LinearDiscriminantAnalysis(n_components=10).fit(X, y)
y_out = cls.predict(X_test)
accuracy = accuracy_score(y_test, y_out)
precision = metrics.precision_score(y_test, y_out, average='macro')
recall = metrics.recall_score(y_test, y_out, average='macro')
f1_score = metrics.f1_score(y_test, y_out, average='macro')
roc_auc = roc_auc_score(y_out, y_test)

X, X_test = data_X.iloc[train_index[1]], data_X.iloc[test_index[1]]
y, y_test = data_y.iloc[train_index[1]], data_y.iloc[test_index[1]]
cls = LinearDiscriminantAnalysis(n_components=10).fit(X, y)
y_out = cls.predict(X_test)
accuracy += accuracy_score(y_test, y_out)
precision += metrics.precision_score(y_test, y_out, average='macro')
recall += metrics.recall_score(y_test, y_out, average='macro')
f1_score += metrics.f1_score(y_test, y_out, average='macro')
roc_auc += roc_auc_score(y_out, y_test)
log_entry = pd.DataFrame([['LDA', accuracy/2, precision/2, recall/2, f1_score/2, roc_auc/2]], columns=log_cols)
log = log.append(log_entry)

#Random Forest Classifier
train_index, test_index = rs.split(data_X, data_y)
X, X_test = data_X.iloc[train_index[0]], data_X.iloc[test_index[0]]
y, y_test = data_y.iloc[train_index[0]], data_y.iloc[test_index[0]]
cls = RandomForestClassifier(max_depth=6,min_samples_leaf=6, random_state=0).fit(X, y)
y_out = cls.predict(X_test)
accuracy = accuracy_score(y_test, y_out)
precision = metrics.precision_score(y_test, y_out, average='macro')
recall = metrics.recall_score(y_test, y_out, average='macro')
f1_score = metrics.f1_score(y_test, y_out, average='macro')
roc_auc = roc_auc_score(y_out, y_test)

X, X_test = data_X.iloc[train_index[1]], data_X.iloc[test_index[1]]
y, y_test = data_y.iloc[train_index[1]], data_y.iloc[test_index[1]]
cls = RandomForestClassifier(max_depth=6,min_samples_leaf=6, random_state=0).fit(X, y)
y_out = cls.predict(X_test)
accuracy += accuracy_score(y_test, y_out)
precision += metrics.precision_score(y_test, y_out, average='macro')
recall += metrics.recall_score(y_test, y_out, average='macro')
f1_score += metrics.f1_score(y_test, y_out, average='macro')
roc_auc += roc_auc_score(y_out, y_test)
log_entry = pd.DataFrame([['RFC', accuracy/2, precision/2, recall/2, f1_score/2, roc_auc/2]], columns=log_cols)
log = log.append(log_entry)

#Gradient Boost
train_index, test_index = rs.split(data_X, data_y)
X, X_test = data_X.iloc[train_index[0]], data_X.iloc[test_index[0]]
y, y_test = data_y.iloc[train_index[0]], data_y.iloc[test_index[0]]
cls = GradientBoostingClassifier().fit(X, y)
y_out = cls.predict(X_test)
accuracy = accuracy_score(y_test, y_out)
precision = metrics.precision_score(y_test, y_out, average='macro')
recall = metrics.recall_score(y_test, y_out, average='macro')
f1_score = metrics.f1_score(y_test, y_out, average='macro')
roc_auc = roc_auc_score(y_out, y_test)

X, X_test = data_X.iloc[train_index[1]], data_X.iloc[test_index[1]]
y, y_test = data_y.iloc[train_index[1]], data_y.iloc[test_index[1]]
cls = GradientBoostingClassifier().fit(X, y)
y_out = cls.predict(X_test)
accuracy += accuracy_score(y_test, y_out)
precision += metrics.precision_score(y_test, y_out, average='macro')
recall += metrics.recall_score(y_test, y_out, average='macro')
f1_score += metrics.f1_score(y_test, y_out, average='macro')
roc_auc += roc_auc_score(y_out, y_test)
log_entry = pd.DataFrame([['GD', accuracy/2, precision/2, recall/2, f1_score/2, roc_auc/2]], columns=log_cols)
log = log.append(log_entry)

#Adaptive Boosting Classifier
train_index, test_index = rs.split(data_X, data_y)
X, X_test = data_X.iloc[train_index[0]], data_X.iloc[test_index[0]]
y, y_test = data_y.iloc[train_index[0]], data_y.iloc[test_index[0]]
cls = AdaBoostClassifier().fit(X, y)
y_out = cls.predict(X_test)
accuracy = accuracy_score(y_test, y_out)
precision = metrics.precision_score(y_test, y_out, average='macro')
recall = metrics.recall_score(y_test, y_out, average='macro')
f1_score = metrics.f1_score(y_test, y_out, average='macro')
roc_auc = roc_auc_score(y_out, y_test)

X, X_test = data_X.iloc[train_index[1]], data_X.iloc[test_index[1]]
y, y_test = data_y.iloc[train_index[1]], data_y.iloc[test_index[1]]
cls = AdaBoostClassifier().fit(X, y)
y_out = cls.predict(X_test)
accuracy += accuracy_score(y_test, y_out)
precision += metrics.precision_score(y_test, y_out, average='macro')
recall += metrics.recall_score(y_test, y_out, average='macro')
f1_score += metrics.f1_score(y_test, y_out, average='macro')
roc_auc += roc_auc_score(y_out, y_test)
log_entry = pd.DataFrame([['ADA', accuracy/2, precision/2, recall/2, f1_score/2, roc_auc/2]], columns=log_cols)
log = log.append(log_entry)

#NN
train_index, test_index = rs.split(data_X, data_y)
X, X_test = data_X.iloc[train_index[0]], data_X.iloc[test_index[0]]
y, y_test = data_y.iloc[train_index[0]], data_y.iloc[test_index[0]]
cls = MLPClassifier(solver='adam',alpha=1e-6,hidden_layer_sizes=(10,25,10),random_state=1).fit(X, y)
y_out = cls.predict(X_test)
accuracy = accuracy_score(y_test, y_out)
precision = metrics.precision_score(y_test, y_out, average='macro')
recall = metrics.recall_score(y_test, y_out, average='macro')
f1_score = metrics.f1_score(y_test, y_out, average='macro')
roc_auc = roc_auc_score(y_out, y_test)

X, X_test = data_X.iloc[train_index[1]], data_X.iloc[test_index[1]]
y, y_test = data_y.iloc[train_index[1]], data_y.iloc[test_index[1]]
cls = MLPClassifier(solver='adam',alpha=1e-6,hidden_layer_sizes=(10,25,10),random_state=1).fit(X, y)
y_out = cls.predict(X_test)
accuracy += accuracy_score(y_test, y_out)
precision += metrics.precision_score(y_test, y_out, average='macro')
recall += metrics.recall_score(y_test, y_out, average='macro')
f1_score += metrics.f1_score(y_test, y_out, average='macro')
roc_auc += roc_auc_score(y_out, y_test)
log_entry = pd.DataFrame([['NN', accuracy/2, precision/2, recall/2, f1_score/2, roc_auc/2]], columns=log_cols)
log = log.append(log_entry)

print(log)
