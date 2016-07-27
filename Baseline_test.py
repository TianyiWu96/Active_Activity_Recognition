from sklearn import *
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier

from sklearn.cross_validation import *
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
def seperate_feature_label(df):
    labels = df['activity']
    features = df.drop('activity',axis=1)
    features = features.drop('User',axis=1)
    # print(features)
    return features,labels

def select(data, key_value_pairs, return_all=False):
   for key in key_value_pairs:
        select = data[key] == key_value_pairs[key]
        if(return_all == False):  return data[select]
        else:
          other = data[select==False]
          return data[select], other
#validation with 
def Leave_one_person_out(classifier,users ,df):
    for i in range(len(users)):
                testUser=users[i]
                train_all, test_all=select(df,{'User':testUser},True)
                train_x,train_y=seperate_feature_label(train_all)
                test_x, test_y=seperate_feature_label(test_all)
                classifier.fit(train_x,train_y)
                predictions = classifier.predict(test_x)


def normalize(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(x_scaled)
    return df_normalized

#data with labels, 
def Supervised_learner(df,name=None):
    users=df['User'].unique()
    classifiers ={}
    classifiers['RandomForestClassifier'] = RandomForestClassifier(n_estimators=5)
    # classifiers['PolyKernal-SVC'] = svm.SVC(kernel='poly', max_iter=20000)
    # classifiers['KNeighborsClassifier'] = KNeighborsClassifier(n_neighbors=5)
    # classifiers['LinearSVC'] = svm.LinearSVC()
    # classifiers['Kmeans']= KMeans(n_clusters=5, init='random', max_iter=3000, n_init=10, tol=0.0001)
    # df=Feature_select(df)
    for algorithm, classifier in classifiers.items():
        # test_all=pd.DataFrame()
        # accuracy_all=None
        accuracy =[]
        # df=Feature_select(df)
        #seperate test,train data:
        for i in range(len(users)-1):
                testUser=users[i]
                print(testUser)
                train_all, test_all= select(df,{'User':testUser},True)
                train_x,train_y=seperate_feature_label(train_all)
                # print(train_x)
                # print(train_y)
                test_x, test_y=seperate_feature_label(test_all)
                # print(test_x)
                if(algorithm == 'Kmeans'):
                    test_x = normalize(test_x)
                    train_x = normalize(train_x)
                    classifier.fit(train_x)
                else:
                    classifier.fit(train_x,train_y)
                y_pred = classifier.predict(test_x)
                # print(confusion_matrix(test_y,y_pred))
                # targetnames=['Lie','Sit','stand','iron','break','vacuum','break','ascend stairs','break','descend stairs','break','normal walk','break','nordic walk','break','cycle','break','run','break','rope jump']
                # print(classification_report(test_y,y_pred))
                accuracy.append(accuracy_score(y_pred,test_y))
                
        print ('Leave one person out \n%s Accuracy: %.2f%% (%.2f)  ' % (algorithm, np.average(accuracy), np.std(accuracy)))
def cluster_visualize(df):
     classifiers['Kmeans']= KMeans(n_clusters=5, init='random', max_iter=3000, n_init=10, tol=0.0001)
     for algorithm, classifier in classifiers.items():
        # test_all=pd.DataFrame()
        # accuracy_all=None
        accuracy =[]
        # df=Feature_select(df)
        #seperate test,train data:
        for i in range(len(users)-1):
                testUser=users[i]
                print(testUser)
                train_all, test_all= select(df,{'User':testUser},True)
                train_x,train_y=seperate_feature_label(train_all)
                # print(train_x)
                # print(train_y)
                test_x, test_y=seperate_feature_label(test_all)
                # print(test_x)
                if(algorithm == 'Kmeans'):
                    test_x = normalize(test_x)
                    train_x = normalize(train_x)
                    classifier.fit(train_x)
                else:
                    classifier.fit(train_x,train_y)
                y_pred = classifier.predict(test_x)
                # print(confusion_matrix(test_y,y_pred))
                # targetnames=['Lie','Sit','stand','iron','break','vacuum','break','ascend stairs','break','descend stairs','break','normal walk','break','nordic walk','break','cycle','break','run','break','rope jump']
                # print(classification_report(test_y,y_pred))
                accuracy.append(accuracy_score(y_pred,test_y))
#perform feature reduction using SVC
def Feature_select(df):
    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
    X,y = seperate_feature_label(df)
    feature_list=np.array(df.columns.values)
    forest.fit(X, y)  
    importances = forest.feature_importances_
    for i in range(len(importances)):
            if(importances[i]<0.01):
               df.drop(df.columns[i])
               # print(features_for_all.columns[i])
    return df
    # indices = np.argsort(importances)[::-1]
    # std = np.std([tree.feature_importances_ for tree in forest.estimators_],  axis=0)
    # plt.figure()
    # plt.title("Feature importances")
    # plt.bar(range(X.shape[1]), importances[indices],
    #        color="r", yerr=std[indices], align="center")
    # plt.xticks(range(X.shape[1]), feature_list[indices])
    # plt.xlim([-1, X.shape[1]])
    # plt.show()
