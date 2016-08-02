from sklearn import *
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
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
    features = df.drop('activity', axis=1)
    features = features.drop('User', axis=1)
    return features, labels


def select(data, key_value_pairs, return_all=False):
    for key in key_value_pairs:
        select = data[key] == key_value_pairs[key]
        if (return_all == False):
            return data[select]
        else:
            other = data[select == False]
            return data[select], other

def Leave_one_person_out(classifier, users, df):
    for i in range(len(users)):
        testUser = users[i]
        train_all, test_all = select(df, {'User': testUser}, True)
        train_x, train_y = seperate_feature_label(train_all)
        test_x, test_y = seperate_feature_label(test_all)
        classifier.fit(train_x, train_y)
        predictions = classifier.predict(test_x)


def normalize(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = pd.DataFrame(min_max_scaler.fit_transform(df))
    return x_scaled


# data with labels,
def Supervised_learner(df, accuracy_tab,name=None):
    users = df['User'].unique()
    classifiers = {}
    classifiers['RandomForestClassifier'] = RandomForestClassifier(n_estimators=15)
    classifiers['PolyKernal-SVC'] = svm.SVC(kernel='poly', max_iter=20000)
    classifiers['KNeighborsClassifier'] = KNeighborsClassifier(n_neighbors=6)
    classifiers['LinearSVC'] = svm.LinearSVC()
    classifiers['Kmeans']= KMeans(n_clusters=6, init='random', max_iter=3000, n_init=6, tol=0.0001)
    # df=Feature_select(df)
    for algorithm, classifier in classifiers.items():
        accuracy = []
        for i in range(len(users)):
            testUser = users[i]
            train_all, test_all = select(df, {'User': testUser}, True)
            train_x, train_y = seperate_feature_label(train_all)
            test_x, test_y = seperate_feature_label(test_all)
            if (algorithm == 'Kmeans'):
                test_x = normalize(test_x)
                train_x = normalize(train_x)
                classifier.fit(train_x)
            else:
                classifier.fit(train_x, train_y)
            y_pred = classifier.predict(test_x)
            # print(classification_report(test_y,y_pred))
            accuracy.append(accuracy_score(y_pred, test_y))
        accuracy_tab[algorithm]= np.average(accuracy)
        print(np.average(accuracy))
        print ('Leave one person out %s Accuracy: %.4f (%.2f)  ' % (algorithm, np.average(accuracy), np.std(accuracy)))
    return accuracy_tab  

def Feature_select(df):
    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
    X, y = seperate_feature_label(df)
    feature_list = np.array(df.columns.values)
    forest.fit(X, y)
    importances = forest.feature_importances_
    droplist=[]
    for i in range(len(importances)):
        if (importances[i] < 0.023):
            droplist.append(i)
    df.drop(df.columns[droplist], axis=1, inplace=True)
    indices = np.argsort(importances)[::-1]
    indices= indices[:10]
   
    plt.figure()
    plt.title("Feature importance Ranking")
    plt.bar(range(10), importances[indices],alpha=0.4,color="blue", align="center")
    plt.xticks(range(10), feature_list[indices])
    plt.xlabel('Feature type')
    plt.ylabel('Relative importance')
    plt.xlim([-1, 10])
    plt.show()
    return df