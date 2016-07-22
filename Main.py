# import csv
import numpy as np
import csv as csv 
from sklearn import svm
import warnings 
from argparse import ArgumentParser
from path import Path
from collections import defaultdict
import pandas as pd
import os,glob
from pandas import *
from math import *
from scipy.fftpack import fft
from numpy import mean, sqrt, square
import scipy
from scipy.signal import *
from scipy.stats import mode
from sklearn import *
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import *
from load_PAMAP2 import Loading_PAMAP2
#from load_HAPT import Loading_HAPT
from feature_generate import *
from evaluation import *
from sklearn.metrics import classification_report
# from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import *
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
#%matplotlib inline

HAPT_folder="HAPT Data Set/RawData"
PAMAP2_folder="PAMAP2_Dataset/Protocol"
# Datasets_description=
# {
#      'HAPT':   f=50HZ  activity_number: 6  Users: 30
#      'PAMAP2': f=100HZ activity_number: 24 Users:  9

# }
def Loading(dataset):
   data = {}
   data['activity']=list()
   data['timestamp']=list()
   data['x']=list()
   data['y']=list()
   data['z']=list()
   data['User']=list()

   # if(dataset=="HAPT"):
   #    paths=glob.glob(HAPT_folder+'/*.txt')
   #    labelpath =HAPT_folder+'/labels.txt'
   #    fnewdata=Loading_HAPT(paths,labelpath,data)
   #    return fnewdata
   new = None
   if(dataset=="PAMAP2"):
      paths=glob.glob(PAMAP2_folder+'/*.dat')
      print str(paths)
      id=1
      for filepath in paths: 
              data=Loading_PAMAP2(filepath,id,data)
              new=pd.DataFrame.from_dict(data)
              # print(new)
              id=id+1
          # return piece
      return new
#return any specified column or one column and rest of it
def select(data,key_value_pairs,return_all=False):

   for key in key_value_pairs:
        select = data[key] == key_value_pairs[key]
        if(return_all == False):  return data[select]
        else:
          other = data[select==False]
          return data[select], other

def seperate_feature_label(df):
    labels=df['activity']
    features=df.drop('activity',axis=1) 
    features=df.drop('User',axis=1)
    return features,labels

def Leave_one_person_out(classifier,users ,df):
    for algorithm, classifier in classifiers.items(): 
        for i in range(len(users)):
                testUser=users[i]
                train_all, test_all=select(df,{'User':testUser},True)
                train_x,train_y=seperate_feature_label(train_all)
                test_x, test_y=seperate_feature_label(test_all)
                classifier.fit(train_x,train_y)
                predictions = classifier.predict(test_x)
    return predictions, test_y

# def Supervised_learning():

if __name__ == '__main__':
    data=Loading('PAMAP2')
    print('Loaded')
    frequency=100
    features_seperate={} #sperate feature for each user
    features_for_all=pd.DataFrame()
    users=data['User'].unique() #list of all users
    for user in users:
        select_user=select(data,{'User':user})
        activities=data['activity'].unique()
        for activity in activities: #one user and one activity
            select_activity= select(select_user,{'activity':activity})
            # print(select_activity)
            #smoothing first:
            #sliding windowing
            features_seperate[user]= sliding_window(select_activity,5*frequency,0.5)
            features_for_all=pd.concat([features_for_all,features_seperate[user]])

    # print(features_for_all)
    ##Baseline model
    classifiers = {}      
    classifiers['RandomForestClassifier'] = RandomForestClassifier(n_estimators=5)
    classifiers['Multi-SVC'] = svm.SVC(kernel='poly', max_iter=20000)
    classifiers['DecisionTreeClassifier'] = DecisionTreeClassifier(max_depth=None,min_samples_split=1)
    # classifiers['MLP']=MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    classifiers['KNeighborsClassifier'] = KNeighborsClassifier(n_neighbors=5)
    classifiers['LinearSVC'] = svm.LinearSVC()

    # classifiers['kMeans']=KMeans(n_clusters=8, init='k-means++', max_iter=3000, random_state=None,tol=0.0001)
    # Classification
    features,labels= seperate_feature_label(features_for_all)
    for algorithm, classifier in classifiers.items():
      classification_results = cross_validation.cross_val_score(classifier, features, labels, cv=10)
      # print （classifier.__class__.__name__, '\t & \t', classification_results.mean().round(2), '\t & \t', classification_results.std().round(2), ' \\\\'）
      print(algorithm, "Accuracy: %0.2f (+/- %0.2f)" % (classification_results.mean(), classification_results.std() * 2))
    #unsupervised['kMeans']= KMeans(n_clusters=8, init='k-means++', max_iter=3000, random_state=None,tol=0.0001)]
    #for algorithm, classifier in unsupervised.items():
      #classification_results = cross_validation.cross_val_score(classifier, features, labels, cv=10)
      # print （classifier.__class__.__name__, '\t & \t', classification_results.mean().round(2), '\t & \t', classification_results.std().round(2), ' \\\\'）
      print(algorithm, "Accuracy: %0.2f (+/- %0.2f)" % (classification_results.mean(), classification_results.std() * 2))
    # kf= KFold(9, n_folds=4, shuffle=False, random_state=None)
    # for classifier in classifiers.items():
    #   for train,test in kf：
    #      train_x,test_x = features[train], features[test]
    #      train_y,train_y =features[train],labels[test]
    #      classifier.fit(train_x,train_y)
    #      predictions = classifier.predict(test_x)
    #   print('K-fold_validation:' classifier, classification_report(test_y,predictions))



        



    





 