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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from load_PAMAP2 import Loading_PAMAP2
from load_HAPT import Loading_HAPT
from feature_generate import *
from evaluation import *
# from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import *
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
#%matplotlib inline
# def parse_args():
#     argparser = ArgumentParser()
#     argparser.add_argument("--HAPT_path", default="/Users/LilyWU/Documents/activity_recognition_repo/HAPT Data Set/RawData")
#     argparser.add_argument("--PAMAP2_path", default="/Users/LilyWU/Documents/activity_recognition_repo/PAMAP2_Dataset/Protocol")
#     argparser.add_argument("--",default="/Users/LilyWU/Documents/activity_recognition_repo/Activity recognition exp 2/save")
#     # argparser.add_argument("--save_path", default="patches/")
#     return argparser.parse_args()
HAPT_folder="HAPT Data Set/RawData"
PAMAP2_folder="PAMAP2_Dataset/Protocol"
# Datasets_description=
# {
#      'HAPT':   f=50HZ  activity_number: 6  Users: 30
#      'PAMAP2': f=100HZ activity_number: 24 Users:  9

# }
def Loading(dataset):
   data={}
   data['activity']=list()
   data['timestamp']=list()
   data['x']=list()
   data['y']=list()
   data['z']=list()
   data['User']=list()

   if(dataset=="HAPT"):
      paths=glob.glob(HAPT_folder+'/*.txt')
      labelpath =HAPT_folder+'/labels.txt'
      fnewdata=Loading_HAPT(paths,labelpath,data)
      return newdata

   if(dataset=="PAMAP2"):
      paths=glob.glob(PAMAP2_folder+'/*.txt') 
      id=1
      for filepath in paths: 
              data=Loading_PAMAP2(filepath,id,data)
              new=pd.DataFrame.from_dict(data)
              # print(new)
              id=id+1
          # return piece
      return new

def select(data,key_value_pairs,return_all=False):
   for key in key_value_pairs:
        if(return_all==False):
          select= data[key] == key_value_pairs[key]
          return data[select]
      # print(data[select])
        else:
          other=data[select==False]
          return data[select],other

def seperate_feature_label(df):
    labels=df['activity']
    features=df.drop('activity','User',axis=1) 
    return features,labels

def Leave_one_person_out(user,df):
    training, testing=select(df,{'User':user},True)
    return training,testing

# def Supervised_learning():

if __name__ == '__main__':
    data=Loading('PAMAP2')
    print('Loaded')
    frequency=100
    features_seperate={}
    features_for_all=pd.DataFrame()
    users=data['User'].unique()
    for user in users:
        select_user=select(data,{'User':user})
        activities=data['activity'].unique()
        for activity in activities:
            select_activity= select(select_user,{'activity':activity})
            # print(select_activity)
            #smoothing first:
            #sliding windowing
            features_seperate[user]= sliding_window(select_activity,5*frequency,0.5)
            features_for_all=pd.concat([features_for_all,features_seperate[user]])
    ##Baseline model
    perform_percent=50
    features_part=features_for_all.iloc()
    classifiers = {}      
    # classifiers['RandomForestClassifier'] = RandomForestClassifier(n_estimators=5)
    # classifiers['svc'] = svm.SVC(kernel='poly', max_iter=20000)
    # classifiers['MLP']=MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    # classifiers['KNeighborsClassifier'] = KNeighborsClassifier(n_neighbors=5)
    # classifiers['LinearSVC'] = svm.LinearSVC()
    classifiers['kMeans']=KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001)
    
#### Classification

    for algorithm, classifier in classifiers.items(): 
            for user in features_for_all['User']:
                train_all,test_all=Leave_one_person_out(user,features_for_all)
                train_x,train_y=seperate_feature_label(train_all)
                test_x,test_y=seperate_feature_label(test_all)
                results=pd.DataFrame()
                print('Classification Begin, leave out:', user)
                print('Perform:',algorithm)
                classifier.fit(train_x)
                predictions = classifier.predict(test_x)
                results.append(pd.DataFrame({
                    'prediction_score': predictions,
                    'prediction': np.sign(predictions),
                    'reference': testing_y,
                    'user': user,
                    'algorithm': algorithm,
                }))
    print(computing_result_metrics(results))


        



    





 