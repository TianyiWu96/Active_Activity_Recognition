import numpy as np
import csv as csv 
from sklearn import svm
import warnings 
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
from sklearn import preprocessing
from load_PAMAP2 import Loading_PAMAP2
from load_HAPT import Loading_HAPT
from feature_generate import *
from evaluation import *
from Baseline_test import *
from sklearn.metrics import classification_report
# from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import *
from sklearn.feature_selection import *
from semi_supervised import *
import os.path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
# matplotlib inline
HAPT_folder="HAPT Data Set/RawData/"
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

   if(dataset=="HAPT"):
      new=pd.DataFrame.from_dict(Loading_HAPT(HAPT_folder,data))
      return new

   if(dataset=="PAMAP2"):
      paths=glob.glob(PAMAP2_folder+'/*.txt')
      id=1
      for filepath in paths: 
              data=Loading_PAMAP2(filepath,id,data)
              new=pd.DataFrame.from_dict(data)
              id=id+1
      return new
#return any specified column or one column and rest of it
def select(data,key_value_pairs,return_all=False):

   for key in key_value_pairs:
        select = data[key] == key_value_pairs[key]
        if(return_all == False):  
          return data[select]
        else:
          other = data[select ==False]
          return data[select], other
def generate_features(data,user,activity,frequency):
        select_user=select(data,{'User':user})
        select_user['x']=butter_filter(select_user['x'],  0.4, frequency)
        select_user['y']=butter_filter(select_user['y'],  0.4, frequency)
        select_user['z']=butter_filter(select_user['z'],  0.4, frequency)
        select_activity =select(select_user,{'activity':activity})
        features = sliding_window(select_activity,2*frequency,0.5)
        return features

if __name__ == '__main__':
    dataset='HAPT'
    # filepath="First_5_for_PAMAP2.csv"
    filepath="First_15_user_HAPT.csv"
    if(not os.path.exists(filepath)):
        data =Loading(dataset)
        print('loaded')
        frequency= 50
        features_seperate={} #sperate feature for each user
        features_for_all=pd.DataFrame()
        users=data['User'].unique()
        activities = data['activity'].unique() #list of all users
        for user in range(1,16):
            print(user)
            # features_for_all=pd.DataFrame()
            for activity in activities: #one user and one activity
                features = generate_features(data,user,activity,frequency)
                features_for_all=pd.concat([features_for_all,features])
            # features_for_all=Feature_select(features_for_all)
        features_for_all.to_csv(filepath,header=features_for_all.columns.values.tolist())
    
    data = pd.DataFrame.from_csv(filepath,header=0)
    plt.subplot(231)
    semi_supervised_test2(data, 250, 50)
    plt.subplot(232)
    semi_supervised_test2(data, 300, 50)
    plt.subplot(233)
    semi_supervised_test2(data, 350, 50)
    plt.subplot(234)
    semi_supervised_test2(data, 400, 50)
    plt.subplot(235)
    semi_supervised_test2(data, 450, 50)
    plt.subplot(236)
    semi_supervised_test2(data, 500, 50)
    
    plt.show()
    # plt.savefig('increase labels.png')
    # Feature_select(data)# users=data['User'].unique()
    # activities= data['activity'].unique()
    # for user in range(1,2):
    #         # for activity in activities: #one user and one activity
    #           df =select(data,{'User':user})
    #           plotting=plt.figure().gca(projection='3d') 
    #           for activity in activities:
    #               df= select(df,{'activity':activity})
    #               if(activity==1):
    #                   plotting.scatter(df['x'],df['y'],df['z'],color='r')
    #               if(activity==2):
    #                   plotting.scatter(df['x'],df['y'],df['z'],color='b')
                  
    #           plt.show()
    # Supervised_learner(data,accuracy)
    # accuracy.plot()
    # plt.show()
    
    # X,y_pred,y_true=semi_supervised_learner(data,100,3000)
    



    





 