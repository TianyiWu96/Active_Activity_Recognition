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
from load_PAMAP2 import Loading_PAMAP2
from load_HAPT import Loading_HAPT
warnings.filterwarnings("ignore")
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
def Loading(dataset,percentage):
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
      length=percentage*len(paths)//100
      i=0
      for filepath in paths: 
          if i>length:
            break
          else:
              data=Loading_PAMAP2(filepath,id,data)
              new=pd.DataFrame.from_dict(data)
              # print(new)
              id=id+1
          # return piece
      return new

def select(data, key_value_pairs):
   for key in key_value_pairs:
      select= data[key] == key_value_pairs[key]
      # print(data[select])
      return data[select]
     
def sliding_window(df,window_size,ratio):
    feature_rows=[]
    for i in range(0, len(df)-window_size, int(ratio*window_size)):
        window = df.iloc[i:i+window_size]
        feature_row = extract_features_in_window(window)
        feature_rows.append(feature_row)
    return pd.DataFrame(feature_rows)

def extract_features_in_window(df):
    feature_row={}

    df['m'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    
    extract_features_of_one_column(df, 'x', feature_row)
    extract_features_of_one_column(df, 'y', feature_row)
    extract_features_of_one_column(df, 'z', feature_row)
    extract_features_of_one_column(df, 'm', feature_row)

    extract_features_of_two_columns(df, ['x', 'm'], feature_row)
    extract_features_of_two_columns(df, ['y', 'm'], feature_row)
    extract_features_of_two_columns(df, ['z', 'm'], feature_row)
    
    feature_row['User'] = df.iloc[0]['User']
    feature_row['activity'] = df.iloc[0]['activity']

    return feature_row
def extract_features_of_one_column(df,key,feature_row):
    series = df[key]
    extract_statistical_features(series, '1_' + key, feature_row)

def extract_features_of_two_columns(df, columns, feature_row):
    feature_row['2_' + columns[0] + columns[1] + '_correlation'] = df[columns[0]].corr(df[columns[1]])

def extract_statistical_features(series, prefix, feature_row):
    feature_row[prefix + '_mean'] = series.mean()
    feature_row[prefix + '_std'] = series.std()
    feature_row[prefix + '_var'] = series.var()
    feature_row[prefix + '_min'] = series.min()
    feature_row[prefix + '_max'] = series.max()
    feature_row[prefix + '_energy'] = np.mean(series**2)

if __name__ == '__main__':
    data=Loading('PAMAP2',20)
    features=pd.DataFrame()
    users=data['User'].unique()
    for user in users:
        select_user=select(data,{'User':user})
        activities=data['activity'].unique()
        for activity in activities:
            select_activity=select(select_user,{'activity':activity})
            # print(select_activity)
            print(sliding_window(select_activity,512,0.5))
            
    classifiers = {}      

    ##Baseline
    
#### Classification

    





 