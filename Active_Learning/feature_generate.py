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
from sklearn import preprocessing



def normalize(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df)
    df_normalized=pd.DataFrame(x_scaled)
    return df_normalized

def sliding_window(df,window_size,ratio):
    feature_rows = []
    for i in range(0, len(df)-window_size, int(ratio*window_size)):
        window = windowing(df,i,window_size)
        feature_row = extract_features_in_window(window)
        feature_rows.append(feature_row)
    return pd.DataFrame(feature_rows)

def windowing(df,start,window_size):
    return df.iloc[start:start+window_size]

def extract_features_in_window(df):
    feature_row = {}

    df['m'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    
    extract_features_of_one_column(df, 'x', feature_row)
    extract_features_of_one_column(df, 'y', feature_row)
    extract_features_of_one_column(df, 'z', feature_row)
    extract_features_of_one_column(df, 'm', feature_row)

    extract_features_of_two_columns(df, ['x', 'y'], feature_row)
    extract_features_of_two_columns(df, ['y', 'z'], feature_row)
    extract_features_of_two_columns(df, ['z', 'x'], feature_row)

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
    feature_row[prefix + '_skew'] =series.skew()
    feature_row[prefix + '_kurtosis']=series.kurtosis()
    feature_row[prefix + '_energy'] = np.mean(series**2)

    