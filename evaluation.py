
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


def compute_true_positive(df, label, reference_column = 'reference', prediction_column = 'prediction'):
    temp = df[df[reference_column] == df[prediction_column]]
    temp = temp[temp[prediction_column] == label]
    return float(len(temp))

def compute_false_positive(df, label, reference_column = 'reference', prediction_column = 'prediction'):
    temp = df[df[reference_column] != df[prediction_column]]
    temp = temp[temp[prediction_column] == label]
    return float(len(temp))

def compute_false_negative(df, label, reference_column = 'reference', prediction_column = 'prediction'):
    temp = df[df[reference_column] != df[prediction_column]]
    temp = temp[temp[reference_column] == label]
    return float(len(temp))


def compute_precision_for_label(df, label, reference_column = 'reference', prediction_column = 'prediction'):
    tp = compute_true_positive(df, label, reference_column, prediction_column)
    fp = compute_false_positive(df, label, reference_column, prediction_column)
    
    if tp + fp == 0:
        return 0
    
    return tp / (tp + fp)

def compute_recall_for_label(df, label, reference_column = 'reference', prediction_column = 'prediction'):
    tp = compute_true_positive(df, label, reference_column, prediction_column)
    fn = compute_false_negative(df, label, reference_column, prediction_column)
    
    if tp + fn == 0:
        return 0
    
    return tp / (tp + fn)

def compute_accuracy(df, reference_column = 'reference', prediction_column = 'prediction'):
    correct_prediction = df[df[reference_column] == df[prediction_column]]    
    accuracy = len(correct_prediction) / float(len(df))
    return accuracy


def get_confusion_matrix(df, reference_column = 'reference', prediction_column = 'prediction'):

    all_labels = Set(df[reference_column].values.tolist()) | Set(df[prediction_column].values.tolist()) 
    name_to_id_mapping = list(all_labels)
    
    prediction_ids = df[prediction_column].map(lambda x: name_to_id_mapping.index(x))
    reference_ids = df[reference_column].map(lambda x: name_to_id_mapping.index(x))
    
    cm = confusion_matrix(reference_ids, prediction_ids)
    
    confusion_matrix_df = pd.DataFrame(cm, columns = name_to_id_mapping, index = ["reference_" + str(x) for x in name_to_id_mapping])
    return confusion_matrix_df

def computing_result_metrics(df):
    precision = compute_precision_for_label(df, 1)
    recall = compute_recall_for_label(df, 1)

    f1 = 2 * precision * recall / (precision + recall)

    accuracy = compute_accuracy(df)

    return pd.Series({
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            })