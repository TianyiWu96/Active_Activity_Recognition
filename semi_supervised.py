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
from sklearn.multiclass import OneVsRestClassifier
from sklearn import *
from scipy.signal import *
from scipy.stats import mode
from sklearn import *
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import label_propagation
from sklearn.cross_validation import *
from sklearn import preprocessing
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import classification_report, confusion_matrix
from Baseline_test import *
def seperate_feature_label(df):
    labels = df['activity']
    features = df.drop('activity',axis=1)
    features = features.drop('User',axis=1)
    # print(features)
    return features,labels

def balanced_sample_maker(X, y, sample_size, random_seed=None):
    uniq_levels = y.unique()
    uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_seed is None:
        np.random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx
    # oversampling on observations of each label
    balanced_copy_idx = []
    for gb_level, gb_idx in groupby_levels.items():
        over_sample_idx = np.random.choice(gb_idx, size=sample_size, replace=True).tolist()
        balanced_copy_idx+=over_sample_idx
        # print(balanced_copy_idx)
    np.random.shuffle(balanced_copy_idx)
    # print(y.iloc[balanced_copy_idx])
    # print(X.iloc[balanced_copy_idx])
    return balanced_copy_idx

def semi_supervised_learner(df,labeled_points,total_points):
    #split the data according to the classes, generate labelled , unlabelled mark for each and reshuffle.
    
    feature,label = seperate_feature_label(df)
    indices = np.arange(len(feature))
    label_indices= balanced_sample_maker(feature,label,labeled_points/len(label))
    unlabeled_indices=np.delete(indices,np.array(label_indices))
    rng = np.random.RandomState(0)
    rng.shuffle(unlabeled_indices)
    indices=np.concatenate((label_indices,unlabeled_indices[:total_points]))
    n_total_samples = len(indices)
    unlabeled_indices=np.arange(n_total_samples)[labeled_points:]
    X = feature.iloc[indices]
    y = label.iloc[indices]
    for i in range(1):
        y_train = np.copy(y)
        y_train[unlabeled_indices] = -1
        classifier= KNeighborsClassifier(n_neighbors=5)
        classifier.fit(X.iloc[:labeled_points],y.iloc[:labeled_points])
        y_pred = classifier.predict(X.iloc[labeled_points:])
        print('Supervised learing:')
        true_labels = y.iloc[unlabeled_indices]
        print(confusion_matrix(true_labels,y_pred))
        print(accuracy_score(true_labels, y_pred))
        print('Semi-supervised learing:')
        lp_model = label_propagation.LabelSpreading(gamma=0.25, kernel='knn',max_iter=300,n_neighbors=5)
        lp_model.fit(X, y_train)
        predicted_labels = lp_model.transduction_[unlabeled_indices]
        # print('Iteration %i %s' % (i, 70 * '_'))
        print("Label Spreading model: %d labeled & %d unlabeled (%d total)"
              % (labeled_points, n_total_samples - labeled_points, n_total_samples))

        print(confusion_matrix(true_labels, predicted_labels))
        return accuracy_score(true_labels, predicted_labels))
    
