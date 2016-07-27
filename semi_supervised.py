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
from sklearn.cross_validation import *
from sklearn import preprocessing
from sklearn.metrics import classification_report
def seperate_feature_label(df):
    labels = df['activity']
    features = df.drop('activity',axis=1)
    features = features.drop('User',axis=1)
    # print(features)
    return features,labels
def semi_supervised_learner(df):
	 feature,label=seperate_feature_label(df)
	 random_unlabeled_points = np.where(np.random.random_integers(0, 1,size=len(label)))
	 # print(random_unlabeled_points)
	 labels = np.copy(label)
	 labels[random_unlabeled_points] = -1
	 print(labels)
	 test_y=labels[random_unlabeled_points]
	 model = LabelSpreading(kernel='knn', gamma=20, n_neighbors=7, alpha=0.2, max_iter=30, tol=0.001)
	 model.fit(feature, labels)
	 print(feature.iloc[random_unlabeled_points])
	 y_pred=model.predict(feature.iloc[random_unlabeled_points])
	 
	 print(classification_report(test_y,y_pred))
