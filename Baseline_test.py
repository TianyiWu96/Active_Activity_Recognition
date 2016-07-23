from sklearn import *
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
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
import pandas as pd
import numpy as np
def seperate_feature_label(df):
    labels = df['activity']
    features = df.drop('activity',axis=1)
    features = df.drop('User',axis=1)
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
    classifiers['PolyKernal-SVC'] = svm.SVC(kernel='poly', max_iter=20000)
    classifiers['KNeighborsClassifier'] = KNeighborsClassifier(n_neighbors=5)
    classifiers['LinearSVC'] = svm.LinearSVC()
    classifiers['Kmeans']= KMeans(n_clusters=5, init='k-means++', max_iter=3000, random_state=None, tol=0.0001)
    features, labels = seperate_feature_label(df)
    accuracy=[]
    
    for algorithm, classifier in classifiers.items():
        # test_all=pd.DataFrame()
        # accuracy_all=None
        accuracy=[]
        for i in range(len(users)-1):
                testUser=users[i]
                train_all, test_all=select(df,{'User':testUser},True)
                train_x,train_y=seperate_feature_label(train_all)
                test_x, test_y=seperate_feature_label(test_all)
                if(algorithm == 'Kmeans'):
                    test_x= normalize(test_x)
                    train_x=normalize(train_x)
                    classifier.fit(train_x)
                else:
                    classifier.fit(train_x,train_y)
                pred_y = classifier.predict(test_x)
                accuracy.append(accuracy_score((pred_y,test_y)))
            
        print ('\n%s Accuracy: %.2f%% (%.2f)  ' % (algorithm, np.average(accuracy), np.std(accuracy)))
       

 #data with no labels, do not need to be normalized
