from __future__ import division
import matplotlib.pyplot as plt
import time
import numpy as np
import heapq
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from data_process import *
from offline_train import *
import os.path
import pandas as pd
from DataStructure import *
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
from scipy.spatial.distance import euclidean
def semi_random_forest(train):
    X, y_all, unlabeled_indices=semi_supervised_learner(train,80,80)
    clf = RandomForestClassifier(n_estimators=30)
    clf.fit(X, y_all)
    return X, y_all,clf

def plot_Q_hit(usage_list, hit_list,hit_list_semi):
    for i ,x in enumerate(hit_list):
        hit_list[i] = hit_list[i]/ (i+1)
        hit_list_semi[i] = hit_list_semi[i]/ (i+1)
    sample = np.arange(len(hit_list))
    # query=np.arange(len(usage_list))
    plt.subplot(2,1,1)
    act,=plt.plot(sample, hit_list, 'r-', label= 'active learning')
    print('accuracy for active',hit_list)
    semi,=plt.plot(sample,hit_list_semi, 'b-',label='semi-supervised learning')
    # plt.legend([act,semi],loc=4)
    plt.xlabel('samples')
    plt.title('Accuracy gain for user 15')
    plt.ylabel('Accuracy')
    plt.subplot(2,1,2)
    index = np.arange(len(usage_list))  # (len(usage_list))
    plt.plot(index, usage_list, 'r-')
    plt.ylim([0, 3])
    plt.title('query points',fontsize=12)
    plt.show()
def stream_simulation(test_x,test_y,active,threshold):
    buffer = []
    pred=[]
    query_list=[]
    total=[]
    hit=0
    hit_list=[]
    used=[]
    for ind,data in enumerate(test_x):
        start=time.time()
        query,point = active.query_by_similarity(test_x[ind],test_y[ind],threshold)
        pred.append(point.label)
        if(query==True):
            query_list.append(1)
            buffer.append(point)
            hit+=1
            used.append(active.count)
        else:
            query_list.append(0)
            if(point.label==test_y[ind]):
                hit+=1
        if(len(buffer)==5):
            active.update(buffer)
            buffer = []
        total.append(ind)
        hit_list.append(hit)
    return query_list, hit_list
def plot_changes(test_x,test_label):
    rf=RandomForestClassifier(n_estimators=30)
    res=[]
    list=[]
    y=[]
    for ind in range(len(test_x)-1):
        old_x=test_x[ind]
        new_x=test_x[ind+1]
        res.append(euclidean(old_x,new_x))
        list.append(ind)
        y.append(50*int(test_label[ind]))
    dist,=plt.plot(list,res,'r-',label='Distance')
    act,=plt.plot(list,y,'b-',label='Activity')
    plt.legend([dist,act])
    plt.xlabel('time')

    plt.title('Eclidean Distance online changes for features')
    plt.show()

if __name__ == '__main__':
    dataset = 'HAPT'
    filepath = "/Users/LilyWU/Documents/activity_recognition_for_sensor" \
               "/Data/HAPT_First_15_user.csv"
    if (not os.path.exists(filepath)):
        data=write_to(filepath, dataset)
    data = pd.DataFrame.from_csv(filepath, header=0)
    test_id=5
    train_id=8
    train = select(data, {'User': test_id})
    test_data= select(data, {'User': train_id})
    # test_data=test_data
    test_x,test_label=seperate_feature_label(test_data)
    test_x=np.array(test_x)
    test_label=np.array(test_label)
    train_x, train_y, clf = semi_random_forest(train)
    active = Active_learned_Model(train_x, train_y, 30, clf)
    active.init_clusters(train_x, train_y)
    query_list, hit_list=stream_simulation(test_x, test_label, active, 1)
    query_list, hit_list_new = stream_simulation(test_x, test_label, active, 1.3)
    plot_Q_hit(query_list, hit_list_new,hit_list)
    # plot_changes(test_x,test_label)