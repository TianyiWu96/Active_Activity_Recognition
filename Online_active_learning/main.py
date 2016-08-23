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
# #Todo:Error plotting
# def error_plot(usage_list,hit_list,user):
#
# def plot_
def plot_Q_hit_compare(usage_list, hit_list,hit_list_rf, label, user):
    sample = np.arange(len(hit_list))
    # query=np.arange(len(usage_list))
    plt.subplot(2,1,1)
    plt.ylim([0, 1])
    act,=plt.plot(sample, hit_list, 'r-', label= 'Active learning')
    # semi,=plt.plot(sample,hit_list_f, 'b-',label='Model without Query')
    rf,=plt.plot(sample,hit_list_rf, 'g-',label='Random Forest')
    print(label)
    plt.legend([act,rf],loc=4)
    plt.xlabel('Iteration')
    plt.title('Accuracy gain for one user'+str(user),fontsize=15)
    plt.ylabel('Accuracy')
    plt.subplot(2,1,2)
    index = np.arange(len(usage_list))  # (len(usage_list))
    plt.plot(index, usage_list, 'b-')
    plt.ylim([0, 2])
    plt.xlabel('Data stream')
    plt.title('Query Location',fontsize=15)
    plt.show()
def stream_test(test_x,test_y,classifier):
    pred = []
    query_list = []
    total = []
    hit = 0
    hit_list = []
    for ind, data in enumerate(test_x):
        res=int(classifier.predict(data))
        if(res==test_y[ind]):
            hit+=1
        hit_list.append(hit)
    for i, x in enumerate(hit_list):
        hit_list[i] = hit_list[i] / (i + 1)
    return hit_list
def stream_simulation(test_x,test_y,active,flag):
    buffer = []
    query_list=[]
    total=[]
    hit=0
    hit_list=[]
    used=[]
    for ind,data in enumerate(test_x):
        start=time.time()
        query,point = active.query_by_similarity(test_x[ind],test_y[ind],flag)
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
            active.update_RF(buffer)
            buffer = []
        total.append(ind)
        hit_list.append(hit)
    for i, x in enumerate(hit_list):
        hit_list[i] = hit_list[i] / (i + 1)
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

def semi_random_forest(train):
    X, y_all, unlabeled_indices= semi_supervised_learner(train,50,50)
    clf = RandomForestClassifier(n_estimators=30)
    clf.fit(X, y_all)
    return X, y_all,clf

if __name__ == '__main__':
    dataset = 'HAPT'
    filepath = "/Users/LilyWU/Documents/activity_recognition_for_sensor" \
               "/Data/HAPT_First_15_user.csv"
    savepath= "/Users/LilyWU/Documents/activity_recognition_for_sensor/Result/"

    if (not os.path.exists(filepath)):
        data=write_to(filepath, dataset)
    data = pd.DataFrame.from_csv(filepath, header=0)

    test_id=13
    test_data, train= select(data, {'User': test_id}, True)
    # train= select(train, {'User': 5})
    # train_two= select(train, {'User': 2})
    # train_three= select(train, {'User': 3})
    # pd.concat([train,train_two,train_three])
    # print(train)
    # test_data,train= select(data, {'User': train_id})
    test_data=test_data
    test_x,test_label=seperate_feature_label(test_data)
    test_x=np.array(test_x)
    test_label=np.array(test_label)
    train_x, train_y, clf = semi_random_forest(train)

    active = Active_learned_Model(train_x, train_y, 30, clf)
    active.init_clusters(train_x, train_y)
    query_list,a_t= stream_simulation(test_x, test_label, active, True)
    # active = Active_learned_Model(train_x, train_y, 30, clf)
    # active.init_clusters(train_x, train_y)
    # query_list_l, a_f = stream_simulation(test_x, test_label, active, False)
    res={}
    res['Active learning']=a_t
    # res['Without Active learning']=a_f
    a_rf=stream_test(test_x, test_label,clf)
    res['Random Forest']=a_rf
    res['Query']=query_list
    res['label']=test_label.tolist()
    file_name = '50-50 trained for User'+str(test_id)+'30 queries (adaptive threshold).csv'
    res= pd.DataFrame.from_dict(res)
    res.to_csv(savepath+file_name)
    plot_Q_hit_compare(query_list, a_t,a_rf, test_label,test_id)
    # plot_Q_hit(query_list, hit_list_new,test_id)