from __future__ import division
import matplotlib.pyplot as plt
import time
import numpy as np
import heapq
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,f1_scorev
import os.path
import pandas as pd
from data_process import *
from offline_train import *
from DataStructure import *
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
from scipy.spatial.distance import euclidean

def plot_accuracy(query_list,random_list,hit_list,hit_list_random):
    sample = np.arange(len(hit_list))
    acc_r=[]
    acc_a=[]
    for i in range(len(query_list)):
        if query_list[i]==3:
            acc_a.append(hit_list[i])
    for i in range(len(hit_list_random)):
        if random_list[i] ==3:
            acc_r.append(hit_list_random[i])
    fig=plt.figure()
    ax = fig.add_subplot(111)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    sample=np.arange(len(acc_a))
    a,=plt.plot(sample,acc_a,'r-',label='Active Learning ',linewidth=1.5)
    b,=plt.plot(sample,acc_r,'b-',label='Random Selection',linewidth=1.5)
    plt.grid()
    plt.legend(loc=4,prop={'size':15})
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.title('Accuracy gain per query',fontsize=20)
    plt.xlabel('Query Number',fontsize=20)
    plt.ylabel('Accuracy',fontsize=20)
    plt.show()

def plot_Q_hit_compare(usage_list, rand_list, hit_list,hit_list_random,hit_list_rf, label, user):
    # index = np.arange(len(usage_list))
    sample = np.arange(len(hit_list))
    # query=np.arange(len(usage_list))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    axis1=plt.subplot(2,1,1)
    plt.ylim([0, 1])
    act,=plt.plot(sample, hit_list, 'r-', label= 'Active Learning with Thresholds')
    rand, = plt.plot(sample, hit_list_random, 'b-', label='Active Learning Random Sampling')
    rf,=plt.plot(sample,hit_list_rf, 'g-',label='Baseline with Random Forest')


    ticks = np.arange(0, len(usage_list)/60, 5)

    # l=[act, rf, rand]
    # set('Box','off')
    leg=plt.legend(loc=4,prop={'size':15})
    # plt.legend([act, rand, rf], loc=4, )
    plt.xlabel('Data Stream',fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.grid()
    plt.title('Model Update',fontsize=20)
    plt.ylabel('Accuracy',fontsize=20)

    axis2=plt.subplot(212,sharex=axis1)

    index = np.arange(len(usage_list))  # (len(usage_list))
    plt.plot(index, usage_list, 'r-',label='Query indicator')
    # plt.plot(index, rand_list, 'b-')
    plt.plot(index,label,'b-',label='Ground Truth',linewidth=2)
    plt.legend(loc=1,prop={'size':15})
    y_tick = [1, 2, 3, 4, 5, 6]
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.yticks(y_tick,['Walking','Walk upstairs', 'Walk downstairs', 'Sitting', 'Standing', 'Laying'])
    q=0
    for i in range(len(usage_list)):
        if(usage_list[i]!=0): q+=1
    plt.xlabel(str(q) + ' Queries Location', fontsize=20)
    # plt.title('Corresponding '+str(q)+' Queries Location',fontsize=20)
    # plt.xlabel('Data Stream',fontsize=15)
    # plt.ylabel('Query')
    # plt.title('Query at',fontsize=12)
    print('Active:',hit_list[-1])
    print('Random:',hit_list_random[-1])
    print('RF:',hit_list_rf[-1])
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

def stream_simulation_active(test_x,test_y,active,flag,timer):
    buffer = []
    query_list=[]
    total=0
    hit=0
    hit_list=[]
    label_list=[]
    used=[]
    hit_r=0
    y_pred=[]
    f_score=[]
    online_time=[]
    update_time=[]
    hit_random=[]
    for ind,data in enumerate(test_x):
        start=time.time()
        query,point = active.query_by_similarity(test_x[ind],test_y[ind],flag,timer)
        if(query==True):
            query_list.append(3)
            buffer.append(point)
            total+=1
            hit+=1
            y_pred.append(test_y[ind])
            used.append(active.count)
        else:
            query_list.append(0)
            y_pred.append(point.label)
            if(point.label==test_y[ind]):
                hit+=1
        mid=time.time()
        online_time.append(mid-start)
        if(len(buffer)==5):
            active.update_RF(buffer)
            end=time.time()
            update_time.append(end-mid)
            buffer = []
        hit_list.append(hit)

    online_t=np.mean(online_time)
    # print('online',online_t)
    update_t=np.mean(update_time)
    # print('update',update_t)
    for i, x in enumerate(hit_list):
        hit_list[i] = hit_list[i] / (i + 1)
    return  online_t,update_t,query_list, hit_list,y_pred,f_score

def stream_simulation_random(test_x, test_y, active, total,timer):
    rand_list=random_select(test_x,test_label,total)
    buffer=[]
    hit_random=[]
    hit_r=0
    y_pred=[]
    f_score=[]
    for ind, data in enumerate(test_x):
        query, point = active.query_by_similarity(test_x[ind], test_y[ind], False,timer)
        print(point.label)
        if (rand_list[ind] == 3) :
            # print('Random asked',ind)
            hit_r+=1
            y_pred.append(test_y[ind])
            buffer.append(point)
        else:
            y_pred.append(point.label)
            if (point.label == test_y[ind]):
                hit_r += 1
        if (len(buffer) == 5):
            active.update_RF(buffer)
            buffer = []
        hit_random.append(hit_r)
        # f_score.append(f1_score(test_y[:ind], y_pred[:ind], average='weighted'))
    for i, x in enumerate(hit_random):
        hit_random[i] = hit_random[i] / (i + 1)
    return hit_random,rand_list,y_pred,f_score

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

def random_select(test_x,test_label,query_n):
    sample=np.arange(len(test_x))
    sample[:query_n]=3
    sample[query_n:]=0
    rng = np.random.RandomState(0)
    rng.shuffle(sample)
    return sample

def semi_random_forest(train,start,end):
    X, y=seperate_feature_label(train)
    X, y_all,unlabeled_indices= semi_supervised_learner(train,start,end)
    clf = RandomForestClassifier(n_estimators=20)
    clf.fit(X, y_all)

    return X, y_all,clf

if __name__ == '__main__':
    dataset = 'HAPT'
    filepath = '/Users/LilyWU/Documents/activity_recognition_for_sensor/Data/HAPT_First_15_user.csv'
    testpath='/Users/LilyWU/Documents/activity_recognition_for_sensor/Data/HAPT_user_'
    # testpath2='/Users/LilyWU/Documents/activity_recognition_for_sensor/Data/HAPT_user_18.csv'
    savepath= "/Users/LilyWU/Desktop/Csst/Summer/Results/"

    if (not os.path.exists(filepath)):
        data=write_to(filepath, dataset)
    train = pd.DataFrame.from_csv(filepath, header=0)
    accuracy_all=[]
    accuracy_random=[]
    accuracy_rf=[]
    test_id=16
    for test_id in range(16,17):
        test_data = pd.DataFrame.from_csv(testpath+'15_17'+'.csv', header=0)
        test_data=test_data
        print('start',test_id)
        start=50
        end=100
        start_t=time.time()
        # test_data,train= select(train,{'User': test_id})
        test_x,test_label= seperate_feature_label(test_data)
        test_x=np.array(test_x)
        test_label=np.array(test_label)
        train_x, train_y, clf = semi_random_forest(train,start,end)
        threshold=0.8
        query_number = 30
        timer=5
        active = Active_learned_Model(train_x, train_y, query_number, clf, timer)
        active.init_clusters(train_x, train_y)
        mid=time.time()
        online_t, update_t, query_list, active_res,active_y,f_active= stream_simulation_active(test_x, test_label, active, True,timer)
        accuracy_all.append(active_res[-1])

        a_rf = stream_test(test_x, test_label, clf)
        total=0
        for i in range(len(query_list)):
            if query_list[i]==3:
                total+=1
        active = Active_learned_Model(train_x, train_y, query_number, clf, timer)
        active.init_clusters(train_x, train_y)
        random_res,rand_list,random_y,f_random =stream_simulation_random(test_x,test_label,active,total,timer)
        accuracy_random.append(random_res[-1])
        accuracy_rf.append(a_rf[-1])

        # file_name = str(test_id) + 'User Accuracy Comparison_multi-50-50 time-based.csv'
    # print('active',f_score_active,np.mean(f_score_active),np.std(f_score_active))
    # print('random'.f_score_random,np.mean(f_score_random),np.std())
    # if(not os.path.exists(savepath + file_name)):
    #     res = {}
    #     res['Active learning'] = active_res[:-1]
    #     res['Random Forest'] = a_rf[:-1]
    #     res['Random'] =  random_res[:-1]
    #     res['online_t']=online_t
    #     res['start_t']=mid-start_t
    #     res['update_t']=update_t
    #     res['threshold'] = threshold
    #     res['Timer'] = timer
    #     res['query_number'] = query_number
    #     res = pd.DataFrame().from_dict(res)
    #     res.to_csv(savepath + file_name)
    # res = pd.DataFrame().from_csv(savepath + file_name)
    # print(a_rf[-1])
    # print('start time',mid-start_t)
    # print('online time',online_t)
    # print('model update',update_t)
    # print(res)
    plot_accuracy(query_list,rand_list, active_res,random_res)
    # plot_Q_hit_compare(query_list, rand_list,active_res, random_res,a_rf, test_label,test_id)
    plot_Q_hit_compare(query_list, rand_list,active_res, random_res,a_rf, test_label,test_id)
