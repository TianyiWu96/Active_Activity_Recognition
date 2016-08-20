from __future__ import division
import matplotlib.pyplot as plt
from load_PAMAP2 import Loading_PAMAP2
#from load_HAPT import Loading_HAPT
from feature_generate import *
from similarity_check import *
from initial_training import *
import os.path
from DataStructure import *
import time
import numpy as np

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

HAPT_folder = "HAPT Data Set/RawData/"
PAMAP2_folder = "PAMAP2_Dataset/Protocol"


def Loading(dataset):
    data = {}
    data['activity'] = list()
    data['timestamp'] = list()
    data['x'] = list()
    data['y'] = list()
    data['z'] = list()
    data['User'] = list()

    if (dataset == "PAMAP2"):
        paths = glob.glob(PAMAP2_folder + '/*.txt')
        id = 1
        for filepath in paths:
            data = Loading_PAMAP2(filepath, id, data)
            new = pd.DataFrame.from_dict(data)
            id = id + 1
        return new


# return any specified column or one column and rest of it
def select(data, key_value_pairs, return_all=False):
    for key in key_value_pairs:
        select = data[key] == key_value_pairs[key]
        if (return_all == False):
            return data[select]
        else:
            other = data[select == False]
            return data[select], other

def generate_features(data, user, activity, ):
    select_user = select(data, {'User': user})
    select_activity = select(select_user, {'activity': activity})
    # print(select_activity)
    features = sliding_window(select_activity, 2 * frequency, 0.5)
    # print(features)
    return features

def get_cluster(label, cluster_dict):
    for clus_num in cluster_dict:
        c = cluster_dict[clus_num]
        if int(float(c.label)) == int(label):
            return c
    return None

# Done: fixed major error when plotting accuracy
def plot_Q_hit(usage_list, hit_list):
    # print (usage_list)
    # print (hit_list)
    for i ,x in enumerate(hit_list):
        hit_list[i] = hit_list[i]/ (i+1)
        # hit_list_old[i] = hit_list_old[i]/ (i+1)

    sample = np.arange(len(hit_list))
    # query=np.arange(len(usage_list))
    plt.subplot(2,1,1)
    act,=plt.plot(sample, hit_list, 'r-', label= 'active learning')
    # print(sample)
    print('accuracy for active',hit_list)
    # print('accuracy for semi',hit_list_old)
    # semi,=plt.plot(sample,hit_list_old, 'b-',label='semi-supervised learning')
    plt.legend([act])
    plt.xlabel('samples')
    plt.title('Accuracy gain for user 15')
    plt.ylabel('Accuracy')
    plt.subplot(2,1,2)
    index = np.arange(len(usage_list))  # (len(usage_list))
    for ind, e in reversed(list(enumerate(usage_list))):
        try:
            usage_list[ind] = usage_list[ind] - usage_list[ind - 1]
            hit_list[ind] = hit_list[ind] - hit_list[ind - 1]
        except:
            continue
    # print usage_list
    # print hit_list
    plt.plot(index, usage_list, 'r-')
    plt.ylim([0, 3])
    plt.title('query points',fontsize=12)
    plt.show()

def plot_hits(usage_list, hit_list):
    # print usage_list
    # print hit_list
    index = np.arange(300)#(len(usage_list))
    for ind, e in reversed(list(enumerate(usage_list))):
        try:
            usage_list[ind] = usage_list[ind] - usage_list[ind -1]
        except:
            continue
    # print usage_list
    # print hit_list
    plt.plot(index, usage_list[:300], 'r--')
    plt.ylim([0,10])
    plt.show(block= True)

#how many times in start how many times in end
def test_new(new_data, cluster_dict,clf,feature_list,label_list,y_train,unlabeled_indices,threshold):
    used = False
    used_count = 0
    hit = 0
    max_query=50
    usage_list = []
    hit_list = []
    learning_rate = 0.5
    print('start testing new data')
    dist=[]
    chunck_f=[]
    chunck_l=[]
    for ind, data in new_data.iterrows():
        start=time.time()
        res = []
        gaussian_list=[]
        true_label = data['activity']
        true_label = int(float(true_label))
        for clus_num in cluster_dict:
            c = cluster_dict[clus_num]
            # print('similarity checking')
            simalirty_measure = c.compare_point(data, clf)
            res.append((simalirty_measure, clus_num))
        # print(simalirty_measure)
        middle=time.time()
        max = 0
        sec_max=0
        tmp_activity = -1
        for r in res:
            sim = r[0]
            mean = (reduce(lambda x, y: x + y, sim[0])) / len(sim[0])
            if mean > max:
                max = mean
                tmp_activity = int(float(r[1]))
        # tmp_activity = int(float(tmp_activity))

        for r in res:
            sim = r[0]
            mean = (reduce(lambda x, y: x + y, sim[0])) / len(sim[0])
            if(mean > sec_max) & (int(float(r[1])) != tmp_activity) :
                sec_max = mean

        #TODO: confidence based , expected error based criterion
        ####################################
        point = Point.init_from_dict(data, -1)
        assigned_label = -1
        #TODO flexible threshold-> decrease over time, multiple criterion:
        if sec_max == 0 :
            sec_max=1
        if  (max/sec_max < threshold-1/(10*math.log1p(max_query))*(math.log1p(used_count))) | (max==0):
            print 1.1-0.0256*(math.log1p(used_count))
                # (1000+150/math.pow(hit+3,6)):
            print('ask for label')
            c = get_cluster(true_label, cluster_dict)
            #update the centers
            point.set_dist(c.center)
            print(point.dist)
            point.label = tmp_activity
            #TODO weight inverse to the center distance
            # point.weight = alpha*Reward+(1-alpha)*old_weight
            point.weight = 1
            c.add_point(point)
            #TODO: optimize
            c.center_update(point,learning_rate)
            #plotting
            used_count = used_count + 1
            hit = hit + 1
            chunck_f.append(point.features[:38])
            chunck_l.append(point.label)
            learning_rate = 0.5/used_count
            print(learning_rate)
            assigned_label = true_label
        else:
            assigned_label = tmp_activity
            label = tmp_activity
            c = get_cluster(label, cluster_dict)
            point.set_dist(c.center)
            # point.weight=confidence(point)
            point.label = label
            c.add_point(point)
            if true_label == tmp_activity:
                hit = hit + 1
        # c = get_cluster(true_label, cluster_dict)
        # point.set_dist(c.center)
        # dist.append(point.dist)
        usage_list.append(used_count)
        hit_list.append(hit)
        #batch mode
        if((used_count!=0) & (used_count % 5 ==0)):
            # print(y_predict)
            x, y_predict, y_train, unlabeled_indices=online_semi_supervised(feature_list, np.array(chunck_f), np.array(chunck_l), y_train, unlabeled_indices)
            # y_predict =np.concatenate((np.array(label_list),np.array(chunck_l)))
            print(y_predict)
            clf= train_base_classifier(x,y_predict)
        #############################################
        #TODO delete and update the model
        end=time.time()
        print('time:',end-start)
    # plot_Q_hit(usage_list, hit_list,dist)
    return usage_list,hit_list,cluster_dict,clf
#return a dictionary of clusters with with labels.
def init_classes(data):
    seen_labels = dict()
    for index, d in data.iterrows():

        label = str(d['activity'])
        newpoint = Point.init_from_dict(d, label)
        if label in seen_labels:
            cluster = seen_labels[label]
            newpoint.set_dist(cluster.center)
            seen_labels[label].add_point(newpoint)
        else:
            cluster = MyCluster(len(d), label) #TODO: how to define the legnth of the cluster centers?
            seen_labels[label] = cluster
            newpoint.set_dist(cluster.center)
            seen_labels[label].add_point(newpoint)
    # print(seen_labels)
    return seen_labels

if __name__ == '__main__':
    dataset = 'HAPT'
    filepath = "/Users/LilyWU/Documents/activity_recognition_for_sensor" \
               "/Data/HAPT_First_15_user.csv"

    if (not os.path.exists(filepath)):
        data = Loading(dataset)
        frequency = 50
        features_seperate = {}  # sperate feature for each user
        features_for_all = pd.DataFrame()
        users = data['User'].unique()
        activities = data['activity'].unique()  # list of all users
        for user in range(1, 16):
            for activity in activities:  # one user and one activity
                features = generate_features(data, user, activity)
                features_for_all = pd.concat([features_for_all, features])
        features_for_all.to_csv(filepath, header=features_for_all.columns.values.tolist())
    data = pd.DataFrame.from_csv(filepath, header=0)
    new_user, train_data = select(data, {'User': 15}, return_all= True)
    total_point = 300
    x, y_predict, y_train, unlabeled_indices= semi_supervised_learner(train_data,30,total_point)
    new = MyCluster(38,1)
    new.initial_set(x,y_predict, total_point)
    clf = train_base_classifier(x,y_predict)
    train_data = train_data.drop('User', axis=1)
    class_dict= init_classes(train_data)
    # print(class_dict)
    # train_data = train_data.drop('activity', axis=1)
    ####################
    #clf2 = train_base_classifier(current_feature,current_label)
    usage_list, hit_list_semi,cluster_dict,clf = test_new(new_user[:300], class_dict, clf,x,y_predict ,y_train,unlabeled_indices,1.1)
    plot_Q_hit(usage_list, hit_list_semi)

