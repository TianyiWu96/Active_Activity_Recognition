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


# Datasets_description=
# {
#      'HAPT':   f=50HZ  activity_number: 6  Users: 30
#      'PAMAP2': f=100HZ activity_number: 24 Users:  9

# }
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
#
# def test(test_data, cluster_dict, clf):
#     for ind, data in test_data.iterrows():
#         res = []
#         hit  = 0`
#         Q_need = 0
#         for clus_num in cluster_dict:
#             c = cluster_dict[clus_num]
#             simalirty_measure = c.compare_point(data, clf)
#             res.append((simalirty_measure, clus_num))
#
#         mean = reduce(lambda x, y: x + y, res[0]) / len(res[0])
#         point = Point.init_from_dict(data, -1)
#         label = new_user['activity']
#         if res[1] == label:
#             hit = hit + 1
#         if mean < 0.4:
#             Q_need = Q_need + 1
#     return hit, Q_need

# Done: fixed major error when plotting accuracy
def plot_Q_hit(usage_list, hit_list):
    print (usage_list)
    print (hit_list)
    for i ,x in enumerate(hit_list):
        hit_list[i] = hit_list[i]/ (i+1)
    sample=np.arange(len(hit_list))
    plt.subplot(2,1,1)
    plt.plot(sample,hit_list, 'r-')
    plt.xlabel('total points')
    plt.title('Accuracy gain for User 17')
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
    plt.plot(index, usage_list, 'r-', index, hit_list[:len(usage_list)], 'bs')
    plt.ylim([0, 5])
    plt.title('query points',fontsize=12)
    plt.show(block= True)


def plot_hits(usage_list, hit_list):
    # print usage_list
    # print hit_list
    index = np.arange(300)#(len(usage_list))
    for ind, e in reversed(list(enumerate(usage_list))):
        try:
            usage_list[ind] = usage_list[ind] - usage_list[ind -1]
            hit_list[ind] = hit_list[ind] - hit_list[ind -1]
        except:
            continue
    # print usage_list
    # print hit_list
    plt.plot(index, usage_list[:300], 'r--', index, hit_list[:300], 'bs')
    plt.ylim([0,10])
    plt.show(block= True)

#how many times in start how many times in end
def test_new(new_data, cluster_dict,clf,current_set,current_label):
    used = False
    used_count = 0
    hit = 0
    usage_list = []
    hit_list = []
    learning_rate=0.5
    print('start testing new data')
    for ind, data in new_data.iterrows():

        start=time.time()
        res = []
        gaussian_list=[]
        true_label = data['activity']
        true_label = int(float(true_label))
        for clus_num in cluster_dict:
            c = cluster_dict[clus_num]
            #TODO check for membership agreement vs. similarity measurement
            simalirty_measure = c.compare_point(data, clf)
            # gaussian=c.Gaussian_membership(data)
            # gaussian_list.append(gaussian)
            res.append((simalirty_measure, clus_num))
        # print(res)
        # criteria########################
        # print(gaussian_list,true_label)
        middle=time.time()
        max = 0
        tmp_activity = -1
        #this is not a very good strategy for comparison
        for r in res:
            sim = r[0]
            # print(sim)
            #regulization
            mean = (reduce(lambda x, y: x + y, sim[0])) / len(sim[0])
            # print(mean)
            if mean > max:
                max = mean
                tmp_activity = int(float(r[1]))
        tmp_activity = int(float(tmp_activity))
        #TODO: confidence based , expected error based criterion
        ####################################
        point = Point.init_from_dict(data, -1)
        assigned_label = -1
        #TODO flexible threshold-> decrease over time, multiple criterion:

        if max < 0.3:
            print('ask for label')
            c = get_cluster(true_label, cluster_dict)
            #update the centers
            point.set_dist(c.center)
            print(point.dist)
            point.label = tmp_activity
            #TODO weight inverse to the center distance
            # point.weight = alpha*Reward+(1-alpha)*old_weight
            point.weight=1
            c.add_point(point)
            #TODO: optimize
            c.center_update(point,learning_rate)
            #plotting
            used = True
            used_count = used_count + 1
            hit = hit + 1
            learning_rate=0.5/used_count
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

        usage_list.append(used_count)
        hit_list.append(hit)
        #############################################
        #TODO delete and update the model
        current_set.append(point.features[:-2])
        current_label.append(assigned_label)
        end=time.time()
        print('time:',end-start)
        # if ind % 200 == 0 and ind >= 200 :
        #      clf = train_base_classifier(current_set, current_label)

    plot_Q_hit(usage_list, hit_list)
    return cluster_dict
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
    new_user = pd.DataFrame.from_csv("/Users/LilyWU/Documents/activity_recognition_for_sensor/"\
                                      "Data/HAPT_user_17.csv", header=0)
    #TODO: initialization with semi-learning ,test on it
    new= MyCluster(38,1)
    # randomlized
    new.initial_set(train_data, 300)
    x,y = seperate_feature_label(train_data)
    # train a classifier for similarity check
    clf = train_base_classifier(x,y)
    #ignore the users
    train_data = train_data.drop('User', axis=1)
    print('start')
    #clustering with labels
    #TODO: A similar function for update
    class_dict= init_classes(train_data)
    train_data = train_data.drop('activity', axis=1)
    ####################
    current_feature = x.values.tolist()
    current_label = y.values.tolist()
    #clf2 = train_base_classifier(current_feature,current_label)
    test_new(new_user, class_dict, clf,current_feature,current_label)
    #TODO:
