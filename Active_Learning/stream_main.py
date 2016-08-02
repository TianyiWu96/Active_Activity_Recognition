import matplotlib.pyplot as plt
from load_PAMAP2 import Loading_PAMAP2
#from load_HAPT import Loading_HAPT
from feature_generate import *
from similarity_check import *
from initial_training import *
import os.path
from DataStructure import *
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
#         hit  = 0
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


def plot_hits(usage_list, hit_list):
    print usage_list
    print hit_list
    index = np.arange(300)#(len(usage_list))
    for ind, e in reversed(list(enumerate(usage_list))):
        try:
            usage_list[ind] = usage_list[ind] - usage_list[ind -1]
            hit_list[ind] = hit_list[ind] - hit_list[ind -1]
        except:
            continue
    print usage_list
    print hit_list
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
    for ind, data in new_data.iterrows():
        res = []
        true_label = data['activity']
        true_label = int(float(true_label))
        for clus_num in cluster_dict:
            c = cluster_dict[clus_num]
            simalirty_measure = c.compare_point(data, clf)
            res.append((simalirty_measure, clus_num))
        # criteria########################
        max = 0
        tmp_activity = -1
        for r in res:
            sim = r[0]
            mean = (reduce(lambda x, y: x + y, sim[0])) / len(sim[0])
            if mean > max:
                max = mean
                tmp_activity = int(float(r[1]))
        tmp_activity = int(float(tmp_activity))
        ####################################
        point = Point.init_from_dict(data, -1)
        assigned_label = -1
        if max < 0.35:
            c = get_cluster(true_label, cluster_dict)
            point.set_dist(c.center)
            point.label = tmp_activity
            point.weight = 1
            c.add_point(point)
            used = True
            used_count = used_count + 1
            hit = hit + 1
            assigned_label = true_label
        else:
            assigned_label = tmp_activity
            label = tmp_activity
            c = get_cluster(label, cluster_dict)
            point.set_dist(c.center)
            point.label = label
            c.add_point(point)
            if true_label == tmp_activity:
                hit = hit + 1

        usage_list.append(used_count)
        hit_list.append(hit)
        #############################################
        current_set.append(point.features[:-2])
        current_label.append(assigned_label)
        # if ind % 200 == 0 and ind >= 200 :
        #      clf = train_base_classifier(current_set, current_label)

    plot_hits(usage_list, hit_list)
    return cluster_dict

def init_classes(data):
    seen_labels = dict()
    for index, d in data.iterrows():
        label = str (d['activity'])
        point = Point.init_from_dict(d, label)
        if label in seen_labels:
            c = seen_labels[label]
            point.set_dist(c.center)
            seen_labels[label].add_point(point)
        else:
            c = MyCluster(40, label) #TODO be dynamic
            seen_labels[label] = c
            point.set_dist(c.center)
            seen_labels[label].add_point(point)
    return seen_labels
if __name__ == '__main__':
    dataset = 'HAPT'
    filepath = "/Users/ana/Documents/ER lab repo/Active Learning" \
               "/activity_recognition_for_sensor/First_15_user_HAPT.csv"
    testpath = "/Users/ana/Documents/ER lab repo/Active Learning" \
               "/activity_recognition_for_sensor/test_user.csv"
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
    new_user = pd.DataFrame.from_csv("/Users/ana/Documents/ER lab repo/"
                                     "Active Learning/activity_recognition_for_sensor/"
                                     "PAMAP2_Dataset/HAPT_user_20.csv", header=0)
    x,y = inital_set(train_data, 300)
    clf = train_base_classifier(x,y)
    train_data = train_data.drop('User', axis=1)
    class_dict= init_classes(train_data)
    train_data = train_data.drop('activity', axis=1)
    ####################
    print new_user.shape
    current_feature = x.values.tolist()
    current_label = y.values.tolist()
    #clf2 = train_base_classifier(current_feature,current_label)
    ####################
    test_new(new_user, class_dict, clf,current_feature,current_label)


#################################################################################
# X, y_pred, y_true = semi_supervised_learner(data, 300, 3000)
# #newx,newy, = split(data,100)

# if (not os.path.exists(testpath)):
#     data = Loading(dataset)
#     activities = data['activity'].unique()
#     new_user = 16
#     for activity in activities:
#         new_user_feature = generate_features(data, new_user, activity)
#     new_user_feature.to_csv(testpath, header=new_user_feature.columns.values.tolist())

