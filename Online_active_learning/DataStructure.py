from __future__ import division
import heapq
import math
import time
from scipy.spatial.distance import euclidean
from data_process import *
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.semi_supervised import label_propagation

# class Semi_supervised_learner(object):
#     def __init__(self, df, labeled_points,total_points):
#         self.x=[]
#         self.y_true=[]
#         self.y_pred=[]
#         self.new_x=[]
#         self.new_y=[]
#         self.accuracy=[]
#         self.lp_model = label_propagation.LabelSpreading(gamma=0.25, kernel='knn', max_iter=300, n_neighbors=6)
#     @classmethod
#     def initial_set(self,df,labeled_points,total_points):
#         feature, label = seperate_feature_label(df)
#         indices = np.arange(len(feature))
#         label_indices = balanced_sample_maker(feature, label, labeled_points / len(label))
#         unlabeled_indices = np.delete(indices, np.array(label_indices))
#         rng = np.random.RandomState(0)
#         rng.shuffle(unlabeled_indices)
#         indices = np.concatenate((label_indices, unlabeled_indices[:total_points]))
#         n_total_samples = len(indices)
#         unlabeled_indices = np.arange(n_total_samples)[labeled_points:]
#         self.x = np.array(feature.iloc[indices])
#         self.y_true = np.array(label.iloc[indices])
#         y_train = np.copy(self.y_true)
#         y_train[unlabeled_indices] = -1
#         self.lp_model = label_propagation.LabelSpreading(gamma=0.25, kernel='knn', max_iter=300, n_neighbors=6)
#         self.lp_model.fit(self.x, y_train)
#         predicted_labels = self.lp_model.transduction_[unlabeled_indices]
#         self.y_pred = np.concatenate((self.y_true[:labeled_points], predicted_labels))
    # def online_propagate(self,new_x):
class Point(object):
    #feature is array
    def __init__(self, features, label):
        self.weight = 0.5
        self.label = label
        self.features = features
        self.len = len(features)
        self.dist=None
        self.certainty= True
    def set_dist(self,center_point):
        distance=euclidean(self.features,center_point.features)
        self.dist = distance
    @classmethod
    #array like
    def init_from_dict(self, features, label):
        point = Point(features, label)
        return point
    def set_weight_label(self,weight,label):
        self.weight=weight
        self.label=label
class Cluster(object):
    def __init__(self, label):
        self.weight = 0
        self.len= 0
        self.center = Point([0]*38,label)
        self.min_heap_list = []
        self.max_heap_list = []
        self.label = label
        self.count=0
        self.radius=[0]*38
        self.centers = heapq.nsmallest(20, self.min_heap_list)
        self.boudaries= heapq.nsmallest(20, self.max_heap_list)
    def get_center(self):
        return self.center
    def get_boundary_distance(self):
        list=[]
        for point in self.boudaries:
          point = heapq.heappop(self.boudaries)
          list.append(point.dist)
        return list
    def add_point(self,point):
        self.len= point.len
        self.count+=1
        heapq.heappush(self.min_heap_list,(point.dist,point))
        point2 = Point(point.features, point.label)
        point2.dist = -point.dist
        heapq.heappush(self.max_heap_list,(point2.dist,point2))
        for ind, f in enumerate(point.features):
            self.center.features[ind] = (self.center.features[ind]* self.weight + f) / (self.weight + 1)
            # self.radius[ind] = math.sqrt(
            #     (math.pow(self.radius[ind], 2) * self.weight + math.pow((point.features[ind] - self.center[ind]), 2)) / (
            #     self.weight + 1))
        self.weight= self.weight+ 1
    # Todo: test wether it works
    def update(self,newpoint):
        old=self.center.features
        self.add_point(newpoint)
        # for point in self.min_heap_list:
        print(self.min_heap_list)
        for dist,point in self.min_heap_list:
            print('update')
            point.set_dist(self.center)
        print(self.min_heap_list)
        return
    def similarity_check(self,old,new,rf):
        trees = rf.estimators_
        count = 0
        for tree in trees:
            new_res = tree.predict(new.features)
            baseline_res = tree.predict(old.features)
            count = count + 1 if baseline_res == new_res else count
        similarity = count / len(trees)
        # TODO: how to weight?
        return similarity * new.dist
    #feature is array
    def compare_center(self,new_point,clf):
        self.centers = heapq.nsmallest(20, self.min_heap_list)
        new_point.set_dist(self.center)
        sim_center = []
        sim_boundry = []
        self.boudaries = heapq.nsmallest(20, self.max_heap_list)
        for dist,small in self.centers:
            sim_center.append(self.similarity_check(small, new_point, clf))
        for dist,large in self.boudaries:
            sim_boundry.append(self.similarity_check(large, new_point, clf))
        return (sim_center, sim_boundry)

class Active_learned_Model(object):
    def __init__(self, x, y_all,max_query, rf):
        self.clusters=dict()
        self.count=0
        self.rf= rf
        self.x= x
        self.y= y_all
        self.winning_cluster=None
        self.max_query=max_query
    def update_rf(self,new_data,new_label):
        self.x.append(new_data)
        self.y.append(new_label)
        rf= RandomForestClassifier(n_estimators=30)
        rf.fit(self.x, self.y)

    def init_clusters(self,x, y_all):
        seen_labels = self.clusters
        for i in range(len(x)):
            label = y_all[i]
            # print(label)
            newpoint = Point.init_from_dict(x[i], label)
            if label in seen_labels:
                cluster = seen_labels[label]
                #compute the similarity with center
                newpoint.set_dist(cluster.center)
                seen_labels[label].add_point(newpoint)
            else:
                #initialize a new cluster with empty center, label= list
                cluster = Cluster(label)
                seen_labels[label] = cluster
                newpoint.set_dist(cluster.center)
                seen_labels[label].add_point(newpoint)
        self.clusters=seen_labels
        # print(self.clusters)
        return seen_labels

    # def query_by_similarity(self, used_count,max_query,new):
    def query_by_similarity(self, new_x,new_y,threshold):
        tmp_label=-1
        new_point=Point.init_from_dict(new_x,tmp_label)
        res=[]
        str=time.time()
        for label in self.clusters.keys():
            c=self.clusters[label]
            sim = c.compare_center(new_point,self.rf)
            res.append((np.mean(sim),label))
        tmp=time.time()
        # print('time:',tmp-str)
        max=sec_max=0
        for r in res:
            sim = r[0]
            if sim > max:
                max = sim
                tmp_label= r[1]
        for r in res:
            sim = r[0]
            if (sim > sec_max) & (r[1] != tmp_label):
                tmp_label_sec=r[1]
                sec_max = sim
        # print('true, pred:', new_y, tmp_label, tmp_label_sec)
        if(sec_max ==0): sec_max=1
        if (max / sec_max < threshold - (threshold-1) / ( math.log1p(self.max_query)) * (math.log1p(self.count))) | (max == 0):
            # print('query with', max, max/sec_max, max-sec_max)
            self.count+= 1
            new_point.label= new_y
            return True, new_point
        else:
            if(new_y!= tmp_label):
                #print('Wrong',max, max/sec_max, max-sec_max)
                new_point.label=tmp_label
                return False, new_point
    def update(self,buffer):
        for point in buffer:
            c =self.clusters[point.label]
            c.update(point)
            # print(point.features.shape)
            np.concatenate((self.x, np.array([point.features])))
            np.concatenate((self.y, np.array([point.label])))
        clf = RandomForestClassifier(n_estimators=30)
        clf.fit(self.x,self.y)
        self.rf=clf

if __name__ == '__main__':
   pass