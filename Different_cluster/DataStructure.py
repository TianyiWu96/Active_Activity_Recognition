from __future__ import division
import heapq
import math
from initial_training import *
from Baseline_test import *
class Point(object):
    def __init__(self, features, label):
        self.weight = 0.5
        self.label = label
        self.features = features
        self.dist = None

#Euclidian distance
    def set_dist(self,center):
        #distance = math.sqrt(math.pow((X-self.x),2) + math.pow((Y - self.y), 2) + math.pow(Z-self.z), 2)
        dis = 0
        for ind, f in enumerate(center):
            dis = math.pow(f-self.features[ind], 2) + dis
        distance = math.sqrt(dis)
        self.dist = distance

    @classmethod
    def init_from_dict(self, point_dict, label):
        #new_point = Point(point_dict['x'], point_dict['y'], point_dict['z'])
        try:
            point_dict['activity'] = 0
            point_dict['User'] = 0
        except:
            pass
        features = []
        for f in point_dict:
            features.append(f)
        point = Point(features, label)
        return point

    def __cmp__(self, other):
        return cmp(self.dist, other.dist)
# construct the similarity matrix by supervised labels
class MyCluster:
    def __init__(self, len, label):
        self.count = 0
        self.center = [0] * len
        self.min_heap_list = []
        self.max_heap_list = []
        self.label = label
        self.centers = heapq.nsmallest(10, self.min_heap_list)
        self.boudaries = heapq.nsmallest(10, self.max_heap_list)
        self.radius=[0] * len
    def add_point(self, point):
        heapq.heappush(self.min_heap_list, point)
        point2 = Point(point.features, point.label)
        point2.dist = -point.dist
        heapq.heappush(self.max_heap_list, point2)
        new_axis = point.features
        #print type (new_axis)
        for ind, f in enumerate(new_axis):
            self.center[ind] = (self.center[ind] * self.count + f) / (self.count + 1)
            self.radius[ind] = math.sqrt((math.pow(self.radius[ind],2)*self.count + math.pow((point.features[ind]-self.center[ind]),2))/(self.count + 1))

        self.count = self.count + 1
    def center_update(self,newpoint,learning_rate):
        print('updated')
        # print(self.centers)
        for i in range(len(newpoint.features)):
            self.center[i]=self.center[i]+learning_rate*(newpoint.features[i]-self.center[i])
        for point in self.centers:
            # print(point.dist)
            point=heapq.heappop(self.min_heap_list)
            point.set_dist(self.center)
            # print(point.dist)
            heapq.heappush(self.min_heap_list, point)
        for point in self.boudaries:
            point = heapq.heappop(self.max_heap_list)
            point.set_dist(self.center)
            heapq.heappush(self.max_heap_list, point)
        # print(self.centers)
        return

    def initial_set(self, point_list, number):
        indices = np.arange(len(point_list))
        np.random.shuffle(indices)
        for i in range(number):
            feature= point_list.iloc[indices[i]][:37]
            # print(feature)
            label =point_list.iloc[indices[i]][39]
            new_point= Point(feature,label)
            new_point.set_dist(feature)
            self.add_point(new_point)
        return
    def similarity_check(self, old_point, new_point,rf):
        trees = rf.estimators_
        count = 0
        for tree in trees:
            new_res = tree.predict(new_point.features[:-2])
            baseline_res = tree.predict(old_point.features[:-2])
            count = count + 1 if baseline_res == new_res else count
        similarity = count/ len(trees)
        return old_point.weight * similarity
    def Gaussian_membership(self,point_dict):
        new_point = Point.init_from_dict(point_dict, point_dict['activity'])
        new_point.set_dist(self.center)
        sum=0
        list=[]
        for i in range(len(new_point.features)):
            sum+= math.pow((new_point.features[i]-self.center[i]),2)
            list.append(math.pow((new_point.features[i]-self.center[i]),2))
        #TODO:whats wrong?
        sum=sum/np.array(list).var()
        sum= math.exp(-1/2*sum)
        # print('compare gauss with dist',sum, new_point.dist,new_point.label)
        return sum
    def compare_point(self, point_dict,clf):
        new_point = Point.init_from_dict(point_dict, point_dict['activity'])
        # smallest = heapq.nsmallest(10, self.min_heap_list)
        # largest = heapq.nsmallest(10, self.max_heap_list)
        sim_center = []
        sim_boundry = []
        self.centers = heapq.nsmallest(10, self.min_heap_list)
        self.boudaries = heapq.nsmallest(10, self.max_heap_list)
        for small in self.centers:
            sim_center.append(self.similarity_check(small, new_point,clf))
        for large in self.boudaries:
            sim_boundry.append(self.similarity_check(large, new_point,clf))
        return (sim_center,sim_boundry)

    def add_labeled(self, point_dict):
        added_point = self.add_point(point_dict)
        added_point.weight = 1

if __name__ == '__main__':
    pass