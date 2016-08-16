from __future__ import division
import heapq
import math
class Point(object):
    def __init__(self, features, label):
        self.weight = 0.5
        self.label = label
        self.features = features
        self.dist = None

    def set_dist(self,features):
        #distance = math.sqrt(math.pow((X-self.x),2) + math.pow((Y - self.y), 2) + math.pow(Z-self.z), 2)
        dis = 0
        for ind, f in enumerate(features):
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

class MyCluster:
    def __init__(self, len, label):
        self.count = 0
        self.center = [0] * len
        self.min_heap_list = []
        self.max_heap_list = []
        self.label = label

    def add_point(self, point):
        heapq.heappush(self.min_heap_list, point)
        point2 = Point(point.features, point.label)
        point2.dist = -point.dist
        heapq.heappush(self.max_heap_list, point2)
        new_axis = point.features
        #print type (new_axis)
        for ind, f in enumerate(new_axis):
            self.center[ind] = (self.center[ind] * self.count + f) / (self.count + 1)
        self.count = self.count + 1

    def initial_set(self, point_list):
        for point_dict in point_list:
            new_point = Point(point_dict['x'], point_dict['y'], point_dict['z'])
            new_point.set_dist(self.center[0], self.center[1], self.center[2])
            self.add_point(new_point)

    def similarity_check(self, old_point, new_point, rf):
        trees = rf.estimators_
        count = 0
        for tree in trees:
            new_res = tree.predict(new_point.features[:-2])
            baseline_res = tree.predict(old_point.features[:-2])
            count = count + 1 if baseline_res == new_res else count

        similarity = count/ len(trees)
        return old_point.weight * similarity

    def compare_point(self, point_dict,clf):
        new_point = Point.init_from_dict(point_dict, point_dict['activity'])
        smallest = heapq.nsmallest(10, self.min_heap_list)
        largest = heapq.nsmallest(10, self.max_heap_list)
        sim_center = []
        sim_boundry = []
        for small in smallest:
            sim_center.append(self.similarity_check(small, new_point,clf))
        for large in largest:
            sim_boundry.append(self.similarity_check(large, new_point,clf))
        return (sim_center ,sim_boundry)


    def add_labeled(self, point_dict):
        added_point = self.add_point(point_dict)
        added_point.weight = 1

if __name__ == '__main__':
    pass