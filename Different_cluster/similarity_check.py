#count resemblence of new instance to each data in filename
from sklearn.ensemble import RandomForestClassifier
import json
import numpy as np
from sklearn.cross_validation import cross_val_score


def train_base_classifier(features, labels):
    clf = RandomForestClassifier(n_estimators=30)
    # print(features)
    clf.fit(features, labels)
    # classification_results = cross_val_score(clf, features, labels, cv=10)
    # print('start')
    # print classification_results
    return clf

#TODO chekc new node with which nodes?
#TODO how to update?
def add_one(clf, test_features, test_labels, points, labels):
    #print points
    for index, row in test_features.iterrows():
        print index
        count = check_similarity(clf,row, points, labels)
        print count

        print " --------"
        if index> 10 :
            break

def check_similarity(rf, data_point, base_list, labels):
    #print labels
    #print base_list
    print data_point
    trees = rf.estimators_
    res = []
    xxx= []
    activities = labels.tolist()
    for i, base in base_list.iterrows():
        print base['activity']
        count = 0
        for tree in trees:
            new_res = tree.predict(data_point)
            baseline_res = tree.predict(base)
            count = count + 1 if baseline_res == new_res else count
        # if i > 50:
        #     break
        res.append((count, base['activity']))
    return res


def tmp_prepare():
    with open(
            '/Users/ana/Documents/ER lab repo/Active Learning/activity_recognition_for_sensor/Active_Learning/may5.json')as data_file:
        data = json.load(data_file)
    ##create feature array
    sample_set = []
    labels = []
    for sample in data:
        features = []
        heart = sample['heart']
        dust = sample['dust']
        pef = sample['pef']
        fev1 = sample['fev1']
        energy = sample['energy']
        features.append(energy)
        features.append(fev1)
        features.append(pef)
        features.append(dust)
        features.append(heart)
        sample_set.append(features)
        score = sample['score']
        label = 0
        if score < 16:
            label = 0
        elif score < 20 and score > 15:
            label = 1
        elif score >= 20 and score < 26:
            label = 2
        labels.append(label)

    f = np.array(sample_set)
    l = np.array(labels)
    return (f, l)


if __name__ == '__main__':
    clf = RandomForestClassifier(n_estimators=20)
    (features, labels) = tmp_prepare()
    clf = clf.fit(features, labels)
    scores = cross_val_score(clf, features, labels, cv=10)
    #print(scores)
    count = check_similarity(clf, features[0], features[1:])
    #print count
