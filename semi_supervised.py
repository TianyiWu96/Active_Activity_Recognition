import numpy as np
from sklearn import *
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import classification_report, confusion_matrix
from Baseline_test import *
import matplotlib.pyplot as plt
from scipy.interpolate import spline


def seperate_feature_label(df):
    labels = df['activity']
    features = df.drop('activity', axis=1)
    features = features.drop('User', axis=1)
    return features,labels



def balanced_sample_maker(X, y, sample_size, random_seed=None):
    uniq_levels = y.unique()
    uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_seed is None:
        np.random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx
    balanced_copy_idx = []
    for gb_level, gb_idx in groupby_levels.items():
        over_sample_idx = np.random.choice(gb_idx, size=sample_size, replace=True).tolist()
        balanced_copy_idx += over_sample_idx
        # print(balanced_copy_idx)
    np.random.shuffle(balanced_copy_idx)
    return balanced_copy_idx


def semi_supervised_learner(df,labeled_points,total_points):
            #split the data according to the classes, generate labelled , unlabelled mark for each and reshuffle.
    
            feature, label = seperate_feature_label(df)
            indices = np.arange(len(feature))
            label_indices = balanced_sample_maker(feature,label,labeled_points/len(label))
            unlabeled_indices = np.delete(indices,np.array(label_indices))
            rng = np.random.RandomState(0)
            rng.shuffle(unlabeled_indices)
            indices = np.concatenate((label_indices,unlabeled_indices[:total_points]))
            n_total_samples = len(indices)
            unlabeled_indices = np.arange(n_total_samples)[labeled_points:]
            X = feature.iloc[indices]
            y = label.iloc[indices]
            y_train = np.copy(y)
            y_train[unlabeled_indices] = -1
            true_labels = y.iloc[unlabeled_indices]
            lp_model = label_propagation.LabelSpreading(gamma=0.25, kernel='knn',max_iter=300,n_neighbors=6)
            lp_model.fit(X, y_train)
            predicted_labels = lp_model.transduction_[unlabeled_indices]
            y_all = np.concatenate((y.iloc[:labeled_points],predicted_labels))
            print(accuracy_score(true_labels, predicted_labels))
            return X, y_all, y


def semi_supervised_test1(df, labeled_points, total_points):
    #split the data according to the classes, generate labelled , unlabelled mark for each and reshuffle.
    
        feature, label = seperate_feature_label(df)
        accuracy_for_supervise =[]
        accuracy_for_semi_supervise =[]
        x = []
        indices = np.arange(len(feature))
        label_indices = balanced_sample_maker(feature, label, labeled_points/len(label))
        unlabeled_indices = np.delete(indices, np.array(label_indices))
        rng = np.random.RandomState(0)
        rng.shuffle(unlabeled_indices)
        indices = np.concatenate((label_indices, unlabeled_indices[:total_points]))
        n_total_samples = len(indices)
        for i in range(10):
            unlabeled_indices=np.arange(n_total_samples)[labeled_points:]
            X = feature.iloc[indices]
            y = label.iloc[indices]
            y_train = np.copy(y)
            y_train[unlabeled_indices] = -1
            classifier = KNeighborsClassifier(n_neighbors=6)
            classifier.fit(X.iloc[:labeled_points], y.iloc[:labeled_points])

            y_pred = classifier.predict(X.iloc[labeled_points:])
            # y_all = pd.concat(y.iloc[:labeled_points], y_pred)
            true_labels = y.iloc[unlabeled_indices]
            # print(confusion_matrix(true_labels,y_pred))
            print("%d labeled & %d unlabeled (%d total)"
                  % (labeled_points, n_total_samples - labeled_points, n_total_samples))
            accuracy_for_supervise.append(accuracy_score(true_labels, y_pred))
            
            lp_model = label_propagation.LabelSpreading(gamma=0.25, kernel='knn', max_iter=300, n_neighbors=6)
            lp_model.fit(X, y_train)
            predicted_labels = lp_model.transduction_[unlabeled_indices]
            # print('Iteration %i %s' % (i, 70 * '_'))
            accuracy_for_semi_supervise.append(accuracy_score(true_labels, predicted_labels))
            x.append(labeled_points)
            # print(confusion_matrix(true_labels, predicted_labels))
            print('Semi-supervised learning:', accuracy_score(true_labels, predicted_labels))
            labeled_points += 50
        x_sm = np.array(x)
        y_sm = np.array(accuracy_for_supervise)
        y1_sm = np.array(accuracy_for_semi_supervise)
        x_smooth = np.linspace(x_sm.min(), x_sm.max(), 200)
        y_smooth = spline(x, y_sm, x_smooth)
        y1_smooth = spline(x, y1_sm, x_smooth)
        plt.plot(x_smooth, y_smooth)
        plt.plot(x_smooth, y1_smooth)
        plt.xlabel('Label numbers')
        plt.ylabel('Accuracy')
        plt.title('Semi-supervised learning for total '+str(n_total_samples)+' samples ')
        plt.show()
        

def semi_supervised_test2(df , labeled_points, step):
        total_points = labeled_points+50
        feature ,label = seperate_feature_label(df)
        accuracy_for_supervise=[]
        accuracy_for_semi_supervise=[]
        x=[]
        indices = np.arange(len(feature))
        label_indices = balanced_sample_maker(feature,label,labeled_points/len(label))
        unlabeled_indices=np.delete(indices,np.array(label_indices))
        rng = np.random.RandomState(0)
        rng.shuffle(unlabeled_indices)
        indices = np.concatenate((label_indices,unlabeled_indices[:total_points]))
        
        for i in range(int((len(df.index)-labeled_points-50)/step)):
            x.append(total_points)
            unlabeled_index = np.arange(total_points)[labeled_points:]
            X = feature.iloc[indices]
            y = label.iloc[indices]
            y_train = np.copy(y)
            y_train[unlabeled_index] = -1
            classifier = KNeighborsClassifier(n_neighbors=6)
            classifier.fit(X.iloc[:labeled_points],y.iloc[:labeled_points])
            y_pred = classifier.predict(X.iloc[labeled_points:])
            true_labels = y.iloc[unlabeled_index]
            lp_model = label_propagation.LabelSpreading(gamma=0.25, kernel='knn',max_iter=300, n_neighbors=6)
            lp_model.fit(X, y_train)
            predicted_labels = lp_model.transduction_[unlabeled_index]
            accuracy_for_semi_supervise.append(accuracy_score(true_labels, predicted_labels))
            total_points += step
            indices = np.concatenate((indices,unlabeled_indices[(total_points-step):total_points]))
        print('accuracy gain average with %d labels:%d'%(labeled_points, np.average(accuracy_for_semi_supervise)-np.average(accuracy_for_semi_supervise)))
        x_sm = np.array(x)
        y_sm = np.array(accuracy_for_supervise)
        y1_sm=np.array(accuracy_for_semi_supervise)
        x_smooth = np.linspace(x_sm.min(), x_sm.max(), 200)
        y_smooth = spline(x, y_sm, x_smooth)
        y1_smooth = spline(x, y1_sm, x_smooth)
        sup, = plt.plot(x_smooth,y_smooth,'r-',label='kNN')
        semi, = plt.plot(x_smooth,y1_smooth,'b-',label='sem-supervised')
        # plt.legend(handles=[sup, semi],frameon=False,loc=4)
        plt.ylim(0.4, 1)  
        plt.xlabel('Total samples',fontsize=14)
        plt.xticks(fontsize=12)  
        plt.yticks(fontsize=12)  
        plt.ylabel('Accuracy',fontsize=14)
        plt.title('Label Propagation with '+str(labeled_points)+'labels')
        return accuracy_score(true_labels, predicted_labels)
