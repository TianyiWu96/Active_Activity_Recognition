from sklearn.semi_supervised import label_propagation
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from sklearn import *
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.cross_validation import *
from sklearn import preprocessing
import pandas as pd
import numpy as np
import numpy as np
from sklearn.semi_supervised import label_propagation
from scipy.interpolate import spline
import pandas as pd
import os,glob
from pandas import *
from math import *
from scipy.fftpack import fft
from numpy import mean, sqrt, square
from sklearn import preprocessing

import os.path
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt


def write_to(filepath, dataset):
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
    return features_for_all

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
    if (dataset == "HAPT"):
        new = pd.DataFrame.from_dict(Loading_HAPT(HAPT_folder, data))
        return new


def Loading_PAMAP2(filepath, id, table):
    with open(filepath, 'r') as f:
        for line in f:
            # print('New Line')
            tokens = line.split()
            # print(tokens)
            assert len(tokens) == 54
            # Skip the line if no current session and cannot adopt activity from row
            if not tokens[1].isdigit() or tokens[1] == "0" or tokens[6] == "NaN" or tokens[4] == "NaN" or \
                            tokens[5] == "NaN":
                continue
            table['timestamp'].append(float(tokens[0]))
            table['x'].append(float(tokens[4]))
            table['y'].append(float(tokens[5]))
            table['z'].append(float(tokens[6]))
            table['activity'].append(int(tokens[1]))
            table['User'].append(id)

    return table


def Loading_HAPT(foldername, data):
    labelfile = foldername + 'labels.txt'
    with open(labelfile) as f:
        for line in f:
            tokens = line.split()
            experiment_id = tokens[0]
            user_id = tokens[1]
            activity = int(tokens[2])
            start = int(tokens[3]) - 1
            end = int(tokens[4]) - 1
            if (int(experiment_id) < 10):
                experiment_id = '0' + experiment_id
            # print(experiment_id)
            if (int(user_id) < 10):
                user_id = '0' + user_id
            if (activity == 1 or activity == 2 or activity == 3 or activity == 4 or activity == 5 or activity == 6):
                with open(foldername + 'acc_exp' + experiment_id + '_user' + user_id + '.txt') as m:
                    lines = m.readlines()
                    for i in range(start, end + 1):
                        raw = lines[i].split()
                        # print(data)
                        data['x'].append(float(raw[0]))
                        data['y'].append(float(raw[1]))
                        data['z'].append(float(raw[2]))
                        data['User'].append(int(user_id))
                        data['activity'].append(int(activity))
                        data['timestamp'].append(i + 1)
    return data


# return any specified column or one column and rest of it
def select(data, key_value_pairs, return_all=False):
    for key in key_value_pairs:
            select = data[key] == key_value_pairs[key]
            if (return_all == False):
                return data[select]
            else:
                other = data[select == False]
                return data[select], other


def generate_features(data, user, activity,frequency ):
    select_user = select(data, {'User': user})
    select_activity = select(select_user, {'activity': activity})
    # print(select_activity)
    features = sliding_window(select_activity, 2 * frequency, 0.5)
    # print(features)
    return features


def seperate_feature_label(df):
    labels = df['activity']
    features = df.drop('activity', axis=1)
    features = features.drop('User', axis=1)
    # print(features)
    return features, labels


def balanced_sample_maker(X, y, sample_size, random_seed=None):
    uniq_levels = y.unique()
    # uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_seed is None:
        np.random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx
    # oversampling on observations of each label
    balanced_copy_idx = []
    for gb_level, gb_idx in groupby_levels.items():
        over_sample_idx = np.random.choice(gb_idx, size=(sample_size / len(uniq_levels)), replace=True).tolist()
        balanced_copy_idx += over_sample_idx
        # print(balanced_copy_idx)
    np.random.shuffle(balanced_copy_idx)
    return balanced_copy_idx


def inital_set(df, labeled_points):
    feature, label = seperate_feature_label(df)
    indices = np.arange(len(feature))
    label_indices = balanced_sample_maker(feature, label, labeled_points)
    X = feature.iloc[label_indices]
    y = label.iloc[label_indices]
    return X, y


def semi_supervised_learner(df, labeled_points, total_points):
    # split the data according to the classes, generate labelled , unlabelled mark for each and reshuffle.
    feature, label = seperate_feature_label(df)
    indices = np.arange(len(feature))
    label_indices = balanced_sample_maker(feature, label, labeled_points / len(label))
    unlabeled_indices = np.delete(indices, np.array(label_indices))
    rng = np.random.RandomState(0)
    rng.shuffle(unlabeled_indices)
    indices = np.concatenate((label_indices, unlabeled_indices[:total_points]))
    n_total_samples = len(indices)
    unlabeled_indices = np.arange(n_total_samples)[labeled_points:]
    X = np.array(feature.iloc[indices])
    y = label.iloc[indices]
    y_train = np.copy(y)
    y_train[unlabeled_indices] = -1
    # print(y_train)
    true_labels = y.iloc[unlabeled_indices]
    lp_model = label_propagation.LabelSpreading(gamma=0.25, kernel='knn', max_iter=300, n_neighbors=6)
    lp_model.fit(X, y_train)
    predicted_labels = lp_model.transduction_[unlabeled_indices]
    y_all = np.concatenate((y.iloc[:labeled_points], predicted_labels))
    print(accuracy_score(true_labels, predicted_labels))
    return X, y_all, unlabeled_indices


def online_semi_supervised(x_old, x_new, y_new, y_train, unlabeled_indices):
    x_old = np.array(x_old)
    y_train = np.array(y_train)
    print(y_train)
    for i in range(len(x_new)):
        np.insert(x_old, x_new[i])
        np.insert(y_train, y_new[i])
    print(y_train)
    lp_model = label_propagation.LabelSpreading(gamma=0.25, kernel='knn', max_iter=300, n_neighbors=6)
    lp_model.fit(x_old, y_train)
    unlabeled_indices += len(x_new)
    predicted_labels = lp_model.transduction_[unlabeled_indices]
    y_all = np.concatenate((y_new[:unlabeled_indices[0]], predicted_labels))
    return x_old, y_all, y_train, unlabeled_indices


def seperate_feature_label(df):
    labels = df['activity']
    features = df.drop('activity', axis=1)
    features = features.drop('User', axis=1)
    # print(features)
    return features, labels



# validation with
def Leave_one_person_out(classifier, users, df):
    for i in range(len(users)):
        testUser = users[i]
        train_all, test_all = select(df, {'User': testUser}, True)
        train_x, train_y = seperate_feature_label(train_all)
        test_x, test_y = seperate_feature_label(test_all)
        classifier.fit(train_x, train_y)
        predictions = classifier.predict(test_x)

def normalize(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df)
    df_normalized=pd.DataFrame(x_scaled)
    return df_normalized

def sliding_window(df,window_size,ratio):
    feature_rows = []
    for i in range(0, len(df)-window_size, int(ratio*window_size)):
        window = windowing(df,i,window_size)
        feature_row = extract_features_in_window(window)
        feature_rows.append(feature_row)
    return pd.DataFrame(feature_rows)

def windowing(df,start,window_size):
    return df.iloc[start:start+window_size]

def extract_features_in_window(df):
    feature_row = {}

    df['m'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

    extract_features_of_one_column(df, 'x', feature_row)
    extract_features_of_one_column(df, 'y', feature_row)
    extract_features_of_one_column(df, 'z', feature_row)
    extract_features_of_one_column(df, 'm', feature_row)

    extract_features_of_two_columns(df, ['x', 'y'], feature_row)
    extract_features_of_two_columns(df, ['y', 'z'], feature_row)
    extract_features_of_two_columns(df, ['z', 'x'], feature_row)

    extract_features_of_two_columns(df, ['x', 'm'], feature_row)
    extract_features_of_two_columns(df, ['y', 'm'], feature_row)
    extract_features_of_two_columns(df, ['z', 'm'], feature_row)

    feature_row['User'] = df.iloc[0]['User']
    feature_row['activity'] = df.iloc[0]['activity']

    return feature_row
def extract_features_of_one_column(df,key,feature_row):
    series = df[key]
    extract_statistical_features(series, '1_' + key, feature_row)

def extract_features_of_two_columns(df, columns, feature_row):
    feature_row['2_' + columns[0] + columns[1] + '_correlation'] = df[columns[0]].corr(df[columns[1]])

def extract_statistical_features(series, prefix, feature_row):
    feature_row[prefix + '_mean'] = series.mean()
    feature_row[prefix + '_std'] = series.std()
    feature_row[prefix + '_var'] = series.var()
    feature_row[prefix + '_min'] = series.min()
    feature_row[prefix + '_max'] = series.max()
    feature_row[prefix + '_skew'] =series.skew()
    feature_row[prefix + '_kurtosis']=series.kurtosis()
    feature_row[prefix + '_energy'] = np.mean(series**2)


# data with labels,
def Supervised_learner(df, name=None):
    users = df['User'].unique()
    classifiers = {}
    classifiers['RandomForestClassifier'] = RandomForestClassifier(n_estimators=5)
    # classifiers['PolyKernal-SVC'] = svm.SVC(kernel='poly', max_iter=20000)
    # classifiers['KNeighborsClassifier'] = KNeighborsClassifier(n_neighbors=5)
    # classifiers['LinearSVC'] = svm.LinearSVC()
    # classifiers['Kmeans']= KMeans(n_clusters=5, init='random', max_iter=3000, n_init=10, tol=0.0001)
    # df=Feature_select(df)
    for algorithm, classifier in classifiers.items():
        # test_all=pd.DataFrame()
        # accuracy_all=None
        accuracy = []
        # df=Feature_select(df)
        # seperate test,train data:
        for i in range(len(users) - 1):
            testUser = users[i]
            print(testUser)
            train_all, test_all = select(df, {'User': testUser}, True)
            train_x, train_y = seperate_feature_label(train_all)
            # print(train_x)
            # print(train_y)
            test_x, test_y = seperate_feature_label(test_all)
            # print(test_x)
            if (algorithm == 'Kmeans'):
                test_x = normalize(test_x)
                train_x = normalize(train_x)
                classifier.fit(train_x)
            else:
                classifier.fit(train_x, train_y)
            y_pred = classifier.predict(test_x)
            # print(confusion_matrix(test_y,y_pred))
            # targetnames=['Lie','Sit','stand','iron','break','vacuum','break','ascend stairs','break','descend stairs','break','normal walk','break','nordic walk','break','cycle','break','run','break','rope jump']
            # print(classification_report(test_y,y_pred))
            accuracy.append(accuracy_score(y_pred, test_y))

        print(
            'Leave one person out \n%s Accuracy: %.2f%% (%.2f)  ' % (algorithm, np.average(accuracy), np.std(accuracy)))


def semi_supervised_test1(df, labeled_points, total_points):
    # split the data according to the classes, generate labelled , unlabelled mark for each and reshuffle.

    feature, label = seperate_feature_label(df)
    accuracy_for_supervise = []
    accuracy_for_semi_supervise = []
    x = []
    indices = np.arange(len(feature))
    label_indices = balanced_sample_maker(feature, label, labeled_points / len(label))
    unlabeled_indices = np.delete(indices, np.array(label_indices))
    rng = np.random.RandomState(0)
    rng.shuffle(unlabeled_indices)
    indices = np.concatenate((label_indices, unlabeled_indices[:total_points]))
    n_total_samples = len(indices)
    for i in range(10):
        unlabeled_indices = np.arange(n_total_samples)[labeled_points:]
        X = feature.iloc[indices]
        y = label.iloc[indices]
        y_train = np.copy(y)
        y_train[unlabeled_indices] = -1
        # supervised learning
        classifier = KNeighborsClassifier(n_neighbors=6)
        classifier.fit(X.iloc[:labeled_points], y.iloc[:labeled_points])
        y_pred = classifier.predict(X.iloc[labeled_points:])
        y_all = pd.concat(y.iloc[:labeled_points], y_pred)
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
        print('Semi-supervised learing:', accuracy_score(true_labels, predicted_labels))
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
    plt.title('Semi-supervised learning for total ' + str(n_total_samples) + ' samples ')
    plt.show()


def semi_supervised_test2(df, labeled_points, step):
    # for i in range(6):
    total_points = 600
    feature, label = seperate_feature_label(df)
    accuracy_for_supervise = []
    accuracy_for_semi_supervise = []
    x = []
    indices = np.arange(len(feature))
    label_indices = balanced_sample_maker(feature, label, labeled_points / len(label))
    unlabeled_indices = np.delete(indices, np.array(label_indices))
    # print(unlabeled_indices.size)
    # print(unlabeled_indices.size)
    rng = np.random.RandomState(0)
    rng.shuffle(unlabeled_indices)
    indices = np.concatenate((label_indices, unlabeled_indices[:total_points]))
    # n_total_samples = len(indices)

    for i in range(80):
        x.append(total_points)
        unlabeled_index = np.arange(total_points)[labeled_points:]
        # print(unlabeled_index.size)
        X = feature.iloc[indices]
        y = label.iloc[indices]
        y_train = np.copy(y)
        y_train[unlabeled_index] = -1
        # supervised learning
        classifier = KNeighborsClassifier(n_neighbors=6)
        classifier.fit(X.iloc[:labeled_points], y.iloc[:labeled_points])
        y_pred = classifier.predict(X.iloc[labeled_points:])
        true_labels = y.iloc[unlabeled_index]
        # print(confusion_matrix(true_labels,y_pred))
        print("%d labeled & %d unlabeled (%d total)"
              % (labeled_points, total_points - labeled_points, total_points))
        accuracy_for_supervise.append(accuracy_score(true_labels, y_pred))
        lp_model = label_propagation.LabelSpreading(gamma=1, kernel='knn', max_iter=300, n_neighbors=6)
        lp_model.fit(X, y_train)
        predicted_labels = lp_model.transduction_[unlabeled_index]
        # print('Iteration %i %s' % (i, 70 * '_'))
        accuracy_for_semi_supervise.append(accuracy_score(true_labels, predicted_labels))
        print('Semi-supervised learning:', accuracy_score(true_labels, predicted_labels))
        total_points += step  # print(unlabeled_indices[(total_points-50):total_points])
        indices = np.concatenate((indices, unlabeled_indices[(total_points - step):total_points]))

    x_sm = np.array(x)
    y_sm = np.array(accuracy_for_supervise)
    y1_sm = np.array(accuracy_for_semi_supervise)
    x_smooth = np.linspace(x_sm.min(), x_sm.max(), 200)
    y_smooth = spline(x, y_sm, x_smooth)
    y1_smooth = spline(x, y1_sm, x_smooth)
    sup, = plt.plot(x_smooth, y_smooth, label='Supervised learning with kNN')
    semi_l, = plt.plot(x_smooth, y1_smooth, label='Semi-supervised learning using Label Propagation')
    # plt.legend(handles=[sup, semi_l])
    plt.xlabel('Total samples')
    plt.ylabel('Accuracy')
    plt.title('Semi-supervised learning for labeled ' + str(labeled_points) + ' samples ')
    plt.show()
    return accuracy_score(true_labels, predicted_labels)
