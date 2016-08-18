from __future__ import division, print_function
import pandas as pd
from sklearn import *
from initial_training import *
from Baseline_test import *
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
import time
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

def seperate_feature_label(df):
    labels = df['activity']
    features = df.drop('activity', axis=1)
    features = features.drop('User', axis=1)
    # print(features)
    return features, labels

def clustering_compare(feature, label, cluster_number):
    feature = normalize(feature)
    print(label.unique().tolist())
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    clustering_names = [
        'MiniBatchKMeans',
        'SpectralClustering',
        'DBSCAN', 'Birch']
    connectivity = kneighbors_graph(feature, n_neighbors=10, include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    two_means = cluster.MiniBatchKMeans(n_clusters=cluster_number)
    # ward = cluster.AgglomerativeClustering(n_clusters=6, linkage='ward',
    #                                        connectivity=connectivity)
    spectral = cluster.SpectralClustering(n_clusters=cluster_number,
                                          eigen_solver='arpack',
                                          affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=.2)
    # average_linkage = cluster.AgglomerativeClustering(
    #     linkage="average", affinity="cityblock", n_clusters=6,
    #     connectivity=connectivity)

    birch = cluster.Birch(n_clusters=cluster_number)
    clustering_algorithms = [
        two_means, spectral,
        dbscan, birch]
    plt.figure(figsize=(len(clustering_names) * 2 + 3, 9.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)
    plot_num = 1
    for name, algorithm in zip(clustering_names, clustering_algorithms):
        # predict cluster memberships
        t0 = time.time()
        algorithm.fit(feature)
        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(feature)

        # plot
        plt.subplot(4, 5, plot_num)
        plt.scatter(feature.iloc[:, 1], feature.iloc[:, 2], color=colors[y_pred].tolist(), s=10)

        if hasattr(algorithm, 'cluster_centers_'):
            centers = algorithm.cluster_centers_
            center_colors = colors[:len(centers)]
            plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xticks(())
        plt.yticks(())
        plt.title(name, size=18)
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plt.text(.3,.01, 'accuracy:%.2f'% (accuracy_score(y_pred ,label)), transform=plt.gca().transAxes, size=15,verticalalignment='top')
        plot_num += 1
    plt.subplot(4, 5, 5)
    plt.scatter(feature.iloc[:, 2], feature.iloc[:, 4], color=colors[label].tolist(), s=10)

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xticks(())
    plt.yticks(())
    plt.title('True Label', size=18)
    plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
             transform=plt.gca().transAxes, size=15,
             horizontalalignment='right')
    plt.show()

if __name__ == '__main__':
    filepath = "/Users/LilyWU/Documents/activity_recognition_for_sensor/Data/HAPT_First_15_user.csv"
    newuser = "/Users/LilyWU/Documents/activity_recognition_for_sensor/Data/HAPT_user_16.csv"
    testpath = "/Users/LilyWU/Documents/activity_recognition_for_sensor/Data"
    data = pd.DataFrame.from_csv(filepath, header=0)

    feature, label = seperate_feature_label(data)

    cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
        feature, 6, 2, error=0.005, maxiter=1000)
    print(cntr.shape)

    # Show 3-cluster model
    # fig2, ax2 = plt.subplots()
    # ax2.set_title('Trained model')
    newdata = pd.DataFrame.from_csv(newuser, header=0)
    new_feature,new_label=seperate_feature_label(newdata)

    # plt.show()






