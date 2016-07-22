from sklearn import *
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import *
from sklearn import preprocessing
from
def seperate_feature_label(df):
    labels=df['activity']
    features=df.drop('activity',axis=1)
    features=df.drop('User',axis=1)
    return features,labels

def select(data,key_value_pairs,return_all=False):
   for key in key_value_pairs:
        select = data[key] == key_value_pairs[key]
        if(return_all == False):  return data[select]
        else:
          other = data[select==False]
          return data[select], other

def Leave_one_person_out(classifier,users ,df):
    for algorithm, classifier in classifiers.items():
        for i in range(len(users)):
                testUser=users[i]
                train_all, test_all=select(df,{'User':testUser},True)
                train_x,train_y=seperate_feature_label(train_all)
                test_x, test_y=seperate_feature_label(test_all)
                classifier.fit(train_x,train_y)
                predictions = classifier.predict(test_x)
    return predictions, test_y

def normalize(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(x_scaled)
    return df_normalized

def classification(df):
    classifiers = {}
    classifiers['RandomForestClassifier'] = RandomForestClassifier(n_estimators=5)
    classifiers['Multi-SVC'] = svm.SVC(kernel='poly', max_iter=20000)
    classifiers['DecisionTreeClassifier'] = DecisionTreeClassifier(max_depth=None, min_samples_split=1)
    # classifiers['MLP']=MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    classifiers['KNeighborsClassifier'] = KNeighborsClassifier(n_neighbors=5)
    classifiers['LinearSVC'] = svm.LinearSVC()
    # Classification
    features, labels = seperate_feature_label(df)

for algorithm, classifier in classifiers.items():
    classification_results = cross_validation.cross_val_score(classifier, features, labels, cv=10)
    print('Results for cross_val_score')
    print(algorithm, "Accuracy: %0.2f (+/- %0.2f)" % (classification_results.mean(), classification_results.std() * 2))

classifiers['kMeans'] = KMeans(n_clusters=8, init='k-means++', max_iter=3000, random_state=None, tol=0.0001)
for algorithm, classifier in classifiers.items():
    features, labels = normalize()
    print(algorithm, "Accuracy: %0.2f (+/- %0.2f)" % (classification_results.mean(), classification_results.std() * 2))
    # kf = KFold(9, n_folds=10, shuffle=False, random_state=None)
    # for classifier in classifiers.items():
    #   for train,test in kf：
    #      train_x,test_x = features[train], labels[test]
    #      train_y,train_y = features[train],labels[test]
    #      classifier.fit(train_x,train_y)
    #      predictions = classifier.predict(test_x)
    #   print('K-fold_validation:' classifier, classification_report(test_y,predictions))