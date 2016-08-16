from load_PAMAP2 import Loading_PAMAP2
from feature_generate import *
from similarity_check import *
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

path = '/Users/ana/Documents/ER lab repo/Active Learning/activity_recognition_for_sensor/'
HAPT_folder="HAPT Data Set/RawData"
PAMAP2_folder=path + "PAMAP2_Dataset/Protocol"
# Datasets_description=
# {
#      'HAPT':   f=50HZ  activity_number: 6  Users: 30
#      'PAMAP2': f=100HZ activity_number: 24 Users:  9

# }
def Loading(dataset):
   data = {}
   data['activity']=list()
   data['timestamp']=list()
   data['x']=list()
   data['y']=list()
   data['z']=list()
   data['User']=list()

   new = None
   if(dataset=="PAMAP2"):
      paths=glob.glob(PAMAP2_folder+'/*.dat')
      id=1
      for filepath in paths:
              data=Loading_PAMAP2(filepath,id,data)
              #new=pd.DataFrame.from_dict(data)
              id = id+1
      new = pd.DataFrame.from_dict(data)
      return new
#return any specified column or one column and rest of it
def select(data,key_value_pairs,return_all=False):

   for key in key_value_pairs:
        select = data[key] == key_value_pairs[key]
        if(return_all == False):  return data[select]
        else:
          other = data[select==False]
          return data[select], other


def seperate_feature_label(df):
    labels=df['activity']
    features=df.drop('activity',axis=1)
    features=df.drop('User',axis=1)
    return features,labels

def get_features(data, user):
    frequency=100
    features_seperate={} #sperate feature for each user
    features_for_all=pd.DataFrame()
    select_user=select(data,{'User':user})
    activities=[1,2,3,12]#data['activity'].unique()
    labels = []
    for ind, activity in enumerate(activities): #one user and one activity
        #print ind
        select_activity= select(select_user,{'activity':activity})
        features_seperate[user]= sliding_window(select_activity,5*frequency,0.5) #smoothing first --> sliding windowing
        features_for_all=pd.concat([features_for_all,features_seperate[user]])
    print ("------------")
    #print features_for_all
    return features_for_all

# DONE check read dataset
# DONE check feature creation
# DONE select a portion of dataset windows
# DONE train a random forest
# TODO feed based on timing (func : getNext())
# DONE implemenet comparison code
# TODO visualize --> show points in each cluster and added points with their assigned label
def main():
    data=Loading('PAMAP2')
    print('Data Loaded')
    #users=data['User'].unique() #list of all users
    train_feature, train_label = seperate_feature_label(get_features(data, 1))
    test_feature, test_label = seperate_feature_label(get_features(data, 2))
    ########################### here we have the dataset ##################
    clf = train_base_classifier(train_feature, train_label)
    print("clf trained")
    print train_feature.shape
    add_one(clf, test_feature, test_label, train_feature, train_label)


if __name__ == '__main__':
    main()






# data=Loading('PAMAP2')
# print('Data Loaded')
# frequency=100
# features_seperate={} #sperate feature for each user
# features_for_all=pd.DataFrame()
# users=data['User'].unique() #list of all users
# print users
# for user in users:
#     select_user=select(data,{'User':user})
#     activities=data['activity'].unique()
#     for activity in activities: #one user and one activity
#         select_activity= select(select_user,{'activity':activity})
#         #smoothing first --> sliding windowing
#         features_seperate[user]= sliding_window(select_activity,5*frequency,0.5)
#         features_for_all=pd.concat([features_for_all,features_seperate[user]])
