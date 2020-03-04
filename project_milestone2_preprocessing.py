import numpy as np
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import json
from PIL import Image
import requests
from io import BytesIO
import seaborn as sns
from sklearn import model_selection, preprocessing, ensemble
import featureExtractionUtility as featUtil
from scipy import sparse


def draw_box_plot(feature_content_outlier, xlabel_title):
    plt.figure()

    sns.boxplot(x = feature_content_outlier)
    plt.xlabel(xlabel_title , fontsize=12)
    
    
def find_content_outlier(feature_name,feature_to_extract):
    feature_content_outlier = []
    xlabel_title = ''
    if(type(train_data[feature_name].values[0]) == np.int64) or (type(train_data[feature_name].values[0]) == np.float64):
        feature_content_outlier = train_data[feature_name].values
        xlabel_title = feature_name
    elif (type(train_data[feature_name].values[0]) == str):
        if (feature_name == 'created'):
            if(feature_to_extract == 'year'):
                feature_content = pd.to_datetime(train_data[feature_name])
                feature_content_outlier = feature_content.dt.year
                xlabel_title = "created_year"

            if(feature_to_extract == 'month'):
                feature_content = pd.to_datetime(train_data[feature_name])
                feature_content_outlier = feature_content.dt.month
                xlabel_title = "created_month"

            if(feature_to_extract == 'day'):
                feature_content = pd.to_datetime(train_data[feature_name])
                feature_content_outlier = feature_content.dt.day
                xlabel_title = "created_day"

            if(feature_to_extract == 'hour'):
                feature_content = pd.to_datetime(train_data[feature_name])
                feature_content_outlier = feature_content.dt.hour
                xlabel_title = "created_hour"

            if(feature_to_extract == 'minute'):
                feature_content = pd.to_datetime(train_data[feature_name])
                feature_content_outlier = feature_content.dt.minute
                xlabel_title = "created_minute"

            if(feature_to_extract == 'second'):
                feature_content = pd.to_datetime(train_data[feature_name])
                feature_content_outlier = feature_content.dt.second
                xlabel_title = "created_second"

        elif (feature_name == 'street_address') or (feature_name == 'display_address') or (feature_name == 'description'):
            if (feature_to_extract == 'len'):
                for n in train_data[feature_name]:
                    feature_content_outlier.append(len(n))
                xlabel_title = "length of " + feature_name
    
    return feature_content_outlier, xlabel_title


def detect_outlier_zScore(feature_content_outlier, threshold):
    outliers = []
    outliers_indices = []

    mean_1 = np.mean(feature_content_outlier)
    std_1 =np.std(feature_content_outlier)
    
    
    for idx, y in enumerate(feature_content_outlier):
        if (std_1 == 0):
            z_score = 0
        else:
            z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
            outliers_indices.append(train_data.index[idx])
    return outliers, outliers_indices

def detect_outlier_IQR(feature_content_outlier):
    outliers = []
    sorted(feature_content_outlier)
    q1, q3= np.percentile(feature_content_outlier,[25,75])
    iqr = q3 - q1
    lower_bound = q1 -(1.5 * iqr) 
    upper_bound = q3 +(1.5 * iqr) 
    
    
    for y in feature_content_outlier:
        if y > upper_bound or y < lower_bound:
            outliers.append(y)
    return outliers

def Trim_one_percentiel(feature_name):

    llimit = np.percentile(train_data[feature_name].values, 1)
    train_data[feature_name].ix[train_data[feature_name]<llimit] = llimit

    ulimit = np.percentile(train_data[feature_name].values, 99)
    train_data[feature_name].ix[train_data[feature_name]>ulimit] = ulimit


def visualize_plot(feature_name,Trim, color_name):

    plt.figure()

    if(type(train_data[feature_name].values[0]) == np.int64):
        if Trim:
            Trim_one_percentiel(feature_name)

        sns.distplot(train_data[feature_name].values, color = color_name)
        
    elif (type(train_data[feature_name].values[0]) == str):
        if (feature_name == 'created'):
            feature_content = pd.to_datetime(train_data[feature_name])
            Content_to_plot = feature_content.dt.hour.value_counts()
        else:
            Content_to_plot = train_data[feature_name].value_counts()
            
        sns.barplot(Content_to_plot.index, Content_to_plot.values, color = color_name)
    
    plt.xlabel(feature_name, fontsize=12)


def find_missing_values(feature_name,min):
    num_missing = 0
    missing_values_indices = []
    data_type = type(train_data[feature_name].values[0])
    if(data_type == np.int64 or data_type == np.float64):
        for idx, i in enumerate(train_data[feature_name].values):
            if (i < min) :
                num_missing += 1
                missing_values_indices.append(idx)
    elif (data_type == str):
        for idx, n in enumerate(range (0,len(train_data[feature_name].values))):
            i = train_data[feature_name].values[n]
            i_trimmed = i.strip()
            if (feature_name == 'building_id') and (i_trimmed == '0'):
                num_missing += 1
                missing_values_indices.append(train_data.index[idx])
            elif (i_trimmed == ''):
                num_missing += 1
                missing_values_indices.append(train_data.index[idx])
    elif (data_type == list):
        for idx, i in enumerate(train_data[feature_name].values):
            if (i == []):
                num_missing += 1
                missing_values_indices.append(idx)
    
    return num_missing, missing_values_indices


train_data = pd.read_json(r"C:\CMPT459_DataMining\Project\milestone2\train.json\train.json")
test_data = pd.read_json(r"C:\CMPT459_DataMining\Project\milestone2\test.json\test.json")


sns.set()
np.random.seed(0)



############################### Handling missing values #####################################
#############################################################################################

num_attributes = ['bathrooms', 'bedrooms', 'latitude', 'listing_id', 'longitude', 'price']
num_attributes_min = [0, 0, 0, 0, -np.inf, 1]
str_attributes = ['street_address','building_id', 'description', 'display_address', 'manager_id', 'created', 'interest_level']
list_attributes = ['features', 'photos']

attributes_missing_value_removal = ['street_address']

attributes_missing_value_bool= ['display_address']

print("1. Handling missing values in train data ...")

attributes_to_process = attributes_missing_value_removal
for n in range(0,len(attributes_to_process)):
    [num_missing, missing_values_indices] = find_missing_values (attributes_to_process[n], 0)
    # print("Number of missing attributs in ", attributes_to_process[n] , ":   ", num_missing)
    train_data.drop(missing_values_indices, inplace=True)
    # [num_missing, missing_values_indices] = find_missing_values (attributes_to_process[n], 0)
    # print("Number of missing attributs in ", attributes_to_process[n] , ":   ", num_missing)


attributes_to_process = attributes_missing_value_bool
for n in range(0,len(attributes_to_process)):
    train_data['display_address_bool'] = train_data[attributes_to_process[n]].apply(lambda x: x.strip() != '')
    [num_missing, missing_values_indices] = find_missing_values (attributes_to_process[n], 0)
    train_data.drop(missing_values_indices, inplace=True)


#############################################################################################
#############################################################################################


################################### handle outliers #########################################
#############################################################################################
print("2. Handling outliers in train data ...")

int_attributes = ['bathrooms', 'bedrooms', 'latitude', 'listing_id', 'longitude', 'price']
str_attributes = ['display_address', 'street_address', 'description']
dateTime_attribute = ['created']


attributes_missing_value_removal = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price']

attributes_to_process = attributes_missing_value_removal
for n in range(0,len(attributes_to_process)):
    content_outlier = []
    outlier_datapoints = []
    content_outlier = find_content_outlier(attributes_to_process[n],'len')
    if (attributes_to_process[n] == 'bathrooms'):
        [outlier_datapoints, outlier_indices]  = detect_outlier_zScore(content_outlier[0], 4)
    else:
        [outlier_datapoints, outlier_indices]  = detect_outlier_zScore(content_outlier[0], 3)
    # print("\n Z-score: Number of outliers in ",content_outlier[1], ": ", len(outlier_datapoints))
    train_data.drop(outlier_indices, inplace=True)


#############################################################################################
#############################################################################################


################################## Feature Extraction ######################################
############################################################################################

################# train data feature extraction #########################
print("3. Feature extraction in train data ...")

target_num_map = {'high':0, 'medium':1, 'low':2}
train_Y = np.array(train_data['interest_level'].apply(lambda x: target_num_map[x]))
np.save(r"C:\CMPT459_DataMining\Project\milestone2\train.json\train_Y_preprocessed.npy", train_Y)

train_data["num_photos"] = train_data["photos"].apply(len)

train_data["created_dt"] = pd.to_datetime(train_data["created"])
train_data["created_year"] = train_data["created_dt"].dt.year
train_data["created_month"] = train_data["created_dt"].dt.month
train_data["created_day"] = train_data["created_dt"].dt.day
train_data["created_hour"] = train_data["created_dt"].dt.hour
train_data["created_minute"] = train_data["created_dt"].dt.minute
train_data["created_second"] = train_data["created_dt"].dt.second

NumericalClassifier = True

if (NumericalClassifier):
    str_attributes = ["display_address", "street_address", "manager_id", "building_id"]
    for feat in str_attributes:               ## Use this in classifiers that do not accept categorical attributes
        if train_data[feat].dtype=='object':
            labelEnc = preprocessing.LabelEncoder()
            labelEnc.fit(list(train_data[feat].values))
            # classes = labelEnc.classes_
            train_data[feat] = labelEnc.transform(list(train_data[feat].values))



features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price", "listing_id", 
    "num_photos","created_year", "created_month", "created_day", "created_hour", "created_minute", 
    "created_second", "display_address", "manager_id", "building_id", "street_address"]



[num_features, Features_tfidf_vector, tfidf_transformer_features, tfidf_features] = featUtil.extract_text_features_train(train_data, 'features')
train_data["num_features"] = num_features
[num_words_description, description_tfidf_vector, tfidf_transformer_description, tfidf_description] = featUtil.extract_text_features_train(train_data, 'description')
train_data["num_words_description"] = num_words_description

features_to_use.extend(["num_features", "num_words_description"])

train_X = sparse.hstack([train_data[features_to_use], Features_tfidf_vector, description_tfidf_vector]).tocsr()

sparse.save_npz(r"C:\CMPT459_DataMining\Project\milestone2\train.json\train_X_preprocessed.npz", train_X)
# loaded_train_X = sparse.load_npz(r"C:\CMPT459_DataMining\Project\milestone2\train.json\train_preprocessed.npz")

################# test data feature extraction #########################
print("3. Feature extraction in test data ...")

test_data["num_photos"] = test_data["photos"].apply(len)

test_data["created_dt"] = pd.to_datetime(test_data["created"])
test_data["created_year"] = test_data["created_dt"].dt.year
test_data["created_month"] = test_data["created_dt"].dt.month
test_data["created_day"] = test_data["created_dt"].dt.day
test_data["created_hour"] = test_data["created_dt"].dt.hour
test_data["created_minute"] = test_data["created_dt"].dt.minute
test_data["created_second"] = test_data["created_dt"].dt.second


if (NumericalClassifier):
    str_attributes = ["display_address", "street_address", "manager_id", "building_id"]
    for feat in str_attributes:               ## Use this in classifiers that do not accept categorical attributes
        if test_data[feat].dtype=='object':
            labelEnc = preprocessing.LabelEncoder()
            labelEnc.fit(list(test_data[feat].values))
            # classes = labelEnc.classes_
            test_data[feat] = labelEnc.transform(list(test_data[feat].values))





[num_features_test, Features_tfidf_vector_test] = featUtil.extract_text_features_test(test_data, 'features', tfidf_transformer_features, tfidf_features)
test_data["num_features"] = num_features_test
[num_words_description_test , description_tfidf_vector_teset] = featUtil.extract_text_features_test(test_data, 'description', tfidf_transformer_description, tfidf_description)
test_data["num_words_description"] = num_words_description_test

# features_to_use.extend(["num_features", "num_words_description"])

test_X = sparse.hstack([test_data[features_to_use], Features_tfidf_vector_test, description_tfidf_vector_teset]).tocsr()

sparse.save_npz(r"C:\CMPT459_DataMining\Project\milestone2\test.json\test_preprocessed.npz", test_X)
# loaded_train_X = sparse.load_npz(r"C:\CMPT459_DataMining\Project\milestone2\train.json\train_preprocessed.npz")

############################################################################################
############################################################################################




quit()
