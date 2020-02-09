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


train_data = pd.read_json(r"C:\CMPT459_DataMining\Project\two-sigma-connect-rental-listing-inquiries\train.json\train.json")

sns.set()
np.random.seed(0)

################################### Begin Q1, Q2, Q3 #########################################
##############################################################################################

#################### Q1 #######################

# visualize_plot('price', True, 'b')
# visualize_plot('latitude', True, 'y')
# visualize_plot('longitude', True, 'r')

# plt.show()

#################### Q2 #######################

# visualize_plot('created', True, 'g')
# plt.show()

#################### Q3 #######################

# visualize_plot('interest_level', True, 'b')
# plt.show()

##############################################################################################
################################### End Q1, Q2, Q3 ###########################################


################################### Begin Q4 #########################################
######################################################################################

######### Find number of missing values ##########
num_attributes = ['bathrooms', 'bedrooms', 'latitude', 'listing_id', 'longitude', 'price']
num_attributes_min = [0, 0, 0, 0, -np.inf, 1]
str_attributes = ['street_address','building_id', 'description', 'display_address', 'manager_id', 'created', 'interest_level']
list_attributes = ['features', 'photos']

attributes_missing_value_removal = ['street_address']

attributes_missing_value_bool= ['display_address']

# attributes_to_process = num_attributes
# attributes_to_process_min = num_attributes_min
# for n in range(0,len(attributes_to_process)):
#     [num_missing, missing_values_indices] = find_missing_values (attributes_to_process[n], attributes_to_process_min[n])
#     print("Number of missing attributs in ", attributes_to_process[n] , ":   ", num_missing)

# attributes_to_process = str_attributes
# for n in range(0,len(attributes_to_process)):
#     [num_missing, missing_values_indices] = find_missing_values (attributes_to_process[n], 0)
#     print("Number of missing attributs in ", attributes_to_process[n] , ":   ", num_missing)

# attributes_to_process = list_attributes
# for n in range(0,len(attributes_to_process)):
#     [num_missing, missing_values_indices] = find_missing_values (attributes_to_process[n], 0)
#     print("Number of missing attributs in ", attributes_to_process[n] , ":   ", num_missing)


attributes_to_process = attributes_missing_value_removal
for n in range(0,len(attributes_to_process)):
    [num_missing, missing_values_indices] = find_missing_values (attributes_to_process[n], 0)
    print("Number of missing attributs in ", attributes_to_process[n] , ":   ", num_missing)
    train_data.drop(missing_values_indices, inplace=True)
    # [num_missing, missing_values_indices] = find_missing_values (attributes_to_process[n], 0)
    # print("Number of missing attributs in ", attributes_to_process[n] , ":   ", num_missing)


attributes_to_process = attributes_missing_value_bool
for n in range(0,len(attributes_to_process)):
    train_data['display_address_bool'] = train_data[attributes_to_process[n]].apply(lambda x: x.strip() != '')
    [num_missing, missing_values_indices] = find_missing_values (attributes_to_process[n], 0)
    train_data.drop(missing_values_indices, inplace=True)


######################################################################################
################################### End Q4 ###########################################


################################### Begin Q5 #########################################
######################################################################################

int_attributes = ['bathrooms', 'bedrooms', 'latitude', 'listing_id', 'longitude', 'price']
str_attributes = ['display_address', 'street_address', 'description']
dateTime_attribute = ['created']

######### draw boxplots to visualize  outliers ##########
    
# attributes_to_process = int_attributes
# for n in range(0,len(attributes_to_process)):
#     content_outlier = []
#     content_outlier = find_content_outlier(attributes_to_process[n],'')
#     draw_box_plot(content_outlier[0], content_outlier[1])

# attributes_to_process = str_attributes
# for n in range(0,len(attributes_to_process)):
#     content_outlier = []
#     content_outlier = find_content_outlier(attributes_to_process[n],'len')
#     draw_box_plot(content_outlier[0], content_outlier[1])

# attributes_to_process = dateTime_attribute
# for n in range(0,len(attributes_to_process)):
#     content_outlier = []
#     content_outlier = find_content_outlier(attributes_to_process[n],'year')
#     draw_box_plot(content_outlier[0], content_outlier[1])

#     content_outlier = []
#     content_outlier = find_content_outlier(attributes_to_process[n],'month')
#     draw_box_plot(content_outlier[0], content_outlier[1])

#     content_outlier = []
#     content_outlier = find_content_outlier(attributes_to_process[n],'day')
#     draw_box_plot(content_outlier[0], content_outlier[1])

#     content_outlier = []
#     content_outlier = find_content_outlier(attributes_to_process[n],'hour')
#     draw_box_plot(content_outlier[0], content_outlier[1])

#     content_outlier = []
#     content_outlier = find_content_outlier(attributes_to_process[n],'minute')
#     draw_box_plot(content_outlier[0], content_outlier[1])

#     content_outlier = []
#     content_outlier = find_content_outlier(attributes_to_process[n],'second')
#     draw_box_plot(content_outlier[0], content_outlier[1])

# plt.show() 

######### find outliers based on z score and IQR ###############

# attributes_to_process = int_attributes
# for n in range(0,len(attributes_to_process)):
#     content_outlier = []
#     outlier_datapoints = []

#     content_outlier = find_content_outlier(attributes_to_process[n],'')
#     if (attributes_to_process[n] == 'bathrooms'):
#         [outlier_datapoints, outlier_indices]  = detect_outlier_zScore(content_outlier[0], 4)
#     else:
#         [outlier_datapoints, outlier_indices]  = detect_outlier_zScore(content_outlier[0], 3)

#     print("\n Z-score: Number of outliers in ",content_outlier[1], ": ", len(outlier_datapoints))
#     # plt.hist(outlier_datapoints, bins=50)
#     # plt.show()

#     # outlier_datapoints = []
#     # outlier_datapoints  = detect_outlier_IQR(content_outlier[0])
#     # print("IQR: Number of outliers in ",content_outlier[1], ": ", len(outlier_datapoints))
#     # plt.hist(outlier_datapoints, bins=50)
#     # plt.show()

# attributes_to_process = str_attributes
# for n in range(0,len(attributes_to_process)):
#     content_outlier = []
#     outlier_datapoints = []
#     content_outlier = find_content_outlier(attributes_to_process[n],'len')
#     if (attributes_to_process[n] == 'street_address'):
#         [outlier_datapoints, outlier_indices]  = detect_outlier_zScore(content_outlier[0], 8)
#     else:
#         [outlier_datapoints, outlier_indices]  = detect_outlier_zScore(content_outlier[0], 3)
#     print("\n Z-score: Number of outliers in ",content_outlier[1], ": ", len(outlier_datapoints))
#     plt.hist(outlier_datapoints, bins=50)
#     plt.show()

#     outlier_datapoints = []
#     outlier_datapoints  = detect_outlier_IQR(content_outlier[0])
#     print("IQR: Number of outliers in ",content_outlier[1], ": ", len(outlier_datapoints))
#     plt.hist(outlier_datapoints, bins=50)
#     plt.show()


# attributes_to_process = dateTime_attribute
# for n in range(0,len(attributes_to_process)):
#     content_outlier = []
#     outlier_datapoints = []
#     content_outlier = find_content_outlier(attributes_to_process[n],'year')
#     outlier_datapoints  = detect_outlier_zScore(content_outlier[0], 3)
#     print("\n Z-score: Number of outliers in ",content_outlier[1], ": ", len(outlier_datapoints))

#     # outlier_datapoints = []
#     # outlier_datapoints  = detect_outlier_IQR(content_outlier[0])
#     # print("IQR: Number of outliers in ",content_outlier[1], ": ", len(outlier_datapoints))



#     content_outlier = []
#     outlier_datapoints = []
#     content_outlier = find_content_outlier(attributes_to_process[n],'month')
#     outlier_datapoints  = detect_outlier_zScore(content_outlier[0], 3)
#     print("\n Z-score: Number of outliers in ",content_outlier[1], ": ", len(outlier_datapoints))


#     # outlier_datapoints = []
#     # outlier_datapoints  = detect_outlier_IQR(content_outlier[0])
#     # print("IQR: Number of outliers in ",content_outlier[1], ": ", len(outlier_datapoints))



#     content_outlier = []
#     outlier_datapoints = []
#     content_outlier = find_content_outlier(attributes_to_process[n],'day')
#     outlier_datapoints  = detect_outlier_zScore(content_outlier[0], 3)
#     print("\n Z-score: Number of outliers in ",content_outlier[1], ": ", len(outlier_datapoints))


#     # outlier_datapoints = []
#     # outlier_datapoints  = detect_outlier_IQR(content_outlier[0])
#     # print("IQR: Number of outliers in ",content_outlier[1], ": ", len(outlier_datapoints))


#     content_outlier = []
#     outlier_datapoints = []
#     content_outlier = find_content_outlier(attributes_to_process[n],'hour')
#     outlier_datapoints  = detect_outlier_zScore(content_outlier[0], 3)
#     print("\n Z-score: Number of outliers in ",content_outlier[1], ": ", len(outlier_datapoints))


#     # outlier_datapoints = []
#     # outlier_datapoints  = detect_outlier_IQR(content_outlier[0])
#     # print("IQR: Number of outliers in ",content_outlier[1], ": ", len(outlier_datapoints))


#     content_outlier = []
#     outlier_datapoints = []
#     content_outlier = find_content_outlier(attributes_to_process[n],'minute')
#     outlier_datapoints  = detect_outlier_zScore(content_outlier[0], 3)
#     print("\n Z-score: Number of outliers in ",content_outlier[1], ": ", len(outlier_datapoints))

#     # outlier_datapoints = []
#     # outlier_datapoints  = detect_outlier_IQR(content_outlier[0])
#     # print("IQR: Number of outliers in ",content_outlier[1], ": ", len(outlier_datapoints))


#     content_outlier = []
#     outlier_datapoints = []
#     content_outlier = find_content_outlier(attributes_to_process[n],'second')
#     outlier_datapoints  = detect_outlier_zScore(content_outlier[0], 3)
#     print("\n Z-score: Number of outliers in ",content_outlier[1], ": ", len(outlier_datapoints))


#     # outlier_datapoints = []
#     # outlier_datapoints  = detect_outlier_IQR(content_outlier[0])
#     # print("IQR: Number of outliers in ",content_outlier[1], ": ", len(outlier_datapoints))

################ handle outliers ###############

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
    print("\n Z-score: Number of outliers in ",content_outlier[1], ": ", len(outlier_datapoints))
    train_data.drop(outlier_indices, inplace=True)






######################################################################################
################################### End Q5 ###########################################


############################### Feature Extraction ###################################
######################################################################################

train_data["num_photos"] = train_data["photos"].apply(len)

train_data["created_dt"] = pd.to_datetime(train_data["created"])
train_data["created_year"] = train_data["created_dt"].dt.year
train_data["created_month"] = train_data["created_dt"].dt.month
train_data["created_day"] = train_data["created_dt"].dt.day
train_data["created_hour"] = train_data["created_dt"].dt.hour
train_data["created_minute"] = train_data["created_dt"].dt.minute
train_data["created_second"] = train_data["created_dt"].dt.second


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
################ Image feature extraction ################################
## These features are not added to train_X for reasons explained in report
# for photo_list in train_data['photos']:
#     for photo in photo_list:
#         [hor, ver, fd, hog_img] = featUtil.extractSkFeatures('https://photos.renthop.com/2/6811957_3dad56e8bf3477b2900ca39d57df041e.jpg', True)
#         [hist_red, hist_blue, hist_green, hist_gray] = featUtil.extractColorHistogramFeatures('https://photos.renthop.com/2/6811957_3dad56e8bf3477b2900ca39d57df041e.jpg', True)
#         [kp, des, img] = featUtil.extractCv2Features('https://photos.renthop.com/2/6811957_3dad56e8bf3477b2900ca39d57df041e.jpg', 4)
#         stat = featUtil.extractStatFeatures('https://photos.renthop.com/2/6811957_3dad56e8bf3477b2900ca39d57df041e.jpg')
        
##########################################################################

[num_features, Features_tfidf_vector] = featUtil.extract_text_features(train_data, 'features')
train_data["num_features"] = num_features
[num_words_description, description_tfidf_vector] = featUtil.extract_text_features(train_data, 'description')
train_data["num_words_description"] = num_words_description

features_to_use.extend(["num_features", "num_words_description"])

train_X = sparse.hstack([train_data[features_to_use], Features_tfidf_vector, description_tfidf_vector]).tocsr()



######################################################################################
######################################################################################




quit()
