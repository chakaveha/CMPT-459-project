import numpy as np
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import json
from sklearn import preprocessing, ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss
import featureExtractionUtility as featUtil
from scipy import sparse
import xgboost as xgb
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from scipy.sparse import hstack
from scipy import sparse
from sklearn.feature_selection import mutual_info_classif, SelectKBest, SelectPercentile
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
import math
from sklearn.model_selection import RandomizedSearchCV
import statistics
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


def PCADescription_and_toDense(train_X, test_X, train_Y):
    ########################### features in train and test data ###############################
    #   - First 19 features: "bathrooms", "bedrooms", "latitude", "longitude", "price", "listing_id", 
    #      "num_photos","created_year", "created_month", "created_day", "created_hour", "created_minute", 
    #      "created_second", "display_address", "manager_id", "building_id", "street_address", "num_features", 
    #      "num_words_description"
    #   - Second 19 features: tfidf for "features"
    #   - Last 228 features: tfidf for "description"

    # TruncatedSVD to reduce dimension of tfidf features 
    train_x_mainfeatures = train_X[:,:19]
    train_X_features_tfidf = train_X[:,19:38]
    train_X_description_tfidf = train_X[:, 38:]
    test_x_mainfeatures = test_X[:,:19]
    test_X_features_tfidf = test_X[:,19:38]
    test_X_description_tfidf = test_X[:, 38:]

    ## tried PCA and TruncatedSVD, TruncatedSVD produced better results with default param classifiers

    # train_X_description_tfidf = train_X_description_tfidf.todense()
    # test_X_description_tfidf = test_X_description_tfidf.todense()

    # num_description_components = 50
    # svd_description = PCA(svd_solver='arpack', n_components=num_description_components, random_state=42) # whiten = True
    # svd_description.fit(train_X_description_tfidf)

    ### keping 50 components would keep about 50% variance of data
    num_description_components = 50
    svd_description = TruncatedSVD(algorithm='arpack', n_components=num_description_components, random_state=42, n_iter=15)
    svd_description.fit(train_X_description_tfidf)

    # print(svd_description.explained_variance_ratio_) 
    # print(svd_description.explained_variance_ratio_.sum())  

    # svd_features = TruncatedSVD(n_components=18, random_state=42)
    # svd_features.fit(train_X_features_tfidf)
    # print(svd_features.explained_variance_ratio_) 

    train_x_mainfeatures = train_x_mainfeatures.todense()
    train_X_features_tfidf = train_X_features_tfidf.todense()
    test_x_mainfeatures = test_x_mainfeatures.todense()
    test_X_features_tfidf = test_X_features_tfidf.todense()

    train_X_description_tfidf_reducedDimension = svd_description.transform(train_X_description_tfidf)
    test_X_description_tfidf_reducedDimension = svd_description.transform(test_X_description_tfidf)

    train_X = np.concatenate((train_x_mainfeatures, train_X_features_tfidf, train_X_description_tfidf_reducedDimension), axis = 1)
    test_X = np.concatenate((test_x_mainfeatures, test_X_features_tfidf, test_X_description_tfidf_reducedDimension), axis = 1)

    return train_X, test_X

def Feature_Selection_mutual_information(train_X, test_X, train_Y):
    # performing mutual information feature selection 

    # scores = mutual_info_classif(train_X, train_Y, discrete_features='auto', n_neighbors=3, copy=True, random_state=42)
    # scores.sort()

    ### when features were sorted based on their mutual info with class label, 12(13.6%) had scores less than 0.01. Decided to keep 90% 

    percentile_to_keep_mutualInfo = 90  

    mutual_info_feat_sel = SelectPercentile(mutual_info_classif, percentile=percentile_to_keep_mutualInfo).fit(train_X, train_Y)
    train_X = mutual_info_feat_sel.transform(train_X)
    test_X = mutual_info_feat_sel.transform(test_X)
    return train_X, test_X




train_Y = np.load(r"C:\CMPT459_DataMining\Project\milestone2\train.json\train_Y_preprocessed.npy")


################################## Begin second stage of pre-processing ######################################
##############################################################################################################
# train_X = sparse.load_npz(r"C:\CMPT459_DataMining\Project\milestone2\train.json\train_X_preprocessed.npz")
# test_X = sparse.load_npz(r"C:\CMPT459_DataMining\Project\milestone2\test.json\test_preprocessed.npz")
# [train_X, test_X] = PCADescription_and_toDense(train_X, test_X, train_Y)
# [train_X, test_X] = Feature_Selection_mutual_information(train_X, test_X, train_Y)


# np.save(r"C:\CMPT459_DataMining\Project\milestone2\train.json\train_X_preprocessed.npy", train_X)
# np.save(r"C:\CMPT459_DataMining\Project\milestone2\test.json\test_X_preprocessed.npy", test_X)


##############################################################################################################
################################### End second stage of pre-processing #######################################

########################################## Begin Classification ##############################################
##############################################################################################################
train_X = np.load(r"C:\CMPT459_DataMining\Project\milestone2\train.json\train_X_preprocessed.npy")
test_X = np.load(r"C:\CMPT459_DataMining\Project\milestone2\test.json\test_X_preprocessed.npy")

########################################## First Attempt with default values ##########################
test_X_milestone1 = sparse.load_npz(r"C:\CMPT459_DataMining\Project\milestone2\test.json\test_preprocessed.npz").toarray()
test_X_listing_id = (test_X_milestone1[:,5].astype(int))

########### Random Forest ###########
clf_RF = RandomForestClassifier(random_state=0)
scores_RF = cross_val_score(clf_RF, train_X, train_Y, cv = 5, scoring='neg_log_loss')
print("RF without tuning score = ", statistics.mean(scores_RF))
clf_RF.fit(train_X,train_Y)
test_Y = clf_RF.predict_proba(test_X)
test_Y = np.column_stack((test_X_listing_id,test_Y ))
np.savetxt(r"C:\CMPT459_DataMining\Project\milestone3\test.json\test_Y_RF_firstAttempt.csv", test_Y, delimiter=',')

########### AdaBoost ###########
clf_AdaBoost = AdaBoostClassifier(random_state=0)
scores_AdaBoost = cross_val_score(clf_AdaBoost, train_X, train_Y, cv = 5, scoring='neg_log_loss')
print("RF without tuning score = ", statistics.mean(scores_AdaBoost))
clf_AdaBoost.fit(train_X,train_Y)
test_Y = clf_AdaBoost.predict_proba(test_X)
test_Y = np.column_stack((test_X_listing_id,test_Y ))
np.savetxt(r"C:\CMPT459_DataMining\Project\milestone3\test.json\test_Y_AdaBoost_firstAttempt.csv", test_Y, delimiter=',')


########################################## Hyperparameter tuning ####################################
########### Random Forest ###########
criterion = ['entropy']
max_features = list(range(1,train_X.shape[1]+1))
n_estimators = [50, 100, 200, 500]

oob_score = [True, False]

param_distributions = dict(max_features = max_features,
                           criterion = criterion,
                           n_estimators = n_estimators,
                           oob_score = oob_score)

clf_RF = RandomForestClassifier(random_state=0)

random = RandomizedSearchCV(estimator=clf_RF,
                            param_distributions=param_distributions,
                            scoring='neg_log_loss',
                            verbose=1, n_jobs=-1,
                            n_iter=140, cv=5)

random_result = random.fit(train_X, train_Y)
print('Best Score: ', random_result.best_score_)
print('Best Params: ', random_result.best_params_)
tuned_hyperparameters = random_result.best_params_
clf_RF_tuned = RandomForestClassifier(random_state=0, max_features = tuned_hyperparameters['max_features'], criterion = tuned_hyperparameters['criterion'], 
 n_estimators = tuned_hyperparameters['n_estimators'], oob_score = tuned_hyperparameters['oob_score'])

clf_RF_tuned.fit(train_X,train_Y)
test_Y = clf_RF_tuned.predict_proba(test_X)
test_Y = np.column_stack((test_X_listing_id,test_Y ))
np.savetxt(r"C:\CMPT459_DataMining\Project\milestone3\test.json\test_Y_RF_afterParameterTuning_deepTrees2.csv", test_Y, delimiter=',')

########### AdaBoost parameter tuning###########
# clf_RF = RandomForestClassifier(random_state=0, max_depth = 2)
# clf_DT = DecisionTreeClassifier(random_state=0, max_depth = 2)
# # clf_GBC = GradientBoostingClassifier(random_state=0, max_depth = 5)

# base_estimator = [clf_RF, clf_DT]
# n_estimators = [50, 100, 200, 500]
# learning_rate = [1, 0.1, 0.01, 0.0001]

# param_distributions = dict(base_estimator = base_estimator,
#                            n_estimators = n_estimators,
#                            learning_rate = learning_rate)

# clf_AdaBoost = AdaBoostClassifier(random_state=0)

# random = RandomizedSearchCV(estimator=clf_AdaBoost,
#                             param_distributions=param_distributions,
#                             scoring='neg_log_loss',
#                             verbose=1, n_jobs=-1,
#                             n_iter=15, cv=5) # total 32 combinations

# random_result = random.fit(train_X, train_Y)
# print('Best Score: ', random_result.best_score_)
# print('Best Params: ', random_result.best_params_)
# tuned_hyperparameters = random_result.best_params_
# clf_AdaBoost_tuned = AdaBoostClassifier(random_state=0, base_estimator = tuned_hyperparameters['base_estimator'], n_estimators = tuned_hyperparameters['n_estimators'], 
#  learning_rate = tuned_hyperparameters['learning_rate'])
# clf_AdaBoost_tuned.fit(train_X,train_Y)
# test_Y = clf_AdaBoost_tuned.predict_proba(test_X)
# test_Y = np.column_stack((test_X_listing_id,test_Y ))
# np.savetxt(r"C:\CMPT459_DataMining\Project\milestone3\test.json\test_Y_AdaBoost_afterParameterTuning.csv", test_Y, delimiter=',')



################################################### feature selection Phase 2 ##############################################



######################## recursive feature elimination #######################
############## Random Forest #########################
## feature selection

criterion = 'entropy'
n_estimators = 100
# max_features = 50
oob_score = False

clf_RF_tuned = RandomForestClassifier(random_state=0, criterion = criterion, n_estimators = n_estimators, oob_score = oob_score)#, max_features = max_features


rfecv = RFECV(estimator=clf_RF_tuned, step=1, cv=5, scoring='neg_log_loss', verbose=2, n_jobs=-1)
rfecv.fit(train_X, train_Y)

print("Optimal number of features : %d" % rfecv.n_features_)
np.save(r"C:\CMPT459_DataMining\Project\milestone3\train.json\rfecv_support_RF.npy", rfecv.support_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

## train with selected features

feature_support = np.load(r"C:\CMPT459_DataMining\Project\milestone3\train.json\rfecv_support_RF.npy")
train_X = train_X[:,feature_support]
test_X = test_X[:,feature_support]

criterion = 'entropy'
n_estimators = 500
max_features = 50
oob_score = False

clf_RF_tuned = RandomForestClassifier(random_state=0, criterion = criterion, n_estimators = n_estimators, oob_score = oob_score, max_features = max_features)

scores_RF = cross_val_score(clf_RF_tuned, train_X, train_Y, cv = 5, scoring='neg_log_loss')
print("RF with tuning with feature selection score = ", statistics.mean(scores_RF))

clf_RF_tuned.fit(train_X,train_Y)
test_Y = clf_RF_tuned.predict_proba(test_X)
test_Y = np.column_stack((test_X_listing_id,test_Y ))
np.savetxt(r"C:\CMPT459_DataMining\Project\milestone3\test.json\test_Y_RF_afterParameterTuning_deepTrees_featureSelected.csv", test_Y, delimiter=',')
########################################### Using Classification Accuracy #####################################################
########################## without tuning ####################################


############ Random Forest ###########
clf_RF = RandomForestClassifier(random_state=0)
scores_DT = cross_val_score(clf_RF, train_X, train_Y, cv = 5, scoring='accuracy')
print("DT without tuning score = ", statistics.mean(scores_DT))

############### with tuning #############

########### Random Forest ###########

criterion = ['entropy']
max_features = [10, 20, 30, 40, 50, 60, 70, 79]
n_estimators = [100, 200, 500]
oob_score = [False]

param_distributions = dict(max_features = max_features,
                           criterion = criterion,
                           n_estimators = n_estimators,
                           oob_score = oob_score)

clf_RF = RandomForestClassifier(random_state=0)

random = RandomizedSearchCV(estimator=clf_RF,
                            param_distributions=param_distributions,
                            scoring='accuracy',
                            verbose=1, n_jobs=-1,
                            n_iter=10, cv=5) ## total combinations 24

random_result = random.fit(train_X, train_Y)
print('Best Score: ', random_result.best_score_)
print('Best Params: ', random_result.best_params_)

quit()
