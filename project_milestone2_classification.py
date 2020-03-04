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
train_X = sparse.load_npz(r"C:\CMPT459_DataMining\Project\milestone2\train.json\train_X_preprocessed.npz")
test_X = sparse.load_npz(r"C:\CMPT459_DataMining\Project\milestone2\test.json\test_preprocessed.npz")
[train_X, test_X] = PCADescription_and_toDense(train_X, test_X, train_Y)
[train_X, test_X] = Feature_Selection_mutual_information(train_X, test_X, train_Y)


np.save(r"C:\CMPT459_DataMining\Project\milestone2\train.json\train_X_preprocessed.npy", train_X)
np.save(r"C:\CMPT459_DataMining\Project\milestone2\test.json\test_X_preprocessed.npy", test_X)


##############################################################################################################
################################### End second stage of pre-processing #######################################

########################################## Begin Classification ##############################################
##############################################################################################################
train_X = np.load(r"C:\CMPT459_DataMining\Project\milestone2\train.json\train_X_preprocessed.npy")
test_X = np.load(r"C:\CMPT459_DataMining\Project\milestone2\test.json\test_X_preprocessed.npy")

########################################## First Attempt with default values ##########################
test_X_milestone1 = sparse.load_npz(r"C:\CMPT459_DataMining\Project\milestone2\test.json\test_preprocessed.npz").toarray()
test_X_listing_id = (test_X_milestone1[:,5].astype(int))

########### Decision Tree ###########
clf_DT = DecisionTreeClassifier(random_state=0)
scores_DT = cross_val_score(clf_DT, train_X, train_Y, cv = 5, scoring='log_loss')
print("DT without tuning score = ", statistics.mean(scores_DT))
clf_DT.fit(train_X,train_Y)
test_Y = clf_DT.predict_proba(test_X)
test_Y = np.column_stack((test_X_listing_id,test_Y ))
np.savetxt(r"C:\CMPT459_DataMining\Project\milestone2\test.json\test_Y_DT_firstAttempt.csv", test_Y, delimiter=',')

############ Logistic regression ###########
clf_LR = LogisticRegression(random_state=0)#.fit(train_X, train_Y)
scores_LR = cross_val_score(clf_LR, train_X, train_Y, cv = 5, scoring='neg_log_loss')     
print("LR without tuning score = ", statistics.mean(scores_LR))     
clf_LR.fit(train_X,train_Y)   
test_Y = clf_LR.predict_proba(test_X)
test_Y = np.column_stack((test_X_listing_id,test_Y ))
np.savetxt(r"C:\CMPT459_DataMining\Project\milestone2\test.json\test_Y_LR_firstAttempt.csv", test_Y, delimiter=',')

# ########### SVM ###########
# # train_X = train_X[:1000,:]
# # train_Y = train_Y[:1000]
# clf_SVC = svm.SVC(probability = True)
# scores_SVC = cross_val_score(clf_SVC, train_X, train_Y, cv = 5, scoring='neg_log_loss')
# clf_SVC.fit(train_X,train_Y)
# test_Y = clf_SVC.predict_proba(test_X)
# test_Y = np.column_stack((test_X_listing_id,test_Y ))
# np.savetxt(r"C:\CMPT459_DataMining\Project\milestone2\test.json\test_Y_SVM_firstAttempt.csv", test_Y, delimiter=',')

########################################## Hyperparameter tuning ####################################

########### Decision Tree ###########
criterion = ['gini', 'entropy']
max_depths = list(range(1,train_X.shape[1]+1))
min_samples_leaf = np.linspace(0.1, 0.5, 9, endpoint=True)
max_features = list(range(1,train_X.shape[1]+1))
min_samples_split = np.linspace(0.01, 1.0, 100, endpoint=True)

param_distributions = dict(max_depth = max_depths,
                           max_features = max_features,
                           criterion = criterion,
                           min_samples_split = min_samples_split,
                           min_samples_leaf = min_samples_leaf)

clf_DT = DecisionTreeClassifier(random_state=0)

random = RandomizedSearchCV(estimator=clf_DT,
                            param_distributions=param_distributions,
                            scoring='neg_log_loss',
                            verbose=1, n_jobs=-1,
                            n_iter=1000, cv=5)

random_result = random.fit(train_X, train_Y)
print('Best Score: ', random_result.best_score_)
print('Best Params: ', random_result.best_params_)
tuned_hyperparameters = random_result.best_params_
clf_DT_tuned = DecisionTreeClassifier(random_state=0, max_depth = tuned_hyperparameters['max_depth'], max_features = tuned_hyperparameters['max_features'], 
 criterion = tuned_hyperparameters['criterion'], min_samples_split = tuned_hyperparameters['min_samples_split'], 
 min_samples_leaf = tuned_hyperparameters['min_samples_leaf'])
clf_DT_tuned.fit(train_X,train_Y)
test_Y = clf_DT_tuned.predict_proba(test_X)
test_Y = np.column_stack((test_X_listing_id,test_Y ))
np.savetxt(r"C:\CMPT459_DataMining\Project\milestone2\test.json\test_Y_DT_afterParameterTuning.csv", test_Y, delimiter=',')


########### Logistic regression ###########
solver = ['liblinear', 'saga']
C = np.logspace(0, 4, 50)
penalty = ['l1', 'l2']

param_distributions = dict(penalty = penalty,
                           solver = solver,
                           C = C)

clf_LR = LogisticRegression(random_state=0)

random = RandomizedSearchCV(estimator=clf_LR,
                            param_distributions=param_distributions,
                            scoring='neg_log_loss',
                            verbose=1, n_jobs=-1,
                            n_iter=100, cv=5)

random_result = random.fit(train_X, train_Y)
print('Best Score: ', random_result.best_score_)
print('Best Params: ', random_result.best_params_)
tuned_hyperparameters = random_result.best_params_
clf_LR_tuned = LogisticRegression(random_state=0, penalty = tuned_hyperparameters['penalty'], solver = tuned_hyperparameters['solver'], 
 C = tuned_hyperparameters['C'])
clf_LR_tuned.fit(train_X,train_Y)
test_Y = clf_LR_tuned.predict_proba(test_X)
test_Y = np.column_stack((test_X_listing_id,test_Y ))
np.savetxt(r"C:\CMPT459_DataMining\Project\milestone2\test.json\test_Y_LR_afterParameterTuning.csv", test_Y, delimiter=',')

################################################### feature selection SFFS ##############################################

########################## Decision Tree ################
criterion = 'entropy'
max_depth = 50
min_samples_leaf = 0.1
# max_features = 61
min_samples_split = 0.2

clf_DT_tuned = DecisionTreeClassifier(random_state=0, max_depth = max_depth, criterion = criterion, min_samples_split = min_samples_split, 
 min_samples_leaf = min_samples_leaf)

sfs_DT = sfs(clf_DT_tuned,
              k_features= 5, #train_X.shape[1],
              forward=True,
              floating=False,
              verbose=2,
              scoring='neg_log_loss',
              cv=5)

sfs_DT = sfs_DT.fit(train_X, train_Y)

########################## Logistic Regression ################
# solver = 'liblinear'
# C = 11.5
# penalty = 'l1'


# clf_LR_tuned = LogisticRegression(random_state=0, penalty = penalty, solver = solver, C = C)

# sfs_LR = sfs(clf_LR_tuned,
#               k_features= train_X.shape[1],
#               forward=True,
#               floating=False,
#               verbose=2,
#               scoring='neg_log_loss',
#               cv=5)

# sfs_LR = sfs_LR.fit(train_X, train_Y)


######################## recursive feature elimination #######################
############## Decision Tree #########################
criterion = 'entropy'
max_depth = 50
min_samples_leaf = 0.1
# max_features = 61
min_samples_split = 0.2

clf_DT_tuned = DecisionTreeClassifier(random_state=0, max_depth = max_depth, criterion = criterion, min_samples_split = min_samples_split, 
 min_samples_leaf = min_samples_leaf)



rfecv = RFECV(estimator=clf_DT_tuned, step=1, cv=5, scoring='neg_log_loss', verbose=2, n_jobs=-1)
rfecv.fit(train_X, train_Y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

############## Logistic Regression ####################
solver = 'liblinear'
C = 11.5
penalty = 'l1'


clf_LR_tuned = LogisticRegression(random_state=0, penalty = penalty, solver = solver, C = C)


rfecv = RFECV(estimator=clf_LR_tuned, step=1, cv=5, scoring='neg_log_loss', verbose=2, n_jobs=-1)
rfecv.fit(train_X, train_Y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

########################################### Using Classification Accuracy #####################################################
########################## without tuning ####################################
############ Logistic regression ###########
clf_LR = LogisticRegression(random_state=0)
scores_LR = cross_val_score(clf_LR, train_X, train_Y, cv = 5, scoring='accuracy')     
print("LR without tuning score = ", statistics.mean(scores_LR))     

############ Decision Tree ###########
clf_DT = DecisionTreeClassifier(random_state=0)
scores_DT = cross_val_score(clf_DT, train_X, train_Y, cv = 5, scoring='accuracy')
print("DT without tuning score = ", statistics.mean(scores_DT))

############### with tuning #############

########### Decision Tree ###########
criterion = ['gini', 'entropy']
max_depths = list(range(1,train_X.shape[1]+1))
min_samples_leaf = np.linspace(0.1, 0.5, 9, endpoint=True)
max_features = list(range(1,train_X.shape[1]+1))
min_samples_split = np.linspace(0.01, 1.0, 100, endpoint=True)

param_distributions = dict(max_depth = max_depths,
                           max_features = max_features,
                           criterion = criterion,
                           min_samples_split = min_samples_split,
                           min_samples_leaf = min_samples_leaf)

clf_DT = DecisionTreeClassifier(random_state=0)

random = RandomizedSearchCV(estimator=clf_DT,
                            param_distributions=param_distributions,
                            scoring='accuracy',
                            verbose=1, n_jobs=-1,
                            n_iter=1000, cv=5)

random_result = random.fit(train_X, train_Y)
print('Best Score: ', random_result.best_score_)
print('Best Params: ', random_result.best_params_)


########### Logistic regression ###########

solver = ['liblinear', 'saga']
C = np.logspace(0, 2, 10)
penalty = ['l1', 'l2']

param_distributions = dict(penalty = penalty,
                           solver = solver,
                           C = C)

clf_LR = LogisticRegression(random_state=0)

random = RandomizedSearchCV(estimator=clf_LR,
                            param_distributions=param_distributions,
                            scoring='accuracy',
                            verbose=1, n_jobs=-1,
                            n_iter=10, cv=5)

random_result = random.fit(train_X, train_Y)
print('Best Score: ', random_result.best_score_)
print('Best Params: ', random_result.best_params_)

quit()
