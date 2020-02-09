import numpy as np
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import json
from PIL import Image, ImageStat
import requests
from io import BytesIO
#from OpenSSL import SSL
import seaborn as sns
import os
from cv2 import cv2
from skimage.io import imread as imread_sk
from skimage.io import imshow as imshow_sk
from skimage.filters import prewitt_h,prewitt_v
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure

workind_dir = r'C:\CMPT459_DataMining\Project\two-sigma-connect-rental-listing-inquiries\images_sample\images_sample'
os.chdir(workind_dir)

def extractStatFeatures(url):
    img = getImage(url)
    return ImageStat.Stat(img)

def getImage_sk(url, asGray):
    [dirName, fileName] = parseUrl(url)
    if (int(dirName) < 6812267):
        img = imread_sk(os.path.join(workind_dir, dirName, fileName), as_gray=asGray)
    # else:
    #     response = requests.get(url)
    #     img = Image.open(BytesIO(response.content))
    return img

def getImage_cv2(url, mode):
    [dirName, fileName] = parseUrl(url)
    if (int(dirName) < 6812267):
        img = cv2.imread(os.path.join(workind_dir, dirName, fileName), mode)
    # else:
    #     response = requests.get(url)
    #     img = Image.open(BytesIO(response.content))
    return img

def getImage(url):
    [dirName, fileName] = parseUrl(url)
    if (int(dirName) < 6812267):
        img = Image.open(os.path.join(workind_dir, dirName, fileName))
    else:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
    return img

def parseUrl(url):
    indexOfLastSlash = url.rfind('/')
    imageLongName = url[indexOfLastSlash+1:]
    imageLongNameArr = imageLongName.split('_')
    dirName = imageLongNameArr[0]
    fileName = imageLongNameArr[0] + '_' + imageLongNameArr[1]
    return dirName, fileName

def extractSkFeatures(url, asGray):
    img = getImage_sk(url, asGray) # Get image as_gray=False

    hor = prewitt_h(img)
    ver = prewitt_v(img)
    
    img = getImage_sk(url, False) # If you wanna use gray-scale image, remove multichannel=True below; otherwise it errors out
    # resized_img = resize(img, (426, 640)) # Multiplication of 16 (8*2 look at below)
    #creating hog features 
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
    
    return hor, ver, fd, hog_image 

def extractCv2Features(url, mode):
    img = getImage_cv2(url, mode)

    detector = cv2.KAZE_create()
    kp = detector.detect(img)
    kp, des = detector.compute(img, kp)

    return kp, des, img

def extractColorHistogramFeatures(url, mask_center):
    img = getImage_cv2(url, 4)

    if (mask_center):
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[100:(img.shape[0]-100), 100:(img.shape[1]-100)] = 255
        img = cv2.bitwise_and(img,img,mask = mask)

    hist_red = cv2.calcHist([img],[2],None,[256],[0,256])
    hist_blue = cv2.calcHist([img],[1],None,[256],[0,256])
    hist_green = cv2.calcHist([img],[0],None,[256],[0,256])

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist_gray = cv2.calcHist([img_gray],[0],None,[256],[0,256])


    return hist_red, hist_blue, hist_green, hist_gray

def extract_text_features(train_data, feature_name):
    porter = PorterStemmer()
    if (type(train_data[feature_name].values[0]) == list): # 'features'

        ### number of listed features can be used as an extracted feature

        num_features = train_data[feature_name].apply(len)

        ### These lines stem the words in features list and join the words in each listeed feature with '_'
        
        features_joined = []
        for feat_list in train_data[feature_name]:
            features_stemmed = []
            for feat in feat_list:
                stemmed_words = []
                for word in feat.split(" "):
                    stemmed_words.append(porter.stem(word))
                features_stemmed.append("_".join(stemmed_words))
            features_joined.append(" ".join(features_stemmed))

    else: # 'description'
        num_features = (train_data[feature_name].apply(lambda x: len(x.split(" "))))

        features_joined = []
        for feat in train_data[feature_name]:
            stemmed_words = []
            for word in feat.split(" "):
                stemmed_words.append(porter.stem(word))
            features_joined.append(" ".join(stemmed_words))



    #### Now extraxt features ###################
    tfidf = CountVectorizer(stop_words='english', max_df = 0.50, min_df = 0.03) ## if max_df and min_df is not determined, 36701 terms are selected in the vocabulary. After stricting it to 0.50 and 0.05, it was reduced to 338
    word_count_vector = tfidf.fit_transform(features_joined) 
    tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)  ## forms the IDF (dictionary with all terms in all documents with the score of each term)
    tf_idf_vector=tfidf_transformer.transform(word_count_vector) ## finds score of each word in each document

    return num_features, tf_idf_vector

