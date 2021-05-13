import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import pickle
import common

images_path = './vehicle-detection/vehicles/'
car_images_paths = list()
non_car_images_paths = list()
for (dirpath, dirnames, filenames) in os.walk(images_path):
    car_images_paths += [os.path.join(dirpath, file) for file in filenames if file.endswith('.png')]

images_path = 'vehicle-detection/non-vehicles/'
for (dirpath, dirnames, filenames) in os.walk(images_path):
    non_car_images_paths += [os.path.join(dirpath, file) for file in filenames if file.endswith('.png')]

print("Car images found: " + str(len(car_images_paths)))
print("Non-car images found: " + str(len(non_car_images_paths)))

def images_features(image_paths):
    features=  []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_features = common.extract_features(image_rgb)
        features.append(image_features)
    return features


def combine_all_images_features():
    car_features = images_features(car_images_paths)
    non_car_features = images_features(non_car_images_paths)
    return car_features, non_car_features

print("Combining all features")
car_features, non_car_features = combine_all_images_features()
print("Shape of car features list is " + str(len(car_features)))
print("Shape of non-car features list is " + str(len(non_car_features)))
print("Length of features is " + str(len(car_features[0])))

all_features = np.vstack([car_features, non_car_features])
print("Shape of features is ", all_features.shape)
all_labels = np.concatenate([np.ones(len(car_features)), np.zeros(len(non_car_features))])
print("Shape of label is ", all_labels.shape)

features_train, features_test, labels_train, labels_test = train_test_split(all_features, all_labels, test_size=0.2, shuffle=True)
print("Shape of training features is ", features_train.shape)
print("Shape of test features is ", features_test.shape)

scaler = StandardScaler()
feature_scaler = scaler.fit(features_train)
print("Saving the feature scaler for later use")
with open('feature_scaler.pkl', 'wb') as fid:
    pickle.dump(feature_scaler, fid) 
features_train_scaled = feature_scaler.transform(features_train)
features_test_scaled = feature_scaler.transform(features_test)

print(len(features_test_scaled[0]))

# Train a SVM classifer
print("Training SVM Classifier")
svc_classifier = LinearSVC(max_iter=100000)

svc_classifier.fit(features_train_scaled, labels_train)
print("Accuracy of SVC is  ", svc_classifier.score(features_test_scaled, labels_test))

print('Prediction: ', svc_classifier.predict(features_test_scaled[0:20]))
print('Labels    : ', labels_test[:20])

print("Saving the classifier as a pickle file for later use")
with open('classifier.pkl', 'wb') as fid:
    pickle.dump(svc_classifier, fid)  