import os
import sys
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


#
#   Extract all features for all images in a list of image paths
#
#       Notes:
#           cv2.imread will read a png image on a scale of 0 to 255
#           matplotlib will read a png image on a scale of 0 to 1
#           matplotlib will read a jpg image on a scale of 0 to 255
def images_features(image_paths, color):
    features = []
    for image_path in image_paths:
        image = cv2.imread(image_path)

        # extract features will convert from RGB to the desired color space
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_features = common.extract_features(image_rgb, color_space=color)

        features.append(image_features)
    return features

#
#   Get the car and non-car features and return them
#
def combine_all_images_features(color_space):
    car_features = images_features(car_images_paths,color_space)
    non_car_features = images_features(non_car_images_paths,color_space)
    return car_features, non_car_features




#
#   Main processing starts here
#
#

color_space_arg = sys.argv[1]

#
#   Create a list of all vehicle images
#
images_path = './vehicle-detection/vehicles/'
car_images_paths = list()
non_car_images_paths = list()
for (dirpath, dirnames, filenames) in os.walk(images_path):
    car_images_paths += [os.path.join(dirpath, file) for file in filenames if file.endswith('.png')]

#
#   Create a list of all non-vehicle images
#
images_path = 'vehicle-detection/non-vehicles/'
for (dirpath, dirnames, filenames) in os.walk(images_path):
    non_car_images_paths += [os.path.join(dirpath, file) for file in filenames if file.endswith('.png')]

print("Car images found: " + str(len(car_images_paths)))
print("Non-car images found: " + str(len(non_car_images_paths)))


print("Combining all features")
car_features, non_car_features = combine_all_images_features(color_space_arg)

all_features = np.vstack([car_features, non_car_features])
all_labels = np.concatenate([np.ones(len(car_features)), np.zeros(len(non_car_features))])

features_train, features_test, labels_train, labels_test = train_test_split(all_features, all_labels, test_size=0.2, shuffle=True)

scaler = StandardScaler()
feature_scaler = scaler.fit(features_train)

print("Saving the feature scaler for later use")

with open('feature_scaler'+color_space_arg+'.pkl', 'wb') as fid:
    pickle.dump(feature_scaler, fid) 

# Scale both the training and testing features
features_train_scaled = feature_scaler.transform(features_train)
features_test_scaled = feature_scaler.transform(features_test)

# Train a SVM classifer
print("Training SVM Classifier")
svc_classifier = LinearSVC(max_iter=100000)
svc_classifier.fit(features_train_scaled, labels_train)

# Test the classifier
print("Accuracy of SVC is  ", svc_classifier.score(features_test_scaled, labels_test))

print('Prediction: ', svc_classifier.predict(features_test_scaled[0:20]))
print('Labels    : ', labels_test[:20])

print("Saving the classifier as a pickle file for later use")
with open('classifier'+color_space_arg+'.pkl', 'wb') as fid:
    pickle.dump(svc_classifier, fid)  