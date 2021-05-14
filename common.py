import cv2
import numpy as np
from skimage.feature import hog

# HOG Features
pix_per_cell = 8
cell_per_block = 2
orient = 9

def extract_hog_features(img, visualize=False):
    return hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  visualize=visualize,
                                  feature_vector=False,
                                  block_norm="L2-Hys")

def extract_color_histogram_features(img, nbins=32, bins_range=(0, 255)):
    rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges)-1]) / 2 

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features

def bin_spatial(img, color_space='RGB', size=(32, 32)):
  # Convert image to new color space (if specified)
  if color_space != 'RGB':
    if color_space == 'HSV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif color_space == 'LUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif color_space == 'YUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif color_space == 'YCrCb':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
  else:
        feature_image = np.copy(img)             
  
  # Use cv2.resize().ravel() to create the feature vector
  features = cv2.resize(feature_image, size).ravel() 
  
  # Return the feature vector
  return features

def extract_features(img, color_space='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256)):
    spatial_binned_features = bin_spatial(img, color_space=color_space, size=spatial_size)
    rhist, ghist, bhist, bin_centers, hist_features = extract_color_histogram_features(img)
    hog_features = list()
    for channel in range(img.shape[2]):
        channel_hog_features = extract_hog_features(img[:,:,channel])
        hog_features.append(np.ravel(channel_hog_features))
    raveled_hog = np.ravel(hog_features)
    all_features = np.concatenate((spatial_binned_features, hist_features, raveled_hog))
    return all_features
