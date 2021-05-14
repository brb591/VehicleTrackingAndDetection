import cv2
import os
import glob
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from skimage.feature import hog
import common

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
   
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] ==  None:
        y_start_stop[0] = img.shape[0] // 2
    if y_start_stop[1] == None:
        y_start_stop[1] = int(img.shape[0] * 0.9)
    
    window_list = []
    image_width_x = x_start_stop[1] - x_start_stop[0]
    image_width_y = y_start_stop[1] - y_start_stop[0]
     
    windows_x = np.int(1 + (image_width_x - xy_window[0]) / (xy_window[0] * xy_overlap[0]))
    windows_y = np.int(1 + (image_width_y - xy_window[1]) / (xy_window[1] * xy_overlap[1]))
    
    for i in range(0, windows_y):
        y_start = y_start_stop[0] + np.int(i * xy_window[1] * xy_overlap[1])
        for j in range(0, windows_x):
            x_start = x_start_stop[0] + np.int(j * xy_window[0] * xy_overlap[0])
            x1 = np.int(x_start +  xy_window[0])
            y1 = np.int(y_start + xy_window[1])
            window_list.append(((x_start, y_start), (x1, y1)))
    return window_list

def find_cars(image, windows, color_space):
    car_seen_windows = []
    for window in windows:
        start = window[0]
        end = window[1]
        cropped_image = image[start[1]:end[1], start[0]:end[0]]
        
        if(cropped_image.shape[1] == cropped_image.shape[0] and cropped_image.shape[1] != 0):
            cropped_image = cv2.resize(cropped_image, (64, 64))
            features = common.extract_features(cropped_image, color_space)
            normalized_features = feature_scaler.transform([features])
        
            prediction = svc_classifier.predict(normalized_features)
            if(prediction == 1):
                car_seen_windows.append(window)
                
    return car_seen_windows

#
#   Extract all HOG features at once, and then
#
def find_cars3(image, windows, cspace):
  car_seen_windows = []

  top_left = windows[0][0]
  bottom_right = windows[len(windows)-1][1]

  region_of_interest = image[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]

  nxblocks = (region_of_interest.shape[1] // 8) - 1
  nyblocks = (region_of_interest.shape[0] // 8) - 1
  nxsteps = (nxblocks - 7) // 2
  nysteps = (nyblocks - 8) // 2

  ch0 = common.extract_hog_features(region_of_interest[:,:,0])
  ch1 = common.extract_hog_features(region_of_interest[:,:,1])
  ch2 = common.extract_hog_features(region_of_interest[:,:,2])

  for xb in range(nxsteps):
    for yb in range(nysteps):
      ypos = yb*2
      xpos = xb*2

      f0 = ch0[ypos:ypos+7, xpos:xpos+7]
      f1 = ch1[ypos:ypos+7, xpos:xpos+7]
      f2 = ch2[ypos:ypos+7, xpos:xpos+7]

      xleft = xpos*8
      ytop = ypos*8

      cropped_image = region_of_interest[ytop:ytop+64, xleft:xleft+64]

      spatial_binned_features = common.bin_spatial(cropped_image, color_space=cspace, size=(32, 32))
      _, _, _, _, hist_features = common.extract_color_histogram_features(cropped_image)
      hog_features = list()
      hog_features.append(np.ravel(f0))
      hog_features.append(np.ravel(f1))
      hog_features.append(np.ravel(f2))
      raveled_hog = np.ravel(hog_features)
      all_features = np.concatenate((spatial_binned_features, hist_features, raveled_hog))

      normalized_features = feature_scaler.transform([all_features])
        
      prediction = svc_classifier.predict(normalized_features)
      if(prediction == 1):
        y0 = top_left[1] + ytop
        x0 = top_left[0] + xleft
        window = ((x0, y0), (x0+64, y0+64))
        car_seen_windows.append(window)
              
  return car_seen_windows

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
  imcopy = np.copy(img)
  for bbox in bboxes:
      cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)

  return imcopy

def add_heat(heatmap, bbox_list):
  for box in bbox_list:
    # Add += 1 for all pixels inside each bbox
    # Assuming each "box" takes the form ((x1, y1), (x2, y2))
    heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

  # Return updated heatmap
  return heatmap

def draw_labeled_bboxes(img, labels):
  # Iterate through all detected cars
  for car_number in range(1, labels[1]+1):
    # Find pixels with each car_number label value
    nonzero = (labels[0] == car_number).nonzero()
    # Identify x and y values of those pixels
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Define a bounding box based on min/max x and y
    bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
    # Draw the box on the image
    cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
  # Return the image
  return img

# applying a threshold value to the image to filter out low pixel cells

def apply_threshold(heatmap, threshold):
  # Zero out pixels below the threshold
  heatmap[heatmap <= threshold] = 0
  # Return thresholded map
  return heatmap
  
#
# Process a single image or a frame of video
#
#   Find all the windows that the classifier has predicted
#     contain cars.  Than apply the heat map and draw
#     a rectangle around the identified cars.
#
def process_image(image, color_space):
  windows_with_cars = find_cars(image, slide_window(image), color_space)
  heat = np.zeros_like(image[:,:,0]).astype(np.float)
  heat = add_heat(heat, windows_with_cars)
  heat = apply_threshold(heat, 2)
  heatmap = np.clip(heat, 0, 255)
  labels = label(heatmap)
  return draw_labeled_bboxes(np.copy(image), labels)

#
# handle_image runs the pipeline on a single, undistorted image
#
#
def handle_image(fileName, output_dir, color_space):
  # Get the image
  image = cv2.imread(fileName)
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  processed_image = process_image(image_rgb, color_space)
  
  if output_dir is None:
    plt.imshow(processed_image)
    print('Press any key to dismiss')
    plt.show()
  else:
    # Save the image to the output_dir
    output_file = os.path.join(output_dir, os.path.basename(fileName))
    print('Saving image to %s'%output_file)
    cv2.imwrite(output_file, processed_image)
    print('Done.')

#
# handle_video runs the pipeline on each undistorted frame of a video file
#
#
def handle_video(fileName, color_space):
  # Create a VideoCapture object and read from input file
  # If the input is the camera, pass 0 instead of the video file name
  cap = cv2.VideoCapture(fileName)

  # Check if camera opened successfully
  if (cap.isOpened()== False): 
    print("Error opening video stream or file")

  print('Press q to dismiss or wait for the video to end')
  # Read until video is completed
  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
      #Process the frame
      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      processed_frame = process_image(frame_rgb, color_space)
      frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

      # Display the processed frame in a window
      cv2.imshow('Lane Lines',frame_bgr)
      # Press Q on keyboard to  exit
      if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    # Break the loop
    else: 
      break

  # When everything done, release the video capture object
  cap.release()


#
# main function
#
#
#
# Create the parser
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', nargs='+', type=str, help='The input file(s)')
parser.add_argument('-o', '--output', nargs='?', type=str, help='The output directory')
parser.add_argument('-c', '--color', nargs='?', type=str, help='The color space')

# Get the arguments
args = parser.parse_args()

if args.color is None:
  color_space = 'RGB'
else:
  color_space = args.color


with open('feature_scaler'+color_space+'.pkl', 'rb') as fid:
  feature_scaler = pickle.load(fid)

with open('classifier'+color_space+'.pkl', 'rb') as fid:
  svc_classifier = pickle.load(fid)

# Process the test image(s)
for input_file in args.input:
  # Determine the file type
  filename, extension = os.path.splitext(input_file)

  if extension == ".jpg":
    print("Processing Image " + input_file)
    handle_image(input_file, args.output, color_space)
  elif extension == ".png":
    print("Processing Image " + input_file)
    handle_image(input_file, args.output, color_space)
  elif extension == ".mp4":
    print("Processing Video " + input_file)
    handle_video(input_file, color_space)
  else:
    print("Unknown extension: " + extension)


# Closes all the frames
cv2.destroyAllWindows()