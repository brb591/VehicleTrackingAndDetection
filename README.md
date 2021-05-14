# Vehicle Tracking and Detection
## DGMD E-17 - Brian Bauer

## Histogram of Oriented Gradients (HOG)
Explain how (and identify where) you extracted HOG features from the training images.
Explain how you settled on your final choice of HOG parameters.
Describe how (and identify where) you trained a classifier using your selected HOG features (and color features if you used them)

I created a python program ```train_classifier.py``` that takes a single command - the color space to use.  It starts by generating a list of all car images and non-car images.  Each image is transformed into the desired color space and the features are extracted (described above).  Each car image is labeled with a 1 and the non-car images are labeled as 0.

The features and labels are shuffled and split into a training data set and a testing data set.  The classifier is created and trained using the training data, and then is tested using the testing data.

Both the classifier and the scaler are saved for use by the vehicle tracking python program.

I tried training the classifier using all of the color spaces.  The accuracy of each is as follows:
- RGB: 0.9836177474402731
- HSV: 0.9849829351535836
- **LUV: 0.990443686006826**
- HLS: 0.9870307167235495
- YUV: 0.9890784982935154
- YCrCb: 0.9890784982935154

Because of the higher accuracy, I chose to use the LUV color space.

## Sliding Window Search
Describe how (and identify where) you implemented a sliding window search.
How did you decide what scales to search and how much to overlap windows?
Show some examples of test images to demonstrate your pipeline is working.
How did you optimize the performance of your classifier?

## Video Implementation
Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video.
Describe how (and identify where) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

## Discussion
Briefly discuss any problems/issues you faced in your implementation of this project.
Where will your pipeline likely fail?
What could you do to make it more robust?


### Goal
Write a software pipeline to identify vehicles from a front-facing camera on a car.

### Notes on Tips and Tricks
1. Extract HOG features just once for a region of interest
1. Make sure images are scaled correctly
    - All images read during training and classifying are being read in the 0 to 255 scale
1. Be sure to normalize training data
1. Random shuffling of data
