# Vehicle Tracking and Detection
## DGMD E-17 - Brian Bauer

## Histogram of Oriented Gradients (HOG)
The SciKit Image module contains a method for extracting HOG features from an image.  There are a number of parameters that are provided.  First, the number of orientations was set to 9 for this project.  The HOG method groups the gradient directions into 9 histogram buckets, each covering 40 degrees of the circle.  The number of pixels per cell was set to 8, with 2 cells per block.  Since the training data was 64 by 64 pixel images, that means that each image consists of 4 square blocks.  This choice in parameters should allow for further refinement and optimization of this portion of the algorithm, as each block can be divided in half multiple times and still be squares.

In addition to the HOG features, I create features using the image itself.  The NumPy ravel function flattens a 2-dimensional array into a single dimension, and this is used to turn the RGB/HSV/etc values into an array.  I also construct histograms of the color channels of the images and concatenate them together.  The combination of these two feature sets seems to help the classifier understand the colors that are most commonly seen on cars yet not on roads.

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

## Run The Vehicle Tracking Code
```
python3 vehicle_tracking.py -i <input file (image or video)> -c <color space (RGB is default)> -o <output file>
```

## Train The Classifier
```
python3 train_classifier.py <color space>
```

## Discussion
Briefly discuss any problems/issues you faced in your implementation of this project.
Where will your pipeline likely fail?
What could you do to make it more robust?