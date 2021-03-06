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
The code creates overlapping windows within the image, each of which is tested to see if there is a car present.  To improve performace, only the bottom half of each image is analyzed with the exception of the bottom 10% of the image which is ignored because of the likely presence of the hood of the drive car.  Because the training images were all 64 by 64 pixels, each window is also 64 by 64 pixels.  By setting the overlap to 50%, half of one image is combined with half of the next to create a whole image.

## Pipeline Testing Output
The directory ```test_output``` contains images that are the result of running the pipeline on test images that have been used throughout past class assignments.

## Optimizing Classifier
I attempted to optimize the classifier by extracting all HOG features at once and then extracting a subset array for each window that was processed.  This method needs further refinement to be good, as it appears to have a higher rate of false positives.  By adding ```-m optimized``` to the command instruction when using the vehicle tracking program, the optimized classifier is used.  Otherwise, the standard classifier is used.

## Video Implementation
The ```vehicle_tracking.py``` program doesn't save output videos, it only run them and displays the result on screen.  The performance of the optimized classifier does not appar to be worth it.

## Filtering and Reducing False Positives
The output of the classifier can include a lot of false positives, so there is a heatmap function to reduce them.  This works by taking a blank image and incrementing the pixel values on the blank image for each of the pixels in every window that is classified as having a car.

Once all "found" windows have been processed, a threshold is applied to the heatmap.  Only those pixels that have been marked as being "found" more than once are kept, reducing single instances of false positives.

## Run The Vehicle Tracking Code
```
python3 vehicle_tracking.py -i <input file (image or video)> -c <color space (RGB is default)> -o <output file> -m <optimized (default is standard)>
```

## Train The Classifier
```
python3 train_classifier.py <color space>
```

## Discussion
Finding combinations of parameters to feature extraction that resulted in improvements was a challenge.  I expected to see different results.  

The pipeline did not seem to work well with parts of the image that were not on the road.  While I specified a region of interest that excluded the top half of the image, the bottom half frequently contained trees and other shapes that included colors and gradients that matched with cars.  One way to improve this would be to use some of the lane tracking techniques to find the entire road and then limit the region of interest to that area.  This would be more difficult as the windowing techniques assume that the region of interest is a rectangle.