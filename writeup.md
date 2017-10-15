**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### Project Structure

I built the project such that I had a main project file `veh_det_main.py` and a separate project file `vehicle_det_main_funcs.py` which contained the majority of my functions that the main file called. 

### Histogram of Oriented Gradients (HOG)

The code for this step is contained in lines 16 -33 of my `vehicle_det_main_funcs.py` file.

I started by reading in all the `vehicle` and `non-vehicle` images in my main function.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

[car_vs_notcar](./writeup_imgs/car_vs_notcar.png)

I then explored different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:
[HOG1](./writeup_imgs/hog_play1.png)
[HOG2](./writeup_imgs/hog_play2.png)
[HOG3](./writeup_imgs/hog_play3.png)

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters but decided to stick with the default parameters as they were giving me pretty good results later down in the pipeline.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using all features that the pipeline provided (Spatial, Histogram and Hog features). This gave me a pretty large feature vector.

###Sliding Window Search

After I extracted my features to train my SVM on from the pretty small pictures, I moved on to car detection in larger images. For this, I implemented a sliding window search in my `find_cars` function in lines 116 to 126 in my `vehicle_det_main_funcs.py` file. At first I only worked on the 1.5 scale but I found that to not be optimal so I implemented multiple searches on 1, 1.5, and 2.0 scales.

Ultimately I searched on three scales using a RGB 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

[Bounding_boxes](./writeup_imgs/bounding_boxes.png)
---

### Video Implementation

Here's a [link to my video result](./project_video_out.mp4)

### Filtering & Heat Maps

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of test images (right), the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid (left):

### Here are six images and their corresponding heatmaps and bounding boxes:

[Heat_map](./writeup_imgs/heat_map.png)

---

###Discussion

####1. Problems / issues you faced in the implementation  

One issue I encountered at first was that the algorithm kept identifying many false positives. I addressed this by tweaking the parameters, training on the full image set (which took substantially longer), by limiting the search region (to the lower half) and lastly using different scales to achieve better overlaps when a car is identified. I also spent a lot of time, making sure I am in the right color space considering how the images were imported. Ultimately, I got rid of all colorspace conversions and just worked in RGB/BGR.

I think the approach was pretty good. However, I have noticed that dark colors in cars tend to make the algorithms less robust and also one way to improve the algorithm would be to spend some time smoothing the boxes by giving the bounding box sizes some history and averaging. 
