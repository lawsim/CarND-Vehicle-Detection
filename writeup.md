**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[1_1_non_vehicle]: ./writeup_images/1_1_non_vehicle.png
[1_1_vehicle]: ./writeup_images/1_1_vehicle.png
[2_1_testing_search_params]: ./writeup_images/2_1_testing_search_params.png
[2_2_testing_search_params]: ./writeup_images/2_2_testing_search_params.png
[3_1_foundheatmap]: ./writeup_images/3_1_foundheatmap.png
[3_2_foundheatmap]: ./writeup_images/3_2_foundheatmap.png

[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In the first 2 code cells of the IPython notebook I imported many of the needed libraries and read in the dataset using glob.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][1_1_vehicle]
![alt text][1_1_non_vehicle]

In the third cell I implemented many of the functions described and designed throughout the lesson.  I kept these as functions so that they could be re-used later when processing video.  

In the next cell I called the extract_features function created in the above cell to generate the feature vectors for the training data.  In my parameters I am extracting spatial, hist, and hog features on the data.  These are then returned as a numpy array.

Then the sklearn StandardScaler is used to normalize the data and finally I split the data into training and test data using sklearn's train_test_split function.

#### 2. Explain how you settled on your final choice of HOG parameters.

Most of the parameters were also set here which I re-used later in the project when searching for cars so that they matched what I trained on.  I experimented with various parameters before settling on the ones in my "Set parameters and split data" code cell in the notebook.  I went back to the parameter testing/tuning we did in the lesson as a starting point and tweaked from there.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the sklearn LinearSVC() class in the "Train classifier" code section of my notebook using the previously extracted and split features from the training data.



### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented in the find_cars function (see section 35 of lesson).  This function takes in your test image and parameters and performs a search based on the parameters (how I determined what to choose below).  I slightly modified this function to also return the bounding boxes so that they could be used in the multi-detection pipeline when processing video.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?


After this I did some basic eye testing of my classifier by running it against images in the "test_images" folder and against various combinations of y start/stop and scale as described in section 37 of the lesson.  I tweaked this to find 2 combinations of y ranges and scales that seemed to generate good results.

![alt text][2_1_testing_search_params]
![alt text][2_2_testing_search_params]

I ended up searching y from 400 to 520 pixels for scale 1.5 and from 400 to 680 pixels for scale 2.0.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

First I brought in the add_heat, apply_threshold, and draw_labeled_boxes functions from section 37 of the lesson. Then I created a single function, pipeline_find_heatmaps_and_draw which I could later feed into MoviePy to process the video.  This function processes each frame of video, runs it against find_cars and adds the found vehicles to the heat np array.  I use a global variable, last_n_frames_boxes, to keep track of the last 5 frames so I can run across multiple.  

This heatmap is then thresholded and the drawn image is returned.  I also implemented a "debug" variable which has it plot the found heatmaps so that I can evaluate them.

Here are some frames and their corresponding heatmaps:

![alt text][3_1_foundheatmap]
![alt text][3_2_foundheatmap]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I mostly followed the pipeline that we built during the lesson.  I modified a few things and added a way to keep track across multiple frames.

Most of the code from the class was fairly portable to this project so the issues I faced were primarily around getting the parameters to work correctly.  I experimented with various changes and read  how other people did on the forums/Slack before ending up where I did.

I could create more sliding windows/scales to make it more robust.  Also I believe this will fail fairly quickly on seeing new situations/data (almost by design).  I believe to make it more robust you could add more data, section off different parts of the image more.  Additional, perhaps there would be a way to combine some of the techniques here for feature extraction with the deep learning algorithms we used previously in the class to make something even more functional.
