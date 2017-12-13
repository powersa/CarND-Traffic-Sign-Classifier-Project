# **Traffic Sign Recognition**

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/class_example.png "Class Examples"
[image2]: ./examples/training_sample_bar.png "Training Samples"
[image3]: ./examples/validation_sample_bar.png "Validation Samples"
[image4]: ./examples/test_sample_bar.png "Testing Samples"
[image5]: ./examples/images_from_web.png "Web Images"
[image6]: ./examples/softmax_test.png "Softmax"

## Rubric Points
### In the following sections, I will describe how I addressed the [rubric points](https://review.udacity.com/#!/rubrics/481/view).  

---
### Writeup

**[project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)**
**[project code with outputs](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)**


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python to calculate the following dataset metrics:

* Size of training set = 34799
* Size of validation set =
* Size of testing set = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Exploratory Visualization

I plotted a randomly selected instance of each class in a simple layout. I found this representation to be a valuable reference during training and testing.

![alt text][image1]

It's important to understand class distributions in each set of data. If, for example, we found that the distribution of our training set differed significantly from our test set, we should be concerned.

Given that we don't have a uniform distribution of across classs, visualizations like this shed insight into performance of different classes.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I chose not to preprocess my data. Instead, I focused on increasing the predictive strength of my model by including [Batch Normalization](https://arxiv.org/abs/1502.03167) and [Inception Modules](https://arxiv.org/abs/1409.4842). If I had additional time (resources), I would have augmented the data set.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

In my final model, all convolutions are followed by batch normalization and a relu. The model contains the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x8 	|
| Convolution 1x1					|	1x1 stride, same padding, outputs 32x32x16						|
| Convolution 3x3	      	| 1x1 stride,  same padding, outputs 32x32x32 				|
| Inception Module A	    | 1x1, 3x3, 5x5, same padding, outputs 32x32x40|
| Convolution 3x3		| 2x2 stride, same padding, outputs 15x15x40        									|
| Inception Module B				| 1x1, 3x3, 5x5, same padding, outputs 15x15x52       									|
|	Convolution 3x3					|	2x2 stride, same padding, outputs 8x8x52											|
|	Inception Module C					|	1x1, 3x3, 5x5, same padding, outputs 8x8x88											|
|	Convolution 3x3					|	1x1 stride,  same padding, outputs 8x8x4											|
|	Fully-Connected					|	input 256, output 43											|
|	softmax					|	input 43, output 43											|


Each inception module has the following structure:
Stack 1:
Layer 1: 1x1 Convolution

Stack 2:
Layer 1: 1x1 Convolution
Layer 2: 3x3 Convolution

Stack 3:
Layer 1: 1x1 Convolution
Layer 2: 3x3 Convolution

The last layer in each stack is concatenated before the next phase of the model.




#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer, a batch size of 128, 30 epochs and a learning rate of 0.00175

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.974
* test set accuracy of 0.965

I started with a working LeNet architecture. In a relatively short amount of time, I was able to tune the hyperparameters to reach the desired accuracy on the validation set.

From there, I started to explore more recent literature about CNNs. I was particularly intrigued by [Inception Modules](https://arxiv.org/abs/1409.4842), [Batch Normalization](https://arxiv.org/abs/1502.03167) and [The All Convolutional Net](https://arxiv.org/abs/1412.6806). I started to incorporate concepts from each of these articles into my architecture.

I added Batch Normalization before all of activation functions.

I replaced pooling layers with downsampling convolutions.

I added 3 inception modules loosely based on the architecture described in GoogLeNet.

In the end, I probably tried too many new tweak to be truly affective with any one. As the size and complexity of my model increased, the training time did as well. In the end, the training set isn't all that large. While my model has decent accuracy on the validation and test sets, it's clearly overfit. To address this, I would reevaluate my model architecture and probably choose to add in some dropout or pooling layers. I would also augment the data set with random rotation to increase the size of my training data.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5]

11 and 25 are probably harder to classify. Many classes share that shape and color scheme. Additionally, they both contain interior shapes that are relatively complex.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Right of Way      		| Right of Way   									|
| No Entry     			| No Entry										|
| Stop					| Stop											|
| Priority Road      		| Priority Road				 				|
| Road Work			| Bicycle Crossing     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This isn't too disappointing given an accuracy of 96% on the test set. Perhaps with a larger sample of images from the web, this accuracy would increase.

However, it's worth calling out that the training, validation and testing images may have some common characteristics. Were they taken with the same type of camera? We they resized or cropped? Even though my model does little to no preprocessing, it's worth considering whether or not our training data is actually representative of the data we hope to classify.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

Here are the top 5 softmax probabilities for each image:

![alt text][image6]
