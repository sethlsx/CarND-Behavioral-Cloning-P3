**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run.mp4 for the successful running video

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I use the model introduced by Nvidia, which consists of 5 convolution neural network layers with 5x5 and 3x3 filter sizes and depths between 3 and 64 (model.py lines 99-141) 

The model also includes RELU layers to introduce nonlinearity (code line 99, 103, 107, 112, 116), and the data is normalized in the model using a Keras lambda layer (code line 96). 

The model consists 1 flatten layer and 4 full connected layer after the convolutional layers before output result.

The architecture of the model is as follows:
1. Cropping layer, input shape of (160, 320, 3), output shape of (90, 320, 3)
2. Lambda layer, output shape of (90, 320, 3)
3. Convolutional layer 1, kernel size of 3x3, valid padding, depth of 12, activation of RELU, output shape of (88, 318, 12)
4. Convolutional layer 2, kernel size of 3x3, valid padding, depth of 24, activation of RELU, output shape of (86, 316, 24)
5. Max pooling layer, pool size 2x2, output shape of (43, 158, 24)
6. Convolutional layer 3, kernel size of 3x3, valid padding, depth of 36, activation of RELU, output shape of (41, 156, 36)
7. Max pooling layer, pool size 2x2, output shape of (20, 78, 36)
8. Convolutional layer 4, kernel size of 3x3, valid padding, depth of 48, activation of RELU, output shape of (18, 76, 48)
9. Max pooling layer, pool size 2x2, output shape of (9, 38, 48)
10. Convolutional layer 5, kernel size of 3x3, valid padding, depth of 64, activation of RELU, output shape of (7, 36, 64)
11. Max pooling layer, pool size 2x2, output shape of (3, 18, 64)
12. Flatten layer, output shape of (3456, 1)
15. Full connected layer, output shape of (512, 1)
16. Full connected layer, output shape of (256, 1)
17. Full connected layer, output shape of (128, 1)
18. Full connected layer, output shape of (64, 1)
19. Output layer.



#### 2. Attempts to reduce overfitting in the model

The model contains l2 regularizers in each layer in order to reduce overfitting (model.py lines 101 - 142). 


The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 19). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 152).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and drive counter-clockwise around the track. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to apply architectures that worked well on image classification, such as LeNet or the Nvidia model.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it worked well on the traffic sign classification project, which indicates that this model could do well in identifying features in images which could be important in this project considering that we are trying to derivate driving angle according to the pictures captured by cameras.

So I used pretty much the same hyper-parameters as the ones I used in the traffic signs classification project.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

I'm very lucky to find that after 10 epochs, my first model had a low mean squared error on the training set as well as on the validation set. This implied that there was no over-fitting nor under-fitting. 

Then I deployed the trained model to the simulator. And it did work pretty well up until it drove the car into the lake.

Then I switched to the Nvidia model. It is good enough to drive a real car, it should be good enough for a virtual one. I set up the hyper-parameter as mentioned in section 1. Since it is a much comlicated model than LeNet, I reduced the training epochs from 10 to 5. However, after 5 epochs, I find it to be overfitting. Then I reduced epochs to 3 where the mse are close between training and validation sets.

As I switch to new model, I also applied the data pre-processing techniques introduced by the instructors such as cropping, flipping and normalization. I also collect more data by driving 3 laps counter-clockwise. I also add the left and right camera image to the data set.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 90-148) is already introduced in section 1.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 3 laps on track one. Although I did try to stay in the center of the road, I found it not easy. I drifted to the side of the road time to time. Especially when cornering, I have to stayed in the outside part of road before entering the corner, stay in the inside part of the corner and get out of the corner in the outside part of the road. This is partly because it's hard to control the car, partly because I found it interesting as it is like driving in a race. 

After the training, the model often make mistakes at two locations. One is right behind the bridge where the model would drive the car into the open field and the other one is a right turn where the model often drove the car into the lake without turning.

So I launched the simulator again to record specifically how to drive at this two spots.

This is some pictures of my driving:
![alt text][../writeup_pics/center_2018_10_23_15_12_22_615.jpg]
![alt text][../writeup_pics/left_2018_10_28_10_41_18_880.jpg]
![alt text][../writeup_pics/right_2018_10_29_15_45_45_471.jpg]

I also drove on track 2 to collect more data.

To augment the data sat, I also flipped images and angles. For example, here is an image that has then been flipped:

![alt text][../writeup_pics/left_2018_10_28_10_41_18_880_flipped.jpg]


Etc ....

After the collection process, I had 416160(2x69369x3, original + flipped) number of data points. I then preprocessed this data by cropping, flipping and normalization.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as mentioned before. I used an adam optimizer so that manually training the learning rate wasn't necessary.

During the training, I encountered multiple difficult issues.

First, the udacity workspace always went into idle when I am not looking. This happened before the training could finish and made it impossible to train the model. After 3 days, I figured this could be caused by Chrome. So I switched to aws to train my model.

Then I found my model was trained very slowly. It took almost 1 hour and a half to train the model for 1 epoch. I searched the internet for solution. And I found that if I increase the batch size in the code. The model would train even slower. One of my friends suggest that I check the memory of the GPU on aws using 'nvidia-smi'. I checked and reduced the batch size to 16, and the model trained much quicker.(40 miniutes for 1 epoch)

After I trained my model on aws, I found that it can't be run on my local machine. Later I figured out this was caused by the opencv library installed in the environment. After I create a new environment, this was fixed.

Then I modified the model architecture many times to improve the performance. At last I trained a model that worked perfectly.

However, later I found that the model is too complicated that it is very large! It took up to 2Gb space on disk and can't be upload to github. So I had to simplify the model. I add 3 more pooling layers. Then I found out the simplified model worked even better and only took 24mb on disk.

After all these, I run the video.py and make a video of the model running the car around track 1 for 2 laps.

