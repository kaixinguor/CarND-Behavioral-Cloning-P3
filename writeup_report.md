## Behavioral Cloning Project

### The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)
[image0]: ./output_images/model.png "Model in the paper"
[image1]: ./output_images/image_crop_65.png "Cropping"
[image2]: ./output_images/image_flip.png "Flip Image"
[image3]: ./output_images/error.png "Error"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---

###Model Architecture and Training Strategy


I start with this paper:

End to End Learning for Self-Driving Cars. Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, Xin Zhang, Jake Zhao, Karol Zieba, By NIVIDIA 2016.

I think this model would be a good start because it is used to train a network to drive a real self-driving car. They leveraged ~ 100 hours real data and showed that this artichitecture allows to train a good model which gives ~98% autonomy.

The architecture introduced in the paper is like:
![alt text][image0]

The paper use 66x220x3 images in YUV color space.


####1. Model arcthiecture

My network is very similar with NIVIDIA network. It is composed of three convolutional layers and three fully connected layers.

The implementation is in the function `nvidia_model` in lines 52 through 87 in the file `./model.py`.
The final model architecture consists of a convolution neural network with the following layers and layer sizes:

* Input RGB image (3@160x320)
* Cropping layer (3@70x320)
* Normalization layer (3@70x320)
* Conv layer (24 filters with size 5x5, stride 2x2, padding 'valid') + Relu activation. Output is 24@32x158
* Conv layer (36 filters with size 5x5, stride 2x2, padding 'valid') + Relu activation. Output is 36@14x77
* Conv layer (48 filters with size 5x5, stride 2x2, padding 'valid') + Relu activation. Output is 48@5x37
* Conv layer (64 filters with size 3x3, stride 1x1, padding 'valid') + Relu activation. Output is 64@3x35
* Conv layer (64 filters with size 3x3, stride 1x1, padding 'valid') + Relu activation. Output is 64@1x33
* Flatten -> 2112 nodes
* Fully connected + Relu activation. Output 100 nodes.
* Fully connected + Relu activation. Output 50 nodes.
* Fully connected + Relu activation. Output 10 nodes.
* Output -> 1 node

The model take RGB images of original size (with lowest resolution 160x320x3). Cropping and normalization is added directly to the beginning layers to the network so that the computation could leverage the GPU and the same operation will be done on test image automatically.

The network first crops the image to remove upper part which contain major far background like sky, trees which would not be very helpful for driving decision. The lowest part is also removed since it contains car hook.

 The following figure shows a comparison between original images (upper row) and cropped images (bottom):

![alt text][image1]

The second layer does the normalization on the cropped images. Pixel values are normalized to `[-0.5, 0.5]`

The following layers look like the architecture introduced in the paper, 
Dropout layers are added to prevent overfitting.


####2. Data Collection

I used sample images from center camera, left camera and right camera with error correction to do the training. 

I tried to collect own data to train, the result is similar but the driving routine is not very smooth. So then I only use sample images to train the model.


I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

Then I flip every image to augmente dataset.
![alt text][image2]

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:



####3. Creation of the Training Set & Training Process

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16).

I random shuffled the data set and put 20%  of the data into a validation set, in order to get an idea about whether the model is over or under fitting. . For splitting the data I use `sklearn.model_selection.train_test_split` in line 177 in `./model.py`.

The model uses an adam optimizer, so that the learning rate is adapted automatically during training. (`./model.py` line 211). The training process optimizes mean square error between network output and ground truth steering angle.

I sampled evenly 50% of the data since the neighboring images look very similar and less images will reduce training time.

Finally 19284 images are used in training.

I used 20 epochs.

![alt text][image3]


#### 4. Test

The final step was to run the simulator to see how well the car was driving around track one. 

It was actually suprising that using only the sample images to training three epochs already enables the car to drive autonomously around the track without leaving the road.

There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

Here's a [link to my video result](./result_track1.mp4)

### Discussion

*  If the driving condition contains a lot of change in lighting condition, then maybe transform RGB image into other color space will increase robustness (In NVIDIA's paper, they actually use YUV space). But here in the simulator, there is not much change in the illumination. So I use directly RGB color space.


 
