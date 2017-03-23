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
[image4]: ./output_images/angle_hist.png "Angle distribution"
[image5]: ./output_images/angle_hist_sample.png "Angle distribution after sampling"
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

End to End Learning for Self-Driving Cars. Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, Xin Zhang, Jake Zhao, Karol Zieba, By NVIDIA 2016.

I think this model would be a good start because it is used to train a network to predict steering angle of a self-driving car. They leveraged about 100 hours real data and showed that this artichitecture allows to train a good model which could give ~98% autonomy.

The architecture introduced in the paper is like:
![alt text][image0]

The paper use 66x220x3 images in YUV color space.


####1. Model arcthiecture

My network is very similar with NVIDIA's network. It is composed of five convolutional layers and three fully connected layers.

The implementation is in the function `nvidia_model()` in lines 56 through 99 in the file `./model.py`. 
The final model architecture consists the following layers and layer sizes:

* Input RGB image (3@160x320)
* Cropping layer (3@70x320)
* Normalization layer (3@70x320)
* Conv layer (24 filters with size 5x5, stride 2x2, padding 'valid') + Relu activation. Output is 24@33x158
* Conv layer (36 filters with size 5x5, stride 2x2, padding 'valid') + Relu activation. Output is 36@15x77
* Conv layer (48 filters with size 5x5, stride 2x2, padding 'valid') + Relu activation. Output is 48@6x37
* Conv layer (64 filters with size 3x3, stride 1x1, padding 'valid') + Relu activation. Output is 64@4x35
* Conv layer (64 filters with size 3x3, stride 1x1, padding 'valid') + Relu activation. Output is 64@2x33
* Flatten -> 4224 nodes
* Fully connected + Relu activation. Output 100 nodes.
* Fully connected + Relu activation. Output 50 nodes.
* Fully connected + Relu activation. Output 10 nodes.
* Output -> 1 node

The model takes RGB images of original size (with lowest resolution 160x320x3). Cropping and normalization are added directly as beginning layers so that the computation could leverage the GPU and the same operation will be done on test image automatically.

The network first crops the image to remove the upper part which contains majorly far background like sky, trees which would not be very helpful for driving decision. The lowest part is also removed since it contains car hook.

 The following figure shows a comparison between the original images (upper row) and the cropped images (bottom):

![alt text][image1]

The second layer does the normalization on the cropped images. Pixel values are normalized to `[-0.5, 0.5]`

The following layers look like the architecture introduced in the paper, I add dropout layers to prevent overfitting.


####2. Data Collection

I use sample images to train the model. 

Training with my own data can achieve similar behavior and keep the car stay on the track too.  But the trajectory does not look very smooth. I guess the reason is I'm a terrible driver...

Images from the left and the right camera are also added for learning recovering. A correction steering angle is added (0.2 for left camera images and -0.2 for right angle images).

I flip every image (line 127 - 128) to augment data. So that I have more images for training, and the distribution of data for left and right turn is balanced.

![alt text][image2]

The distribution of steering angle in the data is like this, where zero angle images are dominant.
![alt text][image4]

So I removed randomly 80% of the images with zero angle, to let the distribution be more balanced, like this
![alt text][image5]

But I did not observe improvement in the performance.

####3. Creation of the Training Set & Training Process

I random shuffled the data set and put 20% of the data into a validation set, in order to get an idea about whether the model is over or under fitting. . For splitting the data I use `sklearn.model_selection.train_test_split` in line 219 in `./model.py`.

The model uses an adam optimizer, so that the learning rate is adapted automatically during training. (`./model.py` line 229-230). The training process optimizes mean square error between network output and ground truth steering angle.

Finally 21822 images are used in the training, and 5460 images are used in the validation.

I used 20 epochs. The following figure shows that the training error does not decrease after around 16 epochs.

![alt text][image3]

However, in all the experiments, I did not observe much overfitting phenomenon, even when I didnot add dropout layer. And the validation error could be smaller than training error even after sufficient round of training.


#### 4. Test

The final step was to run the simulator to see how well the car was driving around track one. 

It was actually suprising that using only the sample images to train three epochs already can enable the car to drive autonomously around the track without leaving the road.

Here's a [link to my video result](./video.mp4)

### Discussion

* If the driving condition contains a lot of change in lighting, then maybe transforming RGB image into other color space will increase robustness (In NVIDIA's paper, they actually use YUV color space). But here in the simulator, there is not much change in the illumination. So I use directly RGB color space.

* In my experiments, removing a part of sample images with zero steering angle actually had an effect of driving the car more adventuously (drift to close to the lane, but being able to recover it)

* Maybe collecting recovering data from off-the-lane would have better recovering information as it covers more possibilities than just using left and right camera. But using recovering images is crucial. The experiments showed that training only using images from center camera does not work. Leave to future exploration.


 
