import csv
from PIL import Image
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, Dropout

# process image
def process_image(image):
    return image

# read data
csv_file = 'data/driving_log.csv'
path = 'data/IMG'
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        steering_center = float(row[3])

        # create adjusted steering measurements for the side camera images
        correction = 0.2  # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        #sdirectory = "..."  # fill in the path to your training IMG directory
        img_center = process_image(np.asarray(Image.open(path + row[0])))
        img_left = process_image(np.asarray(Image.open(path + row[1])))
        img_right = process_image(np.asarray(Image.open(path + row[2])))

        # add images and angles to data set
        car_images.extend(img_center, img_left, img_right)
        steering_angles.extend(steering_center, steering_left, steering_right)


# build model with keras
model = Sequential()

# convolutional layer
model.add(Convolution2D(nb_filter=32,nb_row=3,nb_col=3,border_mode='valid',input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filter=64,nb_row=3,nb_col=3,border_mode='valid'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())

# fully connected
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(43))
model.add(Activation('softmax'))



# Compile and train the model
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_normalized, y_one_hot, batch_size=128, nb_epoch=2, validation_split=0.2)
print("The validation accuracy is: %.3f.  It should be greater than 0.91" % history.history['val_acc'][-1])




