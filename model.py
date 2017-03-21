import csv
from PIL import Image
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers import Convolution2D, Flatten
from keras.layers import Dense, Activation, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import cv2


# process image
def process_image(image):
    return image


def data_augment(imgs, measurements):
    new_imgs = []
    new_measurements = []
    num_img = len(imgs)
    for i in range(num_img):
        img_flipped = np.fliplr(imgs[i])
        measurement_flipped = -measurements[i]
        new_imgs.append(img_flipped)
        new_measurements.append(measurement_flipped)

    return new_imgs, new_measurements


def read_data():
    # read data
    csv_file = 'data/driving_log.csv'
    path = 'data/IMG'

    car_images = []
    steering_angles = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            steering_center = float(row[3])

            # create adjusted steering measurements for the side camera images
            correction = 0.2  # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            # read in images from center, left and right cameras
            # sdirectory = "..."  # fill in the path to your training IMG directory
            img_center = process_image(np.asarray(Image.open(path + row[0])))
            img_left = process_image(np.asarray(Image.open(path + row[1])))
            img_right = process_image(np.asarray(Image.open(path + row[2])))

            # add images and angles to data set
            car_images.extend(img_center, img_left, img_right)
            steering_angles.extend(steering_center, steering_left, steering_right)
    return car_images, steering_angles


def build_model():
    # build model with keras
    model = Sequential()

    # cropping image
    model.add(Cropping2D(cropping=((70,25),(0,0)),input_shape=(160, 320, 3))) # channels_last

    # data normalization
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    # convolutional layer
    model.add(Convolution2D(8, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(16, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(32, 5, 5, subsample=(2,2), activation='relu'))


    model.add(Flatten())

    # fully connected
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(1))
    return model


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


if __name__ == '__main__':

    driving_log_file = 'data/driving_log.csv'
    img_path = 'data/IMG'

    samples = []
    with open(driving_log_file) as csvfile:
        reader = csv.reader(csvfile)

        for line in reader:
            samples.append(line)
    print(samples[0])

    samples = samples[1:]

    print(len(samples))

    train_samples, validation_samples = train_test_split(samples,test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    # ch, row, col = 3, 80, 320  # Trimmed image format
    #
    #
    #
    # car_images, steering_angles = read_data()
    #
    # new_imgs, new_measurements = data_augment(car_images, steering_angles)

    model = build_model()

    # Compile and train the model
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples),
                        validation_data = validation_generator, nb_val_samples = len(validation_samples),
                        nb_epoch = 3, verbose = 1)
    model.save('model.h5')
    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    print("The validation accuracy is: %.3f.  It should be greater than 0.91" % history_object.history['val_acc'][-1])




