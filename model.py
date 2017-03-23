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
import matplotlib.image as mpimg
from PIL import Image


# process image
def preprocess_data(samples):

    zeroItem = [sample for sample in samples if float(sample[3]) == 0]
    nzeroItem = [sample for sample in samples if float(sample[3]) != 0]

    print("number of sample with zero angle",len(zeroItem))
    print("number of sample with non-zero angle", len(nzeroItem))

    sklearn.utils.shuffle(zeroItem)
    sample_num = np.floor(len(zeroItem)*0.2).astype(int)
    print("sample from zero angle",sample_num)

    xsamples = nzeroItem + zeroItem[0:sample_num]
    print("{0:d} / {1:d} is kept".format(len(xsamples),len(samples)))
    return xsamples


def get_sample_data(csv_file,img_path):
    # read data

    lineIdx = 0
    samples = []
    angles = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if lineIdx == 0: # discard headline
                lineIdx = 1
                continue

            # attach image path
            for i in range(3):
                row[i] = img_path + row[i] .split('/')[-1]

            samples.append(row)
            angles.append(float(row[3]))
    return np.array(samples)


def nvidia_model():
    '''
    build model with keras. Reference paper:
    End to End Learning for Self Driving Cars
    Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D.Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, Xin Zhang, Jake Zhao, Karol Zieba
    By NIVIDIA 2016.
    '''

    model = Sequential()

    # cropping image
    model.add(Cropping2D(cropping=((65,25),(0,0)),input_shape=(160, 320, 3))) # channels_last

    # data normalization
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    # five convolutional layers
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.5))

    # flatten
    model.add(Flatten())

    # three fully connected layers
    model.add(Dense(100,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation='relu'))
    model.add(Dropout(0.5))

    # output layer
    model.add(Dense(1))

    return model


def generator(samples, batch_size=32): # use center, left, right images

    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                for i in range(3):  # use center / left / right images
                    name = batch_sample[i]
                    image = np.asarray(Image.open(name))
                    #center_image = cv2.cvtColor(center_image,cv2.COLOR_RGB2YUV) # this might be needed for generalization
                    angle = float(batch_sample[3])
                    if i==1:
                        angle += 0.2
                    elif i==2:
                        angle -= 0.2
                    images.append(image)
                    angles.append(angle)

                    # flip image
                    images.append(np.fliplr(image))
                    angles.append(-angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def data_exploration():
    driving_log_file = 'data/driving_log.csv'
    img_path = './data/IMG/'
    samples = []
    with open(driving_log_file) as csvfile:
        reader = csv.reader(csvfile)

        for line in reader:
            samples.append(line)
        print(samples[0])
        samples = samples[1:] # discard headline

    sample = samples[2200]
    print(sample)

    f,ax_arr = plt.subplots(2,3,figsize=(18,6))
    imgs = []
    angs = []
    for i in range(3):
        name = img_path +  sample[i].split('/')[-1]
        print(name)
        angle = float(sample[3])
        angs.append(angle)
        if i==1:
            angle += 0.2
        elif i==2:
            angle -=0.2
        # img = cv2.imread(name) #BGR
        #img = mpimg.imread(name) #RGB
        img = np.asarray(Image.open(name))  # RGB
        print(img.shape)
        ax_arr[0,i].imshow(img)
        ax_arr[0,i].set_title(sample[i].split('/')[-1].split('_')[0],fontsize=20)

        cropped_img = img[65:160 - 25, :, :]
        ax_arr[1, i].imshow(cropped_img)
        ax_arr[1, i].set_xlabel('steering {0:.4f}'.format(angle), fontsize=20)

        imgs.append(img)
    plt.savefig('output_images/image_crop_65.png')
    plt.show()


    img = imgs[0]
    angle = angs[0]
    f, ax_arr = plt.subplots(1, 2, figsize=(18, 6))
    ax_arr[0].imshow(img)
    ax_arr[0].set_title('original image',fontsize=20)
    ax_arr[0].set_xlabel('steering {0:.4f}'.format(angle), fontsize=20)

    ax_arr[1].imshow(np.fliplr(img))
    ax_arr[1].set_title('flipped image', fontsize=20)
    ax_arr[0].set_xlabel('steering {0:.4f}'.format(-angle), fontsize=20)

    plt.savefig('output_images/image_flip.png')
    plt.show()



if __name__ == '__main__':

    # read data and display
    # data_exploration()
    # exit(0)

    # read data
    csv_file = './data/driving_log.csv'
    img_path = './data/IMG/'
    samples = get_sample_data(csv_file,img_path)

    samples = preprocess_data(samples)
    angles = [float(sample[3]) for sample in samples]
    # plt.hist(angles)
    # plt.savefig('output_images/angle_hist_sample.png')
    # plt.show()
    # exit(0)

    # samples = samples[np.arange(0,num_sample,2)]
    # angles = angles[np.arange(0,num_sample,2)]

    # randomly split into training - validation set
    rand_state = np.random.randint(0, 100)
    train_samples, validation_samples = train_test_split(samples,test_size=0.2,random_state=rand_state)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    # CNN architecture
    model = nvidia_model()

    # Compile and train the model
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*6,
                        validation_data = validation_generator, nb_val_samples = len(validation_samples)*6,
                        nb_epoch = 20, verbose = 1)
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
    plt.savefig('output_images/error_nvidia1.png')
    plt.show()