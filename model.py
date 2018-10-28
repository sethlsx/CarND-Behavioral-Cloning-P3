#This is the code defining the model and training the model


import numpy as np 
import csv
import cv2
import matplotlib.image as mpimg
import sklearn
from sklearn.model_selection import train_test_split
from random import shuffle

lines = []

with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size = 0.2)

#define a generator
def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = '../data/IMG/' + batch_sample[i].split('/')[-1]
                    image = mpimg.imread(name)
                    images.append(image)
                    if i == 0:
                        angle = float(batch_sample[3])
                    else:
                        angle = float(batch_sample[3]) + (-0.4)*i+0.6
                    angles.append(angle)
            #Flip the image to get more data

            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(angle*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)


'''
images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/'+filename
        image = mpimg.imread(current_path)
        images.append(image)
        if i == 0:
            measurement = float(line[3])
        else:
            measurement = float(line[3]) + (-0.4)*i+0.6
        measurements.append(measurement)





X_train = np.array(images)
y_train = np.array(measurements)
'''
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


#This is the model

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
#from keras.layers.core import Activation
from keras.layers.pooling import MaxPooling2D
from keras import regularizers

model = Sequential()

#Cropping image
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))


#Lambda layer
model.add(Lambda(lambda x: (x / 255) - 0.5))

#Layer 1 (90, 320, 3)
model.add(Conv2D(24, kernel_size = (5, 5), padding = 'valid', activation = 'relu', \
    kernel_regularizer = regularizers.l2(0.01)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(rate = 0.8))

#Layer 2 (43, 158, 3)
model.add(Conv2D(36, kernel_size = (3, 3), padding = 'valid', activation = 'relu', \
    kernel_regularizer = regularizers.l2(0.01)))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#Layer 3 (20, 78, 24)
model.add(Conv2D(48, kernel_size = (3, 3), padding = 'valid', activation = 'relu', \
    kernel_regularizer = regularizers.l2(0.01)))
#model.add(MaxPooling2D(pool_size=(2, 2)))


#Layer 4 (9, 38, 36)
model.add(Conv2D(64, kernel_size = (3, 3), padding = 'valid', activation = 'relu', \
    kernel_regularizer = regularizers.l2(0.01)))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#Layer 5 (3, 18, 48)
model.add(Conv2D(128, kernel_size = (3, 3), padding = 'valid', activation = 'relu', \
    kernel_regularizer = regularizers.l2(0.01)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(rate = 0.8))

#Flatten (1, 9, 64)
model.add(Flatten())

#Layer 6, full connected layer (576, 1)
model.add(Dense(1024, activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))
#model.add(Dropout(rate = 0.5))


#Layer 7
model.add(Dense(512, activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))

#Layer 8
model.add(Dense(256, activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))

#Layer 9
model.add(Dense(128, activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))


#Output


model.add(Dense(1))

#Train the model below

model.compile(loss = 'mse', optimizer = 'adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle = True, nb_epoch = 1)


#visualizing the loss

from keras.models import Model
import matplotlib.pyplot as plt

history_object = model.fit_generator(train_generator, steps_per_epoch = len(train_samples), \
    validation_data = validation_generator, validation_steps = len(validation_samples), epochs = 3, verbose = 1)

#Save the model
model.save('model.h5')
print(history_object.history.keys())

#plot the training and validation loss for each epoch

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss.jpg')
#plt.show()
