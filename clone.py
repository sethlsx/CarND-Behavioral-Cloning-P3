#This is the code defining the model and training the model


import numpy as np 
import csv
import cv2
import matplotlib.image as mpimg

lines = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
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
#Flip the image to get more data

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(images)
y_train = np.array(measurements)

#This is the model

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D
#from keras.layers.core import Activation
from keras.layers.pooling import MaxPooling2D

model = Sequential()

#Cropping image
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))


#Lambda layer
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

#Layer 1
model.add(Conv2D(3, kernel_size = (5, 5), padding = 'valid', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Layer 2
model.add(Conv2D(24, kernel_size = (5, 5), padding = 'valid', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Layer 3
model.add(Conv2D(36, kernel_size = (5, 5), padding = 'valid', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


#Layer 4
model.add(Conv2D(48, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Layer 5
model.add(Conv2D(64, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate = 0.6))

#Flatten
model.add(Flatten())

#Layer 6, full connected layer
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(rate = 0.5))


#Layer 7
model.add(Dense(512, activation = 'relu'))

#Layer 8
model.add(Dense(256, activation = 'relu'))

#Layer 9
model.add(Dense(128, activation = 'relu'))


#Output


model.add(Dense(1))

#Train the model below

model.compile(loss = 'mse', optimizer = 'adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle = True, nb_epoch = 1)


#visualizing the loss

from keras.models import Model
import matplotlib.pyplot as plt

history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples), \
    validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch = 5, verbose = 1)

print(history_object.history.keys())

#plot the training and validation loss for each epoch

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show
#Save the model

model.save('model.h5')