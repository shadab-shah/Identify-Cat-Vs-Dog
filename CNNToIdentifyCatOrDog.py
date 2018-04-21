# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 23:15:46 2018

@author: admin
"""

import os
os.chdir('D:\Shadab\MachineLearning\LearningTensorFlow\Dataset')

import pandas
import numpy

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialising the CNN
classifier = Sequential()

#Step1 Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3), activation = 'relu'))

#Step2 Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))


#Step1 Convolution
classifier.add(Convolution2D(32,3,3, activation = 'relu'))

#Step2 Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step3 
classifier.add(Flatten())

#Step4
classifier.add(Dense(output_dim = 128,activation= 'relu'))
classifier.add(Dense(output_dim = 1,activation= 'sigmoid'))

#compiling the CNN
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics = ['accuracy'])

# =============================================================================
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)
# =============================================================================

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                    'training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                                'test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

classifier.fit_generator(
                    training_set,
                    steps_per_epoch=8000,
                    epochs=25,
                    validation_data=test_set,
                    nb_val_samples=2000)

classifier.save('D:\Shadab\MachineLearning\LearningTensorFlow\modelWith2LevelConvution.h5')  # creates a HDF5 file 'model.h5'

from keras.models import load_model
classifier = load_model('D:\Shadab\MachineLearning\LearningTensorFlow\modelWith2LevelConvution.h5')

#Making singel prediction

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis = 0)

result = classifier.predict(test_image)

#what represnt what
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'    

print (prediction) 
    
test_image = image.load_img('single_prediction/cat_or_dog_2.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis = 0)

result = classifier.predict(test_image)

#what represnt what
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'    

print(prediction)