# Convolutional Neural Network // classify images - change images of cats and dogs to whatever you want

# Installing Theano (numerical computations)

# Installing Tensorflow (numerical computations)

# Installing Keras (wraps Theano and Tensorflow -> few lines of code) 

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential # initalize the neural network
from keras.layers import Conv2D # add convolutional layers (images are 2D)
from keras.layers import MaxPooling2D # pooling step
from keras.layers import Flatten # convert pooled feature maps into big feature vector
from keras.layers import Dense # add fully connected layers in nn

# Initializing the CNN
classifier = Sequential()

# Step 1 - Convolution // apply several feature detectors all over image to create feature maps
# highest value in feature map is where feature detector found specific feature in image 
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu')) # no. of filters = no. of feature maps
# input shape: convert all images to same format; size and # of channels (3)

# Step 2 - Pooling // reduce size of feature maps while retaining features
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu')) # don't need input_shape cause we input pooled feature maps to this layer

# Adding second pooling layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Can add more convolutional layers and double feature detectors each time -> great results

# Step 3 - Flattening // pooled feature maps -> single vector // keeps image spatial structure info
# each feature map corresponds to one specific feature -> each node of vector represents info of specific feature
classifier.add(Flatten())

# Step 4 - Full Connection // making ANN of fully connected layers
classifier.add(Dense(units = 128, activation = 'relu')) # 128 is based on experimentation
classifier.add(Dense(units = 1, activation = 'sigmoid')) 

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
# Image augmentation - creates many batches of images and applies random transformations to subsets to increase training size - prevents overfitting
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# apply image augmentation to training set
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64), # target size should match Conv2D input_shape
        batch_size=32,
        class_mode='binary')

# apply image augmentation to test set
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Final results - got 85% on training set and 75% on test set
classifier.fit_generator(
        training_set,
        steps_per_epoch=8000, # no. of images in training set
        epochs=25,
        validation_data=test_set,
        validation_steps=2000) # no. of images in test set

# IMPROVING THE CNN
# Adding convolutional layers is a better option that fully connected layer to increase CNN performance
# Choose higher target size for training and test sets to get more information about pixel patterns

# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64)) # must match training set target size
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, 0) # need to add 1 more input because of batch

result = classifier.predict(test_image) # 0 or 1 
training_set.class_indices # tells mapping between labels and numeric result
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
# IMPROVE CNN PERFORMANCE EVEN MORE: https://www.udemy.com/deeplearning/learn/v4/questions/2276518