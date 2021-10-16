# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import GaussianNoise


# Loading the data
from tensorflow.keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# Scaling the image to make the training faster
X_train = X_train/ 255.0
X_test = X_test/ 255.0

# Adding Gaussian Noise to the images
noise = GaussianNoise(0.25)

# Auto-Encoder Model
units = 28 * 28 # Setting the number of units, ie img_height * img_width

# Encoder Layers
encoder = Sequential()
encoder.add(Flatten(input_shape = [28, 28]))
encoder.add(GaussianNoise(0.2)) # Adding Gaussian Noise
encoder.add(Dense(units = units//2, activation = 'relu'))
encoder.add(Dense(units = units//4, activation = 'relu'))
encoder.add(Dense(units = units//8, activation = 'relu'))
encoder.add(Dense(units = units//16, activation = 'relu'))
encoder.add(Dense(units = units//32, activation = 'relu'))
encoder.add(Dense(units = units//64, activation = 'relu'))

# Decoder Layers
decoder = Sequential()
decoder.add(Dense(units = units//32, input_shape=[units//64], activation = 'relu'))
decoder.add(Dense(units = units//16, activation = 'relu'))
decoder.add(Dense(units = units//8, activation = 'relu'))
decoder.add(Dense(units = units//4, activation = 'relu'))
decoder.add(Dense(units = units//2, activation = 'relu'))
decoder.add(Dense(units = units, activation = 'relu'))
decoder.add(Reshape([28, 28]))

# Creating the Denoiser - Combining the Encoder and Decoder together
denoiser = Sequential([encoder, decoder])
denoiser.compile(loss = 'binary_crossentropy',
                 optimizer='adam',
                 metrics = ['accuracy'])

# Training the model
denoiser.fit(X_train, X_train, epochs=150, verbose=0)

# Testing the model
# Adding noise to test images
noisy_test_images = noise(X_test[:10], training = True)

# Cleaning the test images
clean_test_image = denoiser(noisy_test_images)
