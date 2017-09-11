from __future__ import division, print_function, absolute_import

print("Importing required packages.")

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import build_hdf5_image_dataset

import h5py

import numpy as np

print ("Done.\n")

'''
print("Creating testing hdf5 file")

dataset_file = 'gender_dataset_testing.txt'
build_hdf5_image_dataset(dataset_file, image_shape=(100, 100), mode='file', output_path='gender_dataset_test.h5', categorical_labels=True, normalize=True)

print("Done.\n")
'''

print("Getting testing X and Y")

h5f = h5py.File('glasses_dataset_validation.h5', 'r')
test_x = h5f['X']
test_y = h5f['Y']

print("Done.\n")

print("Reshaping variables.")

test_x = np.array(test_x)
test_y = np.array(test_y)

test_x = test_x.reshape([-1, 100, 40, 3])

print("Done.\n")

print("Creating Neural Network Model : AlexNet.")

print("\tInput")
network = input_data(shape=[None, 100, 40, 3], name='input')

print("\tConvolution")
network = conv_2d(network, 96, 11, strides=4, activation='relu')

print("\tMax Pooling")
network = max_pool_2d(network, 3, strides=2)

print("\tLocal Response Normalization")
network = local_response_normalization(network)

print("\tConvolution")
network = conv_2d(network, 256, 5, activation='relu')

print("\tMax Pooling")
network = max_pool_2d(network, 3, strides=2)

print("\tLocal Response Normalization")
network = local_response_normalization(network)

print("\tConvolution")
network = conv_2d(network, 384, 3, activation='relu')

print("\tConvolution")
network = conv_2d(network, 384, 3, activation='relu')

print("\tConvolution")
network = conv_2d(network, 256, 3, activation='relu')

print("\tMax Pooling")
network = max_pool_2d(network, 3, strides=2)

print("\tLocal Response Normalization")
network = local_response_normalization(network)

print("\tFully Connected")
network = fully_connected(network, 4096, activation='tanh')

print("\tDropout")
network = dropout(network, 0.5)

print("\tFully Connected")
network = fully_connected(network, 4096, activation='tanh')

print("\tDropout")
network = dropout(network, 0.5)

print("\tFully Connected")
network = fully_connected(network, 2, activation='softmax')

print("\tRegression")
network = regression(network, optimizer='momentum', loss='categorical_crossentropy', learning_rate=0.01)

print("Done.\n")

print("Modelling the layers")

model = tflearn.DNN(network, checkpoint_path='model_alexnet', max_checkpoints=1, tensorboard_verbose=2)

print("Done.\n")

print("Loading Model.")

model.load('glasses_modelFit.model')

print("Done.")

print("Predicting.")

prediction = model.predict(test_x)
accuracy = model.evaluate(test_x, test_y)

print('Prediction: ', prediction, sep='\n')
print('Accuracy: ', accuracy)

print("Done.\n")

print("END.")