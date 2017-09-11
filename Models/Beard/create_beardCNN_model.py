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

print("Creating training hdf5 file")

dataset_file = 'beard_dataset_train.txt'
build_hdf5_image_dataset(dataset_file, image_shape=(100, 100), mode='file', output_path='beard_dataset_train.h5', categorical_labels=True, normalize=True)

print("Done.\n")

print("Creating validating hdf5 file")

dataset_file = 'beard_dataset_validation.txt'
build_hdf5_image_dataset(dataset_file, image_shape=(100, 100), mode='file', output_path='beard_dataset_validation.h5', categorical_labels=True, normalize=True)

print("Done.\n")

print("Getting train X and Y")

h5f = h5py.File('beard_dataset_train.h5', 'r')
X = h5f['X']
Y = h5f['Y']

print("Done.\n")

print("Getting validation X and Y")

h5f = h5py.File('beard_dataset_validation.h5', 'r')
validation_x = h5f['X']
validation_y = h5f['Y']

print("Done.\n")

print("Reshaping variables.")

X = np.array(X)
Y = np.array(Y)
validation_x = np.array(validation_x)
validation_y = np.array(validation_y)

X = X.reshape([-1, 100, 100, 3])
validation_x = validation_x.reshape([-1, 100, 100, 3])

print("Done.\n")

print("Creating Neural Network Model : AlexNet.")

print("\tInput")
network = input_data(shape=[None, 100, 100, 3], name='input')

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

print("Starting to fit.")

model.fit(X, Y, n_epoch=500, validation_set=(validation_x, validation_y), snapshot_step=500, show_metric=True, run_id='MIL')

model.save('beard_modelFit.model')

print("Done.\n")

print("END.")