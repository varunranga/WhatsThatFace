from __future__ import division, print_function, absolute_import

print("Importing required packages.")

# Supressing Log Messages

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Using pickle to save all files related to database and program data.

import pickle

# To manipulate images and tensors.

import numpy as np

# Used to get files from a directory on the computer.

import glob

# OpenCV for feature extraction.

import cv2

# Used to build the dataset, for testing and prediction of class.

import h5py

# Using K-Means Algorithm for grouping of Nose Length, Nose Width, Eye Length, Eye Width, and Skin Luminosity.

from sklearn import cluster

# Tensorflow is being used to reset the graph, i.e., to clear the stack of tensorflow, 
# 	because the scope of tensorflow variables are global and it interferes with other functions.

import tensorflow as tf

# TFLearn is the package used for all Convolution Neural Network.

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import build_hdf5_image_dataset

# Using TKinter for GUI.
# Calling PIL (pillow) package for Image Processing using TKinter.

from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter.filedialog import askopenfilename, askopenfilenames, asksaveasfile
from tkinter.messagebox import showerror

# All NLTK tools for extracting features from the text input.

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tree import Tree
from nltk import PorterStemmer

print("Done.")
print()


# Loading all clustering models.

print("Loading all clustering models.")

L_fileObj = open('Models/Skin Color/L_model.pickle', 'rb')
L_kmeans = pickle.load(L_fileObj)
L_fileObj.close()

noseLengthFile = open('Models/Nose Length/noseLength.pickle', 'rb')
noseLength_kmeans = pickle.load(noseLengthFile)
noseLengthFile.close()

noseWidthFile = open('Models/Nose Width/noseWidth.pickle', 'rb')
noseWidth_kmeans = pickle.load(noseWidthFile)
noseWidthFile.close()

eyeLengthFile = open('Models/Eye Length/eyeLength.pickle', 'rb')
eyeLength_kmeans = pickle.load(eyeLengthFile)
eyeLengthFile.close()

eyeWidthFile = open('Models/Eye Width/eyeWidth.pickle', 'rb')
eyeWidth_kmeans = pickle.load(eyeWidthFile)
eyeWidthFile.close()

print("Done.")
print()


# Loading all haarCascade Classifiers

print("Load all haarCascade Classifiers.")

faceCascadeClassifier = cv2.CascadeClassifier('haarCascades/haarcascade_frontalface_default.xml')
eyeCascadeClassifier = cv2.CascadeClassifier('haarCascades/haarcascade_mcs_eyepair_big.xml')
noseCascadeClassifier = cv2.CascadeClassifier('haarCascades/haarcascade_mcs_nose.xml')
mouthCascadeClassifier = cv2.CascadeClassifier('haarCascades/haarcascade_mcs_mouth.xml')

print("Done.")
print()



# Defining all Global Functions


def getOnlySkin(image):
    
	# Lower end of the HSV value for skin.
    low = (3,50,50)

	# Higher end of the HSV value for skin.
    up = (33,255,255)

    # Convert OpenCV Matrix to HSV values of the image.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Get only those pixels within the specified range.
    skinMask=cv2.inRange(hsv, low, up)

    # Helps in forming a custom shape of the face.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    
	# 'Erode away' the boundaries of the foreground object.
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)

    # Used after erosion to because erosion removes away some pixels and shrink the object.
    # 	This will add some white pixels, joining the broken pieces of the image.
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

  	# Replaces the center pixel with the average of the neighbourhood pixel values
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(image, image, mask = skinMask)

    # Convert the pixels back to standard BGR values.
    skin = cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)

    return skin

# Gets the average value of H,L,S of an image.
def getAverageColor(image):
	count = 0
	sumh = 0
	suml = 0
	sums = 0
	for j in range(len(image)):
	    for k in range(len(image[0])):
	        h, l, s = image[j][k]
	        if(h > 0):
	            sumh += h
	            suml += l
	            sums += s
	            count += 1
	
	# unsuccessful face extraction, leading to count being 0.
	try:
		sumh /= count
		suml /= count
		sums /= count
	except:
		print ("Unable to extract face")
		sumh = 0
		suml = 0
		sums = 0

	return (sumh, suml, sums)




# Input : Image Object from OpenCV, Required File Name, Cropping points, Size of new image.
# Process : Crops image and save the file with the Required File Name.
# Output : None.
def saveNewFeature(fileName, image, x, y, w, h, newSizeX, newSizeY):
	newImage = image[y:y+h, x:x+w]
	newImage = cv2.resize(newImage, (newSizeX, newSizeY))
	cv2.imwrite(fileName, newImage)

# Input : Image Object from OpenCV, Face Cascade Classifier, Required File Name.
# Process : Finds face using 'detectMultiScale'.
#			Crops image and save the file with the Required File Name.
# Output : File Name.
def extractFace(image, faceCascadeClassifier, fileName):
	fileName = fileName.split('/')[-1]
	scaleFactor = 1.2
	minNeighbours = 1
	face = faceCascadeClassifier.detectMultiScale(image, scaleFactor, minNeighbours)
	if (len(face)):
		x, y, w, h = face[-1]
		fileName = 'Extracted/Face/'+fileName+'.jpg'
		saveNewFeature(fileName, image, x, y, w, h, 100, 100)
		return fileName

def extractColorFace(image, faceCascadeClassifier, fileName):
	fileName = fileName.split('/')[-1]
	scaleFactor = 1.2
	minNeighbours = 1
	face = faceCascadeClassifier.detectMultiScale(image, scaleFactor, minNeighbours)
	if (len(face)):
		x, y, w, h = face[-1]
		fileName = 'Extracted/ColorFace/'+fileName+'.jpg'
		saveNewFeature(fileName, image, x, y, w, h, 100, 100)
		return fileName

# Input : Image Object from OpenCV, Face Cascade Classifier, Required File Name.
# Process : Finds face using 'detectMultiScale'.
#			Crops image and save the file with the Required File Name.
# Output : File Name.
def extractBeard(image, faceCascadeClassifier, fileName):
	fileName = fileName.split('/')[-1]
	scaleFactor = 1.2
	minNeighbours = 1
	face = faceCascadeClassifier.detectMultiScale(image, scaleFactor, minNeighbours)
	if (len(face)):
		x, y, w, h = face[-1]
		fileName = 'Extracted/Beard/'+fileName+'.jpg'
		saveNewFeature(fileName, image, x, y, w, h+50, 100, 100)
		return fileName

# Input : Image Object from OpenCV, Eye Cascade Classifier, Required File Name.
# Process : Finds eye using 'detectMultiScale'.
#			Crops image and save the file with the Required File Name.
# Output : File Name.
def extractEye(image, eyeCascadeClassifier, fileName):
	fileName = fileName.split('/')[-1]
	scaleFactor = 1.2
	minNeighbours = 1
	eye = eyeCascadeClassifier.detectMultiScale(image, scaleFactor, minNeighbours)
	if (len(eye)):
		x, y, w, h = eye[-1]
		fileName = 'Extracted/Eye/'+fileName+'.jpg'
		saveNewFeature(fileName, image, x, y, w, h, 100, 40)
		return fileName,h,w

# Input : Image Object from OpenCV, Nose Cascade Classifier, Required File Name.
# Process : Finds nose using 'detectMultiScale'.
#			Crops image and save the file with the Required File Name.
# Output : File Name, Height of nose and Width of the nose.
def extractNose(image, noseCascadeClassifier, fileName):
	fileName = fileName.split('/')[-1]
	scaleFactor = 1.1
	minNeighbours = 1
	nose = noseCascadeClassifier.detectMultiScale(image, scaleFactor, minNeighbours)
	if (len(nose)):
		x, y, w, h = nose[-1]
		fileName = 'Extracted/Nose/'+fileName+'.jpg'
		saveNewFeature(fileName, image, x, y, w, h, 60, 60)
		return fileName,h,w

# Input : Image Object from OpenCV, Mouth Cascade Classifier, Required File Name.
# Process : Finds mouth using 'detectMultiScale'.
#			Crops image and save the file with the Required File Name.
# Output : File Name.
def extractMoustache(image, mouthCascadeClassifier, fileName):
	fileName = fileName.split('/')[-1]
	scaleFactor = 1.2
	minNeighbours = 1
	mouth = mouthCascadeClassifier.detectMultiScale(image, scaleFactor, minNeighbours)
	if (len(mouth)):
		x, y, w, h = mouth[-1]
		fileName = 'Extracted/Moustache/'+fileName+'.jpg'
		saveNewFeature(fileName, image, x, y, w, h, 70, 40)
		return fileName

# Input : File Name to be processed.
# Process : Extracts the features for that particular image whose file path is given.
# Output : Nose Length, Nose Width, Eye Length, Eye Width
def extractForOneImage(fileName):
	image = cv2.imread(fileName)


	success_extractColorFace = extractColorFace(image, faceCascadeClassifier, fileName[:-4])

	if (success_extractColorFace is not None):
		print (success_extractColorFace, "Color Face has been extracted")

	image = cv2.imread(fileName, 0)

	print('Processing', fileName)

	success_extractBeard = extractBeard(image, faceCascadeClassifier, fileName.split('/')[-1][:-4])

	if (success_extractBeard is not None):
		print (success_extractBeard, "Beard Face has been extracted")

	success_extractFace = extractFace(image, faceCascadeClassifier, fileName[:-4])

	if (success_extractFace is not None):
		print (success_extractFace, "Face has been extracted")
		image = cv2.imread(success_extractFace)

	# Use only face image after this point

	# It may not detect eyes, therefore use a try except block
	success_extractEye = extractEye(image, eyeCascadeClassifier, fileName.split('/')[-1][:-4])

	if (success_extractEye is not None):
		success_extractEye, eyeLength, eyeWidth = success_extractEye
		print (success_extractEye, "Eyes have been extracted")
	else:
		eyeLength = 16 # mean value
		eyeWidth = 67 # mean value

	# It may not detect nose, therefore use a try except block
	success_extractNose = extractNose(image, noseCascadeClassifier, fileName.split('/')[-1][:-4])
	
	if (success_extractNose is not None):
		success_extractNose, noseLength, noseWidth = success_extractNose
		print (success_extractNose, "Nose has been extracted")
	else:
		noseLength = 25 # mean value
		noseWidth = 31 # mean value

	# It may not detect mouth, therefore use a try except block
	try:
		success_extractMoustache = extractMoustache(image, mouthCascadeClassifier, fileName.split('/')[-1][:-4])
	except:
		success_extractMoustache = None

	if (success_extractMoustache is not None):
		print (success_extractMoustache, "Moustache has been extracted")

	# Assigning the lengths and widths to a numpy variable so that it can be used for clustering. 

	noseLength = np.array(noseLength)
	noseWidth = np.array(noseWidth)
	eyeLength = np.array(eyeLength)
	eyeWidth = np.array(eyeWidth)	

	# Reshaping to avoid DeprecatedError in KMeans() from sklearn. 

	noseLength = noseLength.reshape(-1, 1)
	noseWidth = noseWidth.reshape(-1, 1)
	eyeLength = eyeLength.reshape(-1, 1)
	eyeWidth = eyeWidth.reshape(-1, 1)

	return (noseLength, noseWidth, eyeLength, eyeWidth)


# Input : File Name to Classifier the gender of the person
# Process : Creates the AlexNet Model and Loads pre-trained data. Uses the model to predict.
# Output : Confidence of prediction of the classes. 
def genderClassification(image):

	print("Creating Neural Network Model : AlexNet for Gender")

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

	print("Loading Model.")

	model.load('Models/Gender/gender_modelFit.model')

	print("Done.")

	print("Creating testing text file")

	# Use a string to create the .txt file to be used to create the HDFS file
	# 	to build the binary data of the image.
	# Assign a random class, it will not matter as we are not testing for accuracy.

	textString = image + ' 0'

	fileObj = open('Models/Gender/gender_dataset_test.txt', 'w')
	fileObj.write(textString)
	fileObj.close()

	print("Done.")

	print("Creating testing hdf5 file")

	dataset_file = 'Models/Gender/gender_dataset_test.txt'
	build_hdf5_image_dataset(dataset_file, image_shape=(100, 100), mode='file', output_path='Models/Gender/gender_dataset_test.h5', categorical_labels=False, normalize=True)

	print("Done.\n")

	print("Getting testing X")

	h5f = h5py.File('Models/Gender/gender_dataset_test.h5', 'r')
	test_x = h5f['X']

	print("Done.\n")

	print("Reshaping variables.")

	test_x = np.array(test_x)

	test_x = test_x.reshape([-1, 100, 100, 3])

	print("Done.\n")

	print("Predicting.")

	prediction = model.predict(test_x)

	print("Done.")

	return prediction

# Input : File Name to Classifier the beard of the person
# Process : Creates the AlexNet Model and Loads pre-trained data. Uses the model to predict.
# Output : Confidence of prediction of the classes. 
def beardClassification(image):

	print("Creating Neural Network Model : AlexNet for Beard")

	print("\tInput")
	network2 = input_data(shape=[None, 100, 100, 3], name='input')

	print("\tConvolution")
	network2 = conv_2d(network2, 96, 11, strides=4, activation='relu')

	print("\tMax Pooling")
	network2 = max_pool_2d(network2, 3, strides=2)

	print("\tLocal Response Normalization")
	network2 = local_response_normalization(network2)

	print("\tConvolution")
	network2 = conv_2d(network2, 256, 5, activation='relu')

	print("\tMax Pooling")
	network2 = max_pool_2d(network2, 3, strides=2)

	print("\tLocal Response Normalization")
	network2 = local_response_normalization(network2)

	print("\tConvolution")
	network2 = conv_2d(network2, 384, 3, activation='relu')

	print("\tConvolution")
	network2 = conv_2d(network2, 384, 3, activation='relu')

	print("\tConvolution")
	network2 = conv_2d(network2, 256, 3, activation='relu')

	print("\tMax Pooling")
	network2 = max_pool_2d(network2, 3, strides=2)

	print("\tLocal Response Normalization")
	network2 = local_response_normalization(network2)

	print("\tFully Connected")
	network2 = fully_connected(network2, 4096, activation='tanh')

	print("\tDropout")
	network2 = dropout(network2, 0.5)

	print("\tFully Connected")
	network2 = fully_connected(network2, 4096, activation='tanh')

	print("\tDropout")
	network2 = dropout(network2, 0.5)

	print("\tFully Connected")
	network2 = fully_connected(network2, 2, activation='softmax')

	print("\tRegression")
	network2 = regression(network2, optimizer='momentum', loss='categorical_crossentropy', learning_rate=0.01)

	print("Done.\n")

	print("Modelling the layers")

	model2 = tflearn.DNN(network2, checkpoint_path='model_alexnet', max_checkpoints=1, tensorboard_verbose=2)

	print("Done.\n")

	print("Loading Model.")

	model2.load('Models/Beard/beard_modelFit.model')

	print("Done.")

	print("Creating testing text file")

	# Use a string to create the .txt file to be used to create the HDFS file
	# 	to build the binary data of the image.
	# Assign a random class, it will not matter as we are not testing for accuracy.

	textString = image + ' 0'

	fileObj = open('Models/Beard/beard_dataset_test.txt', 'w')
	fileObj.write(textString)
	fileObj.close()

	print("Creating testing hdf5 file")

	dataset_file = 'Models/Beard/beard_dataset_test.txt'
	build_hdf5_image_dataset(dataset_file, image_shape=(100, 100), mode='file', output_path='Models/Beard/beard_dataset_test.h5', categorical_labels=False, normalize=True)

	print("Done.\n")

	print("Getting testing X")

	h5f = h5py.File('Models/Beard/beard_dataset_test.h5', 'r')
	test_x = h5f['X']

	print("Done.\n")

	print("Reshaping variables.")

	test_x = np.array(test_x)

	test_x = test_x.reshape([-1, 100, 100, 3])

	print("Done.\n")

	print("Predicting.")

	prediction = model2.predict(test_x)

	print("Done.")

	return prediction

# Input : File Name to Classifier the glasses of the person
# Process : Creates the AlexNet Model and Loads pre-trained data. Uses the model to predict.
# Output : Confidence of prediction of the classes.
def glassesClassification(image):

	print("Creating Neural Network Model : AlexNet for Glasses")

	print("\tInput")
	network1 = input_data(shape=[None, 100, 100, 3], name='input')

	print("\tConvolution")
	network1 = conv_2d(network1, 96, 11, strides=4, activation='relu')

	print("\tMax Pooling")
	network1 = max_pool_2d(network1, 3, strides=2)

	print("\tLocal Response Normalization")
	network1 = local_response_normalization(network1)

	print("\tConvolution")
	network1 = conv_2d(network1, 256, 5, activation='relu')

	print("\tMax Pooling")
	network1 = max_pool_2d(network1, 3, strides=2)

	print("\tLocal Response Normalization")
	network1 = local_response_normalization(network1)

	print("\tConvolution")
	network1 = conv_2d(network1, 384, 3, activation='relu')

	print("\tConvolution")
	network1 = conv_2d(network1, 384, 3, activation='relu')

	print("\tConvolution")
	network1 = conv_2d(network1, 256, 3, activation='relu')

	print("\tMax Pooling")
	network1 = max_pool_2d(network1, 3, strides=2)

	print("\tLocal Response Normalization")
	network1 = local_response_normalization(network1)

	print("\tFully Connected")
	network1 = fully_connected(network1, 4096, activation='tanh')

	print("\tDropout")
	network1 = dropout(network1, 0.5)

	print("\tFully Connected")
	network1 = fully_connected(network1, 4096, activation='tanh')

	print("\tDropout")
	network1 = dropout(network1, 0.5)

	print("\tFully Connected")
	network1 = fully_connected(network1, 2, activation='softmax')

	print("\tRegression")
	network1 = regression(network1, optimizer='momentum', loss='categorical_crossentropy', learning_rate=0.01)

	print("Done.\n")

	print("Modelling the layers")

	model1 = tflearn.DNN(network1, checkpoint_path='model_alexnet', max_checkpoints=1, tensorboard_verbose=2)

	print("Done.\n")

	print("Loading Model.")

	model1.load('Models/Glasses/glasses_modelFit.model')

	print("Done.")

	print("Creating testing text file")

	textString = image+' 0'

	fileObj = open('Models/Glasses/glasses_dataset_test.txt', 'w')
	fileObj.write(textString)
	fileObj.close()

	print("Creating testing hdf5 file")

	# Use a string to create the .txt file to be used to create the HDFS file
	# 	to build the binary data of the image.
	# Assign a random class, it will not matter as we are not testing for accuracy.

	dataset_file = 'Models/Glasses/glasses_dataset_test.txt'
	build_hdf5_image_dataset(dataset_file, image_shape=(100, 100), mode='file', output_path='Models/Glasses/glasses_dataset_test.h5', categorical_labels=False, normalize=True)

	print("Done.\n")

	print("Getting testing X")

	h5f = h5py.File('Models/Glasses/glasses_dataset_test.h5', 'r')
	test_x = h5f['X']

	print("Done.\n")

	print("Reshaping variables.")

	test_x = np.array(test_x)

	test_x = test_x.reshape([-1, 100, 100, 3])

	print("Done.\n")

	print("Predicting.")

	prediction = model1.predict(test_x)

	print("Done.")

	return prediction

# Input : File Name to Classifier the moustache of the person
# Process : Creates the AlexNet Model and Loads pre-trained data. Uses the model to predict.
# Output : Confidence of prediction of the classes.
def moustacheClassification(image):

	print("Creating Neural Network Model : AlexNet for Moustache")

	print("\tInput")
	network3 = input_data(shape=[None, 100, 100, 3], name='input')

	print("\tConvolution")
	network3 = conv_2d(network3, 96, 11, strides=4, activation='relu')

	print("\tMax Pooling")
	network3 = max_pool_2d(network3, 3, strides=2)

	print("\tLocal Response Normalization")
	network3 = local_response_normalization(network3)

	print("\tConvolution")
	network3 = conv_2d(network3, 256, 5, activation='relu')

	print("\tMax Pooling")
	network3 = max_pool_2d(network3, 3, strides=2)

	print("\tLocal Response Normalization")
	network3 = local_response_normalization(network3)

	print("\tConvolution")
	network3 = conv_2d(network3, 384, 3, activation='relu')

	print("\tConvolution")
	network3 = conv_2d(network3, 384, 3, activation='relu')

	print("\tConvolution")
	network3 = conv_2d(network3, 256, 3, activation='relu')

	print("\tMax Pooling")
	network3 = max_pool_2d(network3, 3, strides=2)

	print("\tLocal Response Normalization")
	network3 = local_response_normalization(network3)

	print("\tFully Connected")
	network3 = fully_connected(network3, 4096, activation='tanh')

	print("\tDropout")
	network3 = dropout(network3, 0.5)

	print("\tFully Connected")
	network3 = fully_connected(network3, 4096, activation='tanh')

	print("\tDropout")
	network3 = dropout(network3, 0.5)

	print("\tFully Connected")
	network3 = fully_connected(network3, 2, activation='softmax')

	print("\tRegression")
	network3 = regression(network3, optimizer='momentum', loss='categorical_crossentropy', learning_rate=0.01)

	print("Done.\n")

	print("Modelling the layers")

	model3 = tflearn.DNN(network3, checkpoint_path='model_alexnet', max_checkpoints=1, tensorboard_verbose=2)

	print("Done.\n")

	print("Loading Model.")

	model3.load('Models/Moustache/moustache_modelFit.model')

	print("Done.")

	print("Creating testing text file")

	# Use a string to create the .txt file to be used to create the HDFS file
	# 	to build the binary data of the image.
	# Assign a random class, it will not matter as we are not testing for accuracy.

	textString = image + ' 0'

	fileObj = open('Models/Moustache/moustache_dataset_test.txt', 'w')
	fileObj.write(textString)
	fileObj.close()

	print("Creating testing hdf5 file")

	dataset_file = 'Models/Moustache/moustache_dataset_test.txt'
	build_hdf5_image_dataset(dataset_file, image_shape=(100, 100), mode='file', output_path='Models/Moustache/moustache_dataset_test.h5', categorical_labels=False, normalize=True)

	print("Done.\n")

	print("Getting testing X")

	h5f = h5py.File('Models/Moustache/moustache_dataset_test.h5', 'r')
	test_x = h5f['X']

	print("Done.\n")

	print("Reshaping variables.")

	test_x = np.array(test_x)

	test_x = test_x.reshape([-1, 100, 100, 3])

	print("Done.\n")

	print("Predicting.")

	prediction = model3.predict(test_x)

	print("Done.")

	return prediction

# Input : Values of Eye Length and Eye Width 
# Process : Uses the global eyeLength_kmeans and eyeWidth_kmeans sklearn.KMeans() object to predict the
# 	class of the Eye Length and Eye Width provided.
# Output : Class of Eye Length and Eye Width 
def eyeClassification(eyeLength, eyeWidth):
	eyeLengthClass = eyeLength_kmeans.predict(eyeLength)
	eyeWidthClass = eyeWidth_kmeans.predict(eyeWidth)
	return eyeLengthClass, eyeWidthClass

# Input : Values of Nose Length and Nose Width 
# Process : Uses the global noseLength_kmeans and noseWidth_kmeans sklearn.KMeans() object to predict the
# 	class of the Nose Length and Nose Width provided.
# Output : Class of Nose Length and Nose Width 
def noseClassification(noseLength, noseWidth):
	noseLengthClass = noseLength_kmeans.predict(noseLength)
	noseWidthClass = noseWidth_kmeans.predict(noseWidth)
	return noseLengthClass, noseWidthClass

# Not yet filled up.
def faceShapeClassification(image):
	pass

# Not yet filled up.
def raceClassification(image):
	pass

# Skin classifier based on l-value only.
# Not that accurate due to external lighting.
def skinClassification(image):
	image=cv2.imread(image)
	skin=getOnlySkin(image)
	hls=cv2.cvtColor(skin,cv2.COLOR_BGR2HLS)
	l=[int(getAverageColor(hls)[1])]
	l=np.array(l)
	l=l.reshape(-1,1)
	Lclass=L_kmeans.predict(l)
	return Lclass

# Input : Image File Path.
# Process : Extracts the features and passes the feature image to respective neural networks.
# Output : Returns a dictionary with the features and their classes.
def featureExtraction(image):
	noseLength, noseWidth, eyeLength, eyeWidth = extractForOneImage(image)
	
	# The file name only, without full directory.
	image = image.split('/')[-1]

	# Reset tensorflow stack.
	tf.reset_default_graph()

	# Classify gender.
	print("Predicting Gender.")
	gender = genderClassification('Extracted/Face/'+image)

	# Reset tensorflow stack.
	tf.reset_default_graph()

	# Classify glasses.
	print("Predicting Glasses.")
	glasses = glassesClassification('Extracted/Face/'+image)

	# Reset tensorflow stack.
	tf.reset_default_graph()
	
	# Classify beard.
	print("Predicting Beard.")
	beard = beardClassification('Extracted/Face/'+image)
	
	# Reset tensorflow stack.
	tf.reset_default_graph()
	
	# Classify moustache.
	print("Predicting Moustache.")
	moustache = moustacheClassification('Extracted/Face/'+image)
	
	# Reset tensorflow stack.
	tf.reset_default_graph()
	
	# Classify skin color.
	print("Predicting Skin Color Class.")
	skinClass = skinClassification('Extracted/ColorFace/'+image)
	
	# Running eye length and eye width classifier.
	print("Predicting Eye Length and Width.")
	eyeLengthClass, eyeWidthClass = eyeClassification(eyeLength, eyeWidth)
	
	# Running eye length and eye width classifier.
	print("Predicting Nose Length and Width.")
	noseLengthClass, noseWidthClass = noseClassification(noseLength, noseWidth)

	# Not yet finished.
	# faceShape = faceShapeClassification('Extracted/Face/'+image)
	# race = raceClassification('Extracted/Face/'+image)

	# Converting all numerical classes to strings.

	print("Converting numerical classes to strings.")
	
	if(gender[0][0] >= 0.50):
		genderString = 'Male'
	else:
		genderString = 'Female'

	if(glasses[0][0] >= 0.50):
		glassesString = 'No'
	else:
		glassesString = 'Yes'

	if(beard[0][0] >= 0.60):
		beardString = 'No'
	else:
		beardString = 'Yes'

	if(moustache[0][0] >= 0.90):
		moustacheString = 'No'
	else:
		moustacheString = 'Yes'

	if(noseLengthClass[0] == 5 or noseLengthClass[0] == 6):
		noseLengthString = 'Short'
	elif(noseLengthClass[0] == 0 or noseLengthClass[0] == 4):
		noseLengthString = 'Long'
	else:
		noseLengthString = 'Medium'

	if(noseWidthClass[0] == 2 or noseWidthClass[0] == 6):
		noseWidthString = 'Narrow'
	elif(noseWidthClass[0] == 1 or noseWidthClass[0] == 5):
		noseWidthString = 'Wide'
	else:
		noseWidthString = 'Medium'

	if(eyeLengthClass[0] == 4 or eyeLengthClass[0] == 1):
		eyeLengthString = 'Small'
	elif(eyeLengthClass[0] == 2 or eyeLengthClass[0] == 6):
		eyeLengthString = 'Big'
	else:
		eyeLengthString='Medium'

	if(eyeWidthClass[0] == 0 or eyeWidthClass[0] == 5):
		eyeWidthString = 'Narrow'
	elif(eyeWidthClass[0] == 3 or eyeWidthClass[0] ==  2):
		eyeWidthString = 'Wide'
	else:
		eyeWidthString = 'Medium'

	if(skinClass[0] == 3):
		skinString='Black'
	elif(skinClass[0] == 4 or skinClass[0] == 0 or skinClass[0] == 6):
		skinString='Brown'
	else:
		skinString='White'

	# Dictionary to be returned.

	predictFeatures={}

	predictFeatures['gender'] = genderString
	predictFeatures['glasses'] = glassesString
	predictFeatures['moustache'] = moustacheString
	predictFeatures['beard'] = beardString
	predictFeatures['noseLength'] = noseLengthString
	predictFeatures['noseWidth'] = noseWidthString
	predictFeatures['eyeLength'] = eyeLengthString
	predictFeatures['eyeWidth'] = eyeWidthString
	predictFeatures['skinColor'] = skinString

	return predictFeatures

# Input : Texual Description
# Process : Stemming, Sentence Tokenizer, Word Tokenizer, Part of Speech Tagging.
#			Looks for the type of part of speech, and particular words and saves them in a dictionary/
# Output : Dictionary containing the facial features of the person.
def getNLPFeatures(description):

	print("Extracting features for Text.")

	# This dictionary contains everything just in the case when the user does not give all details.

	features = {
		'Gender' : ['Male','Female'],
		'Race' : ['Indian'],
		'Face Shape' : [],
		'Glasses' : ['No','Yes'],
		'Moustache' : ['No','Yes'],
		'Beard' : ['No','Yes'],
		'Nose Length' : ['Short','Medium','Long'],
		'Nose Width' : ['Narrow','Medium','Wide'],
		'Skin Color' : ['Black','Brown','White'],
		'Eye Length' : ['Small','Medium','Big'],
		'Eye Width' : ['Narrow','Medium','Wide']
	}

	porterStemmer = PorterStemmer()
	stemmed = [porterStemmer.stem(word) for sentence in sent_tokenize(description) for word in word_tokenize(sentence)]

	# Create a new string after stemming.

	description = ""
	for stem in stemmed:
		description += stem + ' '

	tokenized = sent_tokenize(description)

	try:
		for i in tokenized:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)
			negation = False
			currentFeature = ""
			currentAdjectives = []
			for word, pos in tagged:
				if (pos == 'PRP'):
					if (word.lower() == 'he'):
						features['Gender'] = ['Male']
					elif (word.lower() == 'she'):
						features['Gender'] = ['Female']
				elif (pos == 'NN'):
					if (word.lower() == 'glass'):
						if (negation == True):
							features['Glasses'] = ['No']
						else:
							features['Glasses'] = ['Yes']
					elif (word.lower() == 'beard'):
						if (negation == True):
							features['Beard'] = ['No']
						else:
							features['Beard'] = ['Yes']
					elif (word.lower() == 'moustach'):
						if (negation == True):
							features['Moustache'] = ['No']
						else:
							features['Moustache'] = ['Yes']
					elif (word.lower() == 'eye'):
						currentFeature = 'eye'
						if (len(currentAdjectives) != 0):
							for adjective in currentAdjectives:
								if (adjective.lower() == 'big'):
									if (negation == True):
										features['Eye Length'] = ['Small','Medium']
									else:
										features['Eye Length'] = ['Big']
								elif (adjective.lower() == 'small'):
									if (negation == True):
										features['Eye Length'] = ['Big','Medium']
									else:
										features['Eye Length'] = ['Small']
								elif (adjective.lower() == 'narrow'):
									if (negation == True):
										features['Eye Width'] = ['Wide','Medium']
									else:
										features['Nose Width'] = ['Narrow']
								elif (adjective.lower() == 'wide'):
									if (negation == True):
										features['Eye Width'] = ['Narrow','Medium']
									else:
										features['Eye Width'] = ['Wide']
					elif (word.lower() == 'nose'):
						currentFeature = 'nose'
						if (len(currentAdjectives) != 0):
							for adjective in currentAdjectives:
								if (adjective.lower() == 'long'):
									if (negation == True):
										features['Nose Length'] = ['Short','Medium']
									else:
										features['Nose Length'] = ['Long']
								elif (adjective.lower() == 'short'):
									if (negation == True):
										features['Nose Length'] = ['Long','Medium']
									else:
										features['Nose Length'] = ['Short']
								elif (adjective.lower() == 'narrow'):
									if (negation == True):
										features['Nose Width'] = ['Wide','Medium']
									else:
										features['Nose Width'] = ['Narrow']
								elif (adjective.lower() == 'wide'):
									if (negation == True):
										features['Nose Width'] = ['Narrow','Medium']
									else:
										features['Nose Width'] = ['Wide']
				elif (pos == 'JJ'):
					if (word.lower() == 'fair'):
						if (negation == True):
							features['Skin Color'] = ['Black','Brown']
						else:
							features['Skin Color'] = ['White']						
					elif (word.lower() == 'nose'):
						currentFeature = 'nose'
						if (len(currentAdjectives) != 0):
							for adjective in currentAdjectives:
								if (adjective.lower() == 'long'):
									if (negation == True):
										features['Nose Length'] = ['Short','Medium']
									else:
										features['Nose Length'] = ['Long']
								elif (adjective.lower() == 'short'):
									if (negation == True):
										features['Nose Length'] = ['Long','Medium']
									else:
										features['Nose Length'] = ['Short']
								elif (adjective.lower() == 'narrow'):
									if (negation == True):
										features['Nose Width'] = ['Wide','Medium']
									else:
										features['Nose Width'] = ['Narrow']
								elif (adjective.lower() == 'wide'):
									if (negation == True):
										features['Nose Width'] = ['Narrow','Medium']
									else:
										features['Nose Width'] = ['Wide']
					elif (word.lower() == 'narrow'):
						if (currentFeature == 'nose'):
							if (negation == True):
								features['Nose Width'] = ['Wide','Medium']
							else:
								features['Nose Width'] = ['Narrow']
						elif (currentFeature == 'eye'):
							if (negation == True):
								features['Eye Width'] = ['Wide','Medium']
							else:
								features['Eye Width'] = ['Narrow']
						elif (currentFeature == ''):
							currentAdjectives.append('Narrow')
					elif (word.lower() == 'wide'):
						if (currentFeature == 'nose'):
							if (negation == True):
								features['Nose Width'] = ['Narrow','Medium']
							else:
								features['Nose Width'] = ['Wide']
						elif (currentFeature == 'eye'):
							if (negation == True):
								features['Eye Width'] = ['Narrow','Medium']
							else:
								features['Eye Width'] = ['Wide']
						elif (currentFeature == ''):
							currentAdjectives.append('wide')
					elif (word.lower() == 'big'):
						if (currentFeature == 'eye'):
							if (negation == True):
								features['Eye Length'] = ['Small','Medium']
							else:
								features['Eye Length'] = ['Big']
						elif (currentFeature == ''):
							currentAdjectives.append('big')
					elif (word.lower() == 'small'):
						if (currentFeature == 'eye'):
							if (negation == True):
								features['Eye Length'] = ['Big','Medium']
							else:
								features['Eye Length'] = ['Small']
						elif (currentFeature == ''):
							currentAdjectives.append('small')
					elif (word.lower() == 'short'):
						if (currentFeature == 'nose'):
							if (negation == True):
								features['Nose Width'] = ['Wide','Medium']
							else:
								features['Nose Width'] = ['Narrow']
						if (currentFeature == 'eye'):
							if (negation == True):
								features['Eye Width'] = ['Big','Medium']
							else:
								features['Eye Width'] = ['Small']
						elif (currentFeature == ''):
							currentAdjectives.append('short')
					elif (word[-2:] == 'an'):
						features['Race'] = [str(word[0].upper() + word[1:].lower())]
					elif (word[-2:] == 'ni'):
						features['Race'] = [str(word[0].upper() + word[1:].lower())]
					elif (word.lower() == 'brown'):
						features['Skin Color'] = ['Brown']
					elif (word.lower() == 'dark'):
						features['Skin Color'] = ['Black']
					elif (word.lower() == 'black'):
						features['Skin Color'] = ['Black']
					elif (word.lower() == 'white'):
						features['Skin Color'] = ['White']
				elif (pos == 'RB'):
					if (word.lower() == 'long'):
						if (currentFeature == 'nose'):
							if (negation == True):
								features['Nose Length'] = ['Short','Medium']
							else:
								features['Nose Length'] = ['Long']
						elif (currentFeature == ''):
							currentAdjectives.append('Long')
					elif (word.lower() == 'not'):
						negation = True
				elif (pos == 'DT'):
					if (word.lower() == 'no'):
						negation = True
				elif (pos == '.'):				
					negation = False
					currentFeature = ""
					currentAdjectives = []
			
	except Exception as e:
		print(str(e))

	print(features)

	return features


# All classes relating to GUI

class AddExistingDatabaseWindow(Frame):

	def __init__(self, master = None, database = None):
		Frame.__init__(self, master)
		
		self.master = master

		# New file names
		self.fileNames = set()

		# Dictionary of people already existing in the database
		self.database = database

		self.init_window()

	def init_window(self):
		self.master.title("Add New Database")
		self.pack(fill = BOTH, expand = 1)
		
		text = Label(self, text = "Select images", font=("Helvetica", 12))
		text.pack()
		text.place(x = 20, y = 20)

		button = Button(self, text = 'Browse', width=12, height=4, command = self.openFileBrowser)
		button.place(x = 300, y = 20)

		scrollBar = Scrollbar(root)
		scrollBar.pack(side=RIGHT, fill=Y)

		listBox = Listbox(self, width = 85, height = 30)
		listBox.pack()
		listBox.place(x = 50, y = 100)

		self.listBox = listBox

		listBox.config(yscrollcommand = scrollBar.set)
		scrollBar.config(command = listBox.yview)


	def second_window(self):
		text = Label(self, text = str(len(self.fileNames))+ " image(s) selected.", font=("Helvetica", 12))
		text.pack()
		text.place(x = 20, y = 40)

		button = Button(self, text="Next", width=12, height=4, command = self.startTraining)
		button.place(x = 600, y = 20)


	def startTraining(self):
		fileNames = self.fileNames
		self.master.destroy()

		child = Toplevel()
		child.geometry("850x600")

		subapp = AddExistingTrainingWindow(child, files = fileNames, database = self.database)
		child.mainloop()


	def openFileBrowser(self):
		fnames = askopenfilenames(parent = self.master, filetypes=(("All Files", "*.*"),
																   ("Windows Bitmap", "*.bmp"),
					                                        	   ("Portable Image Formats", "*.pbm;*.pgm;*ppm"),
					                                               ("Sun Raster", "*.sr;*.ras"),
					                                               ("JPEG", "*.jpeg;*.jpg;*.jpe"),
					                                               ("JPEG 2000", "*.jp2"),
					                                               ("TIFF Files", "*.tiff;*.tif"),
					                                               ("Portable Network Graphics", "*.png")))

		if fnames:
			try:
			    print("Got files!")
			except:                     # naked except is a bad idea
			    showerror("Open Source File", "Failed to read file\n'%s'" % fnames)

		# Split file names
		fileNames = root.tk.splitlist(fnames)

		for fileName in fileNames:
			self.fileNames.add(fileName)

		self.second_window()

		for file in self.fileNames:
			self.listBox.insert(END, file)

class AddExistingTrainingWindow(Frame):

	def __init__(self, master = None, database = None, files = None):
		Frame.__init__(self, master)
		
		self.master = master

		# Dictionary of people who already exists in the database
		self.database = database

		# New files selected goes here.
		self.fileNames = list(files)

		# For the "Next" Button
		self.arrayPos = -1

		# For "Trust the code" checkbox
		self.trustVar = IntVar()

		self.init_window()

	def init_window(self):
		self.master.title("Extract Features")
		self.pack(fill = BOTH, expand = 1)

		button = Button(self, text = 'Start Extracting Features', width = 20, height = 5, command = self.featureExtraction)
		button.place(x = 20, y = 20)


	def featureExtraction(self):

		self.featureExtractionNext()
		
		checkButton = Checkbutton(self, text="Trust the code.", variable=self.trustVar)
		checkButton.pack()
		checkButton.place(x = 250, y = 30)

		button = Button(self, text = 'Next', width=12, height=4, command = self.featureExtractionNext)
		button.place(x = 600, y = 20)

	def featureExtractionTrust(self):

		while (self.trustVar.get() == 1):
			if (self.arrayPos == len(self.fileNames) - 1):
				if (self.arrayPos >= 0):
					# For the last image.
					# This is the tuple which acts like a key for the dictionary.

					tpl = (
						self.genderText.get("1.0",'end-1c'),
						self.raceText.get("1.0",'end-1c'),
						self.glassesText.get("1.0",'end-1c'),
						self.moustacheText.get("1.0",'end-1c'),
						self.beardText.get("1.0",'end-1c'),
						self.noseLengthText.get("1.0",'end-1c'),
						self.noseWidthText.get("1.0",'end-1c'),
						self.eyeLengthText.get("1.0",'end-1c'),
						self.eyeWidthText.get("1.0",'end-1c'),
						self.skinColorText.get("1.0",'end-1c')			
					)
					if (tpl not in self.database):
						self.database[tpl] = set()
					
					self.database[tpl].add(str(self.fileNames[self.arrayPos]))

				button = Button(self, text = 'Save', width=12, height=4, command = self.featureExtractionSave)
				button.place(x = 600, y = 20)
				break
			elif (self.arrayPos < len(self.fileNames)):
				if (self.arrayPos >= 0):

					# This is the tuple which acts like a key for the dictionary.

					tpl = (
						self.genderText.get("1.0",'end-1c'),
						self.raceText.get("1.0",'end-1c'),
						self.glassesText.get("1.0",'end-1c'),
						self.moustacheText.get("1.0",'end-1c'),
						self.beardText.get("1.0",'end-1c'),
						self.noseLengthText.get("1.0",'end-1c'),
						self.noseWidthText.get("1.0",'end-1c'),
						self.eyeLengthText.get("1.0",'end-1c'),
						self.eyeWidthText.get("1.0",'end-1c'),
						self.skinColorText.get("1.0",'end-1c')			
					)
					if (tpl not in self.database):
						self.database[tpl] = set()
					
					self.database[tpl].add(str(self.fileNames[self.arrayPos]))
				
				self.arrayPos += 1

				# Get the classification prediction for that particular image.
				# Goes into the textboxes for displaying.

				featuresDict = featureExtraction(self.fileNames[self.arrayPos])
				genderClass = featuresDict['gender']
				glassesClass = featuresDict['glasses']
				moustacheClass = featuresDict['moustache']
				beardClass = featuresDict['beard']
				noseLengthClass = featuresDict['noseLength']
				noseWidthClass = featuresDict['noseWidth']
				eyeLengthClass = featuresDict['eyeLength']
				eyeWidthClass = featuresDict['eyeWidth']
				skinColorClass = featuresDict['skinColor']

				# race = ''

				self.showImage(img = self.fileNames[self.arrayPos])
				self.showFeatures(gender = genderClass, glasses = glassesClass, moustache = moustacheClass, beard = beardClass, noseLength = noseLengthClass, noseWidth = noseWidthClass, eyeLength = eyeLengthClass, eyeWidth = eyeWidthClass, skinColor = skinColorClass)


	def featureExtractionNext(self):

		if (self.trustVar.get() == 1):
			self.featureExtractionTrust()

		if (self.arrayPos == len(self.fileNames)-1):
			if (self.arrayPos >= 0):
				# For the last image.
				# This is the tuple which acts like a key for the dictionary.
				tpl = (
						self.genderText.get("1.0",'end-1c'),
						self.raceText.get("1.0",'end-1c'),
						self.glassesText.get("1.0",'end-1c'),
						self.moustacheText.get("1.0",'end-1c'),
						self.beardText.get("1.0",'end-1c'),
						self.noseLengthText.get("1.0",'end-1c'),
						self.noseWidthText.get("1.0",'end-1c'),
						self.eyeLengthText.get("1.0",'end-1c'),
						self.eyeWidthText.get("1.0",'end-1c'),
						self.skinColorText.get("1.0",'end-1c')			
					)
				if (tpl not in self.database):
					self.database[tpl] = set()
				
				self.database[tpl].add(str(self.fileNames[self.arrayPos]))

			button = Button(self, text = 'Save', width=12, height=4, command = self.featureExtractionSave)
			button.place(x = 600, y = 20)
		elif (self.arrayPos < len(self.fileNames)):
			if (self.arrayPos >= 0):
				# This is the tuple which acts like a key for the dictionary.
				tpl = (
						self.genderText.get("1.0",'end-1c'),
						self.raceText.get("1.0",'end-1c'),
						self.glassesText.get("1.0",'end-1c'),
						self.moustacheText.get("1.0",'end-1c'),
						self.beardText.get("1.0",'end-1c'),
						self.noseLengthText.get("1.0",'end-1c'),
						self.noseWidthText.get("1.0",'end-1c'),
						self.eyeLengthText.get("1.0",'end-1c'),
						self.eyeWidthText.get("1.0",'end-1c'),
						self.skinColorText.get("1.0",'end-1c')			
					)
				print(str(self.fileNames[self.arrayPos]))
				if (tpl not in self.database):
					self.database[tpl] = set()

				self.database[tpl].add(str(self.fileNames[self.arrayPos]))

			self.arrayPos += 1		

			# Get the classification prediction for that particular image.
			# Goes into the textboxes for displaying.

			featuresDict = featureExtraction(self.fileNames[self.arrayPos])
			genderClass = featuresDict['gender']
			glassesClass = featuresDict['glasses']
			moustacheClass = featuresDict['moustache']
			beardClass = featuresDict['beard']
			noseLengthClass = featuresDict['noseLength']
			noseWidthClass = featuresDict['noseWidth']
			eyeLengthClass = featuresDict['eyeLength']
			eyeWidthClass = featuresDict['eyeWidth']
			skinColorClass = featuresDict['skinColor']

			# race = ''

			self.showImage(img = self.fileNames[self.arrayPos])
			self.showFeatures(gender = genderClass, glasses = glassesClass, moustache = moustacheClass, beard = beardClass, noseLength = noseLengthClass, noseWidth = noseWidthClass, eyeLength = eyeLengthClass, eyeWidth = eyeWidthClass, skinColor = skinColorClass)


	def featureExtractionSave(self):
		file = asksaveasfile(mode='wb', defaultextension='.pickle')
		if file is None:
			print("No file opened")
		pickle.dump(self.database, file)
		file_name = file.name.split('.')[0]

		file.close()

		programFile = open('programData.pickle', 'rb')
		programData = pickle.load(programFile)
		programFile.close()

		programData['history'].insert(0, file_name)
		programData['databases'].append(file_name)

		programFile = open('programData.pickle', 'wb')
		pickle.dump(programData, programFile)
		programFile.close()

		print("Done! Save Successful")

		self.master.destroy()

	def showFeatures(self, gender='', race='Indian', glasses='', moustache='', beard='', noseLength='', noseWidth='', eyeLength='', eyeWidth='', skinColor=''):

		genderLabel = Label(self, text = "Gender", font=("Helvetica", 10))
		genderLabel.pack()
		genderLabel.place(x = 420, y = 100)
		self.genderText = Text(self, height=1, width=50)	
		self.genderText.pack()
		self.genderText.insert(END, gender)
		self.genderText.place(x = 420, y = 120)

		raceLabel = Label(self, text = "Race", font=("Helvetica", 10))
		raceLabel.pack()
		raceLabel.place(x = 420, y = 150)
		self.raceText = Text(self, height=1, width=50)	
		self.raceText.pack()
		self.raceText.insert(END, race)
		self.raceText.place(x = 420, y = 170)

		glassesLabel = Label(self, text = "Glasses", font=("Helvetica", 10))
		glassesLabel.pack()
		glassesLabel.place(x = 420, y = 200)
		self.glassesText = Text(self, height=1, width=50)	
		self.glassesText.pack()
		self.glassesText.insert(END, glasses)
		self.glassesText.place(x = 420, y = 220)

		moustacheLabel = Label(self, text = "Moustache", font=("Helvetica", 10))
		moustacheLabel.pack()
		moustacheLabel.place(x = 420, y = 250)
		self.moustacheText = Text(self, height=1, width=50)	
		self.moustacheText.pack()
		self.moustacheText.insert(END, moustache)
		self.moustacheText.place(x = 420, y = 270)

		beardLabel = Label(self, text = "Beard", font=("Helvetica", 10))
		beardLabel.pack()
		beardLabel.place(x = 420, y = 300)
		self.beardText = Text(self, height=1, width=50)	
		self.beardText.pack()
		self.beardText.insert(END, beard)
		self.beardText.place(x = 420, y = 320)

		noseLengthLabel = Label(self, text = "Nose Length", font=("Helvetica", 10))
		noseLengthLabel.pack()
		noseLengthLabel.place(x = 420, y = 350)
		self.noseLengthText = Text(self, height=1, width=50)	
		self.noseLengthText.pack()
		self.noseLengthText.insert(END, noseLength)
		self.noseLengthText.place(x = 420, y = 370)

		noseWidthLabel = Label(self, text = "Nose Width", font=("Helvetica", 10))
		noseWidthLabel.pack()
		noseWidthLabel.place(x = 420, y = 400)
		self.noseWidthText = Text(self, height=1, width=50)	
		self.noseWidthText.pack()
		self.noseWidthText.insert(END, noseWidth)
		self.noseWidthText.place(x = 420, y = 420)

		eyeLengthLabel = Label(self, text = "Eye Length", font=("Helvetica", 10))
		eyeLengthLabel.pack()
		eyeLengthLabel.place(x = 420, y = 450)
		self.eyeLengthText = Text(self, height=1, width=50)	
		self.eyeLengthText.pack()
		self.eyeLengthText.insert(END, eyeLength)
		self.eyeLengthText.place(x = 420, y = 470)

		eyeWidthLabel = Label(self, text = "Eye Width", font=("Helvetica", 10))
		eyeWidthLabel.pack()
		eyeWidthLabel.place(x = 420, y = 500)
		self.eyeWidthText = Text(self, height=1, width=50)	
		self.eyeWidthText.pack()
		self.eyeWidthText.insert(END, eyeWidth)
		self.eyeWidthText.place(x = 420, y = 520)

		skinColorLabel = Label(self, text = "Skin Color", font=("Helvetica", 10))
		skinColorLabel.pack()
		skinColorLabel.place(x = 420, y = 550)
		self.skinColorText = Text(self, height=1, width=50)	
		self.skinColorText.pack()
		self.skinColorText.insert(END, skinColor)
		self.skinColorText.place(x = 420, y = 570)

	def showImage(self, img = None):
		print(img)

		loadImage = Image.open(img)
		loadImage = loadImage.resize((360, 480), Image.ANTIALIAS)
		render = ImageTk.PhotoImage(loadImage)

		image = Label(self, image=render, width=360, height=480)
		image.image = render
		image.place(x = 40, y = 120)

class ShowRetrievedImages(Frame):

	def __init__(self, master = None, images = None):
		Frame.__init__(self, master)
		self.images = list(images)
		print(self.images)
		self.imagesPos = 0
		self.init_window()

	def init_window(self):
		self.master.title("Showing Retreived Images")
		self.pack(fill=BOTH, expand=1)
		self.showImage(img = self.images[self.imagesPos])
		text = Label(self, text = self.images[self.imagesPos].split('/')[-1][:-4], font=("Helvetica", 14))	
		text.pack()
		text.place(x = 250, y = 550)

		button = Button(self, text = 'Next', width = 20, height = 5, command = self.showNext)
		button.place(x = 375, y = 700)

		button = Button(self, text = 'Previous', width = 20, height = 5, command = self.showPrevious)
		button.place(x = 25, y = 700)


	def showImage(self, img = None):
		print(img)

		loadImage = Image.open(img)
		loadImage = loadImage.resize((360, 480), Image.ANTIALIAS)
		render = ImageTk.PhotoImage(loadImage)

		image = Label(self, image=render, width=360, height=480)
		image.image = render
		image.place(x = 120, y = 40)

	def showNext(self):
		if (self.imagesPos == len(self.images)-1):
			return

		self.imagesPos += 1
		self.showImage(img = self.images[self.imagesPos])
		text = Label(self, text = self.images[self.imagesPos].split('/')[-1][:-4], font=("Helvetica", 14))	
		text.pack()
		text.place(x = 250, y = 550)


	def showPrevious(self):
		if (self.imagesPos == 0):
			return

		self.imagesPos -= 1
		self.showImage(img = self.images[self.imagesPos])
		text = Label(self, text = self.images[self.imagesPos].split('/')[-1][:-4], font=("Helvetica", 14))		
		text.pack()
		text.place(x = 250, y = 550)

class SearchTextWindow(Frame):

	def __init__(self, master = None, files = None, database = None):
		Frame.__init__(self, master)
		
		self.master = master

		# Need to pass the database to the "ShowRetrievedImages" function.
		self.database = database
		
		self.init_window()

	def init_window(self):
		self.master.title("Extract Features")
		self.pack(fill = BOTH, expand = 1)

		label = Label(self, text = "Enter description", font=("Helvetica", 10))
		label.pack()
		label.place(x = 20, y = 20)
		self.text = Text(self, height=10, width=65)	
		self.text.pack()
		self.text.insert(END, '')
		self.text.place(x = 20, y = 50)

		button = Button(self, text = 'Search', width = 20, height = 5, command = self.displaySearch)
		button.place(x = 150, y = 250)

	def displaySearch(self):
		description = self.text.get("1.0",'end-1c')

		features = getNLPFeatures(description)

		setOfImages = set()

		# Order of importance is given here.

		for gender in features['Gender']:
			for skinColor in features['Skin Color']:
				for race in features['Race']:
					for glasses in features['Glasses']:
						for moustache in features['Moustache']:
							for beard in features['Beard']:
								for noseLength in features['Nose Length']:
									for noseWidth in features['Nose Width']:
										for eyeLength in features['Eye Length']:
											for eyeWidth in features['Eye Width']:
												try:
													featureTpl = (
														gender,
														race,
														glasses,
														moustache,
														beard,
														noseLength,
														noseWidth,
														eyeLength,
														eyeWidth,
														skinColor
													)

													for image in self.database[featureTpl]:
														setOfImages.add(image)
												except:
													continue

		child = Toplevel()
		child.geometry("600x800")

		subapp = ShowRetrievedImages(child, images = setOfImages)
		child.mainloop()

class TrainingWindow(Frame):
	
	def __init__(self, master = None, files = None):
		Frame.__init__(self, master)
		
		self.master = master
		self.database = dict()
		self.fileNames = list(files)
		self.arrayPos = -1
		self.trustVar = IntVar()
		self.init_window()

	def init_window(self):
		self.master.title("Extract Features")
		self.pack(fill=BOTH, expand=1)

		button = Button(self, text = 'Start Extracting Features', width = 20, height = 5, command = self.featureExtraction)
		button.place(x = 20, y = 20)


	def featureExtraction(self):

		self.featureExtractionNext()
		
		checkButton = Checkbutton(self, text="Trust the code.", variable=self.trustVar)
		checkButton.pack()
		checkButton.place(x = 250, y = 30)

		button = Button(self, text = 'Next', width=12, height=4, command = self.featureExtractionNext)
		button.place(x = 600, y = 20)

	def featureExtractionTrust(self):

		while (self.trustVar.get() == 1):
			if (self.arrayPos == len(self.fileNames) - 1):
				if (self.arrayPos >= 0):
					tpl = (
						self.genderText.get("1.0",'end-1c'),
						self.raceText.get("1.0",'end-1c'),
						self.glassesText.get("1.0",'end-1c'),
						self.moustacheText.get("1.0",'end-1c'),
						self.beardText.get("1.0",'end-1c'),
						self.noseLengthText.get("1.0",'end-1c'),
						self.noseWidthText.get("1.0",'end-1c'),
						self.eyeLengthText.get("1.0",'end-1c'),
						self.eyeWidthText.get("1.0",'end-1c'),
						self.skinColorText.get("1.0",'end-1c')			
					)
					if (tpl not in self.database):
						self.database[tpl] = set()
					
					self.database[tpl].add(str(self.fileNames[self.arrayPos]))

				button = Button(self, text = 'Save', width=12, height=4, command = self.featureExtractionSave)
				button.place(x = 600, y = 20)
				break
			elif (self.arrayPos < len(self.fileNames)):
				if (self.arrayPos >= 0):
					tpl = (
						self.genderText.get("1.0",'end-1c'),
						self.raceText.get("1.0",'end-1c'),
						self.glassesText.get("1.0",'end-1c'),
						self.moustacheText.get("1.0",'end-1c'),
						self.beardText.get("1.0",'end-1c'),
						self.noseLengthText.get("1.0",'end-1c'),
						self.noseWidthText.get("1.0",'end-1c'),
						self.eyeLengthText.get("1.0",'end-1c'),
						self.eyeWidthText.get("1.0",'end-1c'),
						self.skinColorText.get("1.0",'end-1c')			
					)
					if (tpl not in self.database):
						self.database[tpl] = set()
					
					self.database[tpl].add(str(self.fileNames[self.arrayPos]))
				
				self.arrayPos += 1

				# insert classifiers here

				featuresDict = featureExtraction(self.fileNames[self.arrayPos])
				genderClass = featuresDict['gender']
				glassesClass = featuresDict['glasses']
				moustacheClass = featuresDict['moustache']
				beardClass = featuresDict['beard']
				noseLengthClass = featuresDict['noseLength']
				noseWidthClass = featuresDict['noseWidth']
				eyeLengthClass = featuresDict['eyeLength']
				eyeWidthClass = featuresDict['eyeWidth']
				skinColorClass = featuresDict['skinColor']
				
				# race = ''

				self.showImage(img = self.fileNames[self.arrayPos])
				self.showFeatures(gender = genderClass, glasses = glassesClass, moustache = moustacheClass, beard = beardClass, noseLength = noseLengthClass, noseWidth = noseWidthClass, eyeLength = eyeLengthClass, eyeWidth = eyeWidthClass, skinColor = skinColorClass)



	def featureExtractionNext(self):

		if (self.trustVar.get() == 1):
			self.featureExtractionTrust()

		if (self.arrayPos == len(self.fileNames)-1):
			if (self.arrayPos >= 0):
				tpl = (
						self.genderText.get("1.0",'end-1c'),
						self.raceText.get("1.0",'end-1c'),
						self.glassesText.get("1.0",'end-1c'),
						self.moustacheText.get("1.0",'end-1c'),
						self.beardText.get("1.0",'end-1c'),
						self.noseLengthText.get("1.0",'end-1c'),
						self.noseWidthText.get("1.0",'end-1c'),
						self.eyeLengthText.get("1.0",'end-1c'),
						self.eyeWidthText.get("1.0",'end-1c'),
						self.skinColorText.get("1.0",'end-1c')			
					)
				if (tpl not in self.database):
					self.database[tpl] = set()
				
				self.database[tpl].add(str(self.fileNames[self.arrayPos]))

			button = Button(self, text = 'Save', width=12, height=4, command = self.featureExtractionSave)
			button.place(x = 600, y = 20)
		elif (self.arrayPos < len(self.fileNames)):
			if (self.arrayPos >= 0):
				tpl = (
						self.genderText.get("1.0",'end-1c'),
						self.raceText.get("1.0",'end-1c'),
						self.glassesText.get("1.0",'end-1c'),
						self.moustacheText.get("1.0",'end-1c'),
						self.beardText.get("1.0",'end-1c'),
						self.noseLengthText.get("1.0",'end-1c'),
						self.noseWidthText.get("1.0",'end-1c'),
						self.eyeLengthText.get("1.0",'end-1c'),
						self.eyeWidthText.get("1.0",'end-1c'),
						self.skinColorText.get("1.0",'end-1c')			
					)
				print(str(self.fileNames[self.arrayPos]))
				if (tpl not in self.database):
					self.database[tpl] = set()

				self.database[tpl].add(str(self.fileNames[self.arrayPos]))

			self.arrayPos += 1		

			featuresDict = featureExtraction(self.fileNames[self.arrayPos])
			genderClass = featuresDict['gender']
			glassesClass = featuresDict['glasses']
			moustacheClass = featuresDict['moustache']
			beardClass = featuresDict['beard']
			noseLengthClass = featuresDict['noseLength']
			noseWidthClass = featuresDict['noseWidth']
			eyeLengthClass = featuresDict['eyeLength']
			eyeWidthClass = featuresDict['eyeWidth']
			skinColorClass = featuresDict['skinColor']
			
			# race = ''

			self.showImage(img = self.fileNames[self.arrayPos])
			self.showFeatures(gender = genderClass, glasses = glassesClass, moustache = moustacheClass, beard = beardClass, noseLength = noseLengthClass, noseWidth = noseWidthClass, eyeLength = eyeLengthClass, eyeWidth = eyeWidthClass, skinColor = skinColorClass)


	def featureExtractionSave(self):
		file = asksaveasfile(mode='wb', defaultextension='.pickle')
		if file is None:
			print("No file opened")
		pickle.dump(self.database, file)
		file_name = file.name.split('.')[0]

		file.close()

		programFile = open('programData.pickle', 'rb')
		programData = pickle.load(programFile)
		programFile.close()

		programData['history'].insert(0, file_name)
		programData['databases'].append(file_name)

		programFile = open('programData.pickle', 'wb')
		pickle.dump(programData, programFile)
		programFile.close()

		print("Done! Save Successful")

		self.master.destroy()

	def showFeatures(self, gender='', race='Indian', glasses='', moustache='', beard='', noseLength='', noseWidth='', eyeLength='', eyeWidth='', skinColor=''):

		genderLabel = Label(self, text = "Gender", font=("Helvetica", 10))
		genderLabel.pack()
		genderLabel.place(x = 420, y = 100)
		self.genderText = Text(self, height=1, width=50)	
		self.genderText.pack()
		self.genderText.insert(END, gender)
		self.genderText.place(x = 420, y = 120)

		raceLabel = Label(self, text = "Race", font=("Helvetica", 10))
		raceLabel.pack()
		raceLabel.place(x = 420, y = 150)
		self.raceText = Text(self, height=1, width=50)	
		self.raceText.pack()
		self.raceText.insert(END, race)
		self.raceText.place(x = 420, y = 170)

		glassesLabel = Label(self, text = "Glasses", font=("Helvetica", 10))
		glassesLabel.pack()
		glassesLabel.place(x = 420, y = 200)
		self.glassesText = Text(self, height=1, width=50)	
		self.glassesText.pack()
		self.glassesText.insert(END, glasses)
		self.glassesText.place(x = 420, y = 220)

		moustacheLabel = Label(self, text = "Moustache", font=("Helvetica", 10))
		moustacheLabel.pack()
		moustacheLabel.place(x = 420, y = 250)
		self.moustacheText = Text(self, height=1, width=50)	
		self.moustacheText.pack()
		self.moustacheText.insert(END, moustache)
		self.moustacheText.place(x = 420, y = 270)

		beardLabel = Label(self, text = "Beard", font=("Helvetica", 10))
		beardLabel.pack()
		beardLabel.place(x = 420, y = 300)
		self.beardText = Text(self, height=1, width=50)	
		self.beardText.pack()
		self.beardText.insert(END, beard)
		self.beardText.place(x = 420, y = 320)

		noseLengthLabel = Label(self, text = "Nose Length", font=("Helvetica", 10))
		noseLengthLabel.pack()
		noseLengthLabel.place(x = 420, y = 350)
		self.noseLengthText = Text(self, height=1, width=50)	
		self.noseLengthText.pack()
		self.noseLengthText.insert(END, noseLength)
		self.noseLengthText.place(x = 420, y = 370)

		noseWidthLabel = Label(self, text = "Nose Width", font=("Helvetica", 10))
		noseWidthLabel.pack()
		noseWidthLabel.place(x = 420, y = 400)
		self.noseWidthText = Text(self, height=1, width=50)	
		self.noseWidthText.pack()
		self.noseWidthText.insert(END, noseWidth)
		self.noseWidthText.place(x = 420, y = 420)

		eyeLengthLabel = Label(self, text = "Eye Length", font=("Helvetica", 10))
		eyeLengthLabel.pack()
		eyeLengthLabel.place(x = 420, y = 450)
		self.eyeLengthText = Text(self, height=1, width=50)	
		self.eyeLengthText.pack()
		self.eyeLengthText.insert(END, eyeLength)
		self.eyeLengthText.place(x = 420, y = 470)

		eyeWidthLabel = Label(self, text = "Eye Width", font=("Helvetica", 10))
		eyeWidthLabel.pack()
		eyeWidthLabel.place(x = 420, y = 500)
		self.eyeWidthText = Text(self, height=1, width=50)	
		self.eyeWidthText.pack()
		self.eyeWidthText.insert(END, eyeWidth)
		self.eyeWidthText.place(x = 420, y = 520)

		skinColorLabel = Label(self, text = "Skin Color", font=("Helvetica", 10))
		skinColorLabel.pack()
		skinColorLabel.place(x = 420, y = 550)
		self.skinColorText = Text(self, height=1, width=50)	
		self.skinColorText.pack()
		self.skinColorText.insert(END, skinColor)
		self.skinColorText.place(x = 420, y = 570)

	def showImage(self, img = None):
		print(img)

		loadImage = Image.open(img)
		loadImage = loadImage.resize((360, 480), Image.ANTIALIAS)
		render = ImageTk.PhotoImage(loadImage)

		image = Label(self, image=render, width=360, height=480)
		image.image = render
		image.place(x = 40, y = 120)

class NewDatabaseWindow(Frame):

	def __init__(self, master = None):
		Frame.__init__(self, master)
		
		self.master = master

		# Empty set of files, before choosing the files for testing.
		self.fileNames = set()

		self.init_window()

	def init_window(self):
		self.master.title("Add New Database")
		self.pack(fill = BOTH, expand=1)
		
		text = Label(self, text = "Select images", font = ("Helvetica", 12))
		text.pack()
		text.place(x = 20, y = 20)

		button = Button(self, text = 'Browse', width = 12, height = 4, command = self.openFileBrowser)
		button.place(x = 300, y = 20)

		scrollBar = Scrollbar(root)
		scrollBar.pack(side=RIGHT, fill=Y)

		listBox = Listbox(self, width = 85, height = 30)
		listBox.pack()
		listBox.place(x = 50, y = 100)

		self.listBox = listBox

		listBox.config(yscrollcommand=scrollBar.set)
		scrollBar.config(command=listBox.yview)


	def second_window(self):
		text = Label(self, text = str(len(self.fileNames))+ " image(s) selected.", font = ("Helvetica", 12))
		text.pack()
		text.place(x = 20, y = 40)

		button = Button(self, text="Next", width = 12, height = 4, command = self.startTraining)
		button.place(x = 600, y = 20)


	def startTraining(self):
		fileNames = self.fileNames
		self.master.destroy()

		child = Toplevel()
		child.geometry("850x600")

		subapp = TrainingWindow(child, files = fileNames)
		child.mainloop()


	def openFileBrowser(self):
		fnames = askopenfilenames(parent = self.master, filetypes=(("All Files", "*.*"),
																  ("Windows Bitmap", "*.bmp"),
					                                        	  ("Portable Image Formats", "*.pbm;*.pgm;*ppm"),
					                                              ("Sun Raster", "*.sr;*.ras"),
					                                              ("JPEG", "*.jpeg;*.jpg;*.jpe"),
					                                              ("JPEG 2000", "*.jp2"),
					                                              ("TIFF Files", "*.tiff;*.tif"),
					                                              ("Portable Network Graphics", "*.png")))

		if fnames:
			try:
			    print("""here it comes: self.settings["template"].set(fname)""")
			except:                     # <- naked except is a bad idea
			    showerror("Open Source File", "Failed to read file\n'%s'" % fnames)

		fileNames = root.tk.splitlist(fnames)

		for fileName in fileNames:
			self.fileNames.add(fileName)

		self.second_window()

		for file in self.fileNames:
			self.listBox.insert(END, file)

class MainWindow(Frame):

		def __init__(self, master = None):
			Frame.__init__(self, master)

			# Check for program data file.
			# If not there, create one. (First time use, maybe)

			try:
				programFile = open('programData.pickle', 'rb')
			except:
				programData = {'history':[], 'databases':[]}
				programFile = open('programData.pickle', 'wb')
				pickle.dump(programData, programFile)
				programFile.close()

			programFile = open('programData.pickle', 'rb')
			programData = pickle.load(programFile)
			programFile.close()

			self.programData = programData

			# For when a database may exist. Loading a  database.
			self.database = None
			
			self.master = master
			self.init_window()


		def init_window(self):
			self.master.title("What's that face?")
			self.pack(fill=BOTH, expand=1)

			menuBar = Menu(self.master)
			self.master.config(menu = menuBar)

			file = Menu(menuBar)
			file.add_command(label = 'New Image Database', command = self.newDatabase)
			file.add_command(label = 'Load Image Database', command = self.loadDatabase)
			file.add_command(label = 'Close', command = self.quitProgram)
			menuBar.add_cascade(label = 'File', menu = file)

			edit = Menu(menuBar)
			edit.add_command(label = 'Add New Images', command = self.addToExistingDatabase)
		#	edit.add_command(label = 'Delete Images') # Not yet implemented
			menuBar.add_cascade(label = 'Edit', menu = edit)

			search = Menu(menuBar)
			search.add_command(label = 'Text', command = self.searchText)
		#	search.add_command(label = 'Form (Beta)') # Not yet implemented
		#	search.add_command(label = 'Speech (Beta)') # Not yet implemented
			menuBar.add_cascade(label = 'Search', menu = search)

			help = Menu(menuBar)
			help.add_command(label = 'How to Use')
			help.add_command(label = 'Credits')
			menuBar.add_cascade(label = 'Help', menu = help)

			text = Label(self, text = "What's that face?", font = ("Helvetica", 48))
			text.pack()
			text.place(x = 150, y = 125)

			text = Label(self, text = "Recent Databases", font = ("Helvetica", 16))
			text.pack()
			text.place(x = 300, y = 250)

			if (len(self.programData['history']) == 0):
				text = Label(self, text = "You have no databases.", font = ("Helvetica", 16))
				text.pack()
				text.place(x = 275, y = 320)
	
				text = Label(self, text = "Would you like to create a new database?", font = ("Helvetica", 16))
				text.pack()
				text.place(x = 200, y = 350)

				button = Button(self, text = "Create", width = 15, height = 5, command = self.newDatabase)
				button.place(x = 225, y = 400)

				button = Button(self, text = "Exit", width = 15, height = 5, command = self.quitProgram)
				button.place(x = 415, y = 400)
			else:
				nextButtonX = 50
				nextButtonY = 350
				for i in self.programData['history'][:8]:
					button = Button(self, text=i.split('/')[-1].split('.')[0], width=12, height=5, command = lambda: self.loadDatabaseRecent(fname = i+".pickle"))
					button.place(x = nextButtonX, y = nextButtonY)
					nextButtonX += 190
					if (nextButtonX > 650):
						nextButtonX = 50
						nextButtonY += 125

		def newDatabase(self):
			child = Toplevel()
			child.geometry("800x600")

			subapp = NewDatabaseWindow(child)
			child.mainloop()

		def loadDatabaseRecent(self, fname = None):
			if fname:
				try:
				    print("""here it comes: self.settings["template"].set(fname)""")
				except:                     # <- naked except is a bad idea
				    showerror("Open Source File", "Failed to read file\n'%s'" % fname)

			fileObj = open(fname, 'rb')

			# To get the database name without full path and no extension
			databaseName = fileObj.name.split('/')[-1].split('.')[0]

			self.database = pickle.load(fileObj)
			fileObj.close()

			text = Label(self, text = databaseName + " loaded successfully!", font = ("Helvetica", 10))
			text.pack()
			text.place(x = 500, y = 50)

		def loadDatabase(self):
			fname = askopenfilename(parent = self.master, filetypes=(("All Files", "*.*"),
																  	 ("Pickle", "*.pickle")))

			if fname:
				try:
				    print("""here it comes: self.settings["template"].set(fname)""")
				except:                     # <- naked except is a bad idea
				    showerror("Open Source File", "Failed to read file\n'%s'" % fname)

			fileObj = open(fname, 'rb')
			databaseName = fileObj.name.split('/')[-1].split('.')[0]
			self.database = pickle.load(fileObj)
			fileObj.close()

			text = Label(self, text = databaseName + " loaded successfully!", font = ("Helvetica", 10))
			text.pack()
			text.place(x = 500, y = 50)

		def addToExistingDatabase(self):
			if (self.database == None):
				self.loadDatabase()

			child = Toplevel()
			child.geometry("800x600")

			subapp = AddExistingDatabaseWindow(child, database = self.database)
			child.mainloop()

		def quitProgram(self):
			exit()

		def searchText(self):
			if (self.database == None):
				self.loadDatabase()

			child = Toplevel()
			child.geometry("500x400")

			subapp = SearchTextWindow(child, database = self.database)
			child.mainloop()

# Initialize the starting window 

root = Tk()
root.geometry("800x600")

# Start the first window!

app = MainWindow(root)
root.mainloop()