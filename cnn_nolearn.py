#! /usr/bin/python

from __future__ import print_function

print("Importing libraries...")

import os, sys, urllib, gzip
try:
    import cPickle as pickle
except:
    import pickle

import scipy.io as sio
import matplotlib
matplotlib.use('Agg') # Change matplotlib backend, in case we have no X server running..
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
from IPython.display import Image as IPImage
from PIL import Image

from lasagne.layers import get_output, InputLayer, DenseLayer, Upscale2DLayer, ReshapeLayer
from lasagne.nonlinearities import sigmoid, linear, rectify
from lasagne.updates import nesterov_momentum
from lasagne.objectives import categorical_crossentropy
from nolearn.lasagne import NeuralNet, BatchIterator, PrintLayerInfo

from lasagne.layers import Conv1DLayer as Conv1DLayerSlow
from lasagne.layers import MaxPool1DLayer as MaxPool1DLayerSlow
try:
    from lasagne.layers.cuda_convnet import Conv1DCCLayer as Conv1DLayerFast
    from lasagne.layers.cuda_convnet import MaxPool1DCCLayer as MaxPool1DLayerFast
    print('Using cuda_convnet (faster)')
except ImportError:
    from lasagne.layers import Conv1DLayer as Conv1DLayerFast
    from lasagne.layers import MaxPool1DLayer as MaxPool1DLayerFast
    print('Using lasagne.layers (slower)')


def create_sliding_windows(data, width=20):
	ret = np.zeros([data.size-(width-1), width], dtype=np.float32)
	for i in range(0, data.size-(width-1)):
		ret[i, :] = (data[i:i+width])
	return ret

def load_data(file_name):
	csv_data = np.loadtxt(open(file_name, "rb"),
				delimiter=",", skiprows=1)

	train_data = csv_data[:, 1].astype(np.float32)

	# Gets the input data. We leave the last time-step out: it is going to
	# be the label for the last input window.
	train_wins = train_data[0:399]
	test_wins  = train_data[400:-2]

	# Convert the input data into windows.
	train_wins = np.expand_dims(create_sliding_windows(train_wins), 1)
	test_wins  = np.expand_dims(create_sliding_windows(test_wins), 1)

	train_wins_labels = np.expand_dims(train_data[20:400], 1)
	test_wins_labels  = np.expand_dims(train_data[420:-1], 1)

	return train_wins, train_wins_labels, test_wins, test_wins_labels


recursion_limit = 10000
print("Setting recursion limit to {rl}".format(rl = recursion_limit))
sys.setrecursionlimit(recursion_limit)

print("Loading Yahoo dataset... (actually, for now, only one file)")
file_name = "./project_code/yahoo_dataset/A1Benchmark/real_4.csv"
train_wins, train_wins_labels, test_wins, test_wins_labels = load_data(file_name)

print("Types: {a}, {b}, {c}, {d}".format(
	a=train_wins.dtype, b=train_wins_labels.dtype,
	c=test_wins.dtype, d=test_wins_labels.dtype))


# This should replicate exactly the CNN I am using with the Deep Learning
# Toolbox, with the difference that the DenseLayer in the end has "linear"
# activation.
#
# Also notice that we use `regression=True` when initializing the `NeuralNet`.
# This will make the Loss function a Sum of Squared Errors (by default, lasagne
# would apparently assume I am doing classification and use cross entropy).
layers = [
    # layer dealing with the input data
    (InputLayer, {'shape': (None, 1, 20)}),

    # first stage of our convolutional layers
    (Conv1DLayerFast, {'num_filters': 20, 'filter_size': 11, 'nonlinearity': rectify}),
#    (MaxPool1DLayerFast, {'pool_size': 2}),
    (Conv1DLayerFast, {'num_filters': 10, 'filter_size': 2, 'nonlinearity': rectify}),
#    (MaxPool1DLayerFast, {'pool_size': 2}),
    #(Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    #(Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    #(Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),

    # second stage of our convolutional layers
    #(Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    #(Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    #(Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    #(MaxPool2DLayer, {'pool_size': 2}),

    # two dense layers with dropout
    #(DenseLayer, {'num_units': 64}),
    #(DropoutLayer, {}),
    #(DenseLayer, {'num_units': 64}),

    # the output layer
    (DenseLayer, {'num_units': 1, 'nonlinearity': linear}),
]

print("Creating Neural Network...")
cnn = NeuralNet(
    layers=layers,
    max_epochs=1000,
    
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.975,
    
    regression=True,
    verbose=1
)

print("Training Neural Network...")
cnn.fit(train_wins, train_wins_labels)


print("Testing Neural Network...")
y_pred = cnn.predict(test_wins)

error = y_pred - test_wins_labels
for i, _ in enumerate(y_pred):
	print("y: {a}\ty_pred: {b}\terror: {c}".format(
		a=test_wins_labels[i], b=y_pred[i], c=error[i]))
#print(error)


