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

import theano
import theano.tensor as T
import lasagne
import nolearn.lasagne

try:
    from lasagne.layers.cuda_convnet import Conv1DCCLayer as Conv1DLayerFast
    from lasagne.layers.cuda_convnet import MaxPool1DCCLayer as MaxPool1DLayerFast
    print('Using cuda_convnet (faster)')
except ImportError:
    from lasagne.layers import Conv1DLayer as Conv1DLayerFast
    from lasagne.layers import MaxPool1DLayer as MaxPool1DLayerFast
    print('Using lasagne.layers (slower)')



# -------------------------------------------- FUNCTIONS FOR LOADING THE DATASET

def print_dataset(train_data, train_wins, test_wins,
			train_wins_labels, test_wins_labels):
	print("--- Information on the entire dataset ---")
	print("Size of train_data: {}".format(train_data.shape))

	print("--- Information on the training set ---")
	print("Size of train_wins: {}".format(train_wins.shape))
	print("Size of train_wins_labels: {}".format(train_wins_labels.shape))

	print("Sample of train_wins:")
	print(train_wins[0:10])
	print("Sample of train_wins_labels:")
	print(train_wins_labels[0:10])

	print("--- Information on the test set ---")
	print("Size of test_wins: {}".format(test_wins.shape))
	print("Size of test_wins_labels: {}".format(test_wins_labels.shape))

	print("Sample of test_wins:")
	print(test_wins[0:10])
	print("Sample of test_wins_labels:")
	print(test_wins_labels[0:10])

def create_sliding_windows(data, width=10):
	ret = np.zeros([data.size-(width-1), width], dtype=np.float32)
	for i in range(0, data.size-(width-1)):
		ret[i, :] = (data[i:i+width])
	return ret

def load_data(file_name, sliding_window_length=10):
	csv_data = np.loadtxt(open(file_name, "rb"),
				delimiter=",", skiprows=1)

	train_data = csv_data[:, 1].astype(np.float32)

	# Gets the input data. We leave the last time-step out: it is going to
	# be the label for the last input window.
	train_wins = train_data[0:399]
	test_wins  = train_data[400:-2]

	# Convert the input data into windows.
	train_wins = np.expand_dims(create_sliding_windows(
					train_wins, sliding_window_length), 1)
	test_wins  = np.expand_dims(create_sliding_windows(
					test_wins, sliding_window_length), 1)

	train_wins_labels = np.expand_dims(
				train_data[sliding_window_length:400], 1)
	test_wins_labels  = np.expand_dims(
				train_data[400 + sliding_window_length:-1], 1)

	#print_dataset(train_data, train_wins, test_wins,
	#		train_wins_labels, test_wins_labels)

	return train_wins, train_wins_labels, test_wins, test_wins_labels



# -------------------------------------------------------------- NOLEARN NETWORK

# Here, `nl` indicates "nolearn". It is here in opposition to `lg` in the rest
# of the code, below, which mean "lasagne" (i.e., without `nolearn`)

def create_networknl(sliding_window_length = 10):
	# This should replicate exactly the CNN I am using with the Deep Learning
	# Toolbox, with the difference that the DenseLayer in the end has "linear"
	# activation.
	#
	# Also notice that we use `regression=True` when initializing the `NeuralNet`.
	# This will make the Loss function a Sum of Squared Errors (by default, lasagne
	# would apparently assume I am doing classification and use cross entropy).
	layers = [
		# layer dealing with the input data
		(lasagne.layers.InputLayer, {
				'shape': (None, 1, sliding_window_length)
			}),

		# first stage of our convolutional layers
		(Conv1DLayerFast, {
				'num_filters': 20,
				'filter_size': sliding_window_length,
				'pad':'same',
				'nonlinearity': lasagne.nonlinearities.rectify,
				'W':lasagne.init.Normal(0.1),
				'b':lasagne.init.Constant(1)
			}),

		#(MaxPool1DLayerFast, {'pool_size': 2}),
		(Conv1DLayerFast, {
				'num_filters': 10,
				'filter_size': sliding_window_length,
				'pad':'same',
				'nonlinearity': lasagne.nonlinearities.rectify,
				'W':lasagne.init.Normal(0.1),
				'b':lasagne.init.Constant(1)
			}),

		# the output layer
		(lasagne.layers.ReshapeLayer, {
				'shape': [-1, 100]
			}),
		(lasagne.layers.DenseLayer, {
				'num_units': 1,
				'nonlinearity': lasagne.nonlinearities.linear,
				'b': None
			}),
		]

	print("Creating Neural Network...")
	netnl = nolearn.lasagne.NeuralNet(
		layers = layers,
		max_epochs = 1000,

		update = lasagne.updates.nesterov_momentum,
		update_learning_rate = 0.001,
		update_momentum = 0.0,

		regression = True,
		verbose = 1
		)

	return netnl

def train_networknl(netnl, train_wins, train_wins_labels):
	print("Training Neural Network...")
	return netnl.fit(train_wins, train_wins_labels)

def print_errors_networknl(y_pred, test_wins_labels):
	error = y_pred - test_wins_labels
	for i, _ in enumerate(y_pred):
		print("y: {a}\ty_pred: {b}\terror: {c}".format(
			a=test_wins_labels[i], b=y_pred[i], c=error[i]))
	print(error)

def run_networknl(netnl, train_wins, train_wins_labels,
			test_wins, test_wins_labels):
	netnl(train_networknl(netnl, train_wins, train_wins_labels))

	print("Testing Neural Network...")
	y_pred = netnl.predict(test_wins)

	print_errors_networknl(y_pred, test_wins_labels)


# -------------------------------------------------------------- LASAGNE NETWORK

# Here, `lg` indicates "lasagne". It is here in opposition to `nl` in the rest
# of the code, above, which mean "nolearn" (i.e., using `nolearn`)

def create_networklg(sliding_window_length):
	netlg = lasagne.layers.InputLayer(
			shape = (None, 1, sliding_window_length)
		)

	netlg = Conv1DLayerFast(
			netlg,
			num_filters = 20,
			filter_size = sliding_window_length,
			pad = 'same',
			nonlinearity = lasagne.nonlinearities.rectify,
			W = lasagne.init.Normal(0.1),
			b = lasagne.init.Constant(1)
		)

	netlg = Conv1DLayerFast(
			netlg,
			num_filters = 10,
			filter_size = sliding_window_length,
			pad = 'same',
			nonlinearity = lasagne.nonlinearities.rectify,
			W = lasagne.init.Normal(0.1),
			b = lasagne.init.Constant(1)
		)

	netlg = lasagne.layers.ReshapeLayer(
			netlg,
			shape = [-1, 100]
		)

	netlg = lasagne.layers.DenseLayer(
			netlg,
			num_units = 1,
			nonlinearity = lasagne.nonlinearities.linear,
			b = None
		)

	return netlg

def loss_sum_of_squared_errorslg(prediction, target_var):
	# Takes the squared error
	loss = lasagne.objectives.squared_error(prediction, target_var)

	# Reduces the vector (aggregate) by summing its values. 
	return lasagne.objectives.aggregate(loss, mode='sum')

def train_networklg(netlg, target_var,
			train_wins, train_wins_labels):
	# Because Theano is symbolic, we need to define some variables here
	# - `prediction` is whatever the network will output
	# - `loss` is out training loss (which we want to minimize)
	# - `params` are the variables we can change in the network (e.g.,
	#	connection weights and biases)
	# - `updates` is how to change it (lasagne offers some options already)

	prediction = lasagne.layers.get_output(netlg)
	loss = loss_sum_of_squared_errorslg(prediction, target_var)
	params = lasagne.layers.get_all_params_values(netlg, trainable=True)

	# The default parameters of the lasagne's implementation of the Adam
	# optimizer are exactly the same as those from TensorFlow
	updates = lasagne.updates.adam(loss, params)

	# To monitoring the progress of the network, lasagne's tutorial suggest
	# the definition of these variables
	test_prediction = lasagne.layers.get_output(netlg, deterministic=True)
	test_loss = loss_sum_of_squared_errorslg(test_prediction, target_var)


def print_errors_networklg(y_pred, test_wins_labels):
	pass

def run_networklg(netlg, train_wins, train_wins_labels,
			test_wins, test_wins_labels):

	input_var = T.tensor4('inputs')
	target_var = T.ivector('targets')

	netlg = create_networklg(sliding_window_length)


# ---------------------------------------------------------------- MAIN FUNCTION

if __name__ == '__main__':
	sliding_window_length = 10
	recursion_limit = 10000
	print("Setting recursion limit to {rl}".format(rl = recursion_limit))
	sys.setrecursionlimit(recursion_limit)

	print("Loading Yahoo dataset... (actually, for now, only one file)")
	file_name = "./yahoo_dataset/A3Benchmark/A3Benchmark-TS1.csv"
	train_wins, train_wins_labels, test_wins, test_wins_labels = load_data(
						file_name, sliding_window_length)

	print("Types: {a}, {b}, {c}, {d}".format(
		a=train_wins.dtype, b=train_wins_labels.dtype,
		c=test_wins.dtype, d=test_wins_labels.dtype))

	#netnl = create_networknl(sliding_window_length)
	#run_networknl(netnl, train_wins, train_wins_labels,
	#		test_wins, test_wins_labels)

	netlg = create_networklg(sliding_window_length)
	run_networklg(netlg, train_wins, train_wins_labels,
			test_wins, test_wins_labels)



