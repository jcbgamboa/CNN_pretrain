from __future__ import print_function

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
				'nonlinearity': lasagne.nonlinearities.very_leaky_rectify,
				'W':lasagne.init.Normal(0.1),
				'b':lasagne.init.Constant(1)
			}),

		#(MaxPool1DLayerFast, {'pool_size': 2}),
		(Conv1DLayerFast, {
				'num_filters': 10,
				'filter_size': sliding_window_length,
				'pad':'same',
				'nonlinearity': lasagne.nonlinearities.very_leaky_rectify,
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
		max_epochs = 50,

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

def run_networknl(netnl, train_wins, train_wins_labels,
			test_wins, test_wins_labels):
	netnl = train_networknl(netnl, train_wins, train_wins_labels)

	print("Testing Neural Network...")
	y_pred = netnl.predict(test_wins)

	print_errors_network(test_wins_labels, y_pred)


# -------------------------------------------------------------- LASAGNE NETWORK

# Here, `lg` indicates "lasagne". It is here in opposition to `nl` in the rest
# of the code, above, which mean "nolearn" (i.e., using `nolearn`)

def create_networklg(input_var, sliding_window_length,
			mode = "same_random",
			weights = None):
	number_of_filters = [20, 10]
	if (mode == "same_random"):
		filter_sizes = [sliding_window_length, sliding_window_length]
		last_size = filter_sizes[-1]
		pad = 'same'

	if ("valid" in mode):
		filter_sizes = [sliding_window_length/2, sliding_window_length/2]
		last_size = 2
		pad = 'valid'

	W1 = lasagne.init.Normal(0.1)
	W2 = lasagne.init.Normal(0.1)
	b1 = lasagne.init.Constant(1)
	b2 = lasagne.init.Constant(1)
	if (("weights" in mode) and (weights is not None)):
		W1 = np.squeeze(np.transpose(weights[0], axes=[1,3,2,0]),
				axis=(0,)).astype(np.float32)
		b1 = np.squeeze(weights[1]).astype(np.float32)
		W2 = np.squeeze(np.transpose(weights[2], axes=[1,3,2,0]),
				axis=(0,)).astype(np.float32)
		b2 = np.squeeze(weights[3]).astype(np.float32)

	netlg = lasagne.layers.InputLayer(
			shape = (None, 1, sliding_window_length),
			input_var = input_var
		)
	netlg = Conv1DLayerFast(
			netlg,
			num_filters = number_of_filters[0],
			filter_size = filter_sizes[0],
			pad = pad,
			nonlinearity = lasagne.nonlinearities.rectify,
			W = W1,
			b = b1
		)
	netlg = Conv1DLayerFast(
			netlg,
			num_filters = number_of_filters[1],
			filter_size = filter_sizes[1],
			pad = pad,
			nonlinearity = lasagne.nonlinearities.rectify,
			W = W2,
			b = b2
		)
	netlg = lasagne.layers.ReshapeLayer(
			netlg,
			shape = [-1, number_of_filters[-1] * last_size]
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

def iterate_minibatcheslg(wins, wins_labels, batch_size):
	# A lot based on `iterate_minibatches()` from the Lasagne tutorial
	for start_idx in range(0, len(wins), batch_size):
		curr_elems = slice(start_idx, start_idx + batch_size)
		#print("yielding elements {} to {}".format(
		#	start_idx, start_idx + batch_size))
		yield wins[curr_elems], wins_labels[curr_elems]

def train_networklg(netlg, input_var, target_var,
			train_wins, train_wins_labels, batch_size):
	# Because Theano is symbolic, we need to define some variables here
	# - `prediction` is whatever the network will output
	# - `loss` is out training loss (which we want to minimize)
	# - `params` are the variables we can change in the network (e.g.,
	#	connection weights and biases)
	# - `updates` is how to change it (lasagne offers some options already)
	# (Also: the default parameters of the lasagne's implementation of the
	# Adam optimizer are exactly the same as those from TensorFlow)
	prediction = lasagne.layers.get_output(netlg)

	loss = loss_sum_of_squared_errorslg(prediction, target_var)
	params = lasagne.layers.get_all_params(netlg, trainable=True)
	updates = lasagne.updates.adam(loss, params)
	training_function = theano.function([input_var, target_var],
					loss, updates = updates)

	# To monitoring the progress of the network, lasagne's tutorial suggest
	# the definition of these variables
	test_prediction = lasagne.layers.get_output(netlg, deterministic=True)
	test_loss = loss_sum_of_squared_errorslg(test_prediction, target_var)
	validation_function = theano.function([input_var, target_var],
					[test_loss, prediction])

	# Finally, actually trains the network
	num_iterations = 1000
	for epoch in range(num_iterations):
		train_err = 0
		for batch in iterate_minibatcheslg(train_wins,
					train_wins_labels, batch_size):
			inputs, targets = batch
			train_err += training_function(inputs, targets)

		print("Epoch: {}\tSum of Squared Errors: {}".format(
			epoch, train_err))

	return validation_function

def test_networklg(netlg, input_var, target_var, validation_function,
			test_wins, test_wins_labels, batch_size):
	test_err = 0
	predictions = []

	#print("Length of test_wins: {}".format(len(test_wins)))

	for batch in iterate_minibatcheslg(test_wins,
					test_wins_labels, batch_size):
		inputs, targets = batch
		(err, prediction) = validation_function(inputs, targets)
		test_err += err
		for i in prediction:
			predictions.append(i)

	return test_err, predictions

def run_networklg(train_wins, train_wins_labels,
			test_wins, test_wins_labels,
			mode = "same_random",
			batch_size = None,
			weights = None):

	input_var = T.tensor3('inputs')
	target_var = T.fcol('targets')

	if (batch_size == None):
		batch_size = len(train_wins_labels)

	#print("Will create Lasagne network")
	netlg = create_networklg(input_var, sliding_window_length,
				mode = mode, weights = weights)

	#print("Will train network")
	validation_function = train_networklg(netlg, input_var, target_var,
				train_wins, train_wins_labels, batch_size)

	test_loss, predictions = test_networklg(netlg, input_var, target_var,
			validation_function, test_wins, test_wins_labels, batch_size)

	return test_loss, predictions


# ---------------------------------------------------------------- MAIN FUNCTION

def gen_sliding_windows_mean(time_series):
	for i in time_series:
		print(i, np.avg(i))

def generate_output_files(folder, file_name, test_wins_labels, predictions):
	file_name_stem = "./" + folder + "/" + file_name
	plot_actual_and_predicted(file_name_stem + "_plot_actual_and_predicted",
				test_wins_labels, predictions)
	generate_csv_output(file_name_stem + "output",
				test_wins_labels, predictions)

def generate_csv_output(file_name, y, predictions):
	out = []
	for i in range(len(y)):
		out.append([y[i], predictions[i]])
	np.savetxt(file_name + '.csv', out, delimiter = ',', fmt = '%5.5f')

def plot_actual_and_predicted(file_name, y, pred):
	fig = plt.figure()
	ax = fig.add_subplot(211)
	ax.plot(y)
	plt.ylabel('Actual Values')
	plt.grid(True)
	plt.title('Actual and Predicted Values of TS')

	ax = fig.add_subplot(212)
	ax.plot(pred, color='r')
	plt.ylabel('Predicted Values')
	plt.xlabel('Number of Timestamps')
	plt.grid(True)

	fig.savefig(file_name + '.png', edgecolor=None)

def print_errors_network(test_wins_labels, y_pred):
	error = y_pred - test_wins_labels
	for i, _ in enumerate(y_pred):
		print("y: {a}\ty_pred: {b}\terror: {c}".format(
			a=test_wins_labels[i][0], b=y_pred[i][0], c=error[i][0]))
	print(error)

if __name__ == '__main__':
	# ----------------------------- PARAMETERS
	# Size of the Sliding Window
	sliding_window_length = 10

	# If batch_size is None, use the entire training set [for now, this is
	# what the code is doing]
	batch_size = None

	# The following modes are supported:
	# * 'same_random': Use convolutional layers with `same` padding, and
	#	initialize the weights randomly
	# * 'valid_random': Use convolutional layers with `valid` padding, and
	#	initialize the weights randomly
	# * 'valid_weights': Use convolutional layers with `valid` padding, and
	#	initialize the weights by reading the "weights_input_file"
	mode = 'valid_weights'

	# The name of the file where the weights (generated, e.g., by a CAES or
	# a CDBN) are stored. These are used to initialize the CNN.
	#
	# The file should contain the weights to be used with *all* CNNs that
	# will be trained by this run of the program
	weights_input_file = 'cdbn_models.mat'
	if (weights_input_file):
		weights = sio.loadmat(weights_input_file)['cdbn_models']

	# ----------------------------- TRAIN CNN
	recursion_limit = 10000
	sys.setrecursionlimit(recursion_limit)

	all_losses = []

	input_folder = './yahoo_dataset/A3Benchmark/'
	input_file_name_stem = "A3Benchmark-TS"

	for i in range(1, 7):
		print("Loading Yahoo file {}".format(i))
		input_file = input_folder + input_file_name_stem +\
				'{}'.format(i) + '.csv'

		train_wins, train_wins_labels, \
			test_wins, test_wins_labels = load_data(
					input_file, sliding_window_length)

		gen_sliding_windows_mean(test_wins)
		sys.exit()

		# FIXME: For now, I am only using the Lasagne implementation
		#netnl = create_networknl(sliding_window_length)
		#run_networknl(netnl, train_wins, train_wins_labels,
		#		test_wins, test_wins_labels)

		print("Will run network")
		test_loss, predictions = run_networklg(train_wins,
					train_wins_labels,
					test_wins,
					test_wins_labels,
					mode = mode,
					batch_size = batch_size,
					weights = weights[i-1])

		print("Will output results")
		#print_errors_network(test_wins_labels, predictions)
		generate_output_files('results_yahoo', 'TS{}'.format(i),
					test_wins_labels, predictions)
		all_losses.append(test_loss)

	for i, loss in enumerate(all_losses):
		print("Loss[{}] = {}".format(i, loss))


