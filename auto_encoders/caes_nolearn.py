#! /usr/bin/python

from __future__ import print_function

# Python common stuff
import argparse
import sys
try:
    import cPickle as pickle
except:
    import pickle

import scipy.io as sio
import numpy as np

# Import all stuff from Lasagne
# TODO: somehow clean this
from lasagne.layers import InputLayer, DenseLayer, Upscale2DLayer, ReshapeLayer
from lasagne.nonlinearities import rectify #, leaky_rectify, tanh
from lasagne.updates import nesterov_momentum
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayerFast
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayerFast
    print('Using cuda_convnet (faster)')
except ImportError:
    from lasagne.layers import Conv2DLayer as Conv2DLayerFast
    from lasagne.layers import MaxPool2DLayer as MaxPool2DLayerFast
    print('Using lasagne.layers (slower)')

from nolearn.lasagne import NeuralNet


def load_data(file_name):
	# I need this `load_data` thing because I am getting the "fold" (for
	# cross validation) from MatLab).
	#
	# TODO: eliminate this need
	mat_contents = sio.loadmat(file_name);
	X = mat_contents['CAES_train_data']
	X = np.reshape(X, (-1, 1, 28, 28))
	X_out = X.reshape((X.shape[0], -1))
	return X, X_out


def create_caes(X):
	conv_num_filters1 = 9
	conv_num_filters2 = 16
	filter_size1 = 7
	filter_size2 = 6
	pool_size = 2
	encode_size = 16
	dense_mid_size = 128
	pad_in = 'valid'
	pad_out = 'full'

	# Notice that, by default, Lasagne already uses a Xavier initialization
	layers = [
	    (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),
	    (Conv2DLayerFast, {'num_filters': conv_num_filters1,
				'filter_size': filter_size1, 'pad': pad_in}),
	    (MaxPool2DLayerFast, {'pool_size': pool_size}),
	    (Conv2DLayerFast, {'num_filters': conv_num_filters2,
				'filter_size': filter_size2, 'pad': pad_in}),
	    (MaxPool2DLayerFast, {'pool_size': pool_size}),
	    (ReshapeLayer, {'shape': (([0], -1))}),
	    (DenseLayer, {'name': 'encode', 'num_units': encode_size}),
	    (DenseLayer, {'num_units': 144}),
	    (ReshapeLayer, {'shape': (([0], conv_num_filters2, 3, 3))}),
	    (Upscale2DLayer, {'scale_factor': pool_size}),
	    (Conv2DLayerFast, {'num_filters': conv_num_filters1,
				'filter_size': filter_size2, 'pad': pad_out}),
	    (Upscale2DLayer, {'scale_factor': pool_size}),
	    (Conv2DLayerFast, {'num_filters': 1,
				'filter_size': filter_size1, 'pad': pad_out}),
	    (ReshapeLayer, {'shape': (([0], -1))}),
	]

	return NeuralNet(
	    layers=layers,
	    max_epochs=10,
	    
	    update=nesterov_momentum,
	    update_learning_rate=0.01,
	    update_momentum=0.975,
	    
	    regression=True,
	    verbose=1
	)

def parse_command_line():
	parser = argparse.ArgumentParser(
			description='Simple convolutional autoencoder')
	parser.add_argument('in_file', metavar = 'input_data', type = str,
			nargs = '+',
			help = 'File with the data to be "autoencoded".')
	parser.add_argument('out_file', metavar = 'output_file', type = str,
			nargs = '+',
			help = 'Name of the output file.')

	return parser.parse_args()


def main():
	args = parse_command_line()
	in_file = args.in_file[0]
	out_file = args.out_file[0]
	print(out_file)

	recursion_limit = 10000
	print("Setting recursion limit to {rl}".format(rl = recursion_limit))
	sys.setrecursionlimit(recursion_limit)

	X, X_out = load_data(in_file)
	ae = create_caes(X)
	ae.fit(X, X_out)

	W1 = ae.layers_[1].W.get_value()
	b1 = ae.layers_[1].b.get_value()

	W2 = ae.layers_[3].W.get_value()
	b2 = ae.layers_[3].b.get_value()

	sio.savemat(out_file, {'caes_W1':W1, 'caes_b1':b1,
					'caes_W2':W2, 'caes_b2':b2})

if __name__ == '__main__':
	main()

