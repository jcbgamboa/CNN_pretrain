#! /usr/bin/python

from __future__ import print_function

import os, sys, urllib, gzip
try:
    import cPickle as pickle
except:
    import pickle

import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import numpy as np

from lasagne.layers import InputLayer, DenseLayer, Upscale2DLayer, ReshapeLayer
from lasagne.nonlinearities import rectify, leaky_rectify, tanh
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

from lasagne.layers import Conv2DLayer as Conv2DLayerSlow
from lasagne.layers import MaxPool2DLayer as MaxPool2DLayerSlow
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayerFast
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayerFast
    print('Using cuda_convnet (faster)')
except ImportError:
    from lasagne.layers import Conv2DLayer as Conv2DLayerFast
    from lasagne.layers import MaxPool2DLayer as MaxPool2DLayerFast
    print('Using lasagne.layers (slower)')


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


def create_caes():
	conv_num_filters1 = 9
	conv_num_filters2 = 16
	filter_size1 = 7
	filter_size2 = 6
	pool_size = 2
	encode_size = 16
	dense_mid_size = 128
	pad_in = 'valid'
	pad_out = 'full'
	layers = [
	    (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}), 
	    (Conv2DLayerFast, {'num_filters': conv_num_filters1, 'filter_size': filter_size1, 'pad': pad_in}),
	    (MaxPool2DLayerFast, {'pool_size': pool_size}),
	    (Conv2DLayerFast, {'num_filters': conv_num_filters2, 'filter_size': filter_size2, 'pad': pad_in}),
	    (MaxPool2DLayerFast, {'pool_size': pool_size}),
	    (ReshapeLayer, {'shape': (([0], -1))}),
	    (DenseLayer, {'name': 'encode', 'num_units': encode_size}),
	    (DenseLayer, {'num_units': 144}),
	    (ReshapeLayer, {'shape': (([0], conv_num_filters2, 3, 3))}),
	    (Upscale2DLayer, {'scale_factor': pool_size}),
	    (Conv2DLayerFast, {'num_filters': conv_num_filters1, 'filter_size': filter_size2, 'pad': pad_out}),
	    (Upscale2DLayer, {'scale_factor': pool_size}),
	    (Conv2DLayerSlow, {'num_filters': 1, 'filter_size': filter_size1, 'pad': pad_out}),
	    (ReshapeLayer, {'shape': (([0], -1))}),
	]

	ae = NeuralNet(
	    layers=layers,
	    max_epochs=10,
	    
	    update=nesterov_momentum,
	    update_learning_rate=0.01,
	    update_momentum=0.975,
	    
	    regression=True,
	    verbose=1
	)


if __name__ == '__main__':
	recursion_limit = 10000
	print("Setting recursion limit to {rl}".format(rl = recursion_limit))
	sys.setrecursionlimit(recursion_limit)

	X, X_out = load_data('caes_train_data.mat')
	ae.fit(X, X_out)

	W1 = ae.layers_[1].W.get_value()
	b1 = ae.layers_[1].b.get_value()

	W2 = ae.layers_[3].W.get_value()
	b2 = ae.layers_[3].b.get_value()

	sio.savemat('caes_out.mat', {'caes_W1':W1, 'caes_b1':b1,
					'caes_W2':W2, 'caes_b2':b2})


