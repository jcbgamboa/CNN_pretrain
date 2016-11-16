from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D

import numpy as np
import scipy.io as sio

def create_caes(encoding_layers, decoding_layers,
		input_shape = (1, 28, 28),
		dense_mid_size = 128, activation = 'relu'):
	# `encoding_layers` and `decoding_layers` are lists of tuples of type
	# (n_filters, filter_size_x, filter_size_y, border_mode)

	input_img = x = Input(shape=(1, 28, 28))
	for i in encoding_layers:
		x = Convolution2D(i[0], i[1], i[2], border_mode = i[3],
				activation = activation)(x)
		x = MaxPooling2D((2, 2), border_mode = 'same')(x)

	x = Dense(dense_mid_size, activation = activation)(x)

	for i in decoding_layers:
		x = UpSampling2D((2, 2))(x)
		x = Convolution2D(i[0], i[1], i[2], border_mode = i[3],
				activation = activation)(x)

	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
	return autoencoder

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

#from keras.datasets import mnist
#(x_train, _), (x_test, _) = mnist.load_data()
#
#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#x_train = x_train.reshape((len(x_train), 1, 28, 28))
#x_test = x_test.reshape((len(x_test), 1, 28, 28))
#
#print x_train.shape
#print x_test.shape


if __name__ == '__main__':
	X, X_out = load_data('caes_train_data.mat')

	encoding_layers = [(9, 7, 7, 'valid'), (16, 6, 6, 'valid')]
	decoding_layers = [(16, 6, 6, 'full'), (9, 7, 7, 'full')]]
	ae = create_caes(encoding_layers, decoding_layers)

	autoencoder.fit(X,
		X_out,
                nb_epoch = 50,
                batch_size = 256,
                shuffle = True,
                validation_data = (x_test, x_test),
		callbacks = [TensorBoard(log_dir='./caes_keras_results')])


"""
x = Convolution2D(9, 7, 7, activation='relu',
		border_mode='valid')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(16, 6, 6, activation='relu', border_mode='valid')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
encoded = Dense(dense_mid_size, activation='relu')(x)

x = UpSampling2D((2, 2))(encoded)
x = Convolution2D(16, 6, 6, activation='relu', border_mode='full')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(9, 7, 7, activation='relu', border_mode='full')(x)
"""

