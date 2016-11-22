#! /usr/bin/python3
# Generate a simple CNN with Keras

# It receives as parameter
# * a .mat file with the images to be run
# * and optionally another .mat file with weights

# It runs the CNN and spits
# 1) The loss value for each iteration
# 2) The accuracy on the `test set`

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, \
			Activation, Flatten
from keras.callbacks import TensorBoard

import numpy as np
import scipy.io as sio
import sklearn.preprocessing as skp

import argparse

def create_cnn(pretrained_model_file = None,
		data_shape = (60000, 28, 28, 1)):
	# Loads weights from other models
	if(pretrained_model_file is not None):
		premodel = sio.loadmat(pretrained_model_file)

	input_img = Input(shape=(data_shape[1], data_shape[2], data_shape[3]))
	# Defines the CNN architecture (based on the weights?)
	x = Convolution2D(9, 7, 7, border_mode = 'valid',
				activation = 'sigmoid')(input_img)
	x = MaxPooling2D((2, 2), border_mode = 'same')(x)

	x = Convolution2D(16, 6, 6, border_mode = 'valid',
				activation = 'sigmoid')(x)
	x = MaxPooling2D((2, 2), border_mode = 'same')(x)

	x = Flatten()(x)
	x = Dense(10, activation = 'sigmoid')(x)
	x = Activation('softmax')(x)

	#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	#model.compile(loss='categorical_crossentropy', optimizer=sgd)u

	return Model(input = input_img, output = x)

def label_binarize(trainL, testL):
	binarizer = skp.LabelBinarizer(0, 1, False)
	binarizer.fit(trainL)
	trainL = binarizer.transform(trainL)
	testL = binarizer.transform(testL)
	return trainL, testL

def main():
	args = parse_command_line()
	pretrained_model_file = args.pretrained_model
	in_file = args.in_file
	out_folder = args.out_folder

	dataset = sio.loadmat(in_file)

	print(dataset['testL'].shape)

	# Transposing stuff only because I am using Keras/Tensorflow
	train_data = np.transpose(dataset['train_data'], axes = (3, 0, 1, 2))
	test_data  = np.transpose(dataset['test_data'], axes = (3, 0, 1, 2))
	trainL     = np.transpose(dataset['trainL'])
	# Apparently, I made a mistake and somehow transposed the testL when
	# generating the data
	testL      = dataset['testL']

	trainL, testL = label_binarize(trainL, testL)
	print(train_data.shape)

	# I hope this "free()'s" this variable
	dataset = None

	model = create_cnn(pretrained_model_file, train_data.shape)

	model.compile(optimizer = 'adam',
			loss = 'categorical_crossentropy',
			metrics = ['accuracy', 'categorical_crossentropy'])

	model.fit(train_data, trainL,
		batch_size = 64,
		callbacks = [TensorBoard(log_dir='./caes_keras_results')])



def parse_command_line():
	# TODO: add better description
	description = 'Simple CNN.'
	parser = argparse.ArgumentParser(description = description)
	parser.add_argument('--pretrained_model',
			metavar = 'pretrained_model', type = str,
			help = 'Pretrained weights to be used in CNN. ' +
				'Random initialization is used if left empty.')
	parser.add_argument('in_file', metavar = 'in_file', type = str,
			help = 'File with the dataset to be learnt.')
	parser.add_argument('out_folder', metavar = 'output_folder', type = str,
			help = 'Folder where results should be put.')

	return parser.parse_args()

if __name__ == '__main__':
	main()

