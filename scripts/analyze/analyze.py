import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

def analyse_single_experiment(in_folder):
	# TODO: This is very specific to my experiments. Find a more generic
	#	way to do this.

	# Determines the number of folds
	folds = glob.glob(in_folder + '/fold_*')
	n_folds = len(folds)

	# For each fold, generate a mean of the learning rate
	fold_losses = []
	fold_accuracies = []
	fold_one_minus_accuracies = []
	test_accuracy = np.zeros(n_folds)
	for i, f_name in enumerate(folds):
		# open `per_batch_metrics.csv`
		f = np.loadtxt(f_name + '/per_batch_metrics.csv',
				delimiter = ',')
		test_accuracy[i] = np.loadtxt(f_name + '/accuracy.csv',
				delimiter = ',')

		# insert second column (accuracies) into fold_accuracies
		fold_losses.append(f[:, 0])
		fold_accuracies.append(f[:, 1])
		fold_one_minus_accuracies.append(f[:, 2])

	# Dump mean and std.dev into file
	np_fold_losses = np.array(list(zip(*fold_losses)))
	np_fold_accuracies = np.array(list(zip(*fold_accuracies)))
	np_fold_one_minus_accuracies = np.array(list(zip(*fold_one_minus_accuracies)))
	np.savetxt(in_folder + '/reduced_per_batch_metrics.csv',
		list(zip(np_fold_losses.mean(1),
				np_fold_losses.std(1),
				np_fold_accuracies.mean(1),
				np_fold_accuracies.std(1),
				np_fold_one_minus_accuracies.mean(1),
				np_fold_one_minus_accuracies.std(1))),
		delimiter = ',', fmt = '%f')

	# For each fold, generate a mean and std.dev of the accuracy
	np.savetxt(in_folder + '/reduced_accuracy.csv',
		[test_accuracy.mean(), test_accuracy.std()],
		delimiter = ',', fmt = '%f')

def generate_learning_curves(folder, dataset, position):
	lc_folders = []

	# I use `1 epoch` and `full` because this shouldn't differ with more
	# epochs.
	lc_folders.append(folder + '/caes/' + dataset + '/relu/full/1epochs')
	lc_folders.append(folder + '/caes/' + dataset + '/sigmoid/full/1epochs')
	lc_folders.append(folder + '/cdbn/' + dataset + '/binary/full/1epochs')
	lc_folders.append(folder + '/cdbn/' + dataset + '/gaussian/full/1epochs')
	lc_folders.append(folder + '/random/' + dataset + '/relu/full/1epochs')
	lc_folders.append(folder + '/random/' + dataset + '/sigmoid/full/1epochs')

	# Labels
	labels = ['CAES+ReLU', 'CAES+Sigmoid', 'BernoulliCDBN+Sigmoid',
			'GaussianCDBN+Sigmoid', 'Random+ReLU', 'Random+Sigmoid']

	mean_losses = []
	std_losses = []
	mean_accuracies = []
	std_accuracies = []
	mean_one_minus_accuracies = []
	std_one_minus_accuracies = []
	for f in lc_folders:
		f_data = np.loadtxt(f + '/reduced_per_batch_metrics.csv',
					delimiter = ',')

		curr_loss = f_data[:, 0]
		curr_accuracy = f_data[:, 2]
		curr_one_minus_accuracy = f_data[:, 4]

		mean_losses.append(curr_loss)
		std_losses.append(f_data[:, 1])
		mean_accuracies.append(curr_accuracy)
		std_accuracies.append(f_data[:, 3])
		mean_one_minus_accuracies.append(curr_one_minus_accuracy)
		std_one_minus_accuracies.append(f_data[:, 5])


	# Transform stuff into a vector
	np_mean_losses = np.array(list(zip(*mean_losses)))
	#np_std_losses = np.array(list(zip(*std_losses)))
	np_mean_accuracies = np.array(list(zip(*mean_accuracies)))
	#np_std_accuracies = np.array(list(zip(*std_accuracies)))
	np_mean_one_minus_accuracies = np.array(list(zip(*mean_one_minus_accuracies)))
	#np_std_one_minus_accuracies = np.array(list(zip(*std_one_minus_accuracies)))

	# All of these should have the same length
	np_x = np.array(range(0, np_mean_accuracies.shape[0]))


	# Make images
	#plt.plot(np_x, np_mean_one_minus_accuracies)
	fig = plt.figure()
	for i, l in enumerate(labels):
		plt.plot(np_x, np_mean_one_minus_accuracies[:, i],
				label = l)
	legend = plt.legend(loc = position, fontsize='medium')
	for legobj in legend.legendHandles:
		legobj.set_linewidth(5.0)

	#plt.fill_between(np_x, np_mean_accuracies + np_std_accuracies,
	#			np_mean_accuracies - np_std_accuracies,
	#			alpha = 0.3)
	#plt.show()
	fig.savefig(folder + '/' + dataset + '_cross_fold_mean_per_batch_errors.eps')


def generate_learning_curves_smooth(folder, dataset, position):
	lc_folders = []

	# I use `1 epoch` and `full` because this shouldn't differ with more
	# epochs.
	lc_folders.append(folder + '/caes/' + dataset + '/relu/full/1epochs')
	lc_folders.append(folder + '/caes/' + dataset + '/sigmoid/full/1epochs')
	lc_folders.append(folder + '/cdbn/' + dataset + '/binary/full/1epochs')
	lc_folders.append(folder + '/cdbn/' + dataset + '/gaussian/full/1epochs')
	lc_folders.append(folder + '/random/' + dataset + '/relu/full/1epochs')
	lc_folders.append(folder + '/random/' + dataset + '/sigmoid/full/1epochs')

	# Labels
	labels = ['CAES+ReLU', 'CAES+Sigmoid', 'BernoulliCDBN+Sigmoid',
			'GaussianCDBN+Sigmoid', 'Random+ReLU', 'Random+Sigmoid']

	# All of these should have the same length
	np_x = None
	mean_losses = []
	mean_accuracies = []
	mean_one_minus_accuracies = []
	for f in lc_folders:
		f_data = np.loadtxt(f + '/reduced_per_batch_metrics.csv',
					delimiter = ',')

		curr_loss = f_data[:, 0]
		curr_accuracy = f_data[:, 2]
		curr_one_minus_accuracy = f_data[:, 4]

		np_x = np.array(range(0, f_data[:, 0].shape[0]))
		curr_loss = interp1d(np_x, curr_loss,
					kind = 'cubic')
		curr_accuracy = interp1d(np_x, curr_accuracy,
					kind = 'cubic')
		curr_one_minus_accuracy = interp1d(np_x,
					curr_one_minus_accuracy,
					kind = 'cubic')

		mean_losses.append(curr_loss)
		mean_accuracies.append(curr_accuracy)
		mean_one_minus_accuracies.append(curr_one_minus_accuracy)
		print("Reading interpolations")

	print("Will plot")

	# Make images
	#plt.plot(np_x, np_mean_one_minus_accuracies)
	fig = plt.figure()
	for i, acc in enumerate(mean_one_minus_accuracies):
		plt.plot(np_x, acc(np_x), label = labels[i])
	legend = plt.legend(loc = position, fontsize='medium')
	for legobj in legend.legendHandles:
		legobj.set_linewidth(5.0)

	fig.savefig(folder + '/' + dataset +
			'_cross_fold_mean_per_batch_errors_smooth.eps')


def analyse_cross_experiment(results_folder):
	# TODO: This is very specific to my experiments. Find a more generic
	#	way to do this.
	generate_learning_curves(results_folder, 'cifar', 'lower left')
	generate_learning_curves(results_folder, 'mnist', 'upper right')

	generate_learning_curves_smooth(results_folder, 'cifar', 'lower left')

def main():
	args = parse_command_line()
	in_folder = args.in_folder
	single_experiment = args.single_experiment
	multiple_experiments = args.multiple_experiments

	if (single_experiment):
		analyse_single_experiment(in_folder)
	elif (multiple_experiments):
		analyse_cross_experiment(in_folder)
		

def parse_command_line():
	description = 'Generates images from the results of the Keras CNN.'
	parser = argparse.ArgumentParser(description = description)
	parser.add_argument('in_folder', metavar = 'in_folder', type = str,
		help = 'Where are the files to generate results from?')
	parser.add_argument('--single_experiment',
		dest = 'single_experiment', action='store_true',
		help = 'Generate results based on a certain experiment. ' +
			'Overrides --multiple_experiments.')
	parser.add_argument('--multiple_experiments',
		dest = 'multiple_experiments', action='store_true',
		help = 'Given the folder with results, generates analyses' +
			'through all experiments.')

	return parser.parse_args()

if __name__ == '__main__':
	main()
