import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

def analyse_single_experiment(in_folder, analyse_accuracy = True,
					by_epoch = False):
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
		f = np.loadtxt(f_name + '/per_' +
				('epoch_' if by_epoch else 'batch_') +
				'metrics.csv',
				delimiter = ',')

		if (analyse_accuracy):
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
	np.savetxt(in_folder + '/reduced_per_' +
				('epoch_' if by_epoch else 'batch_') +
				'metrics.csv',
		list(zip(np_fold_losses.mean(1),
				np_fold_losses.std(1),
				np_fold_accuracies.mean(1),
				np_fold_accuracies.std(1),
				np_fold_one_minus_accuracies.mean(1),
				np_fold_one_minus_accuracies.std(1))),
		delimiter = ',', fmt = '%f')

	if (analyse_accuracy):
		np.savetxt(in_folder + '/reduced_accuracy.csv',
			[test_accuracy.mean(), test_accuracy.std()],
			delimiter = ',', fmt = '%f')

def generate_learning_curves(folder, dataset, position,
				use_single_fold = False,
				by_epoch = True):
	lc_folders = []

	if (by_epoch):
		lc_folders.append(folder + '/caes/' + dataset + '/relu/full/50epochs')
		lc_folders.append(folder + '/caes/' + dataset + '/sigmoid/full/50epochs')
		lc_folders.append(folder + '/cdbn/' + dataset + '/binary/full/50epochs')
		lc_folders.append(folder + '/cdbn/' + dataset + '/gaussian/full/50epochs')
		lc_folders.append(folder + '/random/' + dataset + '/relu/full/50epochs')
		lc_folders.append(folder + '/random/' + dataset + '/sigmoid/full/50epochs')
	else:
		lc_folders.append(folder + '/caes/' + dataset + '/relu/full/1epochs')
		lc_folders.append(folder + '/caes/' + dataset + '/sigmoid/full/1epochs')
		lc_folders.append(folder + '/cdbn/' + dataset + '/binary/full/1epochs')
		lc_folders.append(folder + '/cdbn/' + dataset + '/gaussian/full/1epochs')
		lc_folders.append(folder + '/random/' + dataset + '/relu/full/1epochs')
		lc_folders.append(folder + '/random/' + dataset + '/sigmoid/full/1epochs')

	if (use_single_fold):
		for i, f in enumerate(lc_folders):
			lc_folders[i] = f + '/fold_1'

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
		f_data = np.loadtxt(f + '/' +
				('' if use_single_fold else 'reduced_') +
				'per_' +
				('epoch_' if by_epoch else 'batch_') +
				'metrics.csv',
					delimiter = ',')

		if (use_single_fold):
			mean_losses.append(f_data[:, 0])
			mean_accuracies.append(f_data[:, 1])
			mean_one_minus_accuracies.append(f_data[:, 2])
		else:
			mean_losses.append(f_data[:, 0])
			std_losses.append(f_data[:, 1])
			mean_accuracies.append(f_data[:, 2])
			std_accuracies.append(f_data[:, 3])
			mean_one_minus_accuracies.append(f_data[:, 4])
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
	fig = plt.figure()
	#sub = fig.add_subplot(111)
	for i, l in enumerate(labels):
		pl = plt.plot(np_x, np_mean_losses[:, i],
				label = l)
	#if (by_epoch):
	#	sub.set_yscale('log')
	legend = plt.legend(loc = position, fontsize='medium')
	for legobj in legend.legendHandles:
		legobj.set_linewidth(5.0)

	fig.savefig(folder + '/' + dataset +
			('_single_' if use_single_fold else '_cross_') +
			'fold_mean_per_' +
			('epoch_' if by_epoch else 'batch_') +
			'errors.eps')


def analyse_cross_experiment(results_folder):
	# TODO: This is very specific to my experiments. Find a more generic
	#	way to do this.
	generate_learning_curves(results_folder, 'cifar', 'lower left')
	generate_learning_curves(results_folder, 'mnist', 'upper right')

	generate_learning_curves(results_folder, 'cifar', 'lower left',
				use_single_fold = True)
	generate_learning_curves(results_folder, 'mnist', 'upper right',
				use_single_fold = True)

	generate_learning_curves(results_folder, 'cifar', 'lower left',
				by_epoch = True)
	generate_learning_curves(results_folder, 'mnist', 'upper right',
				by_epoch = True)


def main():
	args = parse_command_line()
	in_folder = args.in_folder
	single_experiment = args.single_experiment
	multiple_experiments = args.multiple_experiments
	with_per_epoch = args.with_per_epoch

	if (single_experiment):
		analyse_single_experiment(in_folder)
		if (with_per_epoch):
			analyse_single_experiment(in_folder, by_epoch = True)
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
	parser.add_argument('--with_per_epoch',
		dest = 'with_per_epoch', action='store_true',
		help = 'Activate to also generate results by epoch.')


	return parser.parse_args()

if __name__ == '__main__':
	main()
