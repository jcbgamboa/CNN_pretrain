import numpy as np

# For LaTeX tabe code generation
from string import Template

mnist_full = np.array([
	['results/cnn/caes/mnist/relu/full/1epochs',
	'results/cnn/caes/mnist/sigmoid/full/1epochs',
	'results/cnn/cdbn/mnist/binary/full/1epochs',
	'results/cnn/cdbn/mnist/gaussian/full/1epochs',
	'results/cnn/random/mnist/relu/full/1epochs',
	'results/cnn/random/mnist/sigmoid/full/1epochs'],

	['results/cnn/caes/mnist/relu/full/10epochs',
	'results/cnn/caes/mnist/sigmoid/full/10epochs',
	'results/cnn/cdbn/mnist/binary/full/10epochs',
	'results/cnn/cdbn/mnist/gaussian/full/10epochs',
	'results/cnn/random/mnist/relu/full/10epochs',
	'results/cnn/random/mnist/sigmoid/full/10epochs'],

	['results/cnn/caes/mnist/relu/full/50epochs',
	'results/cnn/caes/mnist/sigmoid/full/50epochs',
	'results/cnn/cdbn/mnist/binary/full/50epochs',
	'results/cnn/cdbn/mnist/gaussian/full/50epochs',
	'results/cnn/random/mnist/relu/full/50epochs',
	'results/cnn/random/mnist/sigmoid/full/50epochs']
	])

cifar_full = np.array([
	['results/cnn/caes/cifar/relu/full/1epochs',
	'results/cnn/caes/cifar/sigmoid/full/1epochs',
	'results/cnn/cdbn/cifar/binary/full/1epochs',
	'results/cnn/cdbn/cifar/gaussian/full/1epochs',
	'results/cnn/random/cifar/relu/full/1epochs',
	'results/cnn/random/cifar/sigmoid/full/1epochs'],

	['results/cnn/caes/cifar/relu/full/10epochs',
	'results/cnn/caes/cifar/sigmoid/full/10epochs',
	'results/cnn/cdbn/cifar/binary/full/10epochs',
	'results/cnn/cdbn/cifar/gaussian/full/10epochs',
	'results/cnn/random/cifar/relu/full/10epochs',
	'results/cnn/random/cifar/sigmoid/full/10epochs'],

	['results/cnn/caes/cifar/relu/full/50epochs',
	'results/cnn/caes/cifar/sigmoid/full/50epochs',
	'results/cnn/cdbn/cifar/binary/full/50epochs',
	'results/cnn/cdbn/cifar/gaussian/full/50epochs',
	'results/cnn/random/cifar/relu/full/50epochs',
	'results/cnn/random/cifar/sigmoid/full/50epochs'],
	])

mnist_reduced = np.array([
	['results/cnn/caes/mnist/relu/reduced/1epochs',
	'results/cnn/caes/mnist/sigmoid/reduced/1epochs',
	'results/cnn/cdbn/mnist/binary/reduced/1epochs',
	'results/cnn/cdbn/mnist/gaussian/reduced/1epochs',
	'results/cnn/random/mnist/relu/reduced/1epochs',
	'results/cnn/random/mnist/sigmoid/reduced/1epochs'],

	['results/cnn/caes/mnist/relu/reduced/10epochs',
	'results/cnn/caes/mnist/sigmoid/reduced/10epochs',
	'results/cnn/cdbn/mnist/binary/reduced/10epochs',
	'results/cnn/cdbn/mnist/gaussian/reduced/10epochs',
	'results/cnn/random/mnist/relu/reduced/10epochs',
	'results/cnn/random/mnist/sigmoid/reduced/10epochs'],

	['results/cnn/caes/mnist/relu/reduced/50epochs',
	'results/cnn/caes/mnist/sigmoid/reduced/50epochs',
	'results/cnn/cdbn/mnist/binary/reduced/50epochs',
	'results/cnn/cdbn/mnist/gaussian/reduced/50epochs',
	'results/cnn/random/mnist/relu/reduced/50epochs',
	'results/cnn/random/mnist/sigmoid/reduced/50epochs'],
	])

cifar_reduced = np.array([
	['results/cnn/caes/cifar/relu/reduced/1epochs',
	'results/cnn/caes/cifar/sigmoid/reduced/1epochs',
	'results/cnn/cdbn/cifar/binary/reduced/1epochs',
	'results/cnn/cdbn/cifar/gaussian/reduced/1epochs',
	'results/cnn/random/cifar/relu/reduced/1epochs',
	'results/cnn/random/cifar/sigmoid/reduced/1epochs'],

	['results/cnn/caes/cifar/relu/reduced/10epochs',
	'results/cnn/caes/cifar/sigmoid/reduced/10epochs',
	'results/cnn/cdbn/cifar/binary/reduced/10epochs',
	'results/cnn/cdbn/cifar/gaussian/reduced/10epochs',
	'results/cnn/random/cifar/relu/reduced/10epochs',
	'results/cnn/random/cifar/sigmoid/reduced/10epochs'],

	['results/cnn/caes/cifar/relu/reduced/50epochs',
	'results/cnn/caes/cifar/sigmoid/reduced/50epochs',
	'results/cnn/cdbn/cifar/binary/reduced/50epochs',
	'results/cnn/cdbn/cifar/gaussian/reduced/50epochs',
	'results/cnn/random/cifar/relu/reduced/50epochs',
	'results/cnn/random/cifar/sigmoid/reduced/50epochs'],
	])

results = {
	'mnist_full'    : mnist_full,
	'cifar_full'    : cifar_full,
	'mnist_reduced' : mnist_reduced,
	'cifar_reduced' : cifar_reduced,
	}

def generate_data_tables(dataset, dataset_size):
	r = results[dataset + '_' + dataset_size]

	# Just so that I can change this in the future
	n_rows = 6
	n_columns = 3

	means = np.zeros([n_columns, n_rows])
	stds  = np.zeros([n_columns, n_rows])

	for i in range(n_columns):
		for j in range(n_rows):
			f_data = np.loadtxt(r[i, j] + '/reduced_accuracy.csv',
							delimiter = ',')
			means[i, j] = f_data[0]
			stds[i, j] = f_data[1]

	return (means.T, stds.T)

table_template = Template("""
\\textbf{Weight Initialization} & \multicolumn{3}{c}{\\textbf{CNN Training Epochs}} \\\\
\hline
			& 1	& 10	& 50 \\\\
CAES+ReLU		& $caes_relu \\\\
CAES+Sigmoid		& $caes_sigmoid \\\\
BernoulliCDBN+Sigmoid	& $bcdbn_sigmoid \\\\
GaussianCDBN+Sigmoid	& $gcdbn_sigmoid \\\\
Random+ReLU		& $rand_relu \\\\
Random+Sigmoid		& $rand_sigmoid
""")

def gen_table_line_latex(means, stds, i, subtract = False, subtract_amount = 0):
	# It will print, in a "latexified" convenient way, the data from row `i`
	# TODO: Can I do this for a generic number of columns?
	if (subtract):
		return \
		('{0:.2f}'.format(subtract_amount - means[i, 0]) + ' $\pm$ ' +
					'{0:.2f}'.format(stds[i, 0]) + '\t& ' +
		'{0:.2f}'.format(subtract_amount - means[i, 1]) + ' $\pm$ ' +
					'{0:.2f}'.format(stds[i, 1]) + '\t& ' +
		'{0:.2f}'.format(subtract_amount - means[i, 2]) + ' $\pm$ ' +
					'{0:.2f}'.format(stds[i, 2]))
	else:
		return \
		('{0:.2f}'.format(means[i, 0]) + ' $\pm$ ' +
			'{0:.2f}'.format(stds[i, 0]) + '\t& ' +
		'{0:.2f}'.format(means[i, 1]) + ' $\pm$ ' +
			'{0:.2f}'.format(stds[i, 1]) + '\t& ' +
		'{0:.2f}'.format(means[i, 2]) + ' $\pm$ ' +
			'{0:.2f}'.format(stds[i, 2]))


def output_latex(means, stds, subtract = False, subtract_amount = 0):
	args = [subtract, subtract_amount]
	return table_template.substitute(
		caes_relu     = gen_table_line_latex(means, stds, 0, *args),
		caes_sigmoid  = gen_table_line_latex(means, stds, 1, *args),
		bcdbn_sigmoid = gen_table_line_latex(means, stds, 2, *args),
		gcdbn_sigmoid = gen_table_line_latex(means, stds, 3, *args),
		rand_relu     = gen_table_line_latex(means, stds, 4, *args),
		rand_sigmoid  = gen_table_line_latex(means, stds, 5, *args),
			)

if __name__ == '__main__':
	print("MNIST -- FULL")
	(means, stds) = generate_data_tables('mnist', 'full')

	print(output_latex(means, stds,
			subtract = True, subtract_amount = 10000))

	print("MNIST -- REDUCED")
	(means, stds) = generate_data_tables('mnist', 'reduced')

	print(output_latex(means, stds,
			subtract = True, subtract_amount = 10000))

	print("CIFAR -- FULL")
	(means, stds) = generate_data_tables('cifar', 'full')

	print(output_latex(means, stds,
			subtract = True, subtract_amount = 10000))

	print("CIFAR -- REDUCED")
	(means, stds) = generate_data_tables('cifar', 'reduced')

	print(output_latex(means, stds,
			subtract = True, subtract_amount = 10000))



