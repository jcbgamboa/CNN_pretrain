#! /bin/bash

MNIST_PATH=./mnist_folds
MNIST_FOLDS=7
MNIST_OUTPUT_PATH=./pretrained_weights/caes/mnist

CIFAR_PATH=./cifar_folds
CIFAR_FOLDS=6
CIFAR_OUTPUT_PATH=./pretrained_weights/caes/cifar

mkdir -p $MNIST_OUTPUT_PATH
mkdir -p $CIFAR_OUTPUT_PATH

for ((i = 1; i <= $MNIST_FOLDS; i++))
do
	echo "python auto_encoders/caes_nolearn.py $MNIST_PATH/fold_$i.mat $MNIST_OUTPUT_PATH/fold_$i.mat"
	python auto_encoders/caes_nolearn.py $MNIST_PATH/fold_$i.mat \
			$MNIST_OUTPUT_PATH/fold_$i.mat
done

for ((i = 1; i <= $CIFAR_FOLDS; i++))
do
	echo "python auto_encoders/caes_nolearn.py $CIFAR_PATH/fold_$i.mat $CIFAR_OUTPUT_PATH/fold_$i.mat --normalize"
	python auto_encoders/caes_nolearn.py $CIFAR_PATH/fold_$i.mat \
			$CIFAR_OUTPUT_PATH/fold_$i.mat --normalize
done


