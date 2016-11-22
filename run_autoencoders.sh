#! /bin/bash

MNIST_PATH=../mnist_folds
MNIST_FOLDS=7

CIFAR_PATH=../cifar_folds
CIFAR_FOLDS=6

for ((i = 0; i < $MNIST_FOLDS; i++))
do
	python caes_nolearn.py $(MNIST_PATH)/fold_$(i).mat
done

for ((i = 0; i < $CIFAR_FOLDS; i++))
do
	python caes_nolearn.py $(CIFAR_PATH)/fold_$(i).mat
done


