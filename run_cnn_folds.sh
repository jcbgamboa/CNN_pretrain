#! /bin/bash

DATASET_PATH=$1
DATASET_FOLDS=$2
DATASET_OUTPUT_PATH=$3

mkdir -p $DATASET_OUTPUT_PATH


# TODO: fix this loop

OUT_DIR=$MNIST_OUTPUT_PATH/$FULL_DATASET/$RANDOM_SIGMOID/1epoch
mkdir -p $OUT_DIR
for ((i = 1; i <= $MNIST_FOLDS; i++))
do
	echo "python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR"
	python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR \
		--n_epochs 1
done

