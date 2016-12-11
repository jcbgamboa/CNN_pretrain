#! /bin/bash

DATASET_PATH=$1
DATASET_FOLDS=$2
OUTPUT_PATH=$3
WEIGHT_INITIALIZATION=$4
DATASET_NAME=$5
NONLINEARITY=$6
N_EPOCHS=$7
DATASET_SIZE=$8


# Parse command line arguments
PRETRAINED_MODEL_FLAGS=""
PARAM_FLAGS=""

if [ $WEIGHT_INITIALIZATION = 'caes' ]
then
	PRETRAINED_MODEL_FLAGS="--pretrained_model ./results/pretrained_weights/$WEIGHT_INITIALIZATION/$DATASET_NAME/$NONLINEARITY"
elif [ $WEIGHT_INITIALIZATION = 'cdbn' ]
then
	# `basename` takes the last folder of a path. It is used here to decide
	# if we should look at the weights of the CDBN with Bernoulli or
	# Gaussian units.
	PRETRAINED_MODEL_FLAGS="--pretrained_model ./results/pretrained_weights/$WEIGHT_INITIALIZATION/$DATASET_NAME/$(basename $OUTPUT_PATH)"
elif [ $WEIGHT_INITIALIZATION = 'random' ]
then
	:
fi

if [ $DATASET_NAME = 'cifar' ]
then
	PARAM_FLAGS="$PARAM_FLAGS --normalize"
elif [ $DATASET_NAME = 'mnist' ]
then
	:
fi

if [ $NONLINEARITY = 'relu' ]
then
	PARAM_FLAGS="$PARAM_FLAGS --use_relu"
elif [ $NONLINEARITY = 'sigmoid' ]
then
	:
fi

PARAM_FLAGS="$PARAM_FLAGS --n_epochs $N_EPOCHS"

if [ $DATASET_SIZE = 'reduced' ]
then
	PARAM_FLAGS="$PARAM_FLAGS --reduce_dataset_to 10000"
elif [ $DATASET_SIZE = 'full' ]
then
	:
fi

OUTPUT_PATH=$OUTPUT_PATH/$DATASET_SIZE/$((N_EPOCHS))epochs


# Calls program
for ((i = 1; i <= $DATASET_FOLDS; i++))
do
	FOLD_OUTPUT_PATH=$OUTPUT_PATH/fold_$i

	if [ $WEIGHT_INITIALIZATION != 'random' ]
	then
		FOLD_PRETRAINED_MODEL_FLAGS=$PRETRAINED_MODEL_FLAGS/fold_$i.mat
	else
		FOLD_PRETRAINED_MODEL_FLAGS=""
	fi

	mkdir -p $FOLD_OUTPUT_PATH

	# For debugging
	echo "python3 networks/cnns/cnn_keras.py $DATASET_PATH/fold_$i.mat $FOLD_OUTPUT_PATH $FOLD_PRETRAINED_MODEL_FLAGS $PARAM_FLAGS"

	python3 networks/cnns/cnn_keras.py $DATASET_PATH/fold_$i.mat $FOLD_OUTPUT_PATH $FOLD_PRETRAINED_MODEL_FLAGS $PARAM_FLAGS
done

