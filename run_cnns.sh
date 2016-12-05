#! /bin/bash
# Is there a better way to do this? This is awful!

MNIST_PATH=./mnist_folds
MNIST_FOLDS=7
MNIST_OUTPUT_PATH=./mnist_results

CIFAR_PATH=./cifar_folds
CIFAR_FOLDS=6
CIFAR_OUTPUT_PATH=./cifar_results

mkdir -p $MNIST_OUTPUT_PATH
mkdir -p $CIFAR_OUTPUT_PATH

FULL_DATASET=full
REDUCED_DATASET=reduced

RANDOM_SIGMOID=rand_sigmoid
RANDOM_RELU=rand_relu
CDBN_BINARY=cdbn_bin
CDBN_GAUSSIAN=cdbn_gaus
CAES_SIGMOID=caes_sigmoid
CAES_RELU=caes_relu

# Summary of for loops
#
# Full dataset  (/full)
# random_weights sigmoid -- 1, 10, 50 /rand_sigmoid/[1epoch, 10epoch, 50 epoch]
# random_weights relu -- 1, 10, 50 /rand_relu/[1epoch, 10epoch, 50 epoch]
# cdbn binary -- 1, 10, 50 /cdbn_binary/[1epoch, 10epoch, 50 epoch]
# cdbn gaussian -- 1, 10, 50 ...
# caes sigmoid -- 1, 10, 50
# caes relu -- 1, 10, 50
#
# 10000 first training examples (/reduced)
# random_weights sigmoid -- 1, 10, 50
# random_weights relu -- 1, 10, 50
# cdbn binary -- 1, 10, 50
# cdbn gaussian -- 1, 10, 50
# caes sigmoid -- 1, 10, 50
# caes relu -- 1, 10, 50

# ---------- RANDOM - FULL - SIGMOID ---------
OUT_DIR=$MNIST_OUTPUT_PATH/$FULL_DATASET/$RANDOM_SIGMOID/1epoch
mkdir -p $OUT_DIR
for ((i = 1; i <= $MNIST_FOLDS; i++))
do
	echo "python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR"
	python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR \
		--n_epochs 1
done

mkdir -p $MNIST_OUTPUT_PATH/$FULL_DATASET/$RANDOM_SIGMOID/10epoch
for ((i = 1; i <= $MNIST_FOLDS; i++))
do
	echo "python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR"
	python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR \
		--n_epochs 10
done

mkdir -p $MNIST_OUTPUT_PATH/$FULL_DATASET/$RANDOM_SIGMOID/50epoch
for ((i = 1; i <= $MNIST_FOLDS; i++))
do
	echo "python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR"
	python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR \
		--n_epochs 50
done

# ---------- RANDOM - FULL - RELU ---------
OUT_DIR=$MNIST_OUTPUT_PATH/$FULL_DATASET/$RANDOM_RELU/1epoch
mkdir -p $OUT_DIR
for ((i = 1; i <= $MNIST_FOLDS; i++))
do
	echo "python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR"
	python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR \
		--n_epochs 1 --use_relu
done

mkdir -p $MNIST_OUTPUT_PATH/$FULL_DATASET/$RANDOM_RELU/10epoch
for ((i = 1; i <= $MNIST_FOLDS; i++))
do
	echo "python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR"
	python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR \
		--n_epochs 10 --use_relu
done

mkdir -p $MNIST_OUTPUT_PATH/$FULL_DATASET/$RANDOM_RELU/50epoch
for ((i = 1; i <= $MNIST_FOLDS; i++))
do
	echo "python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR"
	python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR \
		--n_epochs 50 --use_relu
done

# ---------- CDBN BINARY - FULL ---------
OUT_DIR=$MNIST_OUTPUT_PATH/$FULL_DATASET/$CDBN_BINARY/1epoch
mkdir -p $OUT_DIR
for ((i = 1; i <= $MNIST_FOLDS; i++))
do
	echo "python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR"
	python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR \
		--n_epochs 1 \
		--pretrained_model ./pretrained_weights/cdbn/mnist/binary/cdbn_$i.mat
done

mkdir -p $MNIST_OUTPUT_PATH/$FULL_DATASET/$CDBN_BINARY/10epoch
for ((i = 1; i <= $MNIST_FOLDS; i++))
do
	echo "python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR"
	python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR \
		--n_epochs 10 \
		--pretrained_model ./pretrained_weights/cdbn/mnist/binary/cdbn_$i.mat
done

mkdir -p $MNIST_OUTPUT_PATH/$FULL_DATASET/$CDBN_BINARY/50epoch
for ((i = 1; i <= $MNIST_FOLDS; i++))
do
	echo "python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR"
	python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR \
		--n_epochs 50 \
		--pretrained_model ./pretrained_weights/cdbn/mnist/binary/cdbn_$i.mat
done

# ---------- CDBN GAUSSIAN - FULL ---------
OUT_DIR=$MNIST_OUTPUT_PATH/$FULL_DATASET/$CDBN_GAUSSIAN/1epoch
mkdir -p $OUT_DIR
for ((i = 1; i <= $MNIST_FOLDS; i++))
do
	echo "python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR"
	python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR \
		--n_epochs 1 \
		--pretrained_model ./pretrained_weights/cdbn/mnist/gaussian/cdbn_$i.mat
done

mkdir -p $MNIST_OUTPUT_PATH/$FULL_DATASET/$CDBN_GAUSSIAN/10epoch
for ((i = 1; i <= $MNIST_FOLDS; i++))
do
	echo "python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR"
	python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR \
		--n_epochs 10 \
		--pretrained_model ./pretrained_weights/cdbn/mnist/gaussian/cdbn_$i.mat
done

mkdir -p $MNIST_OUTPUT_PATH/$FULL_DATASET/$CDBN_GAUSSIAN/50epoch
for ((i = 1; i <= $MNIST_FOLDS; i++))
do
	echo "python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR"
	python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR \
		--n_epochs 50 \
		--pretrained_model ./pretrained_weights/cdbn/mnist/gaussian/cdbn_$i.mat
done

# ---------- CAES - FULL - SIGMOID ---------
OUT_DIR=$MNIST_OUTPUT_PATH/$FULL_DATASET/$CAES_SIGMOID/1epoch
mkdir -p $OUT_DIR
for ((i = 1; i <= $MNIST_FOLDS; i++))
do
	echo "python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR"
	python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR \
		--n_epochs 1 \
		--pretrained_model ./pretrained_weights/caes/sigmoid/mnist/fold_$i.mat
done

mkdir -p $MNIST_OUTPUT_PATH/$FULL_DATASET/$CAES_SIGMOID/10epoch
for ((i = 1; i <= $MNIST_FOLDS; i++))
do
	echo "python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR"
	python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR \
		--n_epochs 10 \
		--pretrained_model ./pretrained_weights/caes/sigmoid/mnist/fold_$i.mat
done

mkdir -p $MNIST_OUTPUT_PATH/$FULL_DATASET/$CAES_SIGMOID/50epoch
for ((i = 1; i <= $MNIST_FOLDS; i++))
do
	echo "python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR"
	python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR \
		--n_epochs 50 \
		--pretrained_model ./pretrained_weights/caes/sigmoid/mnist/fold_$i.mat
done

# ---------- CAES - FULL - RELU ---------
OUT_DIR=$MNIST_OUTPUT_PATH/$FULL_DATASET/$CAES_RELU/1epoch
mkdir -p $OUT_DIR
for ((i = 1; i <= $MNIST_FOLDS; i++))
do
	echo "python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR"
	python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR \
		--n_epochs 1 --use_relu \
		--pretrained_model ./pretrained_weights/caes/relu/mnist/fold_$i.mat
done

mkdir -p $MNIST_OUTPUT_PATH/$FULL_DATASET/$CAES_RELU/10epoch
for ((i = 1; i <= $MNIST_FOLDS; i++))
do
	echo "python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR"
	python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR \
		--n_epochs 10 --use_relu \
		--pretrained_model ./pretrained_weights/caes/relu/mnist/fold_$i.mat
done

mkdir -p $MNIST_OUTPUT_PATH/$FULL_DATASET/$CAES_RELU/50epoch
for ((i = 1; i <= $MNIST_FOLDS; i++))
do
	echo "python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR"
	python3 cnns/cnn_keras.py $MNIST_PATH/fold_$i.mat $OUT_DIR \
		--n_epochs 50 --use_relu \
		--pretrained_model ./pretrained_weights/caes/relu/mnist/fold_$i.mat
done


# CIFAR -- DON'T FORGET TO ADD --normalize!!!

