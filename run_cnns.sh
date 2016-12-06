#! /bin/bash

# SUMMARY
#
# Number of epochs for each experiment: 1, 10, 50
# Training set: full, reduced (to 10000 elements)
# Datasets: MNIST, CIFAR
#
# random_weights sigmoid
# random_weights relu
# cdbn binary
# cdbn gaussian
# caes sigmoid
# caes relu



# RANDOM WEIGHTS, FULL DATASET
# ============================

# CIFAR
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/relu random cifar relu 1 full
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/sigmoid random cifar sigmoid 1 full

./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/relu random cifar relu 10 full
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/sigmoid random cifar sigmoid 10 full

./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/relu random cifar relu 50 full
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/sigmoid random cifar sigmoid 50 full

# MNIST
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/relu random mnist relu 1 full
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/sigmoid random mnist sigmoid 1 full

./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/relu random mnist relu 10 full
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/sigmoid random mnist sigmoid 10 full

./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/relu random mnist relu 50 full
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/sigmoid random mnist sigmoid 50 full





# CDBN INITIALIZATION, FULL DATASET
# =================================

# CIFAR
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/binary cdbn cifar sigmoid 1 full
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/gaussian cdbn cifar sigmoid 1 full

./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/binary cdbn cifar sigmoid 10 full
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/gaussian cdbn cifar sigmoid 10 full

./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/binary cdbn cifar sigmoid 50 full
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/gaussian cdbn cifar sigmoid 50 full

# MNIST
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/binary cdbn mnist sigmoid 1 full
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/gaussian cdbn mnist sigmoid 1 full

./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/binary cdbn mnist sigmoid 10 full
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/gaussian cdbn mnist sigmoid 10 full

./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/binary cdbn mnist sigmoid 50 full
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/gaussian cdbn mnist sigmoid 50 full





# CAES INITIALIZATION, FULL DATASET
# =================================

# CIFAR
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/relu caes cifar relu 1 full
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/sigmoid caes cifar sigmoid 1 full

./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/relu caes cifar relu 10 full
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/sigmoid caes cifar sigmoid 10 full

./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/relu caes cifar relu 50 full
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/sigmoid caes cifar sigmoid 50 full

# MNIST
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/relu caes mnist relu 1 full
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/sigmoid caes mnist sigmoid 1 full

./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/relu caes mnist relu 10 full
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/sigmoid caes mnist sigmoid 10 full

./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/relu caes mnist relu 50 full
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/sigmoid caes mnist sigmoid 50 full






# RANDOM WEIGHTS, REDUCED DATASET
# ===============================

# CIFAR
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/relu random cifar relu 1 reduced
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/sigmoid random cifar sigmoid 1 reduced

./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/relu random cifar relu 10 reduced
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/sigmoid random cifar sigmoid 10 reduced

./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/relu random cifar relu 50 reduced
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/sigmoid random cifar sigmoid 50 reduced

# MNIST
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/relu random mnist relu 1 reduced
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/sigmoid random mnist sigmoid 1 reduced

./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/relu random mnist relu 10 reduced
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/sigmoid random mnist sigmoid 10 reduced

./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/relu random mnist relu 50 reduced
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/sigmoid random mnist sigmoid 50 reduced






# CDBN INITIALIZATION, REDUCED DATASET
# ====================================

# CIFAR
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/binary cdbn cifar sigmoid 1 reduced
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/gaussian cdbn cifar sigmoid 1 reduced

./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/binary cdbn cifar sigmoid 10 reduced
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/gaussian cdbn cifar sigmoid 10 reduced

./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/binary cdbn cifar sigmoid 50 reduced
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/gaussian cdbn cifar sigmoid 50 reduced

# MNIST
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/binary cdbn mnist sigmoid 1 reduced
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/gaussian cdbn mnist sigmoid 1 reduced

./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/binary cdbn mnist sigmoid 10 reduced
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/gaussian cdbn mnist sigmoid 10 reduced

./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/binary cdbn mnist sigmoid 50 reduced
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/gaussian cdbn mnist sigmoid 50 reduced






# CAES INITIALIZATION, REDUCED DATASET
# ====================================

# CIFAR
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/relu caes cifar relu 1 reduced
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/sigmoid caes cifar sigmoid 1 reduced

./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/relu caes cifar relu 10 reduced
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/sigmoid caes cifar sigmoid 10 reduced

./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/relu caes cifar relu 50 reduced
./run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/sigmoid caes cifar sigmoid 50 reduced

# MNIST
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/relu caes mnist relu 1 reduced
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/sigmoid caes mnist sigmoid 1 reduced

./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/relu caes mnist relu 10 reduced
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/sigmoid caes mnist sigmoid 10 reduced

./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/relu caes mnist relu 50 reduced
./run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/sigmoid caes mnist sigmoid 50 reduced



