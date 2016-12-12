#! /bin/bash

# SUMMARY
#
# Number of epochs for each experiment: 1, 10, 50
# Training set: full, reduced (to 10000 elements)
# Datasets: MNIST, CIFAR
#
# Conditions:
# random_weights sigmoid
# random_weights relu
# cdbn binary
# cdbn gaussian
# caes sigmoid
# caes relu
#
# Each experiment is run in 6 folds of CIFAR and 7 folds of MNIST.



# RANDOM WEIGHTS, FULL DATASET
# ============================

# CIFAR
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/relu random cifar relu 1 full
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/sigmoid random cifar sigmoid 1 full

./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/relu random cifar relu 10 full
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/sigmoid random cifar sigmoid 10 full

./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/relu random cifar relu 50 full
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/sigmoid random cifar sigmoid 50 full

# MNIST
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/relu random mnist relu 1 full
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/sigmoid random mnist sigmoid 1 full

./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/relu random mnist relu 10 full
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/sigmoid random mnist sigmoid 10 full

./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/relu random mnist relu 50 full
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/sigmoid random mnist sigmoid 50 full





# CDBN INITIALIZATION, FULL DATASET
# =================================

# CIFAR
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/binary cdbn cifar sigmoid 1 full
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/gaussian cdbn cifar sigmoid 1 full

./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/binary cdbn cifar sigmoid 10 full
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/gaussian cdbn cifar sigmoid 10 full

./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/binary cdbn cifar sigmoid 50 full
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/gaussian cdbn cifar sigmoid 50 full

# MNIST
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/binary cdbn mnist sigmoid 1 full
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/gaussian cdbn mnist sigmoid 1 full

./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/binary cdbn mnist sigmoid 10 full
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/gaussian cdbn mnist sigmoid 10 full

./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/binary cdbn mnist sigmoid 50 full
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/gaussian cdbn mnist sigmoid 50 full





# CAES INITIALIZATION, FULL DATASET
# =================================

# CIFAR
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/relu caes cifar relu 1 full
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/sigmoid caes cifar sigmoid 1 full

./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/relu caes cifar relu 10 full
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/sigmoid caes cifar sigmoid 10 full

./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/relu caes cifar relu 50 full
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/sigmoid caes cifar sigmoid 50 full

# MNIST
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/relu caes mnist relu 1 full
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/sigmoid caes mnist sigmoid 1 full

./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/relu caes mnist relu 10 full
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/sigmoid caes mnist sigmoid 10 full

./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/relu caes mnist relu 50 full
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/sigmoid caes mnist sigmoid 50 full






# RANDOM WEIGHTS, REDUCED DATASET
# ===============================

# CIFAR
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/relu random cifar relu 1 reduced
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/sigmoid random cifar sigmoid 1 reduced

./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/relu random cifar relu 10 reduced
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/sigmoid random cifar sigmoid 10 reduced

./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/relu random cifar relu 50 reduced
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/random/cifar/sigmoid random cifar sigmoid 50 reduced

# MNIST
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/relu random mnist relu 1 reduced
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/sigmoid random mnist sigmoid 1 reduced

./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/relu random mnist relu 10 reduced
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/sigmoid random mnist sigmoid 10 reduced

./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/relu random mnist relu 50 reduced
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/random/mnist/sigmoid random mnist sigmoid 50 reduced






# CDBN INITIALIZATION, REDUCED DATASET
# ====================================

# CIFAR
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/binary cdbn cifar sigmoid 1 reduced
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/gaussian cdbn cifar sigmoid 1 reduced

./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/binary cdbn cifar sigmoid 10 reduced
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/gaussian cdbn cifar sigmoid 10 reduced

./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/binary cdbn cifar sigmoid 50 reduced
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/cdbn/cifar/gaussian cdbn cifar sigmoid 50 reduced

# MNIST
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/binary cdbn mnist sigmoid 1 reduced
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/gaussian cdbn mnist sigmoid 1 reduced

./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/binary cdbn mnist sigmoid 10 reduced
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/gaussian cdbn mnist sigmoid 10 reduced

./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/binary cdbn mnist sigmoid 50 reduced
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/cdbn/mnist/gaussian cdbn mnist sigmoid 50 reduced






# CAES INITIALIZATION, REDUCED DATASET
# ====================================

# CIFAR
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/relu caes cifar relu 1 reduced
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/sigmoid caes cifar sigmoid 1 reduced

./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/relu caes cifar relu 10 reduced
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/sigmoid caes cifar sigmoid 10 reduced

./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/relu caes cifar relu 50 reduced
./scripts/run_networks/run_cnn_folds.sh ./datasets/cifar_folds 6 ./results/cnn/caes/cifar/sigmoid caes cifar sigmoid 50 reduced

# MNIST
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/relu caes mnist relu 1 reduced
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/sigmoid caes mnist sigmoid 1 reduced

./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/relu caes mnist relu 10 reduced
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/sigmoid caes mnist sigmoid 10 reduced

./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/relu caes mnist relu 50 reduced
./scripts/run_networks/run_cnn_folds.sh ./datasets/mnist_folds 7 ./results/cnn/caes/mnist/sigmoid caes mnist sigmoid 50 reduced



