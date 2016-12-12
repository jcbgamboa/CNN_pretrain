#! /bin/bash
# TODO: find a better way to do this

set -v

python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/caes/mnist/relu/full/1epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/caes/mnist/sigmoid/full/1epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/cdbn/mnist/binary/full/1epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/cdbn/mnist/gaussian/full/1epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/random/mnist/relu/full/1epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/random/mnist/sigmoid/full/1epochs

python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/caes/cifar/relu/full/1epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/caes/cifar/sigmoid/full/1epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/cdbn/cifar/binary/full/1epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/cdbn/cifar/gaussian/full/1epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/random/cifar/relu/full/1epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/random/cifar/sigmoid/full/1epochs

python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/caes/mnist/relu/full/10epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/caes/mnist/sigmoid/full/10epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/cdbn/mnist/binary/full/10epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/cdbn/mnist/gaussian/full/10epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/random/mnist/relu/full/10epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/random/mnist/sigmoid/full/10epochs

python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/caes/cifar/relu/full/10epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/caes/cifar/sigmoid/full/10epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/cdbn/cifar/binary/full/10epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/cdbn/cifar/gaussian/full/10epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/random/cifar/relu/full/10epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/random/cifar/sigmoid/full/10epochs

python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/caes/mnist/relu/full/50epochs
python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/caes/mnist/sigmoid/full/50epochs
python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/cdbn/mnist/binary/full/50epochs
python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/cdbn/mnist/gaussian/full/50epochs
python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/random/mnist/relu/full/50epochs
python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/random/mnist/sigmoid/full/50epochs

python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/caes/cifar/relu/full/50epochs
python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/caes/cifar/sigmoid/full/50epochs
python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/cdbn/cifar/binary/full/50epochs
python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/cdbn/cifar/gaussian/full/50epochs
python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/random/cifar/relu/full/50epochs
python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/random/cifar/sigmoid/full/50epochs






python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/caes/mnist/relu/reduced/1epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/caes/mnist/sigmoid/reduced/1epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/cdbn/mnist/binary/reduced/1epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/cdbn/mnist/gaussian/reduced/1epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/random/mnist/relu/reduced/1epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/random/mnist/sigmoid/reduced/1epochs

python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/caes/cifar/relu/reduced/1epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/caes/cifar/sigmoid/reduced/1epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/cdbn/cifar/binary/reduced/1epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/cdbn/cifar/gaussian/reduced/1epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/random/cifar/relu/reduced/1epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/random/cifar/sigmoid/reduced/1epochs

python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/caes/mnist/relu/reduced/10epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/caes/mnist/sigmoid/reduced/10epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/cdbn/mnist/binary/reduced/10epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/cdbn/mnist/gaussian/reduced/10epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/random/mnist/relu/reduced/10epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/random/mnist/sigmoid/reduced/10epochs

python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/caes/cifar/relu/reduced/10epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/caes/cifar/sigmoid/reduced/10epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/cdbn/cifar/binary/reduced/10epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/cdbn/cifar/gaussian/reduced/10epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/random/cifar/relu/reduced/10epochs
python3 scripts/analyse/analyse.py --single_experiment \
		results/cnn/random/cifar/sigmoid/reduced/10epochs

python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/caes/mnist/relu/reduced/50epochs
python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/caes/mnist/sigmoid/reduced/50epochs
python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/cdbn/mnist/binary/reduced/50epochs
python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/cdbn/mnist/gaussian/reduced/50epochs
python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/random/mnist/relu/reduced/50epochs
python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/random/mnist/sigmoid/reduced/50epochs

python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/caes/cifar/relu/reduced/50epochs
python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/caes/cifar/sigmoid/reduced/50epochs
python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/cdbn/cifar/binary/reduced/50epochs
python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/cdbn/cifar/gaussian/reduced/50epochs
python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/random/cifar/relu/reduced/50epochs
python3 scripts/analyse/analyse.py --single_experiment --with_per_epoch \
		results/cnn/random/cifar/sigmoid/reduced/50epochs


python3 scripts/analyse/analyse.py --multiple_experiments results/cnn


