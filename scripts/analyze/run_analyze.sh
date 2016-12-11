#! /bin/bash
# TODO: find a better way to do this


python3 scripts/analyze/analyze.py --single_experiment \
		results/cnn/caes/mnist/relu/full/1epochs
python3 scripts/analyze/analyze.py --single_experiment \
		results/cnn/caes/mnist/sigmoid/full/1epochs
python3 scripts/analyze/analyze.py --single_experiment \
		results/cnn/cdbn/mnist/binary/full/1epochs
python3 scripts/analyze/analyze.py --single_experiment \
		results/cnn/cdbn/mnist/gaussian/full/1epochs
python3 scripts/analyze/analyze.py --single_experiment \
		results/cnn/random/mnist/relu/full/1epochs
python3 scripts/analyze/analyze.py --single_experiment \
		results/cnn/random/mnist/sigmoid/full/1epochs

python3 scripts/analyze/analyze.py --single_experiment \
		results/cnn/caes/cifar/relu/full/1epochs
python3 scripts/analyze/analyze.py --single_experiment \
		results/cnn/caes/cifar/sigmoid/full/1epochs
python3 scripts/analyze/analyze.py --single_experiment \
		results/cnn/cdbn/cifar/binary/full/1epochs
python3 scripts/analyze/analyze.py --single_experiment \
		results/cnn/cdbn/cifar/gaussian/full/1epochs
python3 scripts/analyze/analyze.py --single_experiment \
		results/cnn/random/cifar/relu/full/1epochs
python3 scripts/analyze/analyze.py --single_experiment \
		results/cnn/random/cifar/sigmoid/full/1epochs

python3 scripts/analyze/analyze.py --multiple_experiments results/cnn

