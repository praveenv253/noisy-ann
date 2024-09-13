#!/bin/bash -x

set -e

for i in {1..10}; do
	./mnist_train.py --iter $i
	./mnist_test.py --iter $i
	for nl in {0..3}; do
		./mnist_train.py --noisy zero --noisy-layer $nl --reinit --rotate 60 --iter $i
		./mnist_test.py --noisy zero --noisy-layer $nl --reinit --rotate 60 --iter $i
		./compute_covariance.py --noisy-layer $nl --iter $i
		./mnist_train.py --noisy --noisy-layer $nl --reinit --iter $i
		./mnist_test.py --noisy --noisy-layer $nl --reinit --iter $i
		./mnist_train.py --noisy diagonal --noisy-layer $nl --reinit --iter $i
		./mnist_test.py --noisy diagonal --noisy-layer $nl --reinit --iter $i
	done
done
