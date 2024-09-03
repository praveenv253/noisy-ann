#!/bin/bash -x

set -e

for i in {6..10}; do
	./mnist_train.py --iter $i
	./mnist_test.py --iter $i
	./mnist_train.py --noisy zero --reinit --rotate 60 --iter $i
	./mnist_test.py --noisy zero --reinit --rotate 60 --iter $i
	./compute_covariance.py --iter $i
	./mnist_train.py --noisy --reinit --iter $i
	./mnist_test.py --noisy --reinit --iter $i
	./mnist_train.py --noisy diagonal --reinit --iter $i
	./mnist_test.py --noisy diagonal --reinit --iter $i
done
