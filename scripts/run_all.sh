#!/bin/bash -x

set -e

for i in {2..2}; do
	./mnist_train.py --iter $i
	./mnist_test.py --iter $i
	./mnist_train.py --noisy zero --rotate 60 --iter $i
	./mnist_test.py --noisy zero --rotate 60 --iter $i
	./compute_covariance.py --iter $i
	./mnist_train.py --noisy --iter $i
	./mnist_test.py --noisy --iter $i
	./mnist_train.py --noisy diagonal --iter $i
	./mnist_test.py --noisy diagonal --iter $i
done
