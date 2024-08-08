#!/bin/bash -x

for i in {1..10}; do
	./mnist_train.py --iter $i
	#./mnist_train.py --noisy zero --rotate 45 --iter $i
	./mnist_train.py --noisy zero --rotate 60 --iter $i
	./mnist_test.py --rotate all --iter $i
	./compute_covariance.py --iter $i
	./mnist_train.py --noisy --iter $i
	./mnist_test.py --noisy --iter $i
	./mnist_train.py --noisy diagonal --iter $i
	./mnist_test.py --noisy diagonal --iter $i
done
