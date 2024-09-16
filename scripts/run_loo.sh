#!/bin/bash -x

set -e

i=1  # Arbitrary iteration number

for loo_index in {0..9}; do
	./compute_covariance.py --iter $i --loo $loo_index
	./mnist_train.py --noisy --reinit --iter $i --cov-loo $loo_index
	./mnist_test.py --noisy --reinit --iter $i --cov-loo $loo_index --confusion
done
