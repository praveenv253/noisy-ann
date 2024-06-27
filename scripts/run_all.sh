#!/bin/bash -x

./mnist_train.py
./mnist_train.py --noisy zero --rotate 45
./mnist_train.py --noisy zero --rotate 60
./compute_covariance.py
./mnist_train.py --noisy
./mnist_test.py --rotate all
./mnist_test.py --noisy
